"""Lazy CUDA-Q GPU adapter for the existing Qiskit circuit representation.

No PQFM physics is defined here. Circuits are lowered without routing, keeping
XX/YY/ZZ rotations native, then converted to a parameterized CUDA-Q kernel.
``exact`` evaluates observables analytically on either target; ``statevector``
is an alias restricted to the full-vector target. ``shots`` retains sampling.
MPS cutoffs apply to singular values (not Aer's discarded-weight criterion).
"""

from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib import import_module
from numbers import Integral
import os
from threading import RLock
from time import perf_counter

import numpy as np
from qiskit import transpile
from qiskit.circuit import ParameterExpression
from pqfmlib.core.framework import validate_precision


# CUDA-Q targets and MPS environment settings are process-global. Serialize
# PQFMLib operations and restore the caller's target/settings, even on failure.
_CUDA_Q_LOCK = RLock()
_PAULI_ROTATIONS = {"rxx": "XX", "ryy": "YY", "rzz": "ZZ"}
_BASIS = ("rx", "ry", "rz", "h", "x", "cx", *_PAULI_ROTATIONS)


def _load_cudaq():
    try:
        return import_module("cudaq")
    except ImportError as exc:
        raise ImportError(
            "framework='cudaq' requires optional CUDA-Q dependencies. "
            "Install pqfmlib[cudaq] and the compatible NVIDIA GPU runtime."
        ) from exc


def _validate_unitary_circuit(circuit):
    if circuit.num_clbits:
        raise ValueError("CUDA-Q requires a unitary circuit without classical bits.")
    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name in {"measure", "reset", "initialize", "kraus", "superop", "delay"}:
            raise ValueError(f"CUDA-Q does not support circuit instruction {operation.name!r}.")
        if getattr(operation, "condition", None) is not None:
            raise ValueError("CUDA-Q does not support classically controlled instructions.")


def _conservative_mps_bond(circuit):
    """Bound every intermediate Schmidt rank, starting from a product state.

    In the adapter's lowered basis CX and Pauli rotations have operator
    Schmidt rank at most two. Each gate crossing a cut therefore doubles
    its bound; local gates leave it unchanged. Physical cut dimensions cap
    the bound. Parameter values and numerical truncation are never assumed.
    """
    _validate_unitary_circuit(circuit)
    lowered = transpile(circuit, basis_gates=list(_BASIS), optimization_level=0)
    n = lowered.num_qubits
    exponents = [0] * max(0, n - 1)
    for instruction in lowered.data:
        name = instruction.operation.name
        if name == "barrier":
            continue
        if name not in _BASIS:
            raise ValueError(f"Unsupported CUDA-Q instruction: {name!r}.")
        if len(instruction.qubits) < 2:
            continue
        a, b = sorted(lowered.find_bit(q).index for q in instruction.qubits)
        for cut in range(a, b):
            exponents[cut] = min(exponents[cut] + 1, cut + 1, n - cut - 1)
    bound = 1 << max(exponents, default=0)
    if bound > 2**31 - 1:
        raise ValueError(
            "Conservative MPS bond exceeds CUDA-Q's int32 limit; set an explicit "
            "mps_max_bond_dimension if approximation is acceptable."
        )
    return bound


@dataclass
class _CompiledJob:
    kernel: object
    expressions: tuple
    observables: list
    observable_terms: tuple
    num_qubits: int
    pauli_labels: tuple = ()
    sampling_groups: object = None
    shots_aggregate: object = None


def _aggregate_observables(cudaq, observables):
    """Avoid list-observe creating a result with full counts for every term."""
    operator = cudaq.SpinOperator.empty()
    terms = []
    for observable in observables:
        canonical = observable.canonicalize()
        if canonical.term_count != 1:
            raise ValueError('Expected one distinct Pauli term per compiled observable.')
        operator += canonical
        terms.append(canonical if isinstance(canonical, cudaq.SpinOperatorTerm)
                     else next(iter(canonical)))
    return operator, tuple(terms)


def _compile_job(cudaq, prepared, *, measurement_basis=None):
    from pqfmlib.core.execution import validate_circuit_parameter_compatibility

    circuit = prepared.qc_t
    validate_circuit_parameter_compatibility(circuit, prepared.param_order)
    _validate_unitary_circuit(circuit)
    try:
        lowered = transpile(circuit, basis_gates=list(_BASIS), optimization_level=0)
    except Exception as exc:
        raise ValueError("Circuit cannot be lowered to the CUDA-Q unitary gate basis.") from exc
    _validate_unitary_circuit(lowered)
    kernel, angles = cudaq.make_kernel(list)
    qubits = kernel.qalloc(lowered.num_qubits)
    expressions = []
    for instruction in lowered.data:
        operation = instruction.operation
        name = operation.name
        if name == "barrier":
            continue
        if name not in _BASIS:
            raise ValueError(f"Unsupported CUDA-Q instruction: {name!r}.")
        operands = [qubits[lowered.find_bit(q).index] for q in instruction.qubits]
        if name in ("rx", "ry", "rz") or name in _PAULI_ROTATIONS:
            angle = operation.params[0]
            if name in _PAULI_ROTATIONS:
                # Qiskit RPP(theta) = exp(-i theta PP / 2), whereas CUDA-Q
                # exp_pauli(phi, PP) = exp(+i phi PP). Scale the expression
                # BEFORE binding, avoiding extra runtime arithmetic in MLIR.
                angle = -0.5 * angle
            if isinstance(angle, ParameterExpression) and angle.parameters:
                expressions.append(angle)
                angle = angles[len(expressions) - 1]
            else:
                angle = float(angle)
            if name in _PAULI_ROTATIONS:
                kernel.exp_pauli(angle, *operands, _PAULI_ROTATIONS[name])
            else:
                getattr(kernel, name)(angle, *operands)
        else:
            getattr(kernel, name)(*operands)
    if measurement_basis is not None:
        _measure_basis(kernel, qubits, measurement_basis)
    # Global phase has no effect on any expectation value. Never materialize
    # a dense circuit unitary or full state, including for the MPS target.
    observables = []
    observable_terms = []
    term_indices = {}
    for (observable,) in prepared.obs_isa_broadcast:
        if observable.num_qubits != lowered.num_qubits:
            raise ValueError("CUDA-Q observable width must match the prepared circuit.")
        terms = []
        for label, coefficient in observable.simplify().to_list():
            if abs(complex(coefficient).imag) > 1e-12:
                raise ValueError("CUDA-Q requires Hermitian observables with real Pauli coefficients.")
            coefficient = float(np.real(coefficient))
            if coefficient == 0:
                continue
            if label == "I" * lowered.num_qubits:
                # Identity is exactly one and needs no simulation. Scalar
                # identity-only observe is unsupported by CUDA-Q MPS.
                terms.append((-1, coefficient))
                continue
            if label not in term_indices:
                term = 1.0 * cudaq.spin.i(lowered.num_qubits - 1)
                # Qiskit labels put qubit zero at the right-hand end.
                for qubit, axis in enumerate(reversed(label)):
                    if axis != "I":
                        term = term * getattr(cudaq.spin, axis.lower())(qubit)
                term_indices[label] = len(observables)
                observables.append(term)
            terms.append((term_indices[label], coefficient))
        observable_terms.append(tuple(terms))
    # Evaluate distinct, unit-weight Pauli terms. Reconstruct weighted sums
    # ourselves: CUDA-Q 0.16 list-observe mishandles weighted overlapping sums.
    return _CompiledJob(kernel, tuple(expressions), observables,
                        tuple(observable_terms), lowered.num_qubits,
                        pauli_labels=tuple(term_indices))


def _group_paulis(labels):
    """Stable greedy grouping by qubit-wise compatibility (Qiskit label order)."""
    groups = []
    for index, label in enumerate(labels):
        for basis, members in groups:
            if all(a == 'I' or b == 'I' or a == b for a, b in zip(basis, label)):
                for q, axis in enumerate(label):
                    if axis != 'I':
                        basis[q] = axis
                members.append(index)
                break
        else:
            groups.append((list(label), [index]))
    return [(''.join(basis), tuple(members)) for basis, members in groups]


def _measure_basis(kernel, qubits, basis):
    for q, axis in enumerate(reversed(basis)):
        if axis == 'X':
            kernel.h(qubits[q])
        elif axis == 'Y':
            kernel.rx(np.pi / 2, qubits[q])
    kernel.mz(qubits)


def _compile_sampling_groups(cudaq, prepared, compiled):
    groups = []
    for basis, members in _group_paulis(compiled.pauli_labels):
        if compiled.num_qubits > 1:
            kernel, state = cudaq.make_kernel(cudaq.State)
            _measure_basis(kernel, kernel.qalloc(state), basis)
        else:
            # CUDA-Q 0.16 cannot qalloc a one-qubit MPS State.
            kernel = _compile_job(cudaq, prepared, measurement_basis=basis).kernel
        supports = tuple((index, tuple(q for q, axis in enumerate(
            reversed(compiled.pauli_labels[index])) if axis != 'I')) for index in members)
        groups.append((kernel, supports))
    return tuple(groups)


def _sample_expectations(counts, supports, shots, num_qubits):
    """CUDA-Q bitstrings list q0 first; Pauli expectations are shot parities."""
    counts = list(counts.items())
    if sum(count for _, count in counts) != shots:
        raise RuntimeError('CUDA-Q returned an unexpected number of measurement shots.')
    if any(len(bits) != num_qubits or set(bits) - {'0', '1'} for bits, _ in counts):
        raise RuntimeError('CUDA-Q returned invalid measurement bitstrings.')
    return [(index, sum(count * (-1 if sum(bits[q] == '1' for q in support) % 2 else 1)
                        for bits, count in counts) / shots) for index, support in supports]


class CudaqExecutor:
    """Execute unitary PQFMs on cuStateVec or cuTensorNet in single/double precision.

    ``mps_max_bond_dimension=None`` permits the full Schmidt rank instead of
    CUDA-Q's default cap of 64. The existing truncation option is used as an
    absolute singular-value cutoff; relative truncation is effectively disabled. Finite
    bond limits/cutoffs introduce approximations that can differ from Aer.
    Direct concurrent use of CUDA-Q outside PQFMLib must be synchronized by
    the application because CUDA-Q exposes a process-global target.
    """

    def __init__(self, *, mps=False, seed=42, mps_max_bond_dimension=None,
                 mps_truncation_threshold=1e-16, precision="double"):
        validate_precision(precision)
        if mps_max_bond_dimension is not None and (
            isinstance(mps_max_bond_dimension, bool)
            or not isinstance(mps_max_bond_dimension, Integral)
            or not 1 <= mps_max_bond_dimension <= 2**31 - 1
        ):
            raise ValueError("CUDA-Q mps_max_bond_dimension must be a positive 32-bit integer or None.")
        if not np.isfinite(mps_truncation_threshold) or not 0 <= mps_truncation_threshold < 1:
            raise ValueError("CUDA-Q mps_truncation_threshold must be finite and in [0, 1).")
        self.target = "tensornet-mps" if mps else "nvidia"
        self.precision = precision
        self.mps = mps
        self.seed = seed
        self.mps_max_bond_dimension = mps_max_bond_dimension
        self.mps_truncation_threshold = mps_truncation_threshold
        self._prepared = None
        self._compiled = None
        self._bond_prepared = None
        self._automatic_bond = None
        self.last_execution_stats = None
        with _CUDA_Q_LOCK:
            self._cudaq = _load_cudaq()
            if self._cudaq.num_available_gpus() < 1:
                raise RuntimeError(f"CUDA-Q target '{self.target}' requires an accessible NVIDIA GPU.")
            # Initialize the target only when the circuit width is known.
            # Native simulators may retain settings from first initialization.

    @contextmanager
    def _target_context(self, num_qubits, *, batched_mps_observe=False, automatic_bond=None):
        with _CUDA_Q_LOCK:
            cudaq = self._cudaq
            previous_target = cudaq.get_target()
            settings = {}
            if self.mps:
                # CUDA-Q reads an int32 bond limit; physical ranks this large
                # already exceed available GPU memory by many orders of magnitude.
                full_rank = min(1 << (num_qubits // 2), 2**31 - 1)
                settings = {
                    "CUDAQ_MPS_MAX_BOND": str(self.mps_max_bond_dimension or automatic_bond or full_rank),
                    # CUDA-Q requires strictly positive cutoffs. This floor
                    # is below any numerically meaningful singular value.
                    "CUDAQ_MPS_ABS_CUTOFF": str(max(self.mps_truncation_threshold, 1e-300)),
                    "CUDAQ_MPS_RELATIVE_CUTOFF": "1e-300",
                }
                if batched_mps_observe:
                    # CUDA-Q's default TN observe path only records the total
                    # expectation of a sum, so Python list-observe cannot read
                    # individual terms. The path-reuse implementation records
                    # per-term results AND shares contraction planning. Enable
                    # before the thread-local simulator is first initialized.
                    # See CUDA-Q simulator_cutensornet.inc::observe().
                    settings["CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE"] = "TRUE"
            previous_settings = {key: os.environ.get(key) for key in settings}
            try:
                os.environ.update(settings)
                try:
                    cudaq.set_target(self.target, option="fp32" if self.precision == "single" else "fp64")
                except Exception as exc:
                    raise RuntimeError(
                        f"CUDA-Q GPU target '{self.target}' is unavailable; "
                        "check the NVIDIA driver and CUDA/cuQuantum installation."
                    ) from exc
                yield
            finally:
                for key, value in previous_settings.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
                cudaq.set_target(previous_target)

    def execute(self, prepared, theta_values_all, *, expectation_method="shots", shots=4096):
        self.last_execution_stats = None
        if self.mps:
            # CUDA-Q caches MPSSettings in a thread-local native simulator.
            # A fresh worker prevents a prior circuit's bond limit/cutoffs
            # (including those set by another CUDA-Q user) from being reused.
            with ThreadPoolExecutor(max_workers=1) as worker:
                return worker.submit(
                    self._execute, prepared, theta_values_all,
                    expectation_method=expectation_method, shots=shots,
                ).result()
        return self._execute(prepared, theta_values_all,
                             expectation_method=expectation_method, shots=shots)

    def _execute(self, prepared, theta_values_all, *, expectation_method, shots):
        if expectation_method not in ("shots", "statevector", "exact"):
            raise ValueError("CUDA-Q expectation_method must be 'shots', 'statevector' or 'exact'.")
        if expectation_method == "statevector" and self.mps:
            raise ValueError("statevector requires mps=False; use expectation_method='exact' for CUDA-Q MPS.")
        if expectation_method == "shots" and (
            isinstance(shots, bool) or not isinstance(shots, Integral) or shots < 1
        ):
            raise ValueError("CUDA-Q shots must be a positive integer.")
        theta = np.asarray(theta_values_all, dtype=float)
        if theta.ndim != 2 or theta.shape[1] != len(prepared.param_order):
            raise ValueError("CUDA-Q theta must be a 2D matrix with one column per circuit parameter.")
        if not np.isfinite(theta).all():
            raise ValueError("CUDA-Q circuit parameters must be finite.")
        batched_mps_observe = self.mps and expectation_method == "exact"
        started = perf_counter()
        automatic_bond = None
        if self.mps and self.mps_max_bond_dimension is None:
            if self._bond_prepared is not prepared:
                self._automatic_bond = _conservative_mps_bond(prepared.qc_t)
                self._bond_prepared = prepared
                print(
                    f"[pqfmlib CUDA-Q MPS] Teto conservador calculado: "
                    f"{self._automatic_bond} ({prepared.qc_t.num_qubits} qubits; "
                    "mps_max_bond_dimension=None). Critério de truncamento preservado.",
                    flush=True,
                )
            automatic_bond = self._automatic_bond
        stats = dict(n_samples=len(theta), observe_calls=0, get_state_calls=0,
                     precision=self.precision,
                     sample_calls=0, sample_seconds=0.0, n_measurement_groups=0,
                     n_observables=len(prepared.obs_isa_broadcast),
                     shots=int(shots) if expectation_method == "shots" else None,
                     expectation_method=expectation_method,
                     mps_effective_max_bond=(self.mps_max_bond_dimension or automatic_bond) if self.mps else None,
                     binding_seconds=0.0, observe_seconds=0.0, state_seconds=0.0,
                     execution_path="mps_batched_exact" if batched_mps_observe else
                     ("mps_grouped_shots" if self.mps else "statevector"))
        with self._target_context(prepared.qc_t.num_qubits,
                                  batched_mps_observe=batched_mps_observe,
                                  automatic_bond=automatic_bond):
            compile_started = perf_counter()
            if self._prepared is not prepared:
                compiled = _compile_job(self._cudaq, prepared)
                self._prepared, self._compiled = prepared, compiled
            compiled = self._compiled
            aggregate_shots = not self.mps and expectation_method == "shots"
            if aggregate_shots and compiled.observables and compiled.shots_aggregate is None:
                compiled.shots_aggregate = _aggregate_observables(self._cudaq, compiled.observables)
            stats["n_distinct_pauli_terms"] = len(compiled.observables)
            stats["state_kernel_initializations"] = 0
            if self.mps and not batched_mps_observe:
                if compiled.sampling_groups is None:
                    compiled.sampling_groups = _compile_sampling_groups(self._cudaq, prepared, compiled)
                stats["n_measurement_groups"] = len(compiled.sampling_groups)
            stats["compile_seconds"] = perf_counter() - compile_started
            positions = {p.name: i for i, p in enumerate(prepared.param_order)}
            features = np.zeros((len(theta), len(compiled.observable_terms)), dtype=float)
            if not compiled.observables:
                features[:] = [sum(coefficient for _, coefficient in terms)
                               for terms in compiled.observable_terms]
                stats["total_seconds"] = perf_counter() - started
                self.last_execution_stats = stats
                return features, prepared.obs_metadata
            if expectation_method == "shots":
                self._cudaq.set_random_seed(self.seed)
            for row, values in enumerate(theta):
                binding_started = perf_counter()
                angles = [float(expression.bind({
                    p: values[positions[p.name]] for p in expression.parameters
                })) for expression in compiled.expressions]
                stats["binding_seconds"] += perf_counter() - binding_started
                observe_options = {
                    "shots_count": int(shots) if expectation_method == "shots" else -1,
                }
                observe_started = perf_counter()
                if batched_mps_observe:
                    # One kernel execution and MPS factorization per sample.
                    # Do not get_state/qalloc(State) for every observable: that
                    # copies tensors and triggers another MPS factorization.
                    results = self._cudaq.observe(
                        compiled.kernel, compiled.observables, angles, **observe_options,
                    )
                    stats["observe_calls"] += 1
                elif self.mps:
                    expectations = np.zeros(len(compiled.observables))
                    state = None
                    try:
                        if compiled.num_qubits > 1:
                            state_started = perf_counter()
                            state = self._cudaq.get_state(compiled.kernel, angles)
                            stats["state_seconds"] += perf_counter() - state_started
                            stats["get_state_calls"] += 1
                        for kernel, supports in compiled.sampling_groups:
                            sample_started = perf_counter()
                            counts = self._cudaq.sample(
                                kernel, state if compiled.num_qubits > 1 else angles,
                                shots_count=int(shots))
                            stats["sample_seconds"] += perf_counter() - sample_started
                            stats["sample_calls"] += 1
                            stats["state_kernel_initializations"] += int(compiled.num_qubits > 1)
                            for index, value in _sample_expectations(
                                    counts, supports, int(shots), compiled.num_qubits):
                                expectations[index] = value
                    finally:
                        del state
                else:
                    results = self._cudaq.observe(
                        compiled.kernel,
                        compiled.shots_aggregate[0] if aggregate_shots else compiled.observables,
                        angles, **observe_options,
                    )
                    stats["observe_calls"] += 1
                if not self.mps or batched_mps_observe:
                    stats["observe_seconds"] += perf_counter() - observe_started
                    expectations = ([results.expectation(term) for term in compiled.shots_aggregate[1]]
                                    if aggregate_shots else [result.expectation() for result in results])
                features[row] = [sum(coefficient * (1.0 if index == -1 else expectations[index])
                                     for index, coefficient in terms)
                                 for terms in compiled.observable_terms]
        stats["total_seconds"] = perf_counter() - started
        self.last_execution_stats = stats
        return features, prepared.obs_metadata
