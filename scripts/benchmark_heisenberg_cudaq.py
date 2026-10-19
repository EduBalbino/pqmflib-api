"""Opt-in comparison of the PQFMLib adapter and a direct CUDA-Q Heisenberg kernel.

Nothing runs on import. Pass a fitted, ideal CUDA-Q MPS Heisenberg map and the
same already-PCA-transformed rows used by the notebook. No precision, bond,
cutoff or observable changes are made. This is a diagnostic, not a new backend.
"""
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import numpy as np
import pandas as pd
from qiskit.circuit import ParameterExpression
from qiskit.synthesis import OneQubitEulerDecomposer

from pqfmlib import HeisenbergProjectiveQFM
from pqfmlib.maps.heisenberg import make_heisenberg_theta_matrix


def build_direct_kernel(cudaq, circuit):
    """Read the original Heisenberg gates without the production transpiler.

    Only initial one-qubit unitaries, XX/YY/ZZ rotations and barriers are
    accepted. Custom QPY circuits containing other operations fail explicitly.
    Initial unitaries use independent ZYZ decomposition (global phase drops out
    of expectations). CUDA-Q exp_pauli uses exp(+i angle P).
    """
    kernel, parameters = cudaq.make_kernel(list)
    qubits = kernel.qalloc(circuit.num_qubits)
    rotations = []
    decomposer = OneQubitEulerDecomposer('ZYZ')
    for entry in circuit.data:
        op = entry.operation
        operands = [qubits[circuit.find_bit(q).index] for q in entry.qubits]
        if op.name == 'barrier':
            continue
        if op.name == 'unitary' and len(operands) == 1:
            theta, phi, lam = decomposer.angles(op.to_matrix())
            kernel.rz(float(lam), operands[0])
            kernel.ry(float(theta), operands[0])
            kernel.rz(float(phi), operands[0])
        elif op.name in ('rxx', 'ryy', 'rzz'):
            rotations.append(op.params[0])
            # Scale in the kernel, independently of the production binder.
            kernel.exp_pauli(parameters[len(rotations) - 1] * -0.5,
                             *operands, op.name[1:].upper())
        else:
            raise ValueError(f'Direct Heisenberg comparison does not support {op.name!r}.')
    return kernel, tuple(rotations)


def bind_direct_angles(rotations, param_order, theta):
    rows = []
    for values in theta:
        bindings = dict(zip(param_order, values))
        rows.append([
            float(angle.bind({p: bindings[p] for p in angle.parameters}))
            if isinstance(angle, ParameterExpression) and angle.parameters else float(angle)
            for angle in rotations
        ])
    return rows


def benchmark_heisenberg_cudaq_direct(qfm, X, repeats=3):
    """Compare identical observables/circuit/settings; warm-ups are excluded.

    Adapter timing includes transform validation, scaling and parameter binding.
    Direct timing uses pre-bound angles and calls CUDA-Q itself. Both timings
    include a fresh worker and target-context setup per batch, matching the
    production MPS lifecycle. Neither includes PCA, fit or kernel construction.
    The direct path isolates adapter overhead; it does not compare CPU/GPU.
    """
    if not isinstance(qfm, HeisenbergProjectiveQFM):
        raise ValueError('Expected a fitted HeisenbergProjectiveQFM.')
    if (qfm.framework != 'cudaq' or not qfm.mps or qfm.expectation_method != 'exact'
            or not qfm.ideal or not qfm.simulation or qfm.fakebackend):
        raise ValueError('Requires ideal CUDA-Q MPS with expectation_method="exact".')
    if not hasattr(qfm, '_prepared_execution') or qfm._prepared_execution is None:
        raise ValueError('Fit the map first.')
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not len(X) or X.shape[1] != qfm.n_features_in_ or not np.isfinite(X).all():
        raise ValueError('Provide finite, nonempty rows with the fitted input width.')
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError('repeats must be a positive integer.')
    executor = qfm.backend
    tolerance = 2e-5 if executor.precision == "single" else 1e-10
    cudaq = executor._cudaq
    prepared = qfm._prepared_execution
    setup_started = perf_counter()
    kernel, rotations = build_direct_kernel(cudaq, prepared.qc_t)
    theta, _ = make_heisenberg_theta_matrix(
        X, qfm.q_enc, features_per_block=qfm.theta_info['features_per_block'],
        use_tanh_scaling=qfm.use_tanh_scaling)
    angles = bind_direct_angles(rotations, prepared.param_order, theta)
    observables = []
    for (observable,) in prepared.obs_isa_broadcast:
        terms = observable.simplify().to_list()
        if len(terms) != 1 or not np.isclose(terms[0][1], 1.0):
            raise ValueError('Direct comparison requires unit-weight Pauli observables.')
        label = terms[0][0]
        if set(label) == {'I'}:
            raise ValueError('Identity-only observables are not part of this benchmark.')
        term = 1.0 * cudaq.spin.i(qfm.q_enc - 1)
        for q, axis in enumerate(reversed(label)):
            if axis != 'I':
                term *= getattr(cudaq.spin, axis.lower())(q)
        observables.append(term)
    setup_seconds = perf_counter() - setup_started

    def direct_worker():
        # Match production flags and fresh thread-local native simulator.
        with executor._target_context(qfm.q_enc, batched_mps_observe=True):
            return np.asarray([
                [result.expectation() for result in cudaq.observe(
                    kernel, observables, row, shots_count=-1)] for row in angles
            ])

    def direct():
        with ThreadPoolExecutor(max_workers=1) as worker:
            return worker.submit(direct_worker).result()

    paths = {'pqfmlib_adapter': lambda: qfm.transform(X), 'cudaq_direct': direct}
    warmups = {}
    reference = None
    for name, run in paths.items():
        start = perf_counter()
        result = run()
        warmups[name] = perf_counter() - start
        if reference is None:
            reference = result.copy()
        np.testing.assert_allclose(result, reference, atol=tolerance, rtol=tolerance, equal_nan=False)
    timings = {name: [] for name in paths}
    errors = {name: 0.0 for name in paths}
    for repeat in range(repeats):
        # Alternate order to avoid always timing one route first.
        order = list(paths) if repeat % 2 == 0 else list(reversed(paths))
        for name in order:
            start = perf_counter()
            result = paths[name]()
            elapsed = perf_counter() - start
            np.testing.assert_allclose(result, reference, atol=tolerance, rtol=tolerance, equal_nan=False)
            timings[name].append(elapsed)
            errors[name] = max(errors[name], float(np.max(np.abs(result - reference))))
            print(f'{name}: {repeat + 1}/{repeats}: {elapsed:.3f} s', flush=True)
    return pd.DataFrame([
        dict(path=name, q_enc=qfm.q_enc, n_samples=len(X), n_observables=len(observables),
             repeats=repeats, precision=executor.precision, target=executor.target,
             mps_max_bond_dimension=executor.mps_max_bond_dimension,
             mps_truncation_threshold=executor.mps_truncation_threshold,
             warmup_seconds=warmups[name], median_seconds=float(np.median(timings[name])),
             seconds_per_sample=float(np.median(timings[name])) / len(X),
             max_abs_error=errors[name], direct_setup_seconds=setup_seconds,
             includes_parameter_binding=name == 'pqfmlib_adapter')
        for name in paths
    ])


def benchmark_heisenberg_cudaq_shots(qfm, X, repeats=3, *, shots=None,
                                     per_row=True):
    """Compare grouped MPS shots with the old direct scalar-observe baseline.

    Returns one record per route/method/repetition, including a labelled warm-up
    (repeat=-1). Defaults to a fresh execution per row, like row-wise transform.
    Fit, input preprocessing, kernel construction and direct parameter binding
    are excluded. Adapter binding, worker creation and target setup are included.
    Native observe time includes factorization/contractions/sampling; it is NOT
    an isolated measurement of sampling or contraction time. No speed claim or
    statistical convergence assertion is made. Shot errors are reported against
    the adapter's analytic MPS result, which can itself have truncation error.
    Does not change the fitted map, its method, seed, precision or cutoffs.
    """
    from numbers import Integral
    from pqfmlib.core.cudaq_execution import CudaqExecutor, _compile_job

    if not isinstance(qfm, HeisenbergProjectiveQFM):
        raise ValueError('Expected a fitted HeisenbergProjectiveQFM.')
    if (qfm.framework != 'cudaq' or not qfm.mps or not qfm.ideal
            or not qfm.simulation or qfm.fakebackend):
        raise ValueError('Requires ideal CUDA-Q MPS.')
    if getattr(qfm, '_prepared_execution', None) is None:
        raise ValueError('Fit the map first.')
    for name, value in (('repeats', repeats), ('shots', qfm.shots if shots is None else shots)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    shots = int(qfm.shots if shots is None else shots)
    if not isinstance(per_row, bool):
        raise ValueError('per_row must be a boolean.')
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not len(X) or X.shape[1] != qfm.n_features_in_ or not np.isfinite(X).all():
        raise ValueError('Provide finite, nonempty rows with the fitted input width.')
    prepared = qfm._prepared_execution
    theta, _ = make_heisenberg_theta_matrix(
        X, qfm.q_enc, features_per_block=qfm.theta_info['features_per_block'],
        use_tanh_scaling=qfm.use_tanh_scaling)
    executor = CudaqExecutor(mps=True, seed=qfm.seed, precision=qfm.precision,
                             mps_max_bond_dimension=qfm.mps_max_bond_dimension,
                             mps_truncation_threshold=qfm.mps_truncation_threshold)
    tolerance = 2e-5 if executor.precision == "single" else 1e-10
    cudaq = executor._cudaq
    # Construct both kernels before timing. The direct kernel bypasses our
    # transpiler and parameter binder; the direct shots lifecycle is deliberately
    # the old scalar production path, providing a baseline for grouped sampling.
    with executor._target_context(qfm.q_enc):
        executor._compiled = _compile_job(cudaq, prepared)
        executor._prepared = prepared
        kernel, rotations = build_direct_kernel(cudaq, prepared.qc_t)
        observables = []
        for (observable,) in prepared.obs_isa_broadcast:
            terms = observable.simplify().to_list()
            if len(terms) != 1 or not np.isclose(terms[0][1], 1.0):
                raise ValueError('Direct comparison requires unit-weight Pauli observables.')
            label = terms[0][0]
            if set(label) == {'I'}:
                raise ValueError('Identity-only observables are not supported by this diagnostic.')
            term = 1.0 * cudaq.spin.i(qfm.q_enc - 1)
            for q, axis in enumerate(reversed(label)):
                if axis != 'I':
                    term *= getattr(cudaq.spin, axis.lower())(q)
            observables.append(term)
        state_kernel = None
        if qfm.q_enc > 1:
            state_kernel, state_arg = cudaq.make_kernel(cudaq.State)
            state_kernel.qalloc(state_arg)
    angles = bind_direct_angles(rotations, prepared.param_order, theta)
    chunks = [slice(i, i + 1) for i in range(len(X))] if per_row else [slice(None)]
    metric_names = ('observe_calls', 'get_state_calls', 'state_kernel_initializations',
                    'state_seconds', 'observe_seconds', 'binding_seconds',
                    'sample_calls', 'sample_seconds')

    def direct_worker(rows, method):
        metrics = dict.fromkeys(metric_names, 0)
        output = []
        with executor._target_context(qfm.q_enc, batched_mps_observe=method == 'exact'):
            if method == 'shots':
                cudaq.set_random_seed(qfm.seed)
            for row in rows:
                if method == 'exact':
                    start = perf_counter()
                    results = cudaq.observe(kernel, observables, row, shots_count=-1)
                    metrics['observe_seconds'] += perf_counter() - start
                    metrics['observe_calls'] += 1
                else:
                    state = None
                    try:
                        if state_kernel is not None:
                            start = perf_counter()
                            state = cudaq.get_state(kernel, row)
                            metrics['state_seconds'] += perf_counter() - start
                            metrics['get_state_calls'] += 1
                        start = perf_counter()
                        results = [cudaq.observe(
                            state_kernel if state_kernel is not None else kernel,
                            observable, state if state_kernel is not None else row,
                            shots_count=shots) for observable in observables]
                        metrics['observe_seconds'] += perf_counter() - start
                        metrics['observe_calls'] += len(observables)
                        if state_kernel is not None:
                            metrics['state_kernel_initializations'] += len(observables)
                    finally:
                        del state
                output.append([result.expectation() for result in results])
        return np.asarray(output), metrics

    def run(path, method):
        metrics = dict.fromkeys(metric_names, 0)
        outputs = []
        start = perf_counter()
        for chunk in chunks:
            if path == 'pqfmlib_adapter':
                output, _ = executor.execute(prepared, theta[chunk],
                                              expectation_method=method, shots=shots)
                stats = executor.last_execution_stats
            else:
                with ThreadPoolExecutor(max_workers=1) as worker:
                    output, stats = worker.submit(direct_worker, angles[chunk], method).result()
            outputs.append(output)
            for name in metric_names:
                metrics[name] += stats[name]
        return np.concatenate(outputs), perf_counter() - start, metrics

    routes = [(path, method) for method in ('exact', 'shots')
              for path in ('pqfmlib_adapter', 'cudaq_direct')]
    records = []
    reference = None
    for repeat in range(-1, repeats):
        order = routes if repeat == -1 or repeat % 2 == 0 else routes[::-1]
        for path, method in order:
            print(f'{path}/{method}: repeat={repeat}, starting', flush=True)
            result, elapsed, metrics = run(path, method)
            if reference is None:
                reference = result.copy()
            if method == 'exact':
                np.testing.assert_allclose(result, reference, atol=tolerance, rtol=tolerance)
            if result.shape != reference.shape or not np.isfinite(result).all():
                raise ValueError('Invalid feature result from diagnostic route.')
            records.append(dict(
                path=path, expectation_method=method, repeat=repeat, warmup=repeat == -1,
                per_row=per_row, n_samples=len(X), execute_calls=len(chunks),
                q_enc=qfm.q_enc, R=qfm.R, n_observables=len(observables),
                num_blocks=qfm.theta_info['num_blocks'],
                n_distinct_pauli_terms=len(executor._compiled.observables),
                shots=shots if method == 'shots' else None, seed=qfm.seed,
                precision=executor.precision, target=executor.target,
                mps_max_bond_dimension=executor.mps_max_bond_dimension,
                mps_truncation_threshold=executor.mps_truncation_threshold,
                wall_seconds=elapsed, seconds_per_sample=elapsed / len(X),
                max_abs_error=float(np.max(np.abs(result - reference))), **metrics))
            print(f'{path}/{method}: {elapsed:.3f} s', flush=True)
    return pd.DataFrame(records)
