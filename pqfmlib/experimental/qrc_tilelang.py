"""Exact GPU statevectors (TileLang) for qreservoir QRC models.

Executes the same circuit qreservoir/Qulacs would run per row — encoder
followed by reservoir dynamics — but batched on the GPU (FP64, exact
expectations, no sampling), reusing the kernels in
``heisenberg_tilelang``. Supported reservoirs use only single-qubit
rotations and CZ gates (e.g. ``RotationReservoir``); anything else raises
``ValueError`` instead of silently approximating.
"""

import numpy as np
import torch

from pqfmlib.experimental.heisenberg_tilelang import apply_gate, local_expectations

_CZ_MATRIX = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)


def _gate_matrix(gate):
    """Full (dim, dim) matrix for a rotation or CZ gate from qulacs."""
    name = gate.get_name()
    if name == "CZ":
        return _CZ_MATRIX
    if name in ("X-rotation", "Y-rotation", "Z-rotation"):
        return np.asarray(gate.get_matrix(), dtype=complex)
    raise ValueError(f"TileLang QRC backend does not support gate {name!r}")


def build_gate_plan(model):
    """Gate template for a qreservoir ``QELModel``.

    Returns ``(plan, n_qubits)`` where ``plan`` is a list of
    ``(width, q0, q1, matrix_or_None, param_index_or_None)``. Parametric
    encoder rotations (``X-rotation`` in the encoder circuit) carry
    ``matrix=None`` and the input-feature index whose angle they use;
    all other gates carry their fixed matrix.
    """
    reservoir = model.reservoir
    encoder = reservoir.encoder
    if encoder is None:
        raise ValueError("QRC model has no encoder connected")
    ancilla = reservoir.ancilla_num
    n_qubits = reservoir.total_size
    plan = []
    template = encoder.get_circuit(np.zeros(encoder.feature_num))
    qpf = getattr(encoder, "qubits_per_feature", 1)
    axis_of = {"X-rotation": "X", "Y-rotation": "Y", "Z-rotation": "Z"}
    for i in range(template.get_gate_count()):
        gate = template.get_gate(i)
        targets = gate.get_target_index_list()
        controls = gate.get_control_index_list()
        if gate.get_name() in axis_of and not controls:
            # Parametric encoder rotation: axis fixed per position (library
            # random choice for CHE, RotX for Exp), angle = input feature.
            plan.append((1, targets[0], 0, None,
                         (int(targets[0]) // qpf, axis_of[gate.get_name()])))
        elif gate.get_name() == "CZ" and len(targets) == 1 and len(controls) == 1:
            plan.append((2, controls[0], targets[0], _CZ_MATRIX, None))
        else:
            raise ValueError(f"Unsupported encoder gate {gate.get_name()!r}")
    dynamics = reservoir.dynamics_circuit
    for i in range(dynamics.get_gate_count()):
        gate = dynamics.get_gate(i)
        targets = gate.get_target_index_list()
        controls = gate.get_control_index_list()
        if gate.get_name() == "CZ" and len(targets) == 1 and len(controls) == 1:
            plan.append((2, controls[0], targets[0], _CZ_MATRIX, None))
        elif not controls and len(targets) == 1:
            plan.append((1, targets[0], 0, _gate_matrix(gate), None))
        else:
            raise ValueError(f"Unsupported reservoir gate {gate.get_name()!r}")
    return plan, n_qubits

def _is_diagonal(matrix):
    return bool(np.allclose(matrix, np.diag(np.diag(matrix))))


def fuse_diagonal(plan, n_qubits):
    """Fuse maximal runs of diagonal gates (CZ, Z-rotations) into phase passes.

    Diagonal gates commute and multiply as phases per computational-basis
    amplitude, so a run of them is one elementwise pass instead of one
    full statevector pass per gate. Returns a list of
    ``("gate", width, q0, q1, tensor_or_None, param_or_None)`` and
    ``("diag", phase_tensor_or_None)`` steps; non-diagonal gates keep
    their order. The phase vector is precomputed on GPU.
    """
    size = 1 << n_qubits
    idx = np.arange(size)
    steps = []
    phase = None

    def flush():
        nonlocal phase
        if phase is not None:
            tensor = torch.view_as_real(
                torch.tensor(phase, dtype=torch.complex128, device="cuda")
            ).contiguous()
            steps.append(("diag", tensor))
            phase = None

    for width, q0, q1, matrix, param in plan:
        if param is None and _is_diagonal(matrix):
            if phase is None:
                phase = np.ones(size, dtype=complex)
            if width == 2:  # CZ: -1 exactly when both bits are set
                both = ((idx >> q0) & 1) & ((idx >> q1) & 1)
                phase = phase * np.where(both, -1.0, 1.0)
            else:  # diagonal single-qubit gate (e.g. Z-rotation)
                diag = np.diag(matrix)
                phase = phase * np.where(((idx >> q0) & 1) == 0, diag[0], diag[1])
        else:
            flush()
            tensor = None
            if param is None:
                tensor = torch.view_as_real(
                    torch.tensor(matrix, dtype=torch.complex128, device="cuda")
                ).contiguous()
            steps.append(("gate", width, q0, q1, tensor, param))
    flush()
    return steps
def two_local_expectations(state, n_qubits):
    """ZZ and XX expectations on ring edges (i, i+1) from a batched statevector.

    Returns (batch, 2 * n_edges) ordered [ZZ_01, XX_01, ZZ_12, XX_12, ...].
    ZZ is diagonal (probability-weighted); XX pairs amplitudes differing in
    both bits. Exact, matching qulacs Observable("ZZ i j") / ("XX i j").
    """
    psi = torch.view_as_complex(state)  # (batch, size)
    probs = torch.abs(psi) ** 2
    size = psi.shape[1]
    idx = torch.arange(size, device=state.device)
    edges = [(i, (i + 1) % n_qubits) for i in range(n_qubits - 1)] + [(n_qubits - 1, 0)]
    values = []
    for a, b in edges:
        ba = (idx >> a) & 1
        bb = (idx >> b) & 1
        zz = probs @ torch.where(ba == bb, 1.0, -1.0).to(probs.dtype)
        flipped = psi[:, idx ^ (1 << a) ^ (1 << b)]
        xx = torch.einsum("ij,ij->i", torch.conj(psi), flipped).real
        values.append(torch.stack([zz, xx], dim=-1))
    return torch.stack(values, dim=1).reshape(state.shape[0], -1)


def _prepare_plan(plan, n_qubits):
    """Fuse diagonal runs and pre-upload fixed matrices to GPU.

    A leading run of parametric RotX gates on distinct qubits (the
    encoder) is replaced by direct product-state construction, which is
    exact for the |0...0> initial state.
    """
    prefix = 0
    seen = set()
    for width, q0, q1, matrix, param in plan:
        if param is not None and width == 1 and q0 not in seen:
            seen.add(q0)
            prefix += 1
        else:
            break
    if prefix == n_qubits and {step[1] for step in plan[:prefix]} == set(range(prefix)):
        steps = [("product", [step[4] for step in plan[:prefix]])]  # [(param, axis)]
        steps.extend(fuse_diagonal(plan[prefix:], n_qubits))
        return steps
    return fuse_diagonal(plan, n_qubits)


_KERNEL_CACHE = {}


def _kernel(size, width, batch):
    key = (size, width, batch)
    if key not in _KERNEL_CACHE:
        _KERNEL_CACHE[key] = apply_gate(size, width, batch)
    return _KERNEL_CACHE[key]


def _apply_steps(prepared, state, angles, batch, size):
    """Apply fused gate steps to a batched statevector; return final state."""
    param_matrices = {}
    for step in prepared:
        if step[0] == "product":
            params = step[1]  # [(feature_index, axis)] in qubit order
            state_c = torch.ones((batch, 1), dtype=torch.complex128, device="cuda")
            for idx, axis in params:
                th = angles[:, idx:idx + 1] / 2.0
                cos, sin = torch.cos(th), torch.sin(th)
                if axis == "X":
                    v0, v1 = cos, -1j * sin
                elif axis == "Y":
                    v0, v1 = cos, sin
                else:  # Z: RotZ|0> = e^{-i th/2}|0>, no |1> component
                    v0, v1 = cos - 1j * sin, torch.zeros_like(cos, dtype=torch.complex128)
                state_c = torch.stack([state_c * v0, state_c * v1], dim=1).reshape(batch, -1)
            state = torch.view_as_real(state_c).reshape(batch, size, 2).contiguous()
            continue
        if step[0] == "diag":
            phase = step[1]  # (size, 2) real/imag on GPU
            state_c = torch.view_as_complex(state)
            phase_c = torch.complex(phase[:, 0], phase[:, 1])
            state = torch.view_as_real(state_c * phase_c)
            continue
        _, width, q0, q1, tensor, param = step
        if tensor is not None:
            matrix = tensor.expand(batch, -1, -1, -1)
        else:
            if param not in param_matrices:
                param_matrices[param] = _rotation_tensors(param[1], angles[:, param[0]])
            matrix = param_matrices[param]
        output = torch.empty_like(state)
        _kernel(size, width, batch)(
            state, matrix.contiguous(), output, q0, q1 if width == 2 else 0
        )
        state = output
    return state


def _rotation_tensors(axis, angles):
    """Batched Pauli rotation matrices (batch, 2, 2, 2) real/imag on GPU."""
    half = angles / 2.0
    cos, sin = torch.cos(half), torch.sin(half)
    zeros = torch.zeros_like(cos)
    if axis == "X":
        real = torch.stack([torch.stack([cos, zeros], dim=-1),
                            torch.stack([zeros, cos], dim=-1)], dim=-2)
        imag = torch.stack([torch.stack([zeros, -sin], dim=-1),
                            torch.stack([-sin, zeros], dim=-1)], dim=-2)
    elif axis == "Y":
        real = torch.stack([torch.stack([cos, -sin], dim=-1),
                            torch.stack([sin, cos], dim=-1)], dim=-2)
        imag = torch.zeros_like(real)
    elif axis == "Z":
        # diag(e^{-i th/2}, e^{+i th/2}) = (cos - i sin, cos + i sin) on diagonal
        real = torch.stack([torch.stack([cos, zeros], dim=-1),
                            torch.stack([zeros, cos], dim=-1)], dim=-2)
        imag = torch.stack([torch.stack([-sin, zeros], dim=-1),
                            torch.stack([zeros, sin], dim=-1)], dim=-2)
    else:
        raise ValueError(f"Unsupported rotation axis {axis!r}")
    return torch.stack([real, imag], dim=-1)


def simulate_qrc_batch(model, encoded, batch_size=3200, plan=None, n_qubits=None):
    """Local Z/X/Y expectations for a batch of encoded rows, on GPU.

    ``encoded`` has shape ``(n_rows, encoder.feature_num)``. Returns an
    array of shape ``(n_rows, 3 * n_qubits)`` in per-qubit Z, X, Y order,
    matching the observable order built by the QRC notebook.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("TileLang QRC backend requires an available CUDA GPU")
    encoded = np.asarray(encoded, dtype=float)
    if (encoded.ndim != 2 or not np.isfinite(encoded).all()
            or encoded.shape[1] != model.reservoir.encoder.feature_num):
        raise ValueError("Encoded rows must be finite with encoder.feature_num columns")
    if plan is None:
        plan, n_qubits = build_gate_plan(model)
    prepared = _prepare_plan(plan, n_qubits)
    size = 1 << n_qubits
    outputs = []
    for start in range(0, len(encoded), batch_size):
        values = encoded[start:start + batch_size]
        batch = len(values)
        angles = torch.tensor(values, dtype=torch.float64, device="cuda")
        state = torch.zeros((batch, size, 2), dtype=torch.float64, device="cuda")
        state[:, 0, 0] = 1.0
        state = _apply_steps(prepared, state, angles, batch, size)
        outputs.append(local_expectations(state, n_qubits).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty((0, 3 * n_qubits))


class QRCExecutor:
    """Batched TileLang executor for one fitted qreservoir ``QELModel``."""

    precision = "double"
    target = "naive_statevector"
    device = "GPU"

    def __init__(self, model, batch_size=3200, include_two_local=False):
        if not torch.cuda.is_available():
            raise RuntimeError("TileLang QRC backend requires an available CUDA GPU")
        self.model = model
        self.batch_size = batch_size
        self.include_two_local = include_two_local
        self.plan, self.n_qubits = build_gate_plan(model)
        self.prepared = _prepare_plan(self.plan, self.n_qubits)
        self.n_observables = 3 * self.n_qubits + (2 * self.n_qubits if include_two_local else 0)

    def execute(self, encoded):
        """Expectations for encoded rows; see ``simulate_qrc_batch``."""
        encoded = np.asarray(encoded, dtype=float)
        if (encoded.ndim != 2 or not np.isfinite(encoded).all()
                or encoded.shape[1] != self.model.reservoir.encoder.feature_num):
            raise ValueError("Encoded rows must be finite with encoder.feature_num columns")
        size = 1 << self.n_qubits
        outputs = []
        for start in range(0, len(encoded), self.batch_size):
            values = encoded[start:start + self.batch_size]
            batch = len(values)
            angles = torch.tensor(values, dtype=torch.float64, device="cuda")
            state = torch.zeros((batch, size, 2), dtype=torch.float64, device="cuda")
            state[:, 0, 0] = 1.0
            state = _apply_steps(self.prepared, state, angles, batch, size)
            values = local_expectations(state, self.n_qubits)
            if self.include_two_local:
                values = torch.cat([values, two_local_expectations(state, self.n_qubits)], dim=-1)
            outputs.append(values.cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty((0, self.n_observables))


class TemporalQRCExecutor:
    """Sequential pure-state quantum reservoir over batched sequences.

    Architecture (input-driven QRC, TileLang/GPU): total qubits split into
    ``n_enc`` encoding qubits and ``n_mem`` memory qubits. Each timestep,
    encoding rotations RotX(input) are applied to the encoding qubits of the
    carried state, then fixed reservoir dynamics (random rotations + CZ
    ring, same construction as qreservoir ``RotationReservoir``) evolve the
    full state. Expectations are read after dynamics at every timestep.
    This differs from qreservoir ``RCModel`` (density matrices + partial
    trace over encoding qubits); here the pure state carries memory, which
    keeps the computation exactly statevector-simulable on GPU.
    """

    precision = "double"
    target = "naive_statevector"
    device = "GPU"

    def __init__(self, n_enc, n_mem, depth=3, seed=42, batch_size=1024):
        if not torch.cuda.is_available():
            raise RuntimeError("TileLang QRC backend requires an available CUDA GPU")
        self.n_enc, self.n_mem, self.depth = n_enc, n_mem, depth
        self.n_qubits = n_enc + n_mem
        self.batch_size = batch_size
        # Dynamics gates over all qubits, drawn exactly as RotationReservoir does:
        # build the library reservoir circuit with the same seed and extract it.
        from qreservoir.encoders import ExpEncoder
        from qreservoir.reservoirs import RotationReservoir

        np.random.seed(seed)
        dummy = ExpEncoder(self.n_enc, 1, 1)
        reservoir = RotationReservoir(dummy, self.n_mem, depth)
        # Drive steps + dynamics steps prepared from the library circuit.
        drive = [(1, i, 0, None, (i, "X")) for i in range(n_enc)]
        # Encoder CZ ring on encoding qubits (matches ExpEncoder circuit).
        drive += [(2, i, (i + 1) % n_enc, _CZ_MATRIX, None) for i in range(n_enc - 1)]
        if n_enc > 1:
            drive.append((2, n_enc - 1, 0, _CZ_MATRIX, None))
        dynamics_plan = []
        dyn = reservoir.dynamics_circuit
        for i in range(dyn.get_gate_count()):
            gate = dyn.get_gate(i)
            targets = gate.get_target_index_list()
            controls = gate.get_control_index_list()
            if gate.get_name() == "CZ" and len(targets) == 1 and len(controls) == 1:
                dynamics_plan.append((2, controls[0], targets[0], _CZ_MATRIX, None))
            elif not controls and len(targets) == 1:
                dynamics_plan.append((1, targets[0], 0, _gate_matrix(gate), None))
            else:
                raise ValueError(f"Unsupported reservoir gate {gate.get_name()!r}")
        self.prepared = _prepare_plan(drive + dynamics_plan, self.n_qubits)

    def execute(self, sequences):
        """Expectations per timestep for padded sequences.

        ``sequences``: (n_seq, max_len, n_enc) encoded inputs. Returns
        (n_seq, max_len, 3 * n_qubits) in per-qubit Z, X, Y order.
        """
        sequences = np.asarray(sequences, dtype=float)
        if sequences.ndim != 3 or not np.isfinite(sequences).all():
            raise ValueError("Sequences must be a finite 3D array (n_seq, max_len, n_enc)")
        n_seq, max_len, _ = sequences.shape
        size = 1 << self.n_qubits
        out = np.empty((n_seq, max_len, 3 * self.n_qubits))
        for start in range(0, n_seq, self.batch_size):
            chunk = sequences[start:start + self.batch_size]
            batch = len(chunk)
            state = torch.zeros((batch, size, 2), dtype=torch.float64, device="cuda")
            state[:, 0, 0] = 1.0
            for t in range(max_len):
                angles = torch.tensor(chunk[:, t, :], dtype=torch.float64, device="cuda")
                state = _apply_steps(self.prepared, state, angles, batch, size)
                out[start:start + batch, t, :] = local_expectations(state, self.n_qubits).cpu().numpy()
        return out
