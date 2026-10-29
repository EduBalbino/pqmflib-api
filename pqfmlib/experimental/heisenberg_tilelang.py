"""Exact gate-by-gate GPU statevectors with FP64 rounding and no sampling.

Compare against Qiskit or CUDA-Q at absolute tolerance 1e-9. Higher precision
would reduce floating-point rounding; no gates or observables are approximated.
"""

import numpy as np
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(target="cuda", execution_backend="tvm_ffi")
def apply_gate(size: int, width: int, batch: int, parity: bool = False):
    dimension = 1 << width

    @T.prim_func
    def kernel(state: T.Tensor((batch, size, 2), "float64"),
               matrix: T.Tensor((batch, dimension, dimension, 2), "float64"),
               output: T.Tensor((batch, size, 2), "float64"), q0: T.int32, q1: T.int32):
        with T.Kernel(T.ceildiv(size, 128), batch, threads=128) as (block, b):
            i = block * 128 + T.get_thread_binding()
            if i < size:
                row = (i >> q0) & 1
                if width == 2:
                    row += 2 * ((i >> q1) & 1)
                real = T.alloc_local((1,), "float64")
                imag = T.alloc_local((1,), "float64")
                real[0], imag[0] = 0.0, 0.0
                for k in T.serial(2 if parity else dimension):
                    column = row ^ (3 * k) if parity else k
                    source = (i & ~(1 << q0)) | ((column & 1) << q0)
                    if width == 2:
                        source = (source & ~(1 << q1)) | (((column >> 1) & 1) << q1)
                    ar, ai = state[b, source, 0], state[b, source, 1]
                    mr, mi = matrix[b, row, column, 0], matrix[b, row, column, 1]
                    real[0] += mr * ar - mi * ai
                    imag[0] += mr * ai + mi * ar
                output[b, i, 0] = real[0]
                output[b, i, 1] = imag[0]
    return kernel


@tilelang.jit(target="cuda", execution_backend="tvm_ffi")
def apply_cone_gate(size: int, batch: int):
    @T.prim_func
    def kernel(state: T.Tensor((batch, size, 2), "float64"),
               matrix: T.Tensor((batch, 4, 4, 2), "float64"),
               sites: T.Tensor((batch, 2), "int32"),
               output: T.Tensor((batch, size, 2), "float64")):
        with T.Kernel(T.ceildiv(size, 128), batch, threads=128) as (block, b):
            i = block * 128 + T.get_thread_binding()
            if i < size:
                q0, q1 = sites[b, 0], sites[b, 1]
                row = ((i >> q0) & 1) + 2 * ((i >> q1) & 1)
                real = T.alloc_local((1,), "float64")
                imag = T.alloc_local((1,), "float64")
                real[0], imag[0] = 0.0, 0.0
                for column in T.serial(4):
                    source = (i & ~(1 << q0)) | ((column & 1) << q0)
                    source = (source & ~(1 << q1)) | (((column >> 1) & 1) << q1)
                    ar, ai = state[b, source, 0], state[b, source, 1]
                    mr, mi = matrix[b, row, column, 0], matrix[b, row, column, 1]
                    real[0] += mr * ar - mi * ai
                    imag[0] += mr * ai + mi * ar
                output[b, i, 0] = real[0]
                output[b, i, 1] = imag[0]
    return kernel


def local_expectations(state, qubits, target=None):
    values = []
    for qubit in range(qubits) if target is None else (target,):
        pairs = state.reshape(len(state), -1, 2, 1 << qubit, 2)
        ar, ai = pairs[:, :, 0, :, 0], pairs[:, :, 0, :, 1]
        br, bi = pairs[:, :, 1, :, 0], pairs[:, :, 1, :, 1]
        values.extend(((ar * ar + ai * ai - br * br - bi * bi).sum(dim=(1, 2)),
                       (2 * (ar * br + ai * bi)).sum(dim=(1, 2)),
                       (2 * (ar * bi - ai * br)).sum(dim=(1, 2))))
    return torch.stack(values, dim=1)


def affine_rotation_plan(circuit, parameters):
    positions = {parameter: i for i, parameter in enumerate(parameters)}
    plan = {}
    instructions = circuit.data
    i = 0
    while i < len(instructions):
        instruction = instructions[i]
        gate = instruction.operation
        if gate.name == "barrier" or (gate.name == "unitary" and len(instruction.qubits) == 1):
            i += 1
            continue
        group = instructions[i:i + 3]
        if (len(group) != 3 or [item.operation.name for item in group] != ["rzz", "rxx", "ryy"]
                or any(item.qubits != instruction.qubits for item in group)):
            return None
        for j, item in enumerate(group, i):
            angle = item.operation.params[0]
            params = getattr(angle, "parameters", set())
            if len(params) > 1:
                return None
            if params:
                parameter = next(iter(params))
                derivative = angle.gradient(parameter)
                if getattr(derivative, "parameters", set()) or parameter not in positions:
                    return None
                plan[j] = (positions[parameter], float(derivative), float(angle.bind({parameter: 0})))
            else:
                plan[j] = (None, 0.0, float(angle))
        i += 3
    return plan


def simulate_batch(circuit, parameters, values, plan=None):
    bound = None if plan is not None else [circuit.assign_parameters(dict(zip(parameters, row))) for row in values]
    batch, size = len(values), 1 << circuit.num_qubits
    instructions = circuit.data if bound is None else bound[0].data
    gates = []
    i = 0
    while i < len(instructions):
        instruction = instructions[i]
        i += 1
        gate = instruction.operation
        if gate.name == "barrier":
            continue
        sites = [circuit.find_bit(q).index for q in instruction.qubits]
        if len(sites) not in (1, 2):
            raise ValueError(f"Unsupported gate: {gate.name}")
        rotations = instructions[i:i + 2]
        if (gate.name == "rzz" and len(rotations) == 2
                and [item.operation.name for item in rotations] == ["rxx", "ryy"]
                and all(item.qubits == instruction.qubits for item in rotations)):
            if plan is None:
                angles = np.array([[float(c.data[j].operation.params[0]) for j in (i - 1, i, i + 1)]
                                   for c in bound])
            else:
                angles = np.stack([values[:, column] * scale + offset if column is not None
                                   else np.full(batch, offset)
                                   for column, scale, offset in (plan[j] for j in (i - 1, i, i + 1))], axis=1)
            z, x, y = angles.T / 2
            even, odd = np.exp(-1j * z), np.exp(1j * z)
            matrix = np.zeros((batch, 4, 4), dtype=complex)
            matrix[:, 0, 0] = matrix[:, 3, 3] = even * np.cos(x - y)
            matrix[:, 0, 3] = matrix[:, 3, 0] = -1j * even * np.sin(x - y)
            matrix[:, 1, 1] = matrix[:, 2, 2] = odd * np.cos(x + y)
            matrix[:, 1, 2] = matrix[:, 2, 1] = -1j * odd * np.sin(x + y)
            i += 2
        else:
            matrix = (np.broadcast_to(np.asarray(gate.to_matrix()), (batch, 2, 2)) if plan is not None
                      else np.stack([np.asarray(c.data[i - 1].operation.to_matrix()) for c in bound]))
        if len(sites) == 2 and plan is None:
            while i < len(instructions):
                next_instruction = instructions[i]
                if (next_instruction.qubits != instruction.qubits
                        or next_instruction.operation.name == "barrier"):
                    break
                i += 1
                matrix = np.stack([np.asarray(c.data[i - 1].operation.to_matrix()) for c in bound]) @ matrix
        parity = len(sites) == 2 and not matrix[:, [0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 0, 3, 0, 3, 1, 2]].any()
        gates.append((sites, parity, matrix))

    cones = []
    for target in range(circuit.num_qubits):
        active, selected = {target}, []
        for index in range(len(gates) - 1, -1, -1):
            sites = gates[index][0]
            if active.intersection(sites):
                active.update(sites)
                selected.append(index)
        positions = {site: i for i, site in enumerate(sorted(active))}
        cones.append((positions, list(reversed(selected)), positions[target]))
    width = max(2, max(len(positions) for positions, _, _ in cones))
    if not gates:
        return np.tile([1.0, 0.0, 0.0], (batch, circuit.num_qubits))
    if len(cones) * (1 << width) >= size:
        if batch > 8:
            return np.concatenate([simulate_batch(circuit, parameters, values[i:i + 8], plan)
                                   for i in range(0, batch, 8)])
        matrices = {
            gate_width: torch.view_as_real(torch.tensor(
                np.stack([matrix for sites, _, matrix in gates if len(sites) == gate_width]),
                dtype=torch.complex128, device="cuda")).contiguous()
            for gate_width in (1, 2) if any(len(sites) == gate_width for sites, _, _ in gates)
        }
        indices = {1: 0, 2: 0}
        state = torch.zeros((batch, size, 2), dtype=torch.float64, device="cuda")
        state[:, 0, 0] = 1.0
        for sites, parity, _ in gates:
            gate_width = len(sites)
            matrix = matrices[gate_width][indices[gate_width]]
            indices[gate_width] += 1
            output = torch.empty_like(state)
            apply_gate(size, gate_width, batch, parity)(
                state, matrix, output, sites[0], sites[1] if gate_width == 2 else 0)
            state = output
        return local_expectations(state, circuit.num_qubits).cpu().numpy()
    depth = max(len(selected) for _, selected, _ in cones)
    identity = np.broadcast_to(np.eye(4, dtype=complex), (batch, 4, 4))
    all_matrices, all_sites = [], []
    for step in range(depth):
        step_matrices, step_sites = [], []
        for positions, selected, _ in cones:
            if step >= len(selected):
                matrix, local_sites = identity, (0, 1)
            else:
                sites, _, matrix = gates[selected[step]]
                local_sites = tuple(positions[site] for site in sites)
                if len(sites) == 1:
                    local_sites = (local_sites[0], 0 if local_sites[0] else 1)
                    matrix = np.kron(np.eye(2), matrix)
            step_matrices.append(matrix)
            step_sites.extend([local_sites] * batch)
        all_matrices.append(np.concatenate(step_matrices))
        all_sites.append(step_sites)
    matrices = torch.view_as_real(torch.tensor(np.stack(all_matrices), device="cuda")).contiguous()
    sites = torch.tensor(all_sites, dtype=torch.int32, device="cuda")
    lanes, size = batch * len(cones), 1 << width
    state = torch.zeros((lanes, size, 2), dtype=torch.float64, device="cuda")
    state[:, 0, 0] = 1.0
    for step in range(depth):
        output = torch.empty_like(state)
        apply_cone_gate(size, lanes)(state, matrices[step], sites[step], output)
        state = output
    targets = torch.tensor([target for _, _, target in cones], device="cuda")[:, None]
    half = torch.arange(size // 2, device="cuda")[None, :]
    zeros = (half & ((1 << targets) - 1)) | ((half >> targets) << (targets + 1))
    ones = zeros | (1 << targets)
    states = state.reshape(len(cones), batch, size, 2)
    shape = (len(cones), batch, size // 2, 2)
    a = states.gather(2, zeros[:, None, :, None].expand(shape))
    b = states.gather(2, ones[:, None, :, None].expand(shape))
    ar, ai, br, bi = a[..., 0], a[..., 1], b[..., 0], b[..., 1]
    features = torch.stack(((ar * ar + ai * ai - br * br - bi * bi).sum(dim=2),
                            (2 * (ar * br + ai * bi)).sum(dim=2),
                            (2 * (ar * bi - ai * br)).sum(dim=2)), dim=2)
    return features.permute(1, 0, 2).reshape(batch, -1).cpu().numpy()


class TileLangExecutor:
    """Execute independent prepared circuits in batches of statevectors."""

    precision = "double"
    target = "naive_statevector"
    device = "GPU"
    last_execution_stats = None

    def __init__(self):
        if not torch.cuda.is_available():
            raise RuntimeError("TileLang requires an available CUDA GPU")
        self.prepared = None
        self.angle_plan = None

    def execute(self, prepared, theta):
        if self.prepared is not prepared:
            from pqfmlib.maps.heisenberg import heisenberg_chain_observables

            expected, metadata = heisenberg_chain_observables(prepared.qc_t.num_qubits)
            actual = prepared.obs_isa_broadcast
            if (prepared.obs_metadata != metadata or len(actual) != len(expected)
                    or any(len(group) != 1 or not group[0].equiv(obs)
                           for group, obs in zip(actual, expected))):
                raise ValueError("TileLang requires local Z/X/Y observables in site order")
            self.prepared = prepared
            self.angle_plan = affine_rotation_plan(prepared.qc_t, prepared.param_order)
        values = [simulate_batch(prepared.qc_t, prepared.param_order, theta[i:i + 32], self.angle_plan)
                  for i in range(0, len(theta), 32)]
        return np.concatenate(values), prepared.obs_metadata
