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
def apply_cone_gate(size: int, batch: int, cones: int, gates: int, parity: bool):
    @T.prim_func
    def kernel(state: T.Tensor((cones * batch, size, 2), "float64"),
               matrix: T.Tensor((gates, batch, 4, 4, 2), "float64"),
               sites: T.Tensor((cones, 3), "int32"),
               output: T.Tensor((cones * batch, size, 2), "float64")):
        with T.Kernel(T.ceildiv(size, 128), cones * batch, threads=128) as (block, b):
            i = block * 128 + T.get_thread_binding()
            if i < size:
                cone, sample = b // batch, b % batch
                q0, q1, gate = sites[cone, 0], sites[cone, 1], sites[cone, 2]
                row = ((i >> q0) & 1) + 2 * ((i >> q1) & 1)
                real = T.alloc_local((1,), "float64")
                imag = T.alloc_local((1,), "float64")
                real[0], imag[0] = 0.0, 0.0
                for k in T.serial(2 if parity else 4):
                    column = row ^ (3 * k) if parity else k
                    source = (i & ~(1 << q0)) | ((column & 1) << q0)
                    source = (source & ~(1 << q1)) | (((column >> 1) & 1) << q1)
                    ar, ai = state[b, source, 0], state[b, source, 1]
                    mr, mi = matrix[gate, sample, row, column, 0], matrix[gate, sample, row, column, 1]
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


def simulate_batch(circuit, parameters, values, plan=None, instructions=None):
    bound = None if plan is not None else [circuit.assign_parameters(dict(zip(parameters, row))) for row in values]
    batch, size = len(values), 1 << circuit.num_qubits
    if bound is not None:
        instructions = bound[0].data
    elif instructions is None:
        instructions = circuit.data
    gates = []
    rotation_cache = {}
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
            key = tuple(plan[j] for j in (i - 1, i, i + 1)) if plan is not None else None
            if key is not None and key in rotation_cache:
                matrix = rotation_cache[key]
            else:
                if plan is None:
                    angles = np.array([[float(c.data[j].operation.params[0]) for j in (i - 1, i, i + 1)]
                                       for c in bound])
                else:
                    angles = np.stack([values[:, column] * scale + offset if column is not None
                                       else np.full(batch, offset)
                                       for column, scale, offset in key], axis=1)
                z, x, y = angles.T / 2
                even, odd = np.exp(-1j * z), np.exp(1j * z)
                matrix = np.zeros((batch, 4, 4), dtype=complex)
                matrix[:, 0, 0] = matrix[:, 3, 3] = even * np.cos(x - y)
                matrix[:, 0, 3] = matrix[:, 3, 0] = -1j * even * np.sin(x - y)
                matrix[:, 1, 1] = matrix[:, 2, 2] = odd * np.cos(x + y)
                matrix[:, 1, 2] = matrix[:, 2, 1] = -1j * odd * np.sin(x + y)
                if key is not None:
                    rotation_cache[key] = matrix
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

    cones, cone_lookup, observable_cones, observable_targets = [], {}, [], []
    for target in range(circuit.num_qubits):
        active, selected = {target}, []
        for index in range(len(gates) - 1, -1, -1):
            sites = gates[index][0]
            if active.intersection(sites):
                active.update(sites)
                selected.append(index)
        positions = {site: i for i, site in enumerate(sorted(active))}
        selected.reverse()
        key = (tuple(positions), tuple(selected))
        if key not in cone_lookup:
            cone_lookup[key] = len(cones)
            cones.append((positions, selected, positions[target]))
        observable_cones.append(cone_lookup[key])
        observable_targets.append(positions[target])
    maximal = [i for i, (positions, _, _) in enumerate(cones)
               if not any(i != j and set(positions) <= set(other_positions)
                          and (len(positions) < len(other_positions) or j < i)
                          for j, (other_positions, _, _) in enumerate(cones))]
    if len(maximal) < len(cones):
        old_cones = cones
        cones = [old_cones[i] for i in maximal]
        mapping = [next(j for j, (large_positions, _, _) in enumerate(cones)
                        if set(positions) <= set(large_positions))
                   for positions, _, _ in old_cones]
        merged = [set(selected) for _, selected, _ in cones]
        for index, (_, selected, _) in zip(mapping, old_cones):
            merged[index].update(selected)
        cones = [(positions, sorted(merged[i]), target)
                 for i, (positions, _, target) in enumerate(cones)]
        observable_cones = [mapping[index] for index in observable_cones]
        observable_targets = [cones[index][0][target] for target, index in enumerate(observable_cones)]
    width = max(2, max(len(positions) for positions, _, _ in cones))
    if not gates:
        return np.tile([1.0, 0.0, 0.0], (batch, circuit.num_qubits))
    if len(cones) * (1 << width) >= size:
        if batch > 8:
            return np.concatenate([simulate_batch(circuit, parameters, values[i:i + 8], plan, instructions)
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
    if batch > 64 and width > 8:
        return np.concatenate([simulate_batch(circuit, parameters, values[i:i + 64], plan, instructions)
                               for i in range(0, batch, 64)])
    prefix = gates[:circuit.num_qubits]
    product_initial = (len(prefix) == circuit.num_qubits
                       and all(len(sites) == 1 and np.array_equal(matrix, np.broadcast_to(matrix[0], matrix.shape))
                               for sites, _, matrix in prefix)
                       and {sites[0] for sites, _, _ in prefix} == set(range(circuit.num_qubits))
                       and all(item.operation.name == "unitary" for item in instructions[:circuit.num_qubits]))
    if product_initial and any(any(index >= circuit.num_qubits for index in selected) for _, selected, _ in cones):
        initial = {sites[0]: matrix[0, :, 0] for sites, _, matrix in prefix}
        local = np.zeros((len(cones), width, 2), dtype=complex)
        local[:, :, 0] = 1
        for k, (positions, _, _) in enumerate(cones):
            for site, position in positions.items():
                local[k, position] = initial[site]
        bits = (np.arange(1 << width)[:, None] >> np.arange(width)) & 1
        amplitudes = np.take_along_axis(local[:, None, :, :], bits[None, :, :, None], axis=3)
        initial_states = amplitudes[..., 0].prod(axis=2)
        cones = [(positions, [index for index in selected if index >= circuit.num_qubits], target)
                 for positions, selected, target in cones]
    else:
        initial_states = None
    depth = max(len(selected) for _, selected, _ in cones)
    identity = np.broadcast_to(np.eye(4, dtype=complex), (batch, 4, 4))
    unique_matrices, matrix_ids, all_sites = [identity], {}, []
    for step in range(depth):
        step_sites = []
        for positions, selected, _ in cones:
            if step >= len(selected):
                local_sites, gate_id = (0, 1), 0
            else:
                sites, _, matrix = gates[selected[step]]
                local_sites = tuple(positions[site] for site in sites)
                key = (id(matrix), len(sites))
                if key not in matrix_ids:
                    matrix_ids[key] = len(unique_matrices)
                    unique_matrices.append(np.kron(np.eye(2), matrix) if len(sites) == 1 else matrix)
                gate_id = matrix_ids[key]
                if len(sites) == 1:
                    local_sites = (local_sites[0], 0 if local_sites[0] else 1)
            step_sites.append((*local_sites, gate_id))
        all_sites.append(step_sites)
    parity = all(len(gates[index][0]) == 2 and gates[index][1]
                 for _, selected, _ in cones for index in selected)
    matrices = torch.view_as_real(torch.tensor(np.stack(unique_matrices), device="cuda")).contiguous()
    sites = torch.tensor(all_sites, dtype=torch.int32, device="cuda")
    lanes, size = batch * len(cones), 1 << width
    if initial_states is None:
        state = torch.zeros((lanes, size, 2), dtype=torch.float64, device="cuda")
        state[:, 0, 0] = 1.0
    else:
        state = torch.view_as_real(torch.tensor(initial_states, device="cuda")).repeat_interleave(batch, dim=0)
    for step in range(depth):
        output = torch.empty_like(state)
        apply_cone_gate(size, batch, len(cones), len(unique_matrices), parity)(state, matrices, sites[step], output)
        state = output
    targets = torch.tensor(observable_targets, device="cuda")[:, None]
    half = torch.arange(size // 2, device="cuda")[None, :]
    zeros = (half & ((1 << targets) - 1)) | ((half >> targets) << (targets + 1))
    ones = zeros | (1 << targets)
    states = state.reshape(len(cones), batch, size, 2)
    states = states.index_select(0, torch.tensor(observable_cones, device="cuda"))
    shape = (len(observable_cones), batch, size // 2, 2)
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
        self.instructions = None

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
            self.instructions = tuple(prepared.qc_t.data)
        values = [simulate_batch(prepared.qc_t, prepared.param_order, theta[i:i + 320],
                                 self.angle_plan, self.instructions)
                  for i in range(0, len(theta), 320)]
        return np.concatenate(values), prepared.obs_metadata
