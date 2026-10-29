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


def local_expectations(state, qubits):
    values = []
    for qubit in range(qubits):
        pairs = state.reshape(len(state), -1, 2, 1 << qubit, 2)
        ar, ai = pairs[:, :, 0, :, 0], pairs[:, :, 0, :, 1]
        br, bi = pairs[:, :, 1, :, 0], pairs[:, :, 1, :, 1]
        values.extend(((ar * ar + ai * ai - br * br - bi * bi).sum(dim=(1, 2)),
                       (2 * (ar * br + ai * bi)).sum(dim=(1, 2)),
                       (2 * (ar * bi - ai * br)).sum(dim=(1, 2))))
    return torch.stack(values, dim=1)


def simulate_batch(circuit, parameters, values):
    bound = [circuit.assign_parameters(dict(zip(parameters, row))) for row in values]
    batch, size = len(bound), 1 << circuit.num_qubits
    state = torch.zeros((batch, size, 2), dtype=torch.float64, device="cuda")
    state[:, 0, 0] = 1.0
    instructions = bound[0].data
    gates = []
    i = 0
    while i < len(instructions):
        instruction = instructions[i]
        i += 1
        gate = instruction.operation
        if gate.name == "barrier":
            continue
        sites = [bound[0].find_bit(q).index for q in instruction.qubits]
        if len(sites) not in (1, 2):
            raise ValueError(f"Unsupported gate: {gate.name}")
        matrix = np.stack([np.asarray(c.data[i - 1].operation.to_matrix()) for c in bound])
        if len(sites) == 2:
            while i < len(instructions):
                next_instruction = instructions[i]
                if (next_instruction.qubits != instruction.qubits
                        or next_instruction.operation.name == "barrier"):
                    break
                i += 1
                matrix = np.stack([np.asarray(c.data[i - 1].operation.to_matrix()) for c in bound]) @ matrix
        parity = len(sites) == 2 and not matrix[:, [0, 0, 1, 1, 2, 2, 3, 3], [1, 2, 0, 3, 0, 3, 1, 2]].any()
        gates.append((sites, parity, matrix))

    matrices = {
        width: torch.view_as_real(torch.tensor(
            np.stack([matrix for sites, _, matrix in gates if len(sites) == width]),
            dtype=torch.complex128, device="cuda")).contiguous()
        for width in (1, 2) if any(len(sites) == width for sites, _, _ in gates)
    }
    indices = {1: 0, 2: 0}
    for sites, parity, _ in gates:
        width = len(sites)
        matrix = matrices[width][indices[width]]
        indices[width] += 1
        output = torch.empty_like(state)
        apply_gate(size, width, batch, parity)(state, matrix, output, sites[0], sites[1] if width == 2 else 0)
        state = output
    return local_expectations(state, circuit.num_qubits).cpu().numpy()


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
        values = [simulate_batch(prepared.qc_t, prepared.param_order, theta[i:i + 8])
                  for i in range(0, len(theta), 8)]
        return np.concatenate(values), prepared.obs_metadata
