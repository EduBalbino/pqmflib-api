"""Exact gate-by-gate GPU statevectors with FP64 rounding and no sampling.

Compare against Qiskit or CUDA-Q at absolute tolerance 1e-9. Higher precision
would reduce floating-point rounding; no gates or observables are approximated.
"""

import numpy as np
import torch
import tilelang
import tilelang.language as T


@tilelang.jit(target="cuda", execution_backend="tvm_ffi")
def apply_gate(size: int, width: int):
    dimension = 1 << width

    @T.prim_func
    def kernel(state: T.Tensor((size, 2), "float64"),
               matrix: T.Tensor((dimension, dimension, 2), "float64"),
               output: T.Tensor((size, 2), "float64"), q0: T.int32, q1: T.int32):
        with T.Kernel(T.ceildiv(size, 128), threads=128) as block:
            i = block * 128 + T.get_thread_binding()
            if i < size:
                row = (i >> q0) & 1
                if width == 2:
                    row += 2 * ((i >> q1) & 1)
                real = T.alloc_local((1,), "float64")
                imag = T.alloc_local((1,), "float64")
                real[0], imag[0] = 0.0, 0.0
                for column in T.serial(dimension):
                    source = (i & ~(1 << q0)) | ((column & 1) << q0)
                    if width == 2:
                        source = (source & ~(1 << q1)) | (((column >> 1) & 1) << q1)
                    ar, ai = state[source, 0], state[source, 1]
                    mr, mi = matrix[row, column, 0], matrix[row, column, 1]
                    real[0] += mr * ar - mi * ai
                    imag[0] += mr * ai + mi * ar
                output[i, 0] = real[0]
                output[i, 1] = imag[0]
    return kernel


def local_expectations(state, qubits):
    indices = torch.arange(len(state), device=state.device)
    values = []
    for qubit in range(qubits):
        zero = indices[(indices & (1 << qubit)) == 0]
        one = zero | (1 << qubit)
        ar, ai = state[zero, 0], state[zero, 1]
        br, bi = state[one, 0], state[one, 1]
        values.extend(((ar * ar + ai * ai - br * br - bi * bi).sum(),
                       (2 * (ar * br + ai * bi)).sum(),
                       (2 * (ar * bi - ai * br)).sum()))
    return torch.stack(values)


def simulate_row(circuit, parameters, values):
    bound = circuit.assign_parameters(dict(zip(parameters, values)))
    size = 1 << bound.num_qubits
    state = torch.zeros((size, 2), dtype=torch.float64, device="cuda")
    state[0, 0] = 1.0
    for instruction in bound.data:
        gate = instruction.operation
        if gate.name == "barrier":
            continue
        sites = [bound.find_bit(q).index for q in instruction.qubits]
        if len(sites) not in (1, 2):
            raise ValueError(f"Unsupported gate: {gate.name}")
        matrix = torch.tensor(np.asarray(gate.to_matrix()), dtype=torch.complex128, device="cuda")
        matrix = torch.view_as_real(matrix).contiguous()
        output = torch.empty_like(state)
        apply_gate(size, len(sites))(state, matrix, output, sites[0], sites[1] if len(sites) == 2 else 0)
        state = output
    return local_expectations(state, bound.num_qubits).cpu().numpy()


class TileLangExecutor:
    """Execute a prepared circuit once per row, as in the Qiskit exact path."""

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
        values = [simulate_row(prepared.qc_t, prepared.param_order, row) for row in theta]
        return np.asarray(values), prepared.obs_metadata
