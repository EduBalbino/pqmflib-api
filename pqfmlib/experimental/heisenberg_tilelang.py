"""Approximation log.
Naive full-statevector evolution uses native RXX/RYY/RZZ rotations without
truncation, fusion or sampling. FP64 rounding requires comparison with CUDA-Q
at absolute tolerance 1e-9; higher precision would reduce rounding error.
"""

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(target="cuda", execution_backend="tvm_ffi")
def _preparation(size: int, qubits: int):
    @T.prim_func
    def kernel(state: T.Tensor((size, 2), "float64"),
               matrices: T.Tensor((qubits, 2, 2, 2), "float64"),
               out: T.Tensor((size, 2), "float64"), site: T.int32):
        with T.Kernel(T.ceildiv(size, 128), threads=128) as block:
            item = block * 128 + T.get_thread_binding()
            if item < size:
                bit = (item >> site) & 1
                peer = item ^ (1 << site)
                ar, ai = state[item, 0], state[item, 1]
                br, bi = state[peer, 0], state[peer, 1]
                cr, ci = matrices[site, bit, bit, 0], matrices[site, bit, bit, 1]
                dr, di = matrices[site, bit, 1 - bit, 0], matrices[site, bit, 1 - bit, 1]
                out[item, 0] = cr * ar - ci * ai + dr * br - di * bi
                out[item, 1] = cr * ai + ci * ar + dr * bi + di * br
    return kernel


@tilelang.jit(target="cuda", execution_backend="tvm_ffi")
def _rotation(size: int, rotations: int):
    @T.prim_func
    def kernel(state: T.Tensor((size, 2), "float64"),
               phases: T.Tensor((rotations, 2), "float64"),
               out: T.Tensor((size, 2), "float64"),
               axis: T.int32, first: T.int32, second: T.int32, rotation: T.int32):
        with T.Kernel(T.ceildiv(size, 128), threads=128) as block:
            item = block * 128 + T.get_thread_binding()
            if item < size:
                odd = ((item >> first) ^ (item >> second)) & 1
                peer = T.if_then_else(axis == 2, item,
                                      item ^ (1 << first) ^ (1 << second))
                sign = T.if_then_else(axis == 2, 1 - 2 * odd,
                                      T.if_then_else(axis == 1, 2 * odd - 1, 1))
                ar, ai = state[item, 0], state[item, 1]
                br, bi = state[peer, 0], state[peer, 1]
                cosine = phases[rotation, 0]
                sine = phases[rotation, 1] * sign
                out[item, 0] = cosine * ar + sine * bi
                out[item, 1] = cosine * ai - sine * br
    return kernel


@tilelang.jit(target="cuda", execution_backend="tvm_ffi")
def _observables(size: int, qubits: int):
    @T.prim_func
    def kernel(state: T.Tensor((size, 2), "float64"),
               out: T.Tensor((qubits, size // 2, 3), "float64")):
        with T.Kernel(T.ceildiv(size // 2, 128), qubits, threads=128) as (block, site):
            pair = block * 128 + T.get_thread_binding()
            if pair < size // 2:
                stride = 1 << site
                zero = pair // stride * (2 * stride) + pair % stride
                one = zero + stride
                ar, ai = state[zero, 0], state[zero, 1]
                br, bi = state[one, 0], state[one, 1]
                out[site, pair, 0] = ar * ar + ai * ai - br * br - bi * bi
                out[site, pair, 1] = 2 * (ar * br + ai * bi)
                out[site, pair, 2] = 2 * (ar * bi - ai * br)
    return kernel


class HeisenbergTileLang:
    """Naive FP64 full-state evolution, one native Pauli rotation per pass."""

    def __init__(self, circuit, parameters, *, device="cuda"):
        from qiskit.circuit import ParameterExpression

        self.device = torch.device(device)
        self.n_qubits = circuit.num_qubits
        self.size = 1 << self.n_qubits
        parameters = tuple(parameters)
        instructions = iter(item for item in circuit.data if item.operation.name != "barrier")
        matrices = [torch.as_tensor(next(instructions).operation.to_matrix(),
                                   dtype=torch.complex128)
                    for _ in range(self.n_qubits)]
        self.matrices = torch.view_as_real(torch.stack(matrices)).to(self.device)

        self.schedule = []
        columns, coefficients = [], []
        for instruction in instructions:
            axis = ("rxx", "ryy", "rzz").index(instruction.operation.name)
            first, second = (circuit.find_bit(qubit).index for qubit in instruction.qubits)
            expression = instruction.operation.params[0]
            column, slope = 0, 0.0
            if isinstance(expression, ParameterExpression) and expression.parameters:
                parameter, = expression.parameters
                column = parameters.index(parameter)
                slope = float(expression.gradient(parameter))
                offset = float(expression.bind({parameter: 0}))
            else:
                offset = float(expression)
            self.schedule.append((axis, first, second, len(columns)))
            columns.append(column)
            coefficients.append((slope, offset))
        self.columns = torch.tensor(columns, dtype=torch.int64, device=self.device)
        self.coefficients = torch.tensor(coefficients, dtype=torch.float64, device=self.device)
        self.preparation = _preparation(self.size, self.n_qubits)
        self.rotation = _rotation(self.size, len(self.schedule))
        self.observables = _observables(self.size, self.n_qubits)

    def inspect_kernels(self, rows=None):
        kernels = (self.preparation, self.rotation, self.observables)
        blocks = (self.size + 127) // 128
        return {
            "kernel_source_bytes": [len(kernel.get_kernel_source()) for kernel in kernels],
            "threads": [128, 128, 128],
            "blocks": [blocks, blocks, ((self.size // 2 + 127) // 128, self.n_qubits)],
            "state_amplitudes": self.size,
            "preparation_passes": self.n_qubits,
            "native_rotation_passes": len(self.schedule),
        }

    def _transform_row(self, features):
        theta = 2 * torch.pi * torch.tanh(features / 3)
        angles = (theta[self.columns] * self.coefficients[:, 0]
                  + self.coefficients[:, 1]) / 2
        phases = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        state = torch.zeros((self.size, 2), dtype=torch.float64, device=self.device)
        state[0, 0] = 1
        scratch = torch.empty_like(state)
        for site in range(self.n_qubits):
            self.preparation(state, self.matrices, scratch, site)
            state, scratch = scratch, state
        for axis, first, second, rotation in self.schedule:
            self.rotation(state, phases, scratch, axis, first, second, rotation)
            state, scratch = scratch, state
        contributions = torch.empty((self.n_qubits, self.size // 2, 3),
                                    dtype=torch.float64, device=self.device)
        self.observables(state, contributions)
        return contributions.sum(dim=1).flatten()

    def transform(self, features):
        """Return all site-major Z/X/Y means from host float64 feature rows."""
        inputs = torch.as_tensor(features, dtype=torch.float64, device=self.device)
        return torch.stack([self._transform_row(row) for row in inputs]).cpu().numpy()
