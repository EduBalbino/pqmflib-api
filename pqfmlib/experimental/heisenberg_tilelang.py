"""Approximation log.
Double-precision rounding has no general error bound; compare every benchmark
row with CUDA-Q at absolute tolerance 1e-9. Causal-cone restriction is exact and
uses no truncation or sampling.
"""

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(target="cuda", execution_backend="tvm_ffi")
def _local_expectations(angles: int, qubits: int, max_ops: int):
    @T.prim_func
    def kernel(phases: T.Tensor((angles, 2), "float64"),
               instructions: T.Tensor((qubits, max_ops, 4), "int32"),
               widths: T.Tensor((qubits,), "int32"),
               targets: T.Tensor((qubits,), "int32"),
               out: T.Tensor((qubits, 128, 3), "float64")):
        with T.Kernel(qubits, threads=128) as site:
            thread = T.get_thread_binding()
            state = T.alloc_shared((2, 256, 2), "float64")
            for slot in T.Parallel(2):
                for item in T.Parallel(256):
                    state[0, item, slot] = T.if_then_else(
                        (item == 0) & (slot == 0), 1.0, 0.0)
            T.sync_threads()
            for op in T.serial(max_ops):
                kind = instructions[site, op, 0]
                shift = instructions[site, op, 1]
                other_shift = instructions[site, op, 2]
                parameter = instructions[site, op, 3]
                read_side = op & 1
                write_side = 1 - read_side
                for item in T.Parallel(256):
                    bit = (item >> shift) & 1
                    peer = item + (1 - 2 * bit) * (1 << shift)
                    ar, ai = state[read_side, item, 0], state[read_side, item, 1]
                    br, bi = state[read_side, peer, 0], state[read_side, peer, 1]
                    c, s = phases[parameter, 0], phases[parameter, 1]
                    rz_r = ar * c - ai * (2 * bit - 1) * s
                    rz_i = ar * (2 * bit - 1) * s + ai * c
                    sx_r, sx_i = (ar - ai + br + bi) * 0.5, (ar + ai - br + bi) * 0.5
                    sign = 1 - 2 * bit * ((item >> other_shift) & 1)
                    state[write_side, item, 0] = T.if_then_else(
                        kind == 0, rz_r,
                        T.if_then_else(kind == 1, sx_r,
                                       T.if_then_else(kind == 2, br,
                                                      T.if_then_else(kind == 3, sign * ar, ar))))
                    state[write_side, item, 1] = T.if_then_else(
                        kind == 0, rz_i,
                        T.if_then_else(kind == 1, sx_i,
                                       T.if_then_else(kind == 2, bi,
                                                      T.if_then_else(kind == 3, sign * ai, ai))))
                T.sync_threads()

            width = widths[site]
            target = targets[site]
            pair = thread
            stride = 1 << target
            zero = (pair >> target << (target + 1)) + (pair & (stride - 1))
            one = zero + stride
            side = max_ops & 1
            ar, ai = state[side, zero, 0], state[side, zero, 1]
            br, bi = state[side, one, 0], state[side, one, 1]
            valid = pair < (1 << (width - 1))
            out[site, thread, 0] = T.if_then_else(
                valid, ar * ar + ai * ai - br * br - bi * bi, 0.0)
            out[site, thread, 1] = T.if_then_else(
                valid, 2 * (ar * br + ai * bi), 0.0)
            out[site, thread, 2] = T.if_then_else(
                valid, 2 * (ar * bi - ai * br), 0.0)
    return kernel


def _build_plan(circuit, parameters):
    """Build GPU circuit data before simulation."""
    from qiskit.circuit import ParameterExpression

    qubits = circuit.num_qubits
    metadata = circuit.metadata.get("pqfmlib_heisenberg", {})
    if (metadata.get("R") != 2 or metadata.get("encoding_mode") != "multi_axis"
            or metadata.get("interaction_axes") != ["x", "y", "z"]):
        raise ValueError("Expected the complete R=2 multi-axis logical circuit")
    seed, alpha = metadata["seed"], metadata["alpha"]
    from qiskit import transpile
    circuit = transpile(circuit, basis_gates=["rz", "sx", "x", "cz"],
                        optimization_level=0, seed_transpiler=seed)
    operations, coefficients, columns = [], [], []
    for instruction in circuit.data:
        gate = instruction.operation
        sites = tuple(circuit.find_bit(q).index for q in instruction.qubits)
        if gate.name == "rz":
            expression = gate.params[0]
            column, slope = 0, 0.0
            if isinstance(expression, ParameterExpression) and expression.parameters:
                if len(expression.parameters) != 1:
                    raise ValueError("Expected a single affine feature parameter")
                parameter = next(iter(expression.parameters))
                column = parameters.index(parameter)
                slope = float(expression.gradient(parameter))
                offset = float(expression.bind({parameter: 0}))
            else:
                offset = float(expression)
            operations.append((gate.name, sites, len(coefficients)))
            coefficients.append((slope, offset))
            columns.append(column)
        elif gate.name in ("sx", "x", "cz", "id"):
            if len(sites) == 2 and abs(sites[0] - sites[1]) != 1:
                raise ValueError("Expected adjacent chain sites")
            operations.append((gate.name, sites, -1))
        else:
            raise ValueError(f"Unsupported decomposed gate {gate.name}")
    return {
        "schema": "chain-statevector-v1", "qubits": qubits, "seed": seed,
        "alpha": alpha, "repetitions": 2, "noise": False,
        "features": len(parameters), "encoding_mode": "multi_axis",
        "operations": operations,
        "coefficients": torch.tensor(coefficients, dtype=torch.float64),
        "columns": torch.tensor(columns, dtype=torch.int64),
    }


class HeisenbergTileLang:
    """Evolve exact local complex128 statevectors and return site-major means."""

    def __init__(self, circuit, parameters, *, device="cuda"):
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("A CUDA device is required")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.plan = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in _build_plan(circuit, parameters).items()
        }
        if (self.plan.get("schema") != "chain-statevector-v1"
                or self.plan.get("noise") is not False or self.plan.get("repetitions") != 2):
            raise ValueError("Expected an R=2 noiseless chain statevector plan")
        self.n_qubits = self.plan["qubits"]
        cones = []
        gate_kinds = {"rz": 0, "sx": 1, "x": 2, "cz": 3}
        for target in range(self.n_qubits):
            active = {target}
            retained = []
            for name, sites, parameter in reversed(self.plan["operations"]):
                if name == "id":
                    continue
                if len(sites) == 1:
                    if sites[0] in active:
                        retained.append((gate_kinds[name], sites[0], -1,
                                         max(parameter, 0)))
                elif any(site in active for site in sites):
                    active.update(sites)
                    retained.append((gate_kinds[name], sites[0], sites[1], -1))
            local_sites = sorted(active)
            local_index = {site: index for index, site in enumerate(local_sites)}
            operations = []
            for kind, first, second, parameter in reversed(retained):
                shift = local_index[first]
                other_shift = local_index[second] if second >= 0 else 0
                operations.append((kind, shift, other_shift, parameter))
            cones.append((len(local_sites), local_index[target], operations))

        self.cone_widths = [width for width, _, _ in cones]
        if max(self.cone_widths) > 8:
            raise ValueError("Exact local statevector kernels support causal cones up to 8 qubits")
        self.max_ops = max(len(operations) for _, _, operations in cones)
        instructions = torch.zeros((self.n_qubits, self.max_ops, 4),
                                   dtype=torch.int32, device=self.device)
        instructions[:, :, 0].fill_(4)
        widths, targets = [], []
        for site, (width, target, operations) in enumerate(cones):
            widths.append(width)
            targets.append(target)
            if operations:
                instructions[site, :len(operations)] = torch.tensor(
                    operations, dtype=torch.int32, device=self.device)
        self.instructions = instructions
        self.widths = torch.tensor(widths, dtype=torch.int32, device=self.device)
        self.targets = torch.tensor(targets, dtype=torch.int32, device=self.device)
        self.local_expectations = _local_expectations(
            len(self.plan["columns"]), self.n_qubits, self.max_ops)
        self.partial = torch.empty((self.n_qubits, 128, 3), dtype=torch.float64,
                                   device=self.device)

    def inspect_kernels(self, rows=None):
        sizes = [len(self.local_expectations.get_kernel_source())]
        if max(sizes) > 20000:
            raise RuntimeError("Unexpected generated CUDA kernel")
        return {"kernel_source_bytes": sizes, "threads_per_kernel": [128],
                "blocks": [self.n_qubits], "cone_widths": self.cone_widths,
                "max_local_operations": self.max_ops}

    def evolve(self, features):
        if (not isinstance(features, torch.Tensor) or features.shape != (self.plan["features"],)
                or features.dtype != torch.float64 or features.device != self.device):
            raise ValueError(f"Expected CUDA float64 features ({self.plan['features']},) on {self.device}")
        theta = 2 * torch.pi * torch.tanh(features / 3)
        coefficients = self.plan["coefficients"]
        angles = (theta[self.plan["columns"]] * coefficients[:, 0] + coefficients[:, 1]) / 2
        phases = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        self.phases = phases
        return self

    def exact_expectations(self):
        self.local_expectations(self.phases, self.instructions, self.widths,
                                self.targets, self.partial)
        return self.partial.sum(dim=1).flatten()

    def transform(self, features):
        """Transform host feature rows into host float64 Z/X/Y expectations."""
        inputs = torch.as_tensor(features, dtype=torch.float64, device=self.device)
        if inputs.ndim != 2 or inputs.shape[1] != self.plan["features"] or not len(inputs):
            raise ValueError(f"Expected nonempty feature rows with {self.plan['features']} columns")
        if not hasattr(self, "_graph"):
            self._capture_transform()
        outputs = torch.empty((len(inputs), self.n_qubits * 3),
                              dtype=torch.float64, device=self.device)
        for row, output in zip(inputs, outputs):
            self._graph_features.copy_(row)
            self._graph.replay()
            output.copy_(self._graph_output)
        return outputs.cpu().numpy()

    def _capture_transform(self):
        """Capture the fixed gate schedule to avoid per-gate host launches."""
        self._graph_features = torch.zeros(self.plan["features"], dtype=torch.float64,
                                           device=self.device)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            # Populate the CUDA and TileLang caches before capture.
            self.evolve(self._graph_features)
            self.exact_expectations()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            self.evolve(self._graph_features)
            self._graph_output = self.exact_expectations()
        self._graph = graph

    def __call__(self, features):
        return self.evolve(features).exact_expectations()
