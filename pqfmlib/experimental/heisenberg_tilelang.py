"""Approximation log.
Double-precision rounding has no general error bound; compare every benchmark
row with CUDA-Q at absolute tolerance 1e-9. Causal-cone restriction is exact and
uses no truncation or sampling.
"""

import torch
import tilelang
import tilelang.language as T


@tilelang.jit(target="cuda", execution_backend="tvm_ffi")
def _local_expectations(batch_rows: int, exchanges: int, sites: int,
                        width: int, max_ops: int):
    @T.prim_func
    def kernel(phases: T.Tensor((batch_rows, exchanges, 8), "float64"),
               instructions: T.Tensor((sites, max_ops, 3), "int32"),
               op_counts: T.Tensor((sites,), "int32"),
               targets: T.Tensor((sites,), "int32"),
               initial_states: T.Tensor((sites, 1 << width, 2), "float64"),
               out: T.Tensor((batch_rows, sites, 1 << (width - 1), 3), "float64")):
        with T.Kernel(batch_rows, sites, threads=1 << (width - 1)) as (row, site):
            thread = T.get_thread_binding()
            state = T.alloc_shared((2, 1 << width, 2), "float64")
            for item in T.Parallel(1 << width):
                state[0, item, 0] = initial_states[site, item, 0]
                state[0, item, 1] = initial_states[site, item, 1]
            T.sync_threads()
            op_count = op_counts[site]
            for op in T.serial(op_count):
                shift = instructions[site, op, 0]
                other_shift = instructions[site, op, 1]
                parameter = instructions[site, op, 2]
                read_side = op & 1
                write_side = 1 - read_side
                for item in T.Parallel(1 << width):
                    bit = (item >> shift) & 1
                    other_bit = (item >> other_shift) & 1
                    odd = bit ^ other_bit
                    exchange_peer = item ^ (1 << shift) ^ (1 << other_shift)
                    ar, ai = state[read_side, item, 0], state[read_side, item, 1]
                    exchange_br = state[read_side, exchange_peer, 0]
                    exchange_bi = state[read_side, exchange_peer, 1]
                    exchange_parameter = parameter
                    exchange_c_r = T.if_then_else(
                        odd == 0, phases[row, exchange_parameter, 0],
                        phases[row, exchange_parameter, 4])
                    exchange_c_i = T.if_then_else(
                        odd == 0, phases[row, exchange_parameter, 1],
                        phases[row, exchange_parameter, 5])
                    exchange_d_r = T.if_then_else(
                        odd == 0, phases[row, exchange_parameter, 2],
                        phases[row, exchange_parameter, 6])
                    exchange_d_i = T.if_then_else(
                        odd == 0, phases[row, exchange_parameter, 3],
                        phases[row, exchange_parameter, 7])
                    state[write_side, item, 0] = (exchange_c_r * ar - exchange_c_i * ai
                                                  + exchange_d_r * exchange_br - exchange_d_i * exchange_bi)
                    state[write_side, item, 1] = (exchange_c_r * ai + exchange_c_i * ar
                                                  + exchange_d_r * exchange_bi + exchange_d_i * exchange_br)
                T.sync_threads()

            target = targets[site]
            pair = thread
            stride = 1 << target
            zero = (pair >> target << (target + 1)) + (pair & (stride - 1))
            one = zero + stride
            side = op_count & 1
            ar, ai = state[side, zero, 0], state[side, zero, 1]
            br, bi = state[side, one, 0], state[side, one, 1]
            out[row, site, thread, 0] = ar * ar + ai * ai - br * br - bi * bi
            out[row, site, thread, 1] = 2 * (ar * br + ai * bi)
            out[row, site, thread, 2] = 2 * (ar * bi - ai * br)
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
    operations, coefficients, columns = [], [], []
    unitaries = torch.empty((qubits, 2, 2, 2), dtype=torch.float64)
    data = circuit.data
    index = 0
    while index < len(data):
        instruction = data[index]
        gate = instruction.operation
        sites = tuple(circuit.find_bit(q).index for q in instruction.qubits)

        if gate.name == "unitary":
            matrix = gate.to_matrix()
            unitaries[sites[0], :, :, 0] = torch.as_tensor(matrix.real)
            unitaries[sites[0], :, :, 1] = torch.as_tensor(matrix.imag)
            index += 1
            continue

        if gate.name == "rzz":
            triple = data[index:index + 3]
            names = [item.operation.name for item in triple]
            if names != ["rzz", "rxx", "ryy"]:
                raise ValueError("Expected each edge interaction as RZZ, RXX, RYY")
            edge = tuple(circuit.find_bit(q).index for q in triple[0].qubits)
            if any(tuple(circuit.find_bit(q).index for q in item.qubits) != edge
                   for item in triple):
                raise ValueError("Expected the three Pauli rotations on the same edge")
            by_axis = {}
            for item in triple:
                axis = {"rzz": "z", "rxx": "x", "ryy": "y"}[item.operation.name]
                expression = item.operation.params[0]
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
                by_axis[axis] = (column, slope, offset)
            pair_parameters = [by_axis[axis] for axis in ("x", "y", "z")]
            operations.append(("exchange", edge, len(coefficients)))
            columns.append([parameter[0] for parameter in pair_parameters])
            coefficients.append([[parameter[1], parameter[2]]
                                 for parameter in pair_parameters])
            index += 3
            continue

        if gate.name in ("rxx", "ryy"):
            raise ValueError("Pauli rotations must begin with an RZZ interaction")
        if gate.name == "barrier":
            index += 1
            continue
        if gate.name in ("sx", "x", "cz", "id"):
            if len(sites) == 2 and abs(sites[0] - sites[1]) != 1:
                raise ValueError("Expected adjacent chain sites")
            raise ValueError(f"Unexpected native gate in logical plan: {gate.name}")
        else:
            raise ValueError(f"Unsupported logical gate {gate.name}")
        index += 1
    return {
        "schema": "chain-statevector-v3", "qubits": qubits, "seed": seed,
        "alpha": alpha, "repetitions": 2, "noise": False,
        "features": len(parameters), "encoding_mode": "multi_axis",
        "operations": operations,
        "coefficients": torch.tensor(coefficients, dtype=torch.float64),
        "columns": torch.tensor(columns, dtype=torch.int64),
        "unitaries": unitaries,
    }


class HeisenbergTileLang:
    """Evolve exact local complex128 statevectors and return site-major means."""

    def __init__(self, circuit, parameters, *, device="cuda", batch_rows=64):
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("A CUDA device is required")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.plan = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in _build_plan(circuit, parameters).items()
        }
        if (self.plan.get("schema") != "chain-statevector-v3"
                or self.plan.get("noise") is not False or self.plan.get("repetitions") != 2):
            raise ValueError("Expected an R=2 noiseless chain statevector plan")
        if batch_rows < 1:
            raise ValueError("batch_rows must be positive")
        self.batch_rows = batch_rows
        self.n_qubits = self.plan["qubits"]
        cones = []
        for target in range(self.n_qubits):
            active = {target}
            retained = []
            for name, sites, parameter in reversed(self.plan["operations"]):
                if name == "unitary":
                    continue
                if name != "exchange" or len(sites) != 2:
                    raise ValueError(f"Unexpected operation in logical plan: {name}")
                if any(site in active for site in sites):
                    active.update(sites)
                    retained.append((sites[0], sites[1], parameter))
            local_sites = sorted(active)
            local_index = {site: index for index, site in enumerate(local_sites)}
            operations = []
            for first, second, parameter in reversed(retained):
                shift = local_index[first]
                other_shift = local_index[second]
                operations.append((shift, other_shift, parameter))

            basis = torch.arange(1 << len(local_sites), dtype=torch.int64, device=self.device)
            real = torch.ones((len(basis),), dtype=torch.float64, device=self.device)
            imag = torch.zeros_like(real)
            for local, physical in enumerate(local_sites):
                bit = (basis >> local) & 1
                unitary = self.plan["unitaries"][physical]
                factor_r = torch.where(bit == 0, unitary[0, 0, 0], unitary[1, 0, 0])
                factor_i = torch.where(bit == 0, unitary[0, 0, 1], unitary[1, 0, 1])
                real, imag = real * factor_r - imag * factor_i, real * factor_i + imag * factor_r
            initial_state = torch.stack((real, imag), dim=-1)
            cones.append((len(local_sites), local_index[target], operations,
                          local_sites, initial_state))

        self.cone_widths = [cone[0] for cone in cones]
        if max(self.cone_widths) > 8:
            raise ValueError("Exact local statevector kernels support causal cones up to 8 qubits")
        self.groups = {}
        for width in sorted(set(self.cone_widths)):
            site_ids = [site for site, cone in enumerate(cones) if cone[0] == width]
            max_ops = max(len(cones[site][2]) for site in site_ids)
            instructions = torch.zeros((len(site_ids), max_ops, 3),
                                       dtype=torch.int32, device=self.device)
            targets, op_counts, initial_states = [], [], []
            for local_site, site in enumerate(site_ids):
                _, target, operations, _, initial_state = cones[site]
                targets.append(target)
                op_counts.append(len(operations))
                initial_states.append(initial_state)
                if operations:
                    instructions[local_site, :len(operations)] = torch.tensor(
                        operations, dtype=torch.int32, device=self.device)
            self.groups[width] = {
                "sites": torch.tensor(site_ids, dtype=torch.int64, device=self.device),
                "instructions": instructions,
                "op_counts": torch.tensor(op_counts, dtype=torch.int32, device=self.device),
                "targets": torch.tensor(targets, dtype=torch.int32, device=self.device),
                "initial_states": torch.stack(initial_states),
                "max_ops": max_ops,
            }
        self.kernels = {}
        for width in self.groups:
            self._kernel_for(batch_rows, width)

    def _kernel_for(self, batch_rows, width):
        key = (batch_rows, width)
        if key not in self.kernels:
            group = self.groups[width]
            kernel = _local_expectations(
                batch_rows, len(self.plan["columns"]), len(group["sites"]),
                width, group["max_ops"])
            partial = torch.empty((batch_rows, len(group["sites"]),
                                   1 << (width - 1), 3),
                                  dtype=torch.float64, device=self.device)
            self.kernels[key] = kernel, partial
        return self.kernels[key]

    def inspect_kernels(self, rows=None):
        sizes = [len(self._kernel_for(self.batch_rows, width)[0].get_kernel_source())
                 for width in self.groups]
        if max(sizes) > 20000:
            raise RuntimeError("Unexpected generated CUDA kernel")
        return {"kernel_source_bytes": sizes,
                "threads_per_kernel": [1 << (width - 1) for width in self.groups],
                "blocks": [(self.batch_rows, len(group["sites"]))
                           for group in self.groups.values()],
                "cone_widths": self.cone_widths,
                "max_local_operations": [group["max_ops"]
                                          for group in self.groups.values()]}

    def _phases(self, features):
        theta = 2 * torch.pi * torch.tanh(features / 3)
        coefficients = self.plan["coefficients"]
        angles = (theta[:, self.plan["columns"]] * coefficients[:, :, 0]
                  + coefficients[:, :, 1])
        x, y, z = angles.unbind(dim=2)
        half_z = z / 2
        even = (x - y) / 2
        odd = (x + y) / 2
        cos_z, sin_z = torch.cos(half_z), torch.sin(half_z)
        cos_even, sin_even = torch.cos(even), torch.sin(even)
        cos_odd, sin_odd = torch.cos(odd), torch.sin(odd)
        return torch.stack((
            cos_z * cos_even, -sin_z * cos_even,
            -sin_z * sin_even, -cos_z * sin_even,
            cos_z * cos_odd, sin_z * cos_odd,
            sin_z * sin_odd, -cos_z * sin_odd,
        ), dim=-1)

    def _run_batch(self, features):
        count = len(features)
        if count < self.batch_rows:
            padding = features[-1:].expand(self.batch_rows - count, -1)
            features = torch.cat((features, padding), dim=0)
        phases = self._phases(features)
        outputs = torch.empty((self.batch_rows, self.n_qubits, 3),
                              dtype=torch.float64, device=self.device)
        for width, group in self.groups.items():
            kernel, partial = self._kernel_for(self.batch_rows, width)
            kernel(phases, group["instructions"], group["op_counts"], group["targets"],
                   group["initial_states"], partial)
            outputs[:, group["sites"]] = partial.sum(dim=2)
        return outputs.flatten(start_dim=1)[:count]

    def evolve(self, features):
        if (not isinstance(features, torch.Tensor) or features.shape != (self.plan["features"],)
                or features.dtype != torch.float64 or features.device != self.device):
            raise ValueError(f"Expected CUDA float64 features ({self.plan['features']},) on {self.device}")
        self.phases = self._phases(features.unsqueeze(0))
        return self

    def exact_expectations(self):
        outputs = torch.empty((self.n_qubits, 3), dtype=torch.float64, device=self.device)
        for width, group in self.groups.items():
            kernel, partial = self._kernel_for(1, width)
            kernel(self.phases, group["instructions"], group["op_counts"], group["targets"],
                   group["initial_states"], partial)
            outputs[group["sites"]] = partial[0].sum(dim=1)
        return outputs.flatten()

    def transform(self, features):
        """Transform host feature rows into host float64 Z/X/Y expectations."""
        inputs = torch.as_tensor(features, dtype=torch.float64, device=self.device)
        if inputs.ndim != 2 or inputs.shape[1] != self.plan["features"] or not len(inputs):
            raise ValueError(f"Expected nonempty feature rows with {self.plan['features']} columns")
        outputs = torch.empty((len(inputs), self.n_qubits * 3),
                              dtype=torch.float64, device=self.device)
        for start in range(0, len(inputs), self.batch_rows):
            end = min(start + self.batch_rows, len(inputs))
            outputs[start:end] = self._run_batch(inputs[start:end])
        return outputs.cpu().numpy()

    def __call__(self, features):
        return self.evolve(features).exact_expectations()
