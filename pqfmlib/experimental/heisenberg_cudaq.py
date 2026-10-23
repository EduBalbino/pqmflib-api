"""Approximation log.
Exact double-precision CUDA-Q statevector expectations, without sampling or
hardware noise. Floating-point results require comparison at tolerance 1e-9.
"""

from pqfmlib.core.cudaq_execution import CudaqExecutor
from pqfmlib.core.execution import PreparedProjectedFeatureJob
from pqfmlib.maps.heisenberg import (
    heisenberg_chain_observables, make_heisenberg_theta_matrix,
)


class HeisenbergCudaQ:
    """Execute the supplied logical circuit; CUDA-Q owns its native lowering."""

    def __init__(self, circuit, parameters):
        self.parameters = tuple(parameters)
        self.n_qubits = circuit.num_qubits
        observables, metadata = heisenberg_chain_observables(self.n_qubits)
        self.prepared = PreparedProjectedFeatureJob(
            qc_t=circuit, param_order=self.parameters,
            obs_isa_broadcast=[[observable] for observable in observables],
            obs_metadata=metadata,
        )
        self.backend = CudaqExecutor(mps=False, precision="double", seed=42)

    @property
    def target(self):
        return self.backend.target

    def transform(self, features):
        angles, _ = make_heisenberg_theta_matrix(
            features, self.n_qubits, encoding_mode="multi_axis",
            interaction_axes=("x", "y", "z"), use_tanh_scaling=True,
        )
        features, _ = self.backend.execute(
            self.prepared, angles[:, :len(self.parameters)],
            expectation_method="exact", shots=4096,
        )
        return features
