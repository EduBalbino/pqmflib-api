# PQFMLib 0.4.0 — Configurable simulation precision

Heisenberg, CD-Ising and XYZ now accept `precision="single"` (FP32) or
`precision="double"` (FP64). Double precision remains the default, preserving
existing configurations.

```python
from pqfmlib import HeisenbergProjectiveQFM

qfm = HeisenbergProjectiveQFM(
    name_file="fp32_example", q_enc=3,
    framework="cudaq", precision="single",
    simulation=True, ideal=True, fakebackend=False,
    mps=False, expectation_method="exact",
)
```

## Supported configurations

| Simulator | FP32 | FP64 |
|---|---|---|
| Qiskit/Aer statevector, CPU/GPU | Yes | Yes |
| Qiskit/Aer MPS | No | Yes |
| CUDA-Q statevector/MPS, GPU | Yes | Yes |

Compatible Aer fake-backend methods also receive the selected precision.
Unsupported methods and real-QPU execution reject FP32 explicitly.

## Usage and compatibility

- Precision is preserved by cloning and applies to `fit`, `transform`,
  `fit_transform` and `run()`. Changing it after fitting requires refitting.
- Returned feature matrices keep their existing data type. Their dtype does
  not imply that the simulation used FP64.
- Shot counts, MPS bond limits and truncation settings remain unchanged.
- CUDA-Q statistics and Heisenberg diagnostics report the selected precision.
- Run the included example with
  `python examples/heisenberg_cudaq.py --precision single`.

FP32 can introduce larger numerical errors; it does not guarantee equality
through a fixed decimal place. Performance improvements depend on the circuit
and backend and are not claimed by this release.

## Validation

- Full local suite: 167 passed, 38 opt-in GPU tests skipped.
- Precision and CUDA-Q suite with GPU enabled: 105 passed on an NVIDIA
  GeForce RTX 5070 with CUDA-Q `0.16.0.post1`.
- Runnable Heisenberg example: successful in FP32 and FP64.

The full local count includes research tests requiring notebooks excluded
from Git. The GPU results cover the tested circuits, not arbitrary workloads.
