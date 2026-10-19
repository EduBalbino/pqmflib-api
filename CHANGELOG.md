# Changelog

## 0.3.0

- Added optional CUDA-Q GPU simulation to Heisenberg, CD-Ising and XYZ through
  `framework="cudaq"`, using the shared circuit and observable definitions.
  Qiskit remains the default framework and a required dependency.
- Added the `cudaq` installation extra with `cuda-quantum-cu12>=0.16,<0.17`.
  CUDA-Q is loaded only when its backend is initialized; it requires an
  accessible NVIDIA GPU and compatible driver/runtime, with no CPU fallback.
- Supported statevector and MPS execution with CUDA-Q, including finite-shot
  estimates and `expectation_method="exact"` without measurement sampling.
  Both targets currently use double precision. MPS bond limits and truncation
  can still introduce approximation in exact mode.
- Preserved `fit`, `transform`, `fit_transform` and `run()` workflows, feature
  ordering and CSV/NPY exports. Changing the framework after fitting requires
  fitting the map again. Shot-based execution remains the default.
- Preserved Pauli rotations during CUDA-Q translation, batched analytic MPS
  observable evaluation and grouped compatible MPS measurements. Added a
  circuit-derived conservative MPS bond bound and execution statistics.
  These changes do not establish a measured performance improvement.
- Added validation for unsupported CUDA-Q modes: this integration requires
  `simulation=True`, `ideal=True` and `fakebackend=False`; non-ideal simulation,
  fake backends, QPU execution and resource estimation remain unsupported.
- Added regression tests and opt-in GPU integration tests covering framework
  selection, circuit translation, observables, repeated transforms and exports.
- Reorganized the README to introduce the transformer API before framework
  examples. Added `examples/heisenberg_cudaq.py` with local data and moved the
  developer benchmark to `scripts/benchmark_heisenberg_cudaq.py`, with details
  in `docs/cudaq_diagnostics.md`.
- Corrected notebook references in the README and regression tests to match
  the current PCA example and local Creditcard notebook locations.

Validation (2026-10-19): the full local test suite passed with 138 tests and
28 opt-in GPU tests skipped. A separate CUDA-Q run with GPU tests enabled
passed all 66 tests on an NVIDIA GeForce RTX 5070 with CUDA-Q `0.16.0.post1`,
covering statevector and MPS. The Heisenberg CUDA-Q example also completed, producing training features
of shape `(18, 9)` and test features of shape `(6, 9)`. Version consistency
checks passed for `0.3.0`. GPU checks ran outside the sandbox because it blocks
GPU access. These results establish correctness for the tested cases, not a
general performance advantage. The full local suite includes Creditcard tests
and notebooks excluded from Git; those checks require the local research files.

## 0.2.0

- Added opt-in exact statevector expectations to all three PQFMs for noiseless simulation, reusing one state per sample across observables.
- Preserved shot-based execution as the default and added validation for incompatible execution options.
- Updated the hybrid Breast Cancer example with manual cross-validation, reusable fold features, and six-component PCA.
- Moved statevector and notebook regression tests into `tests/`, keeping `examples/` for examples.

## 0.1.1

- Fixed Qiskit observable label ordering across all projected feature maps.
- Configured Aer simulator qubit capacity from `q_enc`.
- Hardened edge-case handling for tabular data, CD-Ising, and Heisenberg.
- Ensured fixed QPY circuits are honored and validated during legacy simulated runs.
- Exposed `pqfmlib.__version__` from the public package API.
- Added regression tests for version consistency and corrected behavior.
