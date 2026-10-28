# PQFMLib

PQFMLib is a Python library for Projected Quantum Feature Maps (PQFMs) for
tabular machine learning. It builds quantum circuits from numerical datasets,
executes them with Qiskit/Aer, optional CUDA-Q/cuQuantum GPU simulation, or
IBM Quantum Runtime, and exports projected
expectation values as classical feature matrices that can be used by standard
machine-learning pipelines.

The central research object in this repository is `XYZProjectiveQFM`: a
feature-map Hamiltonian that combines ideas from CD-Ising encodings and
Heisenberg-style multi-axis interactions, as introduced in [4]. The library
also includes `CDIsingProjectiveQFM` and `HeisenbergProjectiveQFM` as
comparison maps.
`SuperexchangeHeisenbergPQFM` extends Heisenberg with one site-position feature
per qubit and spatially derived isotropic couplings.

## What The Library Does

PQFMLib turns a numeric CSV dataset into quantum features through this workflow:

1. Load a tabular dataset whose last column is the target.
2. For CD-Ising and XYZ, estimate pairwise feature relations with a normalized
   mutual-information matrix $J$. Heisenberg uses direct features or spatial
   superexchange couplings without mutual information.
3. Split features into one or more encoding blocks/layers and axis.
4. Map features to qubits, optionally using QPU connectivity and edge quality.
5. Build a parametrized Hamiltonian circuit.
6. Measure one-local and two-local Pauli observables.
7. Save the resulting projected quantum features to CSV/NPY.

The library supports ideal simulation with Qiskit/Aer or CUDA-Q.
Fake-backend simulation from IBM backends, hardware-topology-aware
simulation, resource estimation, and real IBM Quantum execution remain
available through Qiskit.

## Data Format

Datasets are loaded from:

```text
<data_dir>/<name_file>.csv
```

All columns must be numeric and finite, with no missing values. The last column
is treated as the target $y$; all previous columns are encoded as input
features.

PQFMLib assumes that the dataset has already been preprocessed. In particular,
features should be normalized before running the quantum feature maps. A common
choice is standard-score normalization:

```math
h_i = \bar{x}_{f(i)},
\qquad
\bar{x}_f = \frac{x_f - \mu_f}{w_f},
```

where $\mu_f$ is the feature mean and $w_f$ is the feature standard deviation.

## Quick Example

```python
from pqfmlib import XYZProjectiveQFM

qfm = XYZProjectiveQFM(
    name_file="my_dataset",
    data_dir="./data",
    output_root="./results",
    ideal=True,
    q_enc=4,
    features_per_qubit=3,
    axes=("x", "y", "z"),
    encoding_mode="multi_axis",
    keep_diagonal_terms=True,
    keep_cross_terms=True,
    measure_cross_observables=True,
)

result = qfm.run()
print(result["Xq_all_raw"].shape)
print(result["csv_path"])
```

Runnable examples using the `Toxicity_preprocessed_shuffled` dataset are
available in `examples/`.

## Transformer API

All three map families, including the superexchange facade, also support an
in-memory transformer workflow for simulations:

- `fit(X_train)` prepares the map using the training data and fixes its circuit
  structure for subsequent inputs.
- `transform(X)` executes the fitted map to generate quantum features for each
  input row, without fitting it again.
- `fit_transform(X_train)` fits the map and generates its training features in
  one call.

Split training and test data before fitting preprocessing or the map. Reuse
both fitted objects on validation and test data.

```python
from pqfmlib import XYZProjectiveQFM

qfm = XYZProjectiveQFM(
    name_file="cross_validation",
    output_root="./results",
    simulation=True,
    ideal=True,
    q_enc=4,
    features_per_qubit=2,
    axes=("x", "y"),
)

qfm.fit(X_train)
Xq_train = qfm.transform(X_train)
Xq_val = qfm.transform(X_val)
Xq_test = qfm.transform(X_test)
```

`fit_transform(X_train)` is equivalent to `fit(X_train)` followed by
`transform(X_train)`:

```python
Xq_train = qfm.fit_transform(X_train)
```

The transformer API follows these rules:

- `fit(X, y=None)` returns the fitted map. The optional `y` is accepted for
  scikit-learn-style composition but is not used by the quantum feature maps.
- `transform(X)` returns a NumPy feature matrix and does not save CSV or NPY
  feature files. `fit()` may still save structural artifacts such as blocks or
  physical nodes through the existing reproducibility mechanisms.
- The number of input features must match the training matrix. When training
  uses a pandas DataFrame, later DataFrames must use the same columns in the
  same order.
- Circuit-affecting configuration is frozen by `fit()`. Change the
  configuration and call `fit()` again instead of transforming with a circuit
  built for different settings.
- The transformer API currently requires `simulation=True`. Real-QPU
  submission remains available through the backward-compatible `run()` API.

For CD-Ising and XYZ, `fit()` computes mutual information, feature blocks,
feature assignment, interactions, physical nodes, and circuit structure using
only `X_train`. Validation and test matrices never recompute those quantities.
Their values are used only to construct the parameter matrix for their own
`transform()` call.

Heisenberg does not use mutual information. Its `fit()` freezes only the state
that depends on the training feature dimension, including the number of
blocks, parameter slots, interaction scale, layout, circuit, and observables.
In its default direct mode, each `transform(X)` applies the existing scaling
and padding to the values in that specific `X`. Superexchange uses exactly one
feature per qubit and derives couplings separately for each row, without tanh,
padding, extra blocks, or fitted statistics.

The legacy file-based workflow is unchanged:

```python
result = qfm.run()
```

It still loads `<data_dir>/<name_file>.csv`, treats the last column as the
target, saves feature artifacts, and preserves real-QPU submission behavior.

## Choosing the execution framework

### What is CUDA-Q?

[NVIDIA CUDA-Q](https://nvidia.github.io/cuda-quantum/0.16.0/index.html) is a
quantum programming framework with Python and C++ interfaces for hybrid
applications using CPUs, GPUs and quantum processors. In PQFMLib, it provides
optional NVIDIA GPU simulation through the cuQuantum libraries.

Select `framework="cudaq"` on any of the three feature maps to use this
integration. PQFMLib handles circuit translation and observable evaluation;
you keep the same `fit`, `transform` and `fit_transform` interface without
writing CUDA-Q kernels. The current integration supports ideal simulation.

### Shared construction and separate execution

PQFMLib defines each map's Hamiltonian, circuit parameters and observables
independently of the execution framework. Qiskit supplies the shared circuit
representation; selecting CUDA-Q translates that circuit into a GPU kernel.
Logical qubit indices and feature-column order are preserved.

```text
          Heisenberg / CD-Ising / XYZ
                       |
         Shared circuit and observables
                       |
             +---------+---------+
             |                   |
          Qiskit/Aer           CUDA-Q
             |                   |
      CPU/GPU simulation    GPU simulation
```

All three maps accept `framework="qiskit"` (the default) or
`framework="cudaq"`. The choice changes execution while retaining the same
feature-map construction and transformer interface. CUDA-Q currently supports
only ideal simulation; Qiskit also provides the topology-aware, fake-backend
and IBM Quantum execution modes described under [Execution Modes](#execution-modes).

### Optional CUDA-Q installation

For CUDA-Q simulation, install the optional extra in an environment with an
accessible NVIDIA GPU and a compatible driver:

```bash
pip install -e ".[cudaq]"
```

This extra selects `cuda-quantum-cu12>=0.16,<0.17` and its CUDA 12/cuQuantum
dependencies. CUDA-Q is imported only when its backend is initialized, normally
by `fit()` or `run()`. Qiskit-only installations do not need CUDA-Q. Qiskit
remains a dependency for the shared circuit and observable representation.

Use an isolated environment with matching CUDA package variants: CUDA 12 and
CUDA 13 cuQuantum wheels can install libraries into the same paths. The tested
configuration uses CUDA-Q `0.16.0.post1` for CUDA 12 on an NVIDIA GeForce RTX
5070. See the [NVIDIA installation guide](https://nvidia.github.io/cuda-quantum/latest/using/quick_start.html)
for platform and driver requirements. A GPU-enabled Aer installation is needed
only if you also want Qiskit statevector execution on GPU.

### CUDA-Q practical example

The [Heisenberg CUDA-Q example](examples/heisenberg_cudaq.py) follows the basic
examples: configure `HeisenbergProjectiveQFM` and call `run()`. It reads
`data/Toxicity_preprocessed_shuffled.csv`; edit `name_file` to select your dataset.
After installing the CUDA-Q extra, run from the repository root:

```bash
python examples/heisenberg_cudaq.py
```

The example selects `framework="cudaq"` and `expectation_method="exact"` for
ideal simulation on an NVIDIA GPU. Edit `precision="double"` to `"single"` in
the file for FP32. The same framework selection applies to
`CDIsingProjectiveQFM` and `XYZProjectiveQFM`.

## Choosing `framework` and `mps`

`HeisenbergProjectiveQFM`, `CDIsingProjectiveQFM`, and `XYZProjectiveQFM` all
accept the keyword-only parameter `framework="qiskit"`. The accepted values
are `"qiskit"` and `"cudaq"`; omitting it preserves the existing Qiskit defaults
and behavior. `mps` selects the simulation representation independently of
the framework:

- `mps=False`: statevector simulation stores the full quantum state; memory
  grows exponentially with the number of qubits.
- `mps=True`: matrix product state (MPS) simulation stores a chain of tensors.
  It can reduce memory for circuits with limited entanglement, while bond
  limits and truncation can introduce approximation.

| Configuration | Simulator | Device |
|---|---|---|
| `framework="qiskit", mps=False` | Aer statevector | CPU by default; GPU when configured and available |
| `framework="qiskit", mps=True` | Aer matrix product state | CPU |
| `framework="cudaq", mps=False` | CUDA-Q `nvidia` / cuStateVec | GPU |
| `framework="cudaq", mps=True` | CUDA-Q `tensornet-mps` / cuTensorNet | GPU |

**In this version, CUDA-Q requires `simulation=True`, `ideal=True`, and
`fakebackend=False`.** `ideal=False`, fake-backend simulation, QPU execution,
and resource estimation are unsupported for CUDA-Q and raise clear errors.
These flags are validated before the legacy Qiskit ideal-mode normalization.

CUDA-Q always uses GPU, even though the shared `use_gpu_statevector` default
is `False`. Leave `statevector_device=None` or use `"GPU"`; `"CPU"` is rejected
for CUDA-Q. Missing CUDA-Q dependencies or an unavailable GPU/target produce
errors, with no fallback to Qiskit or CPU.

### Four simulation examples

This complete example uses the same input and Heisenberg configuration in all
four cases, with the default double precision. The same framework options
apply to `CDIsingProjectiveQFM` and `XYZProjectiveQFM`.

```python
import numpy as np
from pqfmlib import HeisenbergProjectiveQFM

X = np.random.default_rng(42).normal(size=(8, 4))
common = dict(
    name_file="framework_demo", q_enc=3, seed=42,
    simulation=True, ideal=True, fakebackend=False,
)

qiskit_sv = HeisenbergProjectiveQFM(
    **common, framework="qiskit", mps=False,
    expectation_method="statevector",
)
qiskit_mps = HeisenbergProjectiveQFM(
    **common, framework="qiskit", mps=True,
    expectation_method="shots", shots=4096,
)
cudaq_sv = HeisenbergProjectiveQFM(
    **common, framework="cudaq", mps=False,
    expectation_method="exact",
)
cudaq_mps = HeisenbergProjectiveQFM(
    **common, framework="cudaq", mps=True,
    expectation_method="exact",
)

Xq_qiskit_sv = qiskit_sv.fit_transform(X)
Xq_qiskit_mps = qiskit_mps.fit_transform(X)
Xq_cudaq_sv = cudaq_sv.fit_transform(X)
Xq_cudaq_mps = cudaq_mps.fit_transform(X)

np.testing.assert_allclose(Xq_cudaq_sv, Xq_qiskit_sv, atol=1e-10, rtol=1e-10)
np.testing.assert_allclose(Xq_cudaq_mps, Xq_qiskit_sv, atol=1e-10, rtol=1e-10)
# Qiskit MPS above uses finite shots, so its features have sampling error.

X_new = np.random.default_rng(43).normal(size=(2, 4))
Xq_new = cudaq_mps.transform(X_new)
```

All four configurations retain `fit(X, y=None)`, `transform(X)`, `fit_transform(X, y=None)`
and `run()`. Feature matrices have shape `(n_samples, n_observables)` with the
same column order and observable metadata for the same PQFM configuration.
Changing `framework`, `mps`, `precision`, or `expectation_method` after fitting
requires a new `fit()`. `run()` retains CSV input, result dictionaries and CSV/NPY output;
use distinct `output_root` values when comparing saved results from different
frameworks.

### Choosing `precision`

All three maps accept the keyword-only option `precision="double"` (FP64,
the default) or `precision="single"` (FP32). This selects the simulator's
internal floating-point precision, independently of the expectation method.

| Simulation | `"single"` (FP32) | `"double"` (FP64) |
|---|---|---|
| Qiskit/Aer statevector, CPU or GPU | Supported | Supported |
| Qiskit/Aer MPS | Unsupported; raises `ValueError` | Supported |
| CUDA-Q statevector or MPS, GPU | Supported | Supported |

For example, after installing CUDA-Q:

```python
import numpy as np
from pqfmlib import HeisenbergProjectiveQFM

X_train = np.random.default_rng(42).normal(size=(8, 4))
pqfm = HeisenbergProjectiveQFM(
    name_file="fp32_demo", q_enc=3,
    framework="cudaq", precision="single",
    simulation=True, ideal=True, fakebackend=False,
    mps=False, expectation_method="exact",
)
X_quantum = pqfm.fit_transform(X_train)

# A precision change requires refitting before further transforms.
pqfm.set_params(precision="double")
X_quantum_double = pqfm.fit_transform(X_train)
```

For Qiskit statevector, use `framework="qiskit"`, `mps=False` and
`expectation_method="statevector"` with the same precision option. The option
is preserved by cloning and applies to both the transformer API and `run()`.

For Aer fake-backend simulations, FP32 requires an explicit compatible
`fakebackend_method`: `"statevector"`, `"density_matrix"`, `"unitary"`,
`"superop"` or `"tensor_network"`. Other requirements of the selected method
(such as noise and device support) still apply. `"automatic"` and methods
without FP32 support are rejected with `precision="single"` to avoid silently
using a different precision. FP32 is also rejected for real-QPU execution;
the default `"double"` does not configure physical hardware precision.

FP32 uses fewer bytes per state/tensor element but introduces more rounding
error; runtime and total memory savings depend on the circuit and backend.
The input preprocessing and returned feature matrices retain their existing
NumPy floating-point types. A `float64` output does not recover accuracy lost
during FP32 simulation. Measurement sampling (`shots`) and MPS truncation are
separate sources of error: changing `precision` does not change shot counts,
bond limits or truncation thresholds. CUDA-Q execution statistics report the
selected precision in `pqfm.backend.last_execution_stats["precision"]` after
a successful execution.

### CUDA-Q numerical behavior and limits

`expectation_method="shots"` remains the default on both frameworks. It
estimates the same observables, but matching seeds and shot counts do not
make the frameworks' random samples identical. Use Qiskit
`expectation_method="statevector"` and CUDA-Q `expectation_method="exact"` for
deterministic comparisons. CUDA-Q also accepts `"statevector"` with
`mps=False`. `shots` is ignored for these deterministic modes.

For CUDA-Q, `"exact"` means expectations without measurement sampling; MPS
truncation and floating-point error can still affect the result:

- `mps_max_bond_dimension=None` computes a conservative bond bound from the
  circuit's two-qubit gates and qubit ordering, bounded by
  `2**(num_qubits // 2)`, rather than using CUDA-Q's default bond cap of 64.
  The selected bound is printed and exposed as `mps_effective_max_bond` in
  `pqfm.backend.last_execution_stats`. If it exceeds the target's 32-bit
  integer limit, execution raises an error requesting an explicit limit.
  Large circuits may need an explicit bond limit to fit GPU memory, at the
  cost of approximation.
- `mps_truncation_threshold` defaults to `1e-16`. In CUDA-Q it is an absolute
  singular-value cutoff, whereas Aer uses a discarded-weight criterion.
  Equal thresholds do not imply identical truncation. Relative truncation is
  effectively disabled; a zero threshold uses a near-zero positive value
  required by the CUDA-Q target.
- The GPU integration tests compare small circuits for all three maps against
  Qiskit FP64 statevector, using `atol=rtol=1e-10` for FP64 and `2e-5` for FP32
  in the precision tests. These are test tolerances, not accuracy guarantees
  for arbitrary circuit sizes or truncation settings.

For implementation details, execution statistics and developer benchmarks, see
[CUDA-Q diagnostics](docs/cudaq_diagnostics.md).

Compatible unitary QPY circuits are supported. Classical bits,
measurement, reset, initialization instructions, noise channels and delays are
rejected by the adapter.

CUDA-Q exposes process-global target settings. PQFMLib serializes its own
CUDA-Q execution and restores target/MPS environment settings afterward.
Applications making concurrent CUDA-Q calls outside PQFMLib must synchronize
those calls too.

### Framework verification

With the development dependencies installed (`pip install -e ".[dev]"`), run
the CPU/optional-dependency regression tests:

```bash
python -m pytest -q tests/test_framework_selection.py tests/test_statevector_execution.py tests/test_cudaq_execution.py tests/test_precision.py
```

GPU tests are opt-in and skipped by the command above. To execute both actual
GPU targets, including all three PQFMs, parameter/observable ordering, repeated
transforms and `run()` exports, as well as FP32/FP64 precision checks:

```bash
PQFMLIB_TEST_CUDAQ_GPU=1 python -m pytest -q tests/test_cudaq_execution.py tests/test_precision.py
```

Once enabled, missing CUDA-Q, a missing GPU or an unavailable target causes a
test failure rather than a skip. The precision tests also exercise Qiskit on
GPU and require a GPU-enabled Aer installation. No credentials or real-QPU
jobs are used.

## Choosing `expectation_method` (since 0.2.0)

All three maps (`XYZProjectiveQFM`, `CDIsingProjectiveQFM`, and
`HeisenbergProjectiveQFM`) accept the keyword-only option
`expectation_method="statevector"`. The default remains `"shots"`. CUDA-Q
additionally accepts `"exact"` for both statevector and MPS.

| Method | How features are computed | When to use it |
|---|---|---|
| `"shots"` | Estimate expectations from finite measurement samples; `shots` controls the sampling budget. | Sampling experiments, noisy simulation, or real hardware. |
| `"statevector"` | Evolve one full state per sample and compute each expectation directly from it. | Noiseless simulation when the full state fits in memory. |
| `"exact"` (CUDA-Q only) | Compute expectations without measurement sampling on the selected statevector or MPS target. | Ideal GPU simulation; MPS truncation can still introduce error. |

`ideal=True` alone does **not** select exact expectations. Set
`expectation_method="statevector"` explicitly to avoid finite-shot sampling,
or use `expectation_method="exact"` with CUDA-Q.

### Supported configurations

| Configuration | Qiskit `statevector` | CUDA-Q `statevector` |
|---|---|---|
| `simulation=True`, `fakebackend=False`, `mps=False`, `ideal=True` | Yes; no hardware credentials needed. | Yes; NVIDIA GPU required. |
| Same configuration with `ideal=False` | Yes; hardware information/credentials may still be needed for topology and assignment. | Unsupported in this version. |
| `fakebackend=True` | No. | No. |
| `simulation=False` | No. | No. |
| `mps=True` | No; this option computes the full vector. | No; use `"exact"` for deterministic MPS expectations. |

Incompatible configurations raise `ValueError`. The method is available on all
three map constructors; it does not change their encoding or observable options.

### Complete example: exact XYZ features

Split the dataset before fitting preprocessing or the map. This example uses
six classical features and runs without credentials on CPU:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pqfmlib import XYZProjectiveQFM

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X[:, :6], y, test_size=0.2, stratify=y, random_state=42,
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pqfm = XYZProjectiveQFM(
    name_file="exact_xyz",
    q_enc=6,
    simulation=True,
    fakebackend=False,
    ideal=True,
    expectation_method="statevector",
    encoding_mode="shared_feature",
    features_per_qubit=1,
    axes=("x", "y", "z"),
    keep_diagonal_terms=False,
    keep_cross_terms=True,
    measure_cross_observables=True,
    use_gpu_statevector=False,
)
X_quantum = pqfm.fit_transform(X_train)
X_quantum_test = pqfm.transform(X_test)
```

### Execution and cost

For `framework="qiskit"`, this option works with both `fit/transform` and `run()`. It requires
`simulation=True`, `fakebackend=False`, and `mps=False`; `ideal` can be either
value. With `ideal=False`, hardware information is still loaded for the map's
existing topology/assignment logic, but simulation remains noiseless.

For each sample, Aer evolves one full state vector and all observables reuse
that vector. The configured CPU/GPU device is used for evolution; observable
expectations are calculated on the CPU. States are processed one at a time,
not retained for the entire dataset. Memory still scales exponentially with
the circuit's qubit count, including any ancillas in loaded circuits.

`shots` is ignored in this mode. Aer executes a single deterministic evolution
per sample (`shots=1` internally), without sampling observable measurements.
Results have no finite-shot sampling error, subject to floating-point accuracy.
Adding observables still costs expectation-value calculations, but does not
repeat state preparation or require measurement groups. Nonunitary circuits
with measurement/reset are unsupported in this mode. The returned feature
matrix and observable ordering are unchanged. Changing execution mode after
`fit()` requires fitting again.

### Switching methods

For a new map, omit `expectation_method` or set it to `"shots"` to retain the
previous behavior. For an existing fitted map, change the option and refit:

```python
pqfm.set_params(expectation_method="shots", shots=4096)
X_quantum_sampled = pqfm.fit_transform(X_train)
```

The methods estimate the same observables, but finite-shot results will not
match exact features bit for bit. Retrain downstream preprocessing/classifiers
when switching modes. Restart a running notebook kernel after updating the
library so it imports the new constructor.

The [hybrid Breast Cancer notebook](examples/xyz_breast_cancer_pca_pipeline.ipynb)
uses exact expectations and prepares PQFM/PCA features once per CV fold,
reusing them across classifier candidates.

### Verification

Run the deterministic exact-execution tests with:

```bash
python -m unittest discover -s tests -p 'test_statevector_execution.py' -v
```

## Execution Modes

PQFMLib separates the execution mode from the hardware topology used to build
the feature map:

- `ideal=True`: runs ideal simulation using the selected `framework`. The encoded qubits are treated as
  fully connected when the map allows it, without enforcing a specific IBM QPU
  coupling map.
- Qiskit only, `ideal=False` and `simulation=True`: simulates the circuit while using the
  structure of the selected `ibm_qpu`. For example, a Heron r2 backend uses its
  heavy-hex topology, so the available two-qubit terms are limited by the QPU
  coupling map even though execution is simulated. This mode is also useful for
  MPS simulation with `mps=True`.
- Qiskit only, `ideal=False`, `simulation=True`, and `fakebackend=True`: simulates from the
  selected IBM backend model through `AerSimulator.from_backend`, including the
  backend topology and fake-backend behavior.
- Qiskit only, `ideal=False` and `simulation=False`: submits the job to the selected IBM
  Quantum backend through Runtime.

For `ideal=False` transformer fits, the physical subgraph, logical edges,
feature assignment, and prepared circuit are selected once and reused by every
`transform()` call. In CD-Ising and XYZ, physical-node selection is a greedy
low-error step; the genetic algorithm then assigns features to logical slots
on that selected subgraph. With `use_edge_error=True`, its fitness combines
mutual information with physical-edge quality.

For Qiskit with `ideal=False` and `fakebackend=False`, the simulator executes the logical circuit, while its
available interaction edges and feature assignment still come from the chosen
QPU topology. With `fakebackend=True`, the prepared circuit is also transpiled
against the backend-derived Aer target, and the resulting layout is reused.

### Fixed fitted state

The existing reproducibility options can initialize transformer state:

- `use_fixed_blocks=True` loads the saved feature assignment and embedded
  interactions instead of recomputing mutual information.
- `use_fixed_phys_nodes=True` loads and validates the saved physical nodes.
- `fixed_circuit_file_name` loads a QPY circuit during `fit()` and reuses that
  same circuit for every `transform()` call.

For full reproduction of a hardware-aware fitted state, combine fixed blocks,
fixed physical nodes, and a fixed circuit. Any component not fixed explicitly
is selected again during the next `fit()`.

Fixed QPY circuits must expose exactly the parameter names expected by the map
configuration fitted from the training feature dimension. An incompatible
circuit is rejected during `fit()` rather than failing during a later
transformation. This validates the parameter interface; users remain
responsible for supplying a circuit produced for the intended Hamiltonian and
backend configuration.

## Hamiltonian Maps

### CD-Ising PQFM

`CDIsingProjectiveQFM` is a counterdiabatic-inspired Ising-glass feature map
based on the approach presented in [1]. In the present implementation, it is extended to support multi-feature
encoding per qubit by increasing the circuit depth. Each circuit block, or
layer, encodes one feature per qubit, allowing multiple features to be assigned
sequentially to the same qubit across different layers.

For each feature block, PQFMLib constructs data-dependent local fields $h_i$
and couplings $J_{ij}$ from the input values and the mutual-information matrix.
The local fields are interpreted as normalized feature values.

A useful way to view the underlying Ising problem Hamiltonian is:

```math
H_{\mathrm{Ising}}(x)
= \sum_i h_i(x) Z_i
+ \sum_{(i,j)} J_{ij} Z_i Z_j .
```

The implemented circuit applies a first-order counterdiabatic term:

```math
H_{\mathrm{CD}}(x,t) =
-2 \dot{\lambda}(t) \alpha_1(t)
[
\sum_i h_i(x)Y_i +
\sum_{i \lt j} J_{ij}(Y_i Z_j + Z_i Y_j)
].
```

The first-order CD coefficient is:

```math
\alpha_1(t) =
-\frac{\sum_i h_i^2 + \sum_{i \lt j} J_{ij}^2}{4R(t)}.
```

with:

```math
R(t) =
(1-\lambda(t))^2
(\sum_i h_i^2 + 4\sum_{i\ne j} J_{ij}^2)
+ \lambda(t)^2
(
\sum_i h_i^4 +
\sum_{i\ne j} J_{ij}^4 +
6\sum_{i\ne j} h_i^2J_{ij}^2 +
6\sum_{i \lt j \lt k}(J_{ij}^2J_{ik}^2 + J_{ij}^2J_{jk}^2 + J_{ik}^2J_{jk}^2)
).
```

In the implementation, the schedule is represented by $s(t)$ and
$\dot{s}(t)$, which play the role of $\lambda(t)$ and $\dot{\lambda}(t)$.
This map is useful as a physically motivated Ising baseline: it encodes
features locally and uses feature-feature relations to activate two-qubit CD
terms on available edges.

### Heisenberg PQFM

`HeisenbergProjectiveQFM` follows the Heisenberg-style projected quantum feature
maps used in [2,3]. It prepares one fixed, seeded random unitary per qubit and
encodes features **only in two-qubit interactions**. It does not learn mutual
information or encode features in local fields. The default remains the
original chain with one feature shared across `XX`, `YY`, and `ZZ` per edge.

| Option | Default | Behavior |
|---|---|---|
| `topology` | `"chain"` | `"chain"`, `"native"`, or `"full"` |
| `interaction_axes` | `("x", "y", "z")` | One to three unique axes; order defines feature assignment in `multi_axis` |
| `encoding_mode` | `"shared_feature"` | One feature per edge, or `"multi_axis"` for one feature per selected axis per edge |
| `measure_2local_diagonal` | `False` | Add separate `ZZ`, `XX`, `YY` observables on each active edge |
| `measure_cross_observables` | `False` | Add separate `XY`, `YX`, `XZ`, `ZX`, `YZ`, `ZY` observables on each active edge |
| `measure_all_pairs` | `False` | Add `ZZ`, `XX`, `YY` for every logical pair, independently of interaction topology; incompatible with cross observables |
| `coupling_mode` | `"direct"` | Use features directly on edges, or `"superexchange"` to derive isotropic couplings from site features |
| `J0`, `ell` | `1.0`, `1.0` | Positive finite coupling scale and localization length for superexchange |

**Topology and feature assignment.** `chain` uses `q_enc - 1` edges and preserves
the original even-edge/odd-edge schedule. `full` requires `ideal=True` and uses
all `q_enc * (q_enc - 1) / 2` unordered pairs, without self-interactions.
`native` requires `ideal=False` and uses all native connections within a
connected subset of `q_enc` physical qubits of the target backend. Select the
subset automatically or supply `use_fixed_phys_nodes=True` and
`fixed_phys_nodes_file="phys_nodes.json"`, containing an ordered list of physical
qubit indices. The selected nodes become the initial transpiler layout; routing
may still change the final layout.

The following feature-to-edge assignment describes the default `direct` mode.
For `full` and `native`, edges are sorted by logical indices `(i, j)`, with `i < j`;
bidirectional hardware connections count once. In `multi_axis`, columns fill
slots by block, then edge, then the order of `interaction_axes`. With `E` edges
and `A` selected axes, block capacity is `E` for `shared_feature` and `E * A` for
`multi_axis`. Excess features create additional blocks; unused slots in the last
block are zero. For example, on a three-qubit chain with axes `("x", "y")`, the
first four multi-axis features fill `XX(0,1)`, `YY(0,1)`, `XX(1,2)`, `YY(1,2)`.

The interaction coefficients have the form

```math
H_b(x) = \sum_{(i,j)\in E} \sum_{a\in A} J_{b,ij}^{(a)}(x)\,\sigma_i^a\sigma_j^a.
```

In `shared_feature`, the selected axes on an edge share the same feature,
allowing, for example, `J(XX + YY)`. In `multi_axis`, they use distinct features,
allowing `Jx*XX + Jy*YY` or `Jx*XX + Jy*YY + Jz*ZZ`. By default, each input feature
is scaled to `2*pi*tanh(x/3)`; `use_tanh_scaling=False` uses the input directly.
Each selected `RXX`, `RYY`, or `RZZ` gate receives angle
`alpha / (2 * num_blocks)` times that scaled value. The block sequence repeats
`R` times with the same parameters. Preprocessing of the input remains the
caller's responsibility.

**Measurements are independent of encoding.** One-local `Z`, `X`, `Y` are always
measured on every qubit, in that order per qubit. Optional two-local measurements
follow sorted edges, with diagonal terms first and then the six cross terms in
the order listed above. With `measure_all_pairs=False`, `n` qubits and `E` edges, output width is
`3*n + 3*E*measure_2local_diagonal + 6*E*measure_cross_observables`.
`XY` and `YX` are separate columns. Selecting only `XX` for encoding does not
remove any requested measurement axes. Cross measurements do not introduce
cross interactions into the encoding Hamiltonian. The legacy `axes` argument
does not select the encoding; use `interaction_axes`.

With `measure_all_pairs=True`, local `Z`, `X`, `Y` are followed by `ZZ`, `XX`,
`YY` for each pair `(i, j)` in lexicographic order with `i < j`, including pairs
that have no interaction edge. Output width is `3*n + 3*n*(n-1)/2`: **975 for
25 qubits** and **2,460 for 40 qubits**. `measure_2local_diagonal` is redundant
in this mode; `measure_cross_observables=True` raises `ValueError`. This readout
also works with direct, anisotropic, and multi-axis Heisenberg encoding without
changing the circuit or Hamiltonian. Column labels remain `q1_z_0`, `q1_x_0`,
`q1_y_0`, ..., `q2_zz_0_1`, `q2_xx_0_1`, `q2_yy_0_1`, ... . Observables are sent
in one batched Estimator PUB, allowing grouping into three global X/Y/Z bases
when supported by the executor. Local BackendEstimatorV2/Aer tests verify three
basis circuits per sample; real Runtime scheduling remains service-managed.

```python
import numpy as np
from pqfmlib import HeisenbergProjectiveQFM

X = np.random.default_rng(42).normal(size=(8, 7))
qfm = HeisenbergProjectiveQFM(
    name_file="heisenberg_full", q_enc=3, ideal=True,
    topology="full", interaction_axes=("x", "y"), encoding_mode="multi_axis",
    measure_2local_diagonal=True, measure_cross_observables=True,
    expectation_method="statevector",
)
Xq = qfm.fit_transform(X)  # shape (8, 36); 6 feature slots per block
```

After `fit`, changing topology, encoding axes/mode, or measurement flags requires
a new `fit`. The [topology example](examples/heisenberg_topologies.py) configures
one map and calls `run()`, like the other basic examples. It reads
`data/Toxicity_preprocessed_shuffled.csv` and defaults to ideal full connectivity,
multi-axis encoding on `x` and `y`, and diagonal plus cross measurements.
Edit the constructor arguments to select another dataset, encoding or topology.
For native connectivity, set `ideal=False`, `topology="native"`, and the target
`ibm_qpu`; retrieving its topology requires configured IBM access.

```bash
python examples/heisenberg_topologies.py
```

**Saving and loading QPY circuits.** `run()` retains its CSV/NPY interface and
also saves `heisenberg_circuit.qpy`, `phys_nodes.json` when available, and
`heisenberg_metadata.json`. When the logical source is available, it also saves
`heisenberg_circuit_logical.qpy`. Simulation result dictionaries include
`metadata_path`; real-QPU submissions additionally produce `job_meta.json`.
The metadata records encoding configuration, edges, parameter names/order,
measurement flags and ordered observable labels, plus physical layouts when
available. Output folders distinguish topology, encoding and optional
measurements; the default configuration keeps its original folder naming.

Set `fixed_circuit_file_name` to a QPY path to reuse either encoding mode:

```python
qfm = HeisenbergProjectiveQFM(
    name_file="my_dataset", q_enc=3,
    ideal=False, simulation=False, ibm_qpu="YOUR_QPU_NAME",
    topology="native", interaction_axes=("x", "y"), encoding_mode="multi_axis",
    fixed_circuit_file_name="saved/heisenberg_circuit.qpy",
    measure_cross_observables=True,
)
# qfm.run() submits a real QPU job using data/my_dataset.csv.
# Supply the same R, alpha, seed and feature/block configuration as the saved circuit.
```

A logical QPY is transpiled for the selected QPU. A compatible transpiled QPY
keeps its gates and layout; native topology derives its physical subset from
the saved initial layout unless a fixed-node file is explicitly supplied. That
file must agree with the saved layout. Observables use the **final** layout,
including routing permutations. An incompatible physical circuit is rejected;
use the logical QPY to compile for another backend. For a native logical QPY,
use the corresponding saved `phys_nodes.json` if automatic selection would
produce a different graph.

The loader checks exact parameter names, qubit/layout consistency, backend
instruction support for physical QPU circuits, and encoding metadata when
present. QPY files must contain exactly one parameterized feature circuit,
without measurements or classical bits. New circuits built by
`build_heisenberg_chain_feature_circuit` carry versioned encoding metadata,
which survives transpilation and QPY serialization. Older QPYs without this
metadata remain accepted with a warning: the caller must provide the correct
encoding configuration because parameter names alone cannot verify axis
assignment, interaction scales, or the edge schedule. `full` remains an ideal
simulation topology, including when loading QPY files.

### Spatial superexchange Heisenberg PQFM

`SuperexchangeHeisenbergPQFM` is a thin facade over the Heisenberg circuit and
execution pipeline. Equivalently, select
`HeisenbergProjectiveQFM(coupling_mode="superexchange", ...)`. The original
Heisenberg defaults are unchanged.

Each input row contains exactly `q_enc = N` finite real features, interpreted
as effective one-dimensional site positions. For every interaction edge:

```math
d_{ij}(x) = |x_i-x_j|,\qquad
J_{ij}(x) = J_0 \exp\!\left(-\frac{2d_{ij}(x)}{\ell}\right),\qquad
H_G(x) = \sum_{(i,j)\in E(G)} J_{ij}(x)(X_iX_j+Y_iY_j+Z_iZ_j).
```

`J0 > 0` and `ell > 0` must be finite, and both default to 1. In the Pauli
convention, `J0` absorbs the spin-to-Pauli factor: physically motivated hopping
`t_ij = t0 * exp(-d_ij/ell)` gives `J0 = t0**2/U`. This is a physics-inspired
encoding of features as positions; they are not real atomic distances.

Normalize/select features in the external training pipeline. Inputs in `[0, 1]`
are convenient, but any finite real positions are accepted. The module applies
no scaler, SHAP, mutual information, or cross-sample statistics.
`use_tanh_scaling` is ignored in this mode, including when explicitly True.
`interaction_axes` must contain all three axes and `encoding_mode` must be
`"shared_feature"`. Each `Jij` multiplies XX, YY and ZZ equally; there are no
local fields or anisotropic couplings. Very small couplings can underflow to
zero at float64 precision.

All `N` features participate on a chain with `N-1` couplings. A full graph has
`N*(N-1)/2` couplings derived from the same `N` positions. Every topology uses
one block with one slot per edge, without padding or extra feature blocks.
Chain scheduling remains even/odd; native/full scheduling follows sorted logical
edges. Haar initialization is fixed by `seed`, with the same layout and circuit
for all rows after fitting. With one block, the inherited angle is
`alpha * Jij / 2` for each RZZ/RXX/RYY, so each repetition corresponds to a
first-order edge-product step with `delta = alpha/4` in the Pauli convention.
`R` repeats that step; it does not divide the angle. To refine Trotter steps at
fixed total time `t`, use `alpha = 4*t/R`.

```python
import numpy as np
import pandas as pd
from sklearn.base import clone
from pqfmlib import SuperexchangeHeisenbergPQFM

X_train = pd.DataFrame([[0., .25, .75], [.1, .4, .8]], columns=["a", "b", "c"])
X_test = pd.DataFrame([[.2, .3, .9]], columns=X_train.columns)
qfm = SuperexchangeHeisenbergPQFM(
    name_file="sites", q_enc=3, J0=1., ell=1., topology="chain",
    measure_all_pairs=True, expectation_method="statevector",
)
qfm.fit(X_train, y=[0, 1])  # y is accepted and ignored
Xq_train = qfm.transform(X_train)  # NumPy (2, 18), no CSV/NPY writes
Xq_test = qfm.transform(X_test)   # NumPy (1, 18), same frozen map
np.testing.assert_allclose(clone(qfm).fit_transform(X_train), Xq_train)
```

Column names and order must match the fitted DataFrame. Changing coupling
parameters, topology, encoding, readout or execution configuration requires a
new `fit`. `fit/transform` use simulation. For the legacy CSV workflow:

```python
# data/sites.csv must have three position columns, then the target column y.
runner = SuperexchangeHeisenbergPQFM(
    name_file="sites", q_enc=3, data_dir="./data", output_root="./results",
    J0=1., ell=1., measure_all_pairs=True, expectation_method="statevector",
)
result = runner.run()
print(result["Xq_all_raw"].shape, result["csv_path"], result["metadata_path"])
```

`run()` saves CSV/NPY features, QPY circuits, JSON metadata and physical layout
when available. Metadata records `coupling_mode`, `J0`, `ell`, edge/slot order,
effective scaling and ordered observables. Superexchange output folders include
the coupling scale/length and optional `allpairs`. Fixed QPY reuse validates
coupling configuration in addition to the inherited circuit checks. Changing
only the readout can reuse the same circuit. Older metadata without a coupling
mode denotes `direct`; QPYs without encoding metadata retain the legacy warning.

| Backend | Configuration |
|---|---|
| Qiskit exact CPU statevector | `expectation_method="statevector"`, `mps=False` |
| Qiskit sampled statevector/MPS | `expectation_method="shots"`, `mps=False/True`; Aer MPS requires double precision |
| CUDA-Q ideal GPU statevector/MPS | `framework="cudaq"`, `expectation_method="exact"`, `mps=False/True`; supports single/double precision |
| Qiskit native topology/fakebackend | `ideal=False`, `topology="native"`, `ibm_qpu="ibm_kingston"`; configured IBM access supplies the real topology |
| Real IBM QPU | `ideal=False`, `simulation=False`, `framework="qiskit"`, `expectation_method="shots"`, then explicit `run()` |

`topology="full"` remains restricted to ideal simulation. CUDA-Q requires an
accessible NVIDIA GPU and ideal simulation. CUDA-Q `exact` removes shot noise
but does not remove MPS truncation. `shots`, `precision`, MPS limits, seeds,
resource estimation and fixed layouts keep their inherited semantics. No real
QPU jobs are submitted by the automated tests or the runnable example.

The [superexchange example](examples/superexchange.py) follows the other basic
examples: configure the constructor and call `run()`. It reads
`data/Toxicity_original_MI_top20_standardized_0_1.csv` (20 feature columns plus
the target), uses a 20-qubit full graph with CUDA-Q exact expectations and all-pairs readout,
and saves artifacts under `./results`. Run it from a directory containing
`data/`, or edit `data_dir`. To select another dataset, set `name_file` and
`q_enc` to match its feature count. Edit the constructor arguments to change
topology, backend or measurements. With `framework="cudaq"`, use
`expectation_method="exact"` for analytical expectations, or keep `"shots"`
for sampling. The in-memory transformer examples are shown above.

```bash
python examples/superexchange.py
python -m unittest discover -s tests -p 'test_superexchange*.py' -v
python -m unittest discover -s tests -p 'test_heisenberg_all_pairs.py' -v
# Requires an accessible GPU; missing GPU/dependencies then cause a test failure.
PQFMLIB_TEST_CUDAQ_GPU=1 python -m unittest discover -s tests -p 'test_superexchange_backends.py' -v
```

### XYZ PQFM

`XYZProjectiveQFM` is the main map in PQFMLib. It generalizes the Ising idea
from one Pauli axis to several Pauli axes and can also include cross-axis
interactions. This generalized two-qubit Hamiltonian and its application as a
projective quantum feature map are presented in [4]. A compact way to write
the full Hamiltonian is:

```math
H_f
= \sum_i
(
h_i^{(x)} X_i
+ h_i^{(y)} Y_i
+ h_i^{(z)} Z_i
)
+ \sum_{i \lt j}
[J_{ij}^{(xx)} X_i X_j
+ J_{ij}^{(yy)} Y_i Y_j
+ J_{ij}^{(zz)} Z_i Z_j
+ J_{ij}^{(xy)}(X_iY_j + Y_iX_j)
+ J_{ij}^{(xz)}(X_iZ_j + Z_iX_j)
+ J_{ij}^{(yz)}(Y_iZ_j + Z_iY_j)].
```

The local fields $h_i^{(x)}$, $h_i^{(y)}$, and $h_i^{(z)}$ are normalized
axis-encoded features:

```math
h_i^{(a)} = \bar{x}_{f_a(i)},
\qquad
\bar{x}_{f_a(i)}
= \frac{x_{f_a(i)} - \mu_{f_a(i)}}{w_{f_a(i)}} ,
\qquad
a \in \{x,y,z\}.
```

The pairwise couplings between qubits $i$ and $j$ form a full axis-correlation
matrix:

```math
J_{ij}
=
\begin{pmatrix}
J_{ij}^{(xx)} & J_{ij}^{(xy)} & J_{ij}^{(xz)} \\
J_{ij}^{(yx)} & J_{ij}^{(yy)} & J_{ij}^{(yz)} \\
J_{ij}^{(zx)} & J_{ij}^{(zy)} & J_{ij}^{(zz)}
\end{pmatrix}.
```

With `keep_diagonal_terms=True`, the map includes the three Ising-like channels
$XX$, $YY$, and $ZZ$. With `keep_cross_terms=True`, it also includes cross-axis
correlations such as $XY$, $XZ$, and $YZ$. In feature-map terms, each qubit has
a local feature vector:

```math
\mathbf{x}_i
=
(
x_i^{(x)},
x_i^{(y)},
x_i^{(z)}
).
```

and each pair of qubits can carry a full correlation matrix between their
axis-encoded features.

## Axis Encoding Modes

`XYZProjectiveQFM` has two encoding modes.

### Multi-Axis Encoding

Use:

```python
encoding_mode="multi_axis"
features_per_qubit=3
axes=("x", "y", "z")
```

In multi-axis mode, each `(qubit, axis)` slot can receive a different tabular
feature:

```math
\begin{aligned}
\text{qubit } i,\ X\text{ axis} &\mapsto f_X(i), \\
\text{qubit } i,\ Y\text{ axis} &\mapsto f_Y(i), \\
\text{qubit } i,\ Z\text{ axis} &\mapsto f_Z(i).
\end{aligned}
```

This means the capacity per layer is:

```math
C_{\text{multi-axis}} = q_{\mathrm{enc}} \, |\mathrm{axes}| .
```

For example, $q_{\mathrm{enc}} = 10$ and `axes=("x", "y", "z")` can encode up to 30
features per layer. This mode is useful when the goal is to compress many
features into a small qubit register while preserving axis-specific structure.

You can also choose only two axes. For example:

```python
encoding_mode="multi_axis"
features_per_qubit=2
axes=("x", "y")
```

In this case, each qubit encodes two different features per layer, one on the
$X$ axis and one on the $Y$ axis.

### Shared-Feature Encoding

Use:

```python
encoding_mode="shared_feature"
features_per_qubit=1
axes=("x", "y", "z")
```

In shared-feature mode, each qubit receives one feature and reuses that same
feature across all selected axes:

```math
\begin{aligned}
\text{qubit } i,\ X\text{ axis} &\mapsto f(i), \\
\text{qubit } i,\ Y\text{ axis} &\mapsto f(i), \\
\text{qubit } i,\ Z\text{ axis} &\mapsto f(i).
\end{aligned}
```

The capacity per layer is:

```math
C_{\text{shared-feature}} = q_{\mathrm{enc}} .
```

This mode is useful when the experiment should compare or combine projections
of the same variable through different Pauli axes.

Shared-feature mode can also use two axes:

```python
encoding_mode="shared_feature"
features_per_qubit=1
axes=("x", "y")
```

Here each qubit still carries one feature, but that same feature is encoded on
both the $X$ and $Y$ axes. Thus, the feature is shared across two axes instead
of three.

## Layer Encoding

PQFMLib can encode more features than fit in one layer. It builds blocks from
the mutual-information matrix and stacks them as Hamiltonian layers. If a
dataset has $n_{\mathrm{features}}$ and one layer has capacity $C$, the number of blocks is
approximately:

```math
n_{\mathrm{blocks}}
= \lceil \frac{n_{\mathrm{features}}}{C} \rceil .
```

The layer mechanism can be mixed with axis encoding:

- `multi_axis` + layers: many different features per qubit per layer.
- `shared_feature` + layers: the same feature is reused across selected axes
  inside each layer, and additional features appear in later layers.
- diagonal + cross terms + layers: each layer can contain the $XX$, $YY$, and
  $ZZ$ Ising channels plus cross-axis couplings.

The physical repetition parameter $m$ controls how many Trotter steps are used
per block. Internally, PQFMLib builds a total circuit depth proportional to:

```math
m_{\mathrm{total}} = m \, n_{\mathrm{blocks}} .
```

## Important XYZ Options

```python
XYZProjectiveQFM(
    features_per_qubit=3,
    axes=("x", "y", "z"),
    encoding_mode="multi_axis",
    keep_diagonal_terms=True,
    keep_cross_terms=True,
    n_keep_terms=None,
    measure_all_zz=False,
    measure_cross_observables=False,
    use_edge_error=True,
    rho_thr=0.0,
)
```

Key options:

- `axes`: choose any unique subset of `("x", "y", "z")`.
- `encoding_mode`: choose `"multi_axis"` or `"shared_feature"`.
- `keep_diagonal_terms`: include same-axis terms such as $XX$, $YY$, $ZZ$.
- `keep_cross_terms`: include mixed-axis terms such as $XY$, $XZ$, $YZ$.
- `n_keep_terms`: keep only the strongest interaction terms in a layer.
- `measure_cross_observables`: measure cross-axis observables in addition to
  default one-local and diagonal observables.
- `use_edge_error`: when running on IBM backends, include edge quality in the
  feature-to-qubit assignment heuristic.
- `use_fixed_blocks` and `use_fixed_phys_nodes`: reproduce a previously saved
  feature assignment or physical layout.

## IBM Quantum Runtime

For real IBM Quantum execution, set a token before running:

```bash
export QISKIT_IBM_TOKEN="your-token"
```

On Windows PowerShell:

```powershell
$env:QISKIT_IBM_TOKEN="your-token"
```

Then set:

```python
ideal=False
simulation=False
ibm_qpu="ibm_kingston"
```

### Retrieving completed QPU jobs

Real QPU execution saves the submitted job ID and observable metadata in
`job_meta.json` inside the execution output folder. To download the results,
use the script for the corresponding map family:

| Map | Retrieval script |
| --- | --- |
| XYZ | [`scripts/get_job_xyz.py`](scripts/get_job_xyz.py) |
| CD-Ising | [`scripts/get_job_cd_ising.py`](scripts/get_job_cd_ising.py) |
| Heisenberg, including superexchange and multi-axis encoding | [`scripts/get_job_heisenberg.py`](scripts/get_job_heisenberg.py) |

**Use `get_job_heisenberg.py` for all Heisenberg variants:** the original
`HeisenbergProjectiveQFM`, `SuperexchangeHeisenbergPQFM` (or
`coupling_mode="superexchange"`), `encoding_mode="multi_axis"`, and selected
interaction axes such as `interaction_axes=("x", "y")` for XX/YY interactions.
The same script handles local, diagonal two-qubit, cross-axis, and all-pairs
observables recorded in the job metadata.

The script reads `job_meta.json`, retrieves the job with `service.job(job_id)`,
and checks that it has completed. It then reads `job.result()[0].data.evs` and
transposes the values into a samples-by-features matrix. Column names come
from `obs_metadata`, so retrieval does not need to rebuild the circuit or
reconstruct its encoding. Keep the original `job_meta.json` with the output
folder; the legacy fallback without observable metadata only infers names
for single-qubit features.

Before running the Heisenberg retrieval script, edit its configuration:

```python
BASE_FOLDER = "path/to/execution_output"  # Contains job_meta.json
NAME_FILE = "dataset_name"              # Without the .csv extension
DATA_DIR = "./data"
```

With the IBM token configured as above, run from the repository root after
installing the package:

```bash
python scripts/get_job_heisenberg.py
```

`DATA_DIR/NAME_FILE.csv` must contain the labels in its last column, with
samples in exactly the same order as the submitted data. The script checks
the sample count, but cannot detect reordered labels. It saves
`qfeatures_all_<NAME_FILE>_heisenberg.csv` (features plus `y`) and
`qfeatures_all_<NAME_FILE>_heisenberg.npy` (features only) in `BASE_FOLDER`.
If the job is still running, rerun the script after it completes.

Interaction axes and measured observables are independent: selecting XX/YY
interactions does not automatically restrict the measured observables.
Superexchange currently requires isotropic XYZ interactions and
`encoding_mode="shared_feature"`; combining superexchange with XX/YY-only
interactions or `multi_axis` encoding is rejected during validation. These
submission constraints do not change which retrieval script to use.

## References

[1] Anton Simen et al., "Digitized Counterdiabatic Quantum Feature Extraction,"
arXiv:2510.13807, 2025. https://arxiv.org/abs/2510.13807

[2] Axel Ciceri et al., "Enhanced fill probability estimates in institutional
algorithmic bond trading using statistical learning algorithms with quantum
computers," arXiv:2509.17715, 2025. https://arxiv.org/abs/2509.17715

[3] Andras Ferenczi et al., "Credit Default Prediction with Projected Quantum
Feature Models and Ensembles," arXiv:2510.01129, 2025.
https://arxiv.org/abs/2510.01129

[4] Rafael Simões do Carmo, Edson Amaro Junior, and Felipe Fanchini,
"Generalized two-qubit Hamiltonian for Projective Quantum Feature Maps,"
arXiv:2606.13641 [quant-ph], 2026. https://arxiv.org/abs/2606.13641

## Repository Structure

```text
pqfmlib/
|-- docs/
|-- examples/
|-- pqfmlib/
|   |-- core/
|   |-- hardware/
|   |-- maps/
|   |-- runners/
|   `-- utils/
|-- scripts/
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```
