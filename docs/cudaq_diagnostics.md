# CUDA-Q implementation and diagnostics

For installation and normal usage, see the [README](../README.md).
Run the diagnostic snippets from the repository root after fitting a map.

The adapter caches the parameterized kernel across `transform()` calls. With
CUDA-Q, XX/YY/ZZ rotations are preserved as native `exp_pauli(-theta/2, ...)`
operations instead of decomposing them in the adapter into CNOTs and
single-qubit gates. The factor `-theta/2` matches Qiskit's rotation convention.
Here, `exp_pauli` describes a Pauli rotation, not a general-purpose matrix
exponential: for a Pauli product P, P squared is the identity, so
`exp(-i theta P / 2) = cos(theta/2) I - i sin(theta/2) P`.

Keeping this operation intact lets the CUDA-Q compiler and simulator choose
how to execute it. Fewer operations emitted by the adapter do not, by
themselves, establish a runtime improvement. Performance relative to an
explicit decomposition must be measured for the chosen circuit, simulator
and GPU. This translation applies independently of the total qubit count;
gate ordering, precision and truncation settings are retained.

With CUDA-Q MPS and `expectation_method="exact"`, all distinct Pauli terms are passed
to a single `observe()` call per sample. The adapter enables
`CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE=TRUE` before initializing the native
simulator: this path provides the per-term expectations needed by list-observe
and reuses contraction planning. See the
[CUDA-Q 0.16 tensor-network options](https://nvidia.github.io/cuda-quantum/0.16.0/using/backends/sims/tnsims.html).
It avoids the earlier `get_state()` followed by a separate state-initializing
kernel for each observable. No dense statevector is extracted.

With CUDA-Q MPS and `expectation_method="shots"`, Pauli terms are grouped by
compatible single-qubit measurement bases. Each group uses one `sample()` call
with `shots` measurements, shared by all terms in that group. Expectations are
computed from bitstring parities, including cross-axis terms and weighted sums.
For more than one qubit, the state is prepared once per input row and loaded
once per group, rather than once per observable. For one qubit, each group
executes the original circuit because CUDA-Q 0.16 cannot load a one-qubit MPS.
For 63 single-qubit X/Y/Z terms, this means three sampling calls and no
`observe()` calls per row. Grouping is deterministic but not guaranteed minimal.
The marginal shot estimators are preserved; shared samples introduce covariance
between compatible observables, and seeded values differ from the previous
scalar implementation. Repeated calls with the same seed and inputs remain
reproducible. Precision, bond limits and truncation settings are unchanged.

`pqfm.backend.last_execution_stats` reports the most recent successful CUDA-Q
batch's call counts and host wall times. `observe_seconds` includes native
circuit execution, MPS factorization and observable evaluation; it does not
isolate contraction time. For 32 samples with 63 observables, the analytic MPS
path issues 32 observe calls instead of 2,016. This call-count reduction is not
a measured speedup. GPU numerical and performance validation of the batched
MPS path must be run in the target environment; the opt-in tests in the README cover
numerical equivalence. Restart notebook kernels after updating the library.

For a direct CUDA-Q diagnostic, [benchmark_heisenberg_cudaq.py](../scripts/benchmark_heisenberg_cudaq.py)
provides `benchmark_heisenberg_cudaq_direct(fitted_map, X, repeats=3)`.
It independently constructs a Heisenberg kernel from the original gates and
compares it with the adapter on the same rows and observables, with identical
GPU MPS settings. Pass already-PCA-transformed inputs when applicable. Warm-ups
are excluded and features must agree within `atol=rtol=1e-10`. The adapter timing
includes parameter preprocessing; the direct route precomputes its angles.
This diagnostic does not run on import and does not compare against CPU. Its
independent circuit construction has CPU matrix-equivalence tests; GPU timing
and numerical validation remain opt-in.

The same module provides `benchmark_heisenberg_cudaq_shots` to diagnose the
grouped MPS sampling path against analytic expectations and independent direct
CUDA-Q kernels. The direct shots route deliberately retains the old scalar
`observe()` implementation as a performance baseline:

```python
from scripts.benchmark_heisenberg_cudaq import benchmark_heisenberg_cudaq_shots

# Use a fitted ideal CUDA-Q MPS map and already-PCA-transformed input rows.
report = benchmark_heisenberg_cudaq_shots(
    pqfm, X_new[:1], repeats=1, shots=4096, per_row=True,
)
print(report.loc[~report.warmup, [
    "path", "expectation_method", "seconds_per_sample", "observe_calls",
    "get_state_calls", "state_kernel_initializations", "state_seconds",
    "observe_seconds", "sample_calls", "sample_seconds", "max_abs_error",
]])
```

Each of the four routes runs once for warm-up (`repeat=-1`) and then `repeats`
times. Start with one input row: scalar MPS shots can be slow. `per_row=True`
creates a fresh execution for each row, matching the backend lifecycle of
row-wise `transform`; `False` executes a batch. The diagnostic excludes fit,
input preprocessing and Python kernel construction, so it is not an end-to-end
`transform` timer. Native initialization/JIT costs can still occur inside timed
execution. The fitted map and its execution statistics are not modified.

Times and counts are recorded per route/repetition. `state_seconds` measures
`get_state`; `observe_seconds` includes native evolution, factorization,
contractions and sampling as applicable. `sample_seconds` measures the grouped
`sample()` calls (including state loading and measurement preparation).
These are not isolated GPU kernel
timings. `state_kernel_initializations` counts calls that initialize a kernel
from a saved MPS, not native SVD operations. Exact routes must agree within
`atol=rtol=1e-10`; shot errors are reported against the analytic MPS reference
without requiring equality across independently constructed kernels. That
reference can itself contain MPS truncation error. Production
`last_execution_stats` also includes the method, shot count, observable count,
distinct Pauli term count, measurement group count, sampling call count/time
and state-kernel initialization count. Statistics are
available only after a successful execution.

