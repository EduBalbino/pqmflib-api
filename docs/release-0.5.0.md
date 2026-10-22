# PQFMLib 0.5.0 — Configurable Heisenberg encodings

The Heisenberg projected quantum feature map now supports additional
interaction encodings, hardware-aware connectivity and optional cross
observables. The default `topology="chain"`, `encoding_mode="shared_feature"`
and `interaction_axes=("x", "y", "z")` preserve the original chain encoding.

## Interaction topology and encoding

- `topology="chain"` uses the existing nearest-neighbor schedule.
- `topology="full"` uses every unordered logical qubit pair and requires
  `ideal=True`.
- `topology="native"` uses the induced graph on a connected subset of physical
  backend qubits and requires `ideal=False`.
- `encoding_mode="shared_feature"` shares one input feature across selected
  axes on each edge. `encoding_mode="multi_axis"` assigns a separate feature
  per selected axis and edge.
- `interaction_axes` selects a unique ordered subset of `x`, `y`, and `z`.
  Its order defines feature assignment for `multi_axis`.

Use `measure_2local_diagonal=True` to add separate `ZZ`, `XX`, and `YY`
observables, and `measure_cross_observables=True` to add `XY`, `YX`, `XZ`,
`ZX`, `YZ`, and `ZY`. These options affect measurements only; they do not
change the encoding interactions.

```python
from pqfmlib import HeisenbergProjectiveQFM

qfm = HeisenbergProjectiveQFM(
    name_file="my_dataset", q_enc=3, ideal=True,
    topology="full", interaction_axes=("x", "y"),
    encoding_mode="multi_axis", measure_cross_observables=True,
    expectation_method="statevector",
)
```

See the [Heisenberg section in the README](../README.md#heisenberg-pqfm) for
feature ordering, output columns, native connectivity, circuit metadata, and
QPY compatibility details. The [topology example](../examples/heisenberg_topologies.py)
shows the `run()` workflow.

## Validation

The local Heisenberg transformer, topology, fixed-circuit,
framework-selection, statevector and precision suites passed: 72 passed and
10 opt-in GPU tests skipped. These checks did not submit jobs to quantum
hardware. The test run emitted Qiskit deprecation warnings.
