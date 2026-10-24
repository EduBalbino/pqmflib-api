# Project rules

- Never modify `pqfmlib/core/`.
- Active Heisenberg benchmarks use exact expectations and a double-precision
  CUDA-Q GPU statevector baseline. Keep shots=4096 configured; exact mode does
  not sample. Sampling benchmarks and their warmups use 4,096 shots per basis.
  Never lower that budget for probes or change encoding/noise silently.
- Measure the complete 35,300-row SIA-SUS dataset in Heisenberg benchmarks. Warm
  each backend on 50 rows separately; do not substitute a smaller timing dataset.
- Default to one measured run. Reuse CUDA-Q reference outputs and timings from
  `experiments/sia/heisenberg_baseline.json`; reject changed inputs or reference
  configuration. Use `--refresh-baseline` to regenerate; label cached timings.
- Save correctness results to `experiments/sia/heisenberg_correctness.json`,
  including checked rows, thresholds and pass/fail status. Use `--check-only`
  to verify without repeating timings; preserve partial results on failure.
- Active Heisenberg uses R=2 on a logical chain without a hardware backend;
  disable hardware noise explicitly. Use double precision for CPU, TileLang
  and the public API. Encode consecutive SIA features in X/Y/Z interaction slots
  and retain every UF indicator. Keep CPU and GPU maps aligned.
- Keep the full runtime math on the GPU.
- Runtime GPU files must not import NumPy. Use NumPy only for the final result
  returned to a host caller.
- Build fixed data offline and load it as GPU tensors.
- Compare completed host-to-host transforms after warmup. Include scaling,
  parameter preparation, transfers and simulation for both backends. Keep fit,
  circuit setup, compilation and dataset preprocessing outside timed runs.
- Cache compiled kernels for fixed shapes. Avoid Python loops, extra copies,
  and syncs.
- Test topology candidates in one process. Compile and warm every candidate
  before timing. Reuse the compiler disk cache. Keep only faster code.
- Inspect generated CUDA/PTX and launch shapes before a full benchmark. Reject
  giant kernels, bad launch shapes, and extra copies first.
- Any file with approximate math must start with an `Approximation log`.
  For each approximation, state its valid range, error bound or measured error,
  and the exact change that would make it more accurate.

# Project identity

- The Git base is `https://github.com/rscarmo/pqfmlib-api.git`.
- The local checkout is `/home/pichau/pqfmlib-api`. Keep the upstream distribution
  and Python import name `pqfmlib`; that package is part of the new API repository.
- Keep the complete upstream API in this checkout. Integrate upstream updates
  by rebasing; do not create reference snapshots, duplicate clones, or archives.

# Focus and documentation

- Active development is GPU-focused Heisenberg with TileLang, exclusively for
  SIA-SUS. Preserve the other upstream maps as API/reference code. Keep useful
  TileLang integration examples under `examples/`; do not
  treat that XYZ study as the Heisenberg MoE implementation. Do not maintain or
  archive SIH. Investigate legacy code only as needed for the requested work.
- Prefer the smallest complete change. Update existing code and documents in
  place. Do not create extra Markdown files, reports, plans, READMEs, decision
  logs, or review artifacts unless explicitly requested.
- Documentation must explain the model, algorithm, behavior, constraints, and
  useful results directly. Omit conversion history, document origins, process
  narration, repeated caveats, and descriptions of the work performed.
- Use Mermaid when it clarifies a computational flow. Keep essential numerical
  and experimental limitations next to the result they qualify.
- Keep verification reproducible through existing commands and tests. Report
  concise outcomes in chat; do not create a document for each check.
- Start `TIMINGS.md` with the run command and shared circuit definition, stating
  that CPU and GPU use the same circuit. Follow with one table numbered #1, #2,
  etc., without dates. Replace superseded measurements. Keep numbered summaries
  below the table very short; do not repeat the shared circuit definition.
- Do not run tests or benchmarks during cleanup, file moves, or documentation
  work unless explicitly requested.
- The predictive comparison target is classical SIA-SUS demand/pressure under
  its temporal protocol. QIMED informs the method. Use the general Heisenberg
  backend; do not retain PCA20 options, Q21 compatibility paths, or old fixtures.
- Keep the method in `MODEL.md` and SIA inputs in `experiments/sia/`. Consolidate
  useful existing code before deleting it. Keep deletion rankings in chat,
  not in new documents; preserve complete upstream source and numerical oracles.
- The active SIA model uses every preprocessed feature without PCA, latent
  compression, or feature selection. Use multi-axis chain encoding with independent X/Y/Z angles per edge:
  50 features require 18 qubits, one zero-padded angle and 54 Z/X/Y outputs.

# Version control

- Use `just hooks` to install Lefthook. Before changing the GPU implementation,
  run `just reference`; then edit and run `just benchmark`. A candidate must
  pass exact correctness and be at least 10% faster than the previous accepted
  TileLang implementation on the same inputs, settings and GPU. After adding
  its TIMINGS row and mathematical explanation, run `just promote` for the
  guarded local commit, normal push to the configured PR branch, and PR sync.
  Do not manually stage, commit, push, or run `just pr` for a new improvement;
  `just benchmark` never publishes. The promotion command validates everything
  before committing and can safely retry a committed improvement. Evidence stays
  local under `.git/heisenberg-performance/`.
- Keep `TIMINGS.md` local, with the table first and `## #N` descriptions below.
  After a qualifying benchmark, update the table with its measured TileLang time
  and speedup, and add the matching numbered description before running `just
  promote`. The command must fail before commit or posting if the latest table
  row does not match the passing candidate or its description lacks a fenced
  mathematical explanation. It updates the private PR body and creates or updates
  each numbered description as a separate comment; repeating it must not duplicate
  comments. `just pr` is available to retry PR synchronization for an accepted
  commit.

- Commit descriptions and PR improvement comments follow: **Was this → With
  this math → Became this**. State the previous computation, show the specific
  mathematical transformation, then explain the resulting computation and what
  it saves in work, memory traffic or launches. Keep the explanation concise
  and concrete; equations must explain the change, not merely decorate it.
  Use readable equations in commit bodies and fenced `math` blocks in GitHub
  Markdown, with `\Re` and `\Im` instead of `\operatorname`.
  Keep routine correctness statistics in JSON outputs, not change summaries.

- Use standard Git for all version-control operations in this repository.
- Do not use GitButler or Jujutsu, or install their hooks or automation here.
- Keep this repository local unless the user explicitly requests publishing.
- Any GitHub repository created for this project must be private and independent,
  without a GitHub fork relationship.
- Preserve a local backup before rewriting history.
