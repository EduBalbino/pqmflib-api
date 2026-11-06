---
name: tilelang
description: Design, review, test, and optimize TileLang kernels while keeping data flow and mathematical assumptions easy to read.
---

# TileLang

Use this skill for TileLang kernel development or review. Check the installed TileLang version before relying on an API: this project's current environment uses 0.1.15.

1. Read the real caller, tensor shapes, dtypes, device, and reference implementation. State the mathematical operation and invariants before changing a kernel.
2. Prefer a small number of readable kernels that directly express the operation. Keep shapes, layouts, and data movement visible; remove redundant intermediate tensors and hidden mutable state when safe.
3. Use TileLang's native tile operations where they fit: `T.Parallel`, `T.alloc_fragment`, `T.copy`, and `T.reduce_sum`. Consult the installed version's API and upstream examples before assuming a feature or layout works on the target GPU. Do not assume newer hardware features exist on the user's device.
4. Check the complete output against an independent reference on representative real inputs and meaningful edge cases, including partial batches and fallback paths. Preserve precision and tolerance; explain any changed numerical behavior.
5. Benchmark the actual workload with identical inputs and synchronized GPU execution. Separate compilation, warm-up, transfer, and steady-state execution. Retain timing results and compare multiple runs when differences are small.
6. Keep changes within the requested scope. Do not commit a performance change unless it satisfies the user's current correctness and improvement gates.

References: [language basics](https://tilelang.com/programming_guides/language_basics.html), [instructions and reductions](https://tilelang.com/programming_guides/instructions.html), [software pipelines](https://tilelang.com/programming_guides/software_pipeline.html), and [official examples](https://github.com/tile-ai/tilelang/tree/main/examples).
