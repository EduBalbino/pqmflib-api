"""Compare exact Heisenberg backends on the complete 35,300-row SIA dataset."""

import json

import numpy as np
import torch

from pqfmlib.experimental.heisenberg_cudaq import HeisenbergCudaQ
from pqfmlib.experimental.heisenberg_contract import EXACT_ATOL, expectation_shape
from experiments.sia.heisenberg_model import (
    SIA_DATASET, chain_circuit, load_sia_input,
)
from experiments.sia.heisenberg_reference import (
    REFERENCE_PATH, TIMING_SCOPE, load_reference, reference_identity, save_reference,
)

ROWS = 35300
CORRECTNESS_PATH = SIA_DATASET.parent / "heisenberg_correctness.json"


def measure(transform, values):
    """Measure completed host-to-host work with the same clock and GPU fences."""
    from time import perf_counter

    torch.cuda.synchronize()
    started = perf_counter()
    result = transform(values)
    torch.cuda.synchronize()
    return result, (perf_counter() - started) * 1000


def check_outputs(outputs, rows, qubits):
    """Require the same host float64 matrix from both backend adapters."""
    shape = expectation_shape(rows, qubits)
    if (not isinstance(outputs, np.ndarray) or outputs.shape != shape
            or outputs.dtype != np.float64):
        raise ValueError(f"Expected a host float64 expectation matrix of shape {shape}")


def run_benchmark(*, output=CORRECTNESS_PATH, check_only=False, refresh_baseline=False):
    from pqfmlib.experimental.heisenberg_tilelang import HeisenbergTileLang

    values = load_sia_input(SIA_DATASET)
    logical, parameters = chain_circuit()
    if (values.shape != (ROWS, len(parameters)) or values.dtype != np.float64
            or not np.isfinite(values).all()):
        raise ValueError(f"Expected all {ROWS:,} rows as finite float64 features")
    identity = reference_identity(values, logical, parameters)
    cached = REFERENCE_PATH.exists() and not refresh_baseline
    if check_only and not cached:
        raise ValueError("Checking requires an existing reference without regeneration")
    if not cached:
        print("Measuring CUDA-Q…", flush=True)
        baseline = HeisenbergCudaQ(logical, parameters)
        warmup = baseline.transform(values[:50])
        check_outputs(warmup, 50, logical.num_qubits)
        expected, elapsed = measure(baseline.transform, values)
        save_reference(REFERENCE_PATH, identity, expected, elapsed,
                       gpu=torch.cuda.get_device_name(), target=baseline.target)
    reference, expected = load_reference(REFERENCE_PATH, identity)
    if not check_only and reference["gpu"] != torch.cuda.get_device_name():
        raise ValueError("Reference timing uses another GPU; regenerate the reference explicitly")
    gpu = HeisenbergTileLang(
        logical, parameters, device=f"cuda:{torch.cuda.current_device()}")
    kernel_info = gpu.inspect_kernels(len(values))
    print(f"TileLang kernels: {kernel_info}", flush=True)
    if max(kernel_info["kernel_source_bytes"]) > 20000:
        raise RuntimeError("Unexpected generated CUDA kernel")
    warmup = gpu.transform(values[:50])
    check_outputs(warmup, 50, logical.num_qubits)
    if check_only:
        outputs, elapsed = gpu.transform(values), None
    else:
        print("Measuring TileLang…", flush=True)
        outputs, elapsed = measure(gpu.transform, values)
    check_outputs(outputs, len(values), logical.num_qubits)
    errors = np.max(np.abs(outputs - expected), axis=1)
    passed = np.isfinite(errors) & (errors <= EXACT_ATOL)
    report = {
        "status": "passed" if passed.all() else "failed", "rows": len(values),
        "threshold": EXACT_ATOL, "per_row_error": [float(e) if np.isfinite(e) else None for e in errors],
        "baseline_cached": cached, "baseline_measured_at": reference["measured_at"],
        "cudaq_ms": float(np.median(reference["wall_ms"])), "tilelang_ms": elapsed,
        "cudaq_rows_per_second": len(values) / (float(np.median(reference["wall_ms"])) / 1000),
        "tilelang_rows_per_second": None if elapsed is None else len(values) / (elapsed / 1000),
        "identity": identity, "scope": TIMING_SCOPE,
        "gpu": torch.cuda.get_device_name(), "kernels": kernel_info,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not passed.all():
        raise RuntimeError(f"Reference comparison failed: {output}")
    if elapsed is not None:
        print(f"TileLang: {report['tilelang_rows_per_second']:.1f} rows/s · "
              f"CUDA-Q: {report['cudaq_rows_per_second']:.1f} rows/s · "
              f"{report['cudaq_ms'] / elapsed:.2f}× CUDA-Q"
              + (" (cached baseline)" if cached else ""))
    else:
        print("Reference comparison passed.")

    return report


if __name__ == "__main__":
    run_benchmark()
