"""Persist and validate the exact SIA CUDA-Q reference."""

import hashlib
import json
from pathlib import Path

import numpy as np
from experiments.sia.heisenberg_model import SIA_DATASET
from pqfmlib.experimental.heisenberg_contract import (
    OUTPUT_ORDER, expectation_shape,
)

REFERENCE_PATH = SIA_DATASET.parent / "heisenberg_baseline.json"
TIMING_SCOPE = (
    "Host float64 feature matrix to completed host float64 expectations. "
    "Includes scaling, parameter preparation, transfers, simulation and output assembly. "
    "Excludes dataset preprocessing, fit, circuit setup, compilation and 50-row warmup."
)
REFERENCE_SOURCES = (
    "experiments/sia/heisenberg_model.py",
    "pqfmlib/experimental/heisenberg_cudaq.py",
    "pqfmlib/experimental/heisenberg_contract.py",
    "pqfmlib/maps/heisenberg.py",
    "pqfmlib/core/observables.py",
    "pqfmlib/core/cudaq_execution.py",
    "pqfmlib/core/execution.py",
    "pqfmlib/core/framework.py",
)


def reference_identity(values, circuit, parameters):
    """Identify the supplied logical circuit, inputs and reference implementation."""
    root = Path(__file__).resolve().parents[2]
    logical = {
        "qubits": circuit.num_qubits,
        "parameters": [str(parameter) for parameter in parameters],
        "global_phase": str(circuit.global_phase),
        "instructions": [
            (item.operation.name,
             [circuit.find_bit(qubit).index for qubit in item.qubits],
             [str(angle) for angle in item.operation.params])
            for item in circuit.data
        ],
    }
    return {
        "schema": 2, "rows": len(values),
        "outputs": expectation_shape(len(values), circuit.num_qubits)[1],
        "circuit_sha256": hashlib.sha256(json.dumps(logical, sort_keys=True).encode()).hexdigest(),
        "input_sha256": hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest(),
        "source_sha256": hashlib.sha256(
            b"".join((root / path).read_bytes() for path in REFERENCE_SOURCES)
        ).hexdigest(),
        "output_order": OUTPUT_ORDER,
    }


def load_reference(path, identity):
    """Read saved exact outputs, rejecting mismatched or corrupt references."""

    reference = json.loads(path.read_text())
    if reference["identity"] != identity:
        raise ValueError("Baseline JSON does not match these inputs or configuration; regenerate the reference explicitly")
    expected = np.asarray(reference["expectations"], dtype="<f8")
    if (expected.shape != (identity["rows"], identity["outputs"])
            or not np.isfinite(expected).all()):
        raise ValueError("Invalid baseline expectations")
    if hashlib.sha256(expected.tobytes()).hexdigest() != reference["expectations_sha256"]:
        raise ValueError("Baseline expectations checksum mismatch")
    return reference, expected


def save_reference(path, identity, expected, elapsed, *, gpu, target):
    """Persist measured outputs and metadata without depending on an adapter."""
    from datetime import datetime, timezone
    from importlib.metadata import version

    expected = np.asarray(expected, dtype="<f8")
    if (expected.shape != (identity["rows"], identity["outputs"])
            or not np.isfinite(expected).all()):
        raise ValueError("Invalid CUDA-Q baseline output")
    reference = {
        "identity": identity,
        "expectations": expected.tolist(),
        "expectations_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
        "wall_ms": [elapsed], "scope": TIMING_SCOPE,
        "gpu": gpu, "target": target,
        "cudaq_version": version("cuda-quantum-cu12"),
        "measured_at": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(reference, allow_nan=False) + "\n")
    temporary.replace(path)
    return reference
