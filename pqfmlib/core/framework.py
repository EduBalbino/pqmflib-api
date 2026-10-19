"""Framework selection without importing optional simulation dependencies."""

from __future__ import annotations


def validate_framework(framework: str) -> None:
    if framework not in ("qiskit", "cudaq"):
        raise ValueError("framework must be 'qiskit' or 'cudaq'.")


def validate_framework_configuration(
    framework: str, *, simulation: bool, ideal: bool, fakebackend: bool,
) -> None:
    """Check CUDA-Q restrictions before legacy ideal-mode normalization."""
    validate_framework(framework)
    if framework == "cudaq" and (not simulation or not ideal or fakebackend):
        raise ValueError(
            "framework='cudaq' requires simulation=True, ideal=True and "
            "fakebackend=False; non-ideal simulation, fake backends and QPU "
            "execution are not supported."
        )
