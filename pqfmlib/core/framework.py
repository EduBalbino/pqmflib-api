"""Framework selection without importing optional simulation dependencies."""

from __future__ import annotations


def validate_framework(framework: str) -> None:
    if framework not in ("qiskit", "cudaq"):
        raise ValueError("framework must be 'qiskit' or 'cudaq'.")


def validate_precision(precision: str) -> None:
    if precision not in ("single", "double"):
        raise ValueError("precision must be 'single' (FP32) or 'double' (FP64).")


def validate_aer_precision(precision: str, method: str) -> None:
    validate_precision(precision)
    if precision == "single" and method not in (
        "statevector", "density_matrix", "unitary", "superop", "tensor_network",
    ):
        raise ValueError(
            f"precision='single' is unsupported for Aer method={method!r}; "
            "choose an explicit FP32-capable method or precision='double'."
        )


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
