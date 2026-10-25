"""Backend-independent logical Heisenberg expectation contract."""

OBSERVABLE_AXES = ("z", "x", "y")
OBSERVABLE_COUNT = len(OBSERVABLE_AXES)
OUTPUT_ORDER = f"site-major {'/'.join(axis.upper() for axis in OBSERVABLE_AXES)}"
EXACT_ATOL = 1e-9


def expectation_shape(rows, qubits):
    """Shape required of exact Heisenberg backend results."""
    return rows, qubits * OBSERVABLE_COUNT
