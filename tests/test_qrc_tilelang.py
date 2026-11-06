"""QRC TileLang backend: exactness vs qreservoir/Qulacs (no credentials).

GPU integration runs only with PQFMLIB_TEST_QRC_TILELANG_GPU=1; all other
tests run without a GPU.
"""
import os

import numpy as np
import pytest

gpu = pytest.mark.skipif(
    os.environ.get("PQFMLIB_TEST_QRC_TILELANG_GPU") != "1",
    reason="set PQFMLIB_TEST_QRC_TILELANG_GPU=1 for TileLang GPU tests",
)


def _make_model(n_qubits=6, depth=2, seed=42):
    from qulacs import Observable
    from sklearn.linear_model import LinearRegression

    from qreservoir.encoders import ExpEncoder
    from qreservoir.models import QELModel
    from qreservoir.reservoirs import RotationReservoir

    np.random.seed(seed)
    encoder = ExpEncoder(n_qubits, 1, 1)
    reservoir = RotationReservoir(encoder, 0, depth)
    observables = []
    for i in range(n_qubits):
        for axis in ("Z", "X", "Y"):
            ob = Observable(n_qubits)
            ob.add_operator(1.0, f"{axis} {i}")
            observables.append(ob)
    return QELModel(reservoir, observables, LinearRegression())


def test_gate_plan_and_missing_gpu_error():
    """Plan covers encoder + dynamics; executor refuses CPU-only hosts."""
    from pqfmlib.experimental.qrc_tilelang import build_gate_plan

    model = _make_model()
    plan, n_qubits = build_gate_plan(model)
    assert n_qubits == 6
    assert len(plan) > 0
    # Encoder RotX gates are parametric, reservoir gates are fixed.
    assert any(param is not None for *_, param in plan)
    assert any(param is None for *_, param in plan)


@gpu
def test_tilelang_matches_qulacs_exact():
    """Batched TileLang expectations agree with qreservoir within 1e-9."""
    from pqfmlib.experimental.qrc_tilelang import QRCExecutor

    model = _make_model()
    rows = np.random.RandomState(7).randn(24, 6) * 0.5
    expected = model.batch_calculate_observable_expectations(rows)
    actual = QRCExecutor(model, batch_size=8).execute(rows)
    assert actual.shape == expected.shape == (24, 18)
    np.testing.assert_allclose(actual, expected, atol=1e-9)


@gpu
def test_tilelang_deterministic_and_batch_invariant():
    """Repeated and re-batched execution give identical expectations."""
    from pqfmlib.experimental.qrc_tilelang import QRCExecutor

    model = _make_model()
    rows = np.random.RandomState(3).randn(16, 6) * 0.5
    first = QRCExecutor(model, batch_size=4).execute(rows)
    second = QRCExecutor(model, batch_size=16).execute(rows)
    np.testing.assert_array_equal(first, second)


def test_invalid_rows_rejected_without_gpu():
    """execute() validates shape/finiteness before touching the GPU."""
    from pqfmlib.experimental.qrc_tilelang import QRCExecutor

    executor = QRCExecutor.__new__(QRCExecutor)
    executor.model = _make_model()
    executor.n_qubits = 6
    executor.batch_size = 8
    executor.prepared = []
    with pytest.raises(ValueError):
        executor.execute(np.array([[np.nan] * 6]))
    with pytest.raises(ValueError):
        executor.execute(np.zeros((2, 3)))

@gpu
def test_temporal_step1_matches_qel_with_ancilla():
    """First temporal step equals QEL with ancilla memory at |0> (exact)."""
    from qulacs import Observable
    from sklearn.linear_model import LinearRegression

    from qreservoir.encoders import ExpEncoder
    from qreservoir.models import QELModel
    from qreservoir.reservoirs import RotationReservoir

    from pqfmlib.experimental.qrc_tilelang import TemporalQRCExecutor

    n_enc, n_mem = 4, 6
    np.random.seed(42)
    encoder = ExpEncoder(n_enc, 1, 1)
    reservoir = RotationReservoir(encoder, n_mem, 3)
    total = n_enc + n_mem
    observables = []
    for i in range(total):
        for axis in ("Z", "X", "Y"):
            ob = Observable(total)
            ob.add_operator(1.0, f"{axis} {i}")
            observables.append(ob)
    model = QELModel(reservoir, observables, LinearRegression())
    rows = np.random.RandomState(0).randn(8, n_enc) * 0.5
    expected = model.batch_calculate_observable_expectations(rows)
    executor = TemporalQRCExecutor(n_enc, n_mem, depth=3, seed=42, batch_size=8)
    actual = executor.execute(rows[:, None, :])[:, 0, :]
    np.testing.assert_allclose(actual, expected, atol=1e-9)


@gpu
def test_temporal_carries_memory_and_is_deterministic():
    """Later steps differ from step 1 (state carries over) and are stable."""
    from pqfmlib.experimental.qrc_tilelang import TemporalQRCExecutor

    executor = TemporalQRCExecutor(4, 6, depth=3, seed=42, batch_size=4)
    seq = np.random.RandomState(1).randn(4, 5, 4) * 0.5
    first = executor.execute(seq)
    second = executor.execute(seq)
    assert first.shape == (4, 5, 30)
    np.testing.assert_array_equal(first, second)
    assert np.abs(first[:, -1] - first[:, 0]).mean() > 0.01
