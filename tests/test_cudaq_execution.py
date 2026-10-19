"""GPU integration: PQFMLIB_TEST_CUDAQ_GPU=1 python -m pytest -q tests/test_cudaq_execution.py.

GPU tests are opt-in. When enabled, missing CUDA-Q/targets are failures rather
than skips. All other tests run without CUDA-Q, credentials or GPU access.
"""

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from qiskit import QuantumCircuit, qpy
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector

from pqfmlib import CDIsingProjectiveQFM, HeisenbergProjectiveQFM, XYZProjectiveQFM
from pqfmlib.core.cudaq_execution import CudaqExecutor, _CompiledJob, _validate_unitary_circuit
from pqfmlib.core.execution import PreparedProjectedFeatureJob
from pqfmlib.core.observables import pauli_observables_axis_pairs


gpu = pytest.mark.skipif(os.environ.get("PQFMLIB_TEST_CUDAQ_GPU") != "1",
                         reason="Set PQFMLIB_TEST_CUDAQ_GPU=1 to run on an NVIDIA GPU.")
MAP_TYPES = (HeisenbergProjectiveQFM, CDIsingProjectiveQFM, XYZProjectiveQFM)
X = np.random.default_rng(29).normal(size=(8, 4))


@pytest.mark.parametrize("map_type", MAP_TYPES)
def test_cudaq_exact_option_and_gpu_only_validation(map_type):
    assert map_type("test", framework="cudaq", mps=True, expectation_method="exact").mps
    with pytest.raises(ValueError, match="expectation_method"):
        map_type("test", framework="qiskit", expectation_method="exact")
    with pytest.raises(ValueError, match="mps=False"):
        map_type("test", framework="cudaq", mps=True, expectation_method="statevector")
    qfm = map_type("test", framework="cudaq", statevector_device="CPU")
    with pytest.raises(ValueError, match="requires a GPU"):
        qfm.setup_backend_and_estimator()


@pytest.mark.parametrize("kwargs", [
    {"mps_max_bond_dimension": 0}, {"mps_max_bond_dimension": 1.5},
    {"mps_max_bond_dimension": True}, {"mps_max_bond_dimension": 2**31},
    {"mps_truncation_threshold": -1}, {"mps_truncation_threshold": float("nan")},
    {"mps_truncation_threshold": 1},
])
def test_invalid_mps_options_do_not_import_cudaq(kwargs):
    with patch("pqfmlib.core.cudaq_execution._load_cudaq") as load:
        with pytest.raises(ValueError, match="CUDA-Q mps_"):
            CudaqExecutor(**kwargs)
    load.assert_not_called()


def test_missing_gpu_has_clear_error():
    cudaq = Mock()
    cudaq.num_available_gpus.return_value = 0
    with patch("pqfmlib.core.cudaq_execution._load_cudaq", return_value=cudaq):
        with pytest.raises(RuntimeError, match="accessible NVIDIA GPU"):
            CudaqExecutor()
    cudaq.set_target.assert_not_called()


def test_target_and_environment_restored_on_failure():
    cudaq = Mock()
    cudaq.num_available_gpus.return_value = 1
    previous_target = cudaq.get_target.return_value
    with patch("pqfmlib.core.cudaq_execution._load_cudaq", return_value=cudaq):
        executor = CudaqExecutor(mps=True)
    with patch.dict(os.environ, {"CUDAQ_MPS_MAX_BOND": "23"}):
        previous_env = dict(os.environ)
        with pytest.raises(RuntimeError, match="test failure"):
            with executor._target_context(8):
                assert os.environ["CUDAQ_MPS_MAX_BOND"] == "16"
                raise RuntimeError("test failure")
        assert dict(os.environ) == previous_env
        cudaq.set_target.assert_called_with(previous_target)
        cudaq.set_target.side_effect = [RuntimeError("missing simulator"), None]
        with pytest.raises(RuntimeError, match="tensornet-mps.*unavailable"):
            with executor._target_context(8):
                pass
        assert dict(os.environ) == previous_env


@pytest.mark.parametrize("previous_reuse", [None, "OFF", "TRUE"])
def test_mps_exact_batches_observables_and_restores_native_option(previous_reuse):
    """Protect against the 32*63 launches regression without needing a GPU."""
    cudaq = Mock()
    cudaq.num_available_gpus.return_value = 1
    key = "CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE"
    with patch.dict(os.environ):
        if previous_reuse is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous_reuse
        before = dict(os.environ)
        with patch("pqfmlib.core.cudaq_execution._load_cudaq", return_value=cudaq):
            executor = CudaqExecutor(mps=True)
        observables = [object() for _ in range(63)]
        expected = np.linspace(-0.9, 0.9, 63)
        terms = tuple(((i, 1.0),) for i in range(63)) + (((-1, .2), (2, -.7), (0, .5)),)
        compiled = _CompiledJob(object(), (), observables, terms, 21)
        prepared = SimpleNamespace(qc_t=QuantumCircuit(21), param_order=(), obs_metadata=object(),
                                   obs_isa_broadcast=[None] * len(terms))

        def observe(kernel, operators, angles, **kwargs):
            assert os.environ[key] == "TRUE"
            assert kernel is compiled.kernel
            assert operators is observables
            assert kwargs == {"shots_count": -1}
            return [SimpleNamespace(expectation=lambda value=value: value) for value in expected]

        cudaq.observe.side_effect = observe
        with patch("pqfmlib.core.cudaq_execution._compile_job", return_value=compiled) as compile_job:
            result, metadata = executor.execute(prepared, np.empty((32, 0)), expectation_method="exact")
            np.testing.assert_allclose(result[:, :63], np.tile(expected, (32, 1)))
            np.testing.assert_allclose(result[:, 63], .2 - .7 * expected[2] + .5 * expected[0])
            assert metadata is prepared.obs_metadata
            assert cudaq.observe.call_count == 32
            assert executor.last_execution_stats["observe_calls"] == 32
            assert executor.last_execution_stats["get_state_calls"] == 0
            assert executor.last_execution_stats["execution_path"] == "mps_batched_exact"
            executor.execute(prepared, np.empty((1, 0)), expectation_method="exact")
            compile_job.assert_called_once()
            assert dict(os.environ) == before
            cudaq.observe.side_effect = RuntimeError("observe failed")
            with pytest.raises(RuntimeError, match="observe failed"):
                executor.execute(prepared, np.empty((1, 0)), expectation_method="exact")
            assert executor.last_execution_stats is None
            assert dict(os.environ) == before
        cudaq.get_state.assert_not_called()
        cudaq.make_kernel.assert_not_called()


@pytest.mark.parametrize("operation", ["reset", "initialize", "measure", "delay"])
def test_nonunitary_or_classical_circuits_rejected(operation):
    qc = QuantumCircuit(1)
    if operation == "initialize":
        qc.initialize([1, 0], 0)
    elif operation == "measure":
        qc.measure_all()
    elif operation == "delay":
        qc.delay(10, 0)
    else:
        qc.reset(0)
    with pytest.raises(ValueError, match="CUDA-Q"):
        _validate_unitary_circuit(qc)


@gpu
@pytest.mark.parametrize("mps", [False, True])
@pytest.mark.parametrize("map_type", MAP_TYPES)
def test_all_maps_match_qiskit_and_preserve_lifecycle(map_type, mps):
    kwargs = dict(name_file="compare", q_enc=3, seed=11)
    if map_type is XYZProjectiveQFM:
        kwargs.update(measure_cross_observables=True, features_per_qubit=1,
                      encoding_mode="shared_feature", axes=("z", "x", "y"))
    elif map_type is HeisenbergProjectiveQFM:
        kwargs.update(measure_2local_diagonal=True, alpha=0.9, R=3)
    else:
        kwargs.update(m=2, measure_all_zz=True)
    reference = map_type(**kwargs, expectation_method="statevector")
    expected = reference.fit_transform(X)
    actual_map = map_type(**kwargs, framework="cudaq", mps=mps, expectation_method="exact")
    actual = actual_map.fit_transform(X)
    if mps:
        assert actual_map.backend.last_execution_stats["observe_calls"] == len(X)
        assert actual_map.backend.last_execution_stats["get_state_calls"] == 0
    assert actual_map.backend.target == ("tensornet-mps" if mps else "nvidia")
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert actual_map.obs_metadata == reference.obs_metadata
    cached_kernel = actual_map.backend._compiled.kernel
    np.testing.assert_allclose(actual_map.transform(X[:2]), expected[:2], atol=1e-10, rtol=1e-10)
    assert actual_map.backend._compiled.kernel is cached_kernel
    actual_map.set_params(framework="qiskit")
    with pytest.raises(RuntimeError, match="configuration changed"):
        actual_map.transform(X[:1])


@gpu
@pytest.mark.parametrize("mps", [False, True])
def test_adapter_parameter_order_pauli_endianness_and_zero_parameters(mps):
    import cudaq

    parameters = ParameterVector("theta", 12)
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.rx(0.23, 2)
    qc.ry(parameters[10] * 0.7 + parameters[2], 1)
    qc.rzz(-2 * parameters[2], 0, 2)
    qc.ryy(parameters[10], 2, 1)
    qc.rxx(parameters[2], 1, 0)
    obs, meta = pauli_observables_axis_pairs(3, measure_cross_observables=True)
    obs += [SparsePauliOp.from_list([("III", 0.2), ("XYZ", -0.7), ("YIX", 0.5)])]
    prepared = PreparedProjectedFeatureJob(qc, (parameters[10], parameters[2]), [[o] for o in obs], meta)
    values = np.array([[0.83, -0.41], [-0.2, 0.53]])
    expected = np.array([[Statevector.from_instruction(qc.assign_parameters(
        dict(zip(prepared.param_order, row)))).expectation_value(o).real for o in obs] for row in values])
    previous_target = cudaq.get_target()
    executor = CudaqExecutor(mps=mps, mps_truncation_threshold=0)
    actual, returned_metadata = executor.execute(prepared, values, expectation_method="exact")
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10)
    assert returned_metadata is meta
    assert str(cudaq.get_target()) == str(previous_target)
    constant = QuantumCircuit(3)
    constant.x(0)
    prepared = PreparedProjectedFeatureJob(constant, (), [[o] for o in obs[:3]], meta)
    actual, _ = executor.execute(prepared, np.empty((2, 0)), expectation_method="exact")
    expected = [Statevector.from_instruction(constant).expectation_value(o).real for o in obs[:3]]
    np.testing.assert_allclose(actual, [expected, expected], atol=1e-12)


@gpu
@pytest.mark.parametrize("mps", [False, True])
@pytest.mark.parametrize("map_type", MAP_TYPES)
def test_run_exports_same_features_as_qiskit(map_type, mps):
    with tempfile.TemporaryDirectory() as folder:
        frame = pd.DataFrame(X)
        frame["y"] = np.arange(len(X)) % 2
        frame.to_csv(Path(folder) / "data.csv", index=False)
        kwargs = dict(name_file="data", data_dir=folder, q_enc=2)
        reference = map_type(**kwargs, output_root=str(Path(folder) / "qiskit"),
                             expectation_method="statevector").run()
        qfm = map_type(**kwargs, output_root=str(Path(folder) / "cudaq"),
                       framework="cudaq", mps=mps, expectation_method="exact")
        actual = qfm.run()
        assert actual.keys() == reference.keys()
        np.testing.assert_allclose(actual["Xq_all_raw"], reference["Xq_all_raw"], atol=1e-10, rtol=1e-10)
        np.testing.assert_array_equal(np.load(actual["npy_path"]), actual["Xq_all_raw"])
        actual_frame = pd.read_csv(actual["csv_path"])
        assert list(actual_frame.columns) == list(pd.read_csv(reference["csv_path"]).columns)
        np.testing.assert_array_equal(actual_frame["y"], frame["y"])


@gpu
@pytest.mark.parametrize("mps", [False, True])
def test_shots_repeatability_and_convergence(mps):
    qfm = HeisenbergProjectiveQFM("shots", q_enc=2, framework="cudaq", mps=mps,
                                 shots=16384, alpha=1.2)
    first = qfm.fit_transform(X[:3])
    np.testing.assert_array_equal(qfm.transform(X[:3]), first)
    reference = HeisenbergProjectiveQFM("exact", q_enc=2, alpha=1.2,
                                      expectation_method="statevector").fit_transform(X[:3])
    np.testing.assert_allclose(first, reference, atol=0.04, rtol=0)


@gpu
def test_statevector_alias_and_alternating_gpu_targets():
    maps = [HeisenbergProjectiveQFM("switch", q_enc=2, framework="cudaq", mps=mps,
                                   expectation_method=method)
            for mps, method in ((False, "statevector"), (True, "exact"))]
    expected = maps[0].fit_transform(X[:2])
    np.testing.assert_allclose(maps[1].fit_transform(X[:2]), expected, atol=1e-10)
    np.testing.assert_allclose(maps[0].transform(X[:2]), expected, atol=1e-10)


@gpu
def test_fixed_qpy_circuit_uses_same_parameters_and_observables():
    reference = HeisenbergProjectiveQFM("fixed", q_enc=2, expectation_method="statevector")
    expected = reference.fit_transform(X[:2])
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "fixed.qpy"
        with path.open("wb") as stream:
            qpy.dump(reference._prepared_execution.qc_t, stream)
        for mps in (False, True):
            actual = HeisenbergProjectiveQFM("fixed", q_enc=2, framework="cudaq", mps=mps,
                                             expectation_method="exact", fixed_circuit_file_name=str(path))
            np.testing.assert_allclose(actual.fit_transform(X[:2]), expected, atol=1e-10)


@gpu
def test_mps_does_not_reuse_a_smaller_circuits_bond_limit():
    executor = CudaqExecutor(mps=True)
    for n in (2, 4, 2, 4):
        qc = QuantumCircuit(n)
        for i in range(n // 2):
            qc.h(i)
            qc.cx(i, i + n // 2)
        obs, meta = pauli_observables_axis_pairs(n)
        prepared = PreparedProjectedFeatureJob(qc, (), [[o] for o in obs], meta)
        actual, _ = executor.execute(prepared, np.empty((1, 0)), expectation_method="exact")
        state = Statevector.from_instruction(qc)
        expected = [[state.expectation_value(o).real for o in obs]]
        np.testing.assert_allclose(actual, expected, atol=1e-10)


@gpu
def test_one_qubit_mps_and_qiskit_gpu_remain_available():
    from qiskit_aer import AerSimulator

    qc = QuantumCircuit(1)
    qc.rx(0.71, 0)
    qc.ry(-0.33, 0)
    obs, meta = pauli_observables_axis_pairs(1)
    prepared = PreparedProjectedFeatureJob(qc, (), [[o] for o in obs], meta)
    actual, _ = CudaqExecutor(mps=True).execute(prepared, np.empty((1, 0)), expectation_method="exact")
    expected = [[Statevector.from_instruction(qc).expectation_value(o).real for o in obs]]
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    # Check coexistence with an existing Aer GPU installation when available.
    if "GPU" in AerSimulator().available_devices():
        qc.save_statevector()
        result = AerSimulator(method="statevector", device="GPU").run(qc).result()
        assert result.success
        np.testing.assert_allclose([[Statevector(result.get_statevector()).expectation_value(o).real
                                     for o in obs]], expected, atol=1e-10)


class _RecordingKernel:
    """Record CUDA-Q calls, then evaluate their documented matrices in Qiskit."""
    def __init__(self):
        self.operations = []
        self.num_qubits = 0

    def qalloc(self, n):
        self.num_qubits = n
        return list(range(n))

    def exp_pauli(self, angle, *args):
        self.operations.append(('exp_pauli', angle, args[:-1], args[-1]))

    def __getattr__(self, name):
        if name in ('rx', 'ry', 'rz'):
            return lambda angle, qubit: self.operations.append((name, angle, (qubit,), None))
        if name in ('h', 'x', 'cx'):
            return lambda *qubits: self.operations.append((name, None, qubits, None))
        raise AttributeError(name)

    def to_circuit(self, params, values):
        from scipy.linalg import expm
        from qiskit.quantum_info import Pauli
        qc = QuantumCircuit(self.num_qubits)
        mapping = dict(zip(params, values))
        for name, angle, qubits, word in self.operations:
            if hasattr(angle, 'parameters'):
                angle = float(angle.bind({p: mapping[p] for p in angle.parameters}))
            if name == 'exp_pauli':
                # Independent matrix definition, not a call to Qiskit's rpp.
                qc.unitary(expm(1j * float(angle) * Pauli(word).to_matrix()), list(qubits))
            elif angle is not None:
                getattr(qc, name)(angle, *qubits)
            else:
                getattr(qc, name)(*qubits)
        return qc


def _recording_cudaq():
    class Arguments:
        def __init__(self):
            self.parameters = ParameterVector('native', 0)

        def __getitem__(self, index):
            if len(self.parameters) <= index:
                self.parameters.resize(index + 1)
            return self.parameters[index]

    kernel, arguments = _RecordingKernel(), Arguments()
    return SimpleNamespace(make_kernel=lambda *args: (kernel, arguments)), kernel, arguments


@pytest.mark.parametrize('axis', ['xx', 'yy', 'zz'])
def test_native_pauli_rotation_sign_factor_constants_and_parameter_expressions(axis):
    from pqfmlib.core.cudaq_execution import _compile_job
    from qiskit.quantum_info import Operator
    parameters = ParameterVector('p', 12)
    qc = QuantumCircuit(3)
    qc.h(0)
    getattr(qc, 'r' + axis)(.37, 2, 0)
    getattr(qc, 'r' + axis)(parameters[10] * .7 - parameters[2], 0, 1)
    prepared = PreparedProjectedFeatureJob(qc, (parameters[10], parameters[2]), [], [])
    cudaq, kernel, arguments = _recording_cudaq()
    compiled = _compile_job(cudaq, prepared)
    assert [op[0] for op in kernel.operations].count('exp_pauli') == 2
    assert not any(op[0] == 'cx' for op in kernel.operations)
    for row in [[1.7, -.4], [-2.1, .7]]:
        bindings = dict(zip(prepared.param_order, row))
        values = [float(e.bind({p: bindings[p] for p in e.parameters})) for e in compiled.expressions]
        converted = kernel.to_circuit(arguments.parameters, values)
        assert Operator(converted).equiv(Operator(qc.assign_parameters(bindings)), atol=1e-12)


@pytest.mark.parametrize('map_type', MAP_TYPES)
def test_native_translator_preserves_complete_small_pqfm_unitary_on_cpu(map_type):
    from dataclasses import replace
    from pqfmlib.core.cudaq_execution import _compile_job
    from qiskit.quantum_info import Operator
    qfm = map_type('native', q_enc=3, expectation_method='statevector')
    qfm.fit(X)
    prepared = replace(qfm._prepared_execution, obs_isa_broadcast=[])
    cudaq, kernel, arguments = _recording_cudaq()
    compiled = _compile_job(cudaq, prepared)
    values = np.random.default_rng(9).uniform(-2, 2, len(prepared.param_order))
    bindings = dict(zip(prepared.param_order, values))
    angles = [float(e.bind({p: bindings[p] for p in e.parameters})) for e in compiled.expressions]
    converted = kernel.to_circuit(arguments.parameters, angles)
    assert Operator(converted).equiv(Operator(prepared.qc_t.assign_parameters(bindings)), atol=1e-11)


def test_21q_heisenberg_retains_120_native_two_qubit_gates_without_simulation():
    from pqfmlib.core.cudaq_execution import _compile_job
    from pqfmlib.maps.heisenberg import build_heisenberg_chain_feature_circuit
    qc, params, _ = build_heisenberg_chain_feature_circuit(21, num_blocks=1, R=2)
    cudaq, kernel, _ = _recording_cudaq()
    _compile_job(cudaq, PreparedProjectedFeatureJob(qc, tuple(params), [], []))
    assert sum(op[0] == 'exp_pauli' for op in kernel.operations) == 120
    assert sum(op[0] == 'cx' for op in kernel.operations) == 0


def test_independent_direct_kernel_matches_original_heisenberg_on_cpu():
    import importlib.util
    from pqfmlib.maps.heisenberg import build_heisenberg_chain_feature_circuit
    from qiskit.quantum_info import Operator
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'benchmark_heisenberg_cudaq.py'
    spec = importlib.util.spec_from_file_location('direct_heisenberg_benchmark', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    qc, params, _ = build_heisenberg_chain_feature_circuit(3, num_blocks=2, R=3, alpha=.9)
    cudaq, _, arguments = _recording_cudaq()
    kernel, rotations = module.build_direct_kernel(cudaq, qc)
    values = np.random.default_rng(2).uniform(-2, 2, (2, len(params)))
    rows = module.bind_direct_angles(rotations, params, values)
    for values_row, angles in zip(values, rows):
        converted = kernel.to_circuit(arguments.parameters, angles)
        assert Operator(converted).equiv(Operator(qc.assign_parameters(dict(zip(params, values_row)))), atol=1e-11)
    qc.reset(0)
    with pytest.raises(ValueError, match='does not support'):
        module.build_direct_kernel(cudaq, qc)


@gpu
def test_independent_direct_cudaq_comparison_matches_adapter():
    import importlib.util
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'benchmark_heisenberg_cudaq.py'
    spec = importlib.util.spec_from_file_location('direct_heisenberg_benchmark_gpu', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    qfm = HeisenbergProjectiveQFM('direct', q_enc=3, framework='cudaq', mps=True,
                                 expectation_method='exact', mps_truncation_threshold=0,
                                 measure_2local_diagonal=True)
    qfm.fit(X)
    report = module.benchmark_heisenberg_cudaq_direct(qfm, X[:2], repeats=1)
    assert report.path.tolist() == ['pqfmlib_adapter', 'cudaq_direct']
    assert (report.max_abs_error < 1e-10).all()
    assert (report.target == 'tensornet-mps').all()


@pytest.mark.parametrize('width', [1, 3])
def test_mps_shots_diagnostics_count_grouped_calls_and_state_initializations(width):
    cudaq = Mock()
    cudaq.num_available_gpus.return_value = 1
    cudaq.make_kernel.return_value = (Mock(), object())
    cudaq.sample.return_value = {'0' * width: 13, '1' * width: 4}
    compiled = _CompiledJob(object(), (), [object(), object()],
                            (((0, 1.),), ((1, 1.),)), width,
                            sampling_groups=((object(), ((0, (0,)), (1, (0,)))),))
    prepared = SimpleNamespace(qc_t=QuantumCircuit(width), param_order=(),
                               obs_metadata=object(), obs_isa_broadcast=[None, None])
    with patch('pqfmlib.core.cudaq_execution._load_cudaq', return_value=cudaq), \
         patch('pqfmlib.core.cudaq_execution._compile_job', return_value=compiled):
        executor = CudaqExecutor(mps=True)
        result, _ = executor.execute(prepared, np.empty((3, 0)), shots=17)
    np.testing.assert_array_equal(result, np.full((3, 2), 9 / 17))
    stats = executor.last_execution_stats
    assert stats['observe_calls'] == cudaq.observe.call_count == 0
    assert stats['sample_calls'] == cudaq.sample.call_count == 3
    assert stats['n_measurement_groups'] == 1
    assert stats['get_state_calls'] == cudaq.get_state.call_count == (3 if width > 1 else 0)
    assert stats['state_kernel_initializations'] == (3 if width > 1 else 0)
    assert stats['n_observables'] == stats['n_distinct_pauli_terms'] == 2
    assert stats['shots'] == 17
    assert stats['expectation_method'] == 'shots'
    assert stats['observe_seconds'] >= 0
    assert stats['state_seconds'] >= 0
    for call in cudaq.sample.call_args_list:
        assert call.kwargs == {'shots_count': 17}


@pytest.mark.parametrize('per_row', [False, True])
def test_shots_benchmark_reports_routes_errors_and_lifecycle_without_gpu(per_row):
    import importlib.util
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'benchmark_heisenberg_cudaq.py'
    spec = importlib.util.spec_from_file_location('shots_benchmark', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    qfm = HeisenbergProjectiveQFM('diagnostic', q_enc=2, mps=True)
    qfm.fit(X)
    qfm.framework = 'cudaq'
    original_backend = qfm.backend
    original_method = qfm.expectation_method
    n_obs = len(qfm._prepared_execution.obs_isa_broadcast)
    cudaq = Mock()
    cudaq.num_available_gpus.return_value = 1
    cudaq.make_kernel.return_value = (Mock(), object())
    for axis in ('i', 'x', 'y', 'z'):
        getattr(cudaq.spin, axis).return_value = 1.0

    def observe(kernel, operators, arg, *, shots_count):
        value = .25 if shots_count == -1 else .35
        result = SimpleNamespace(expectation=lambda: value)
        return [result] * len(operators) if isinstance(operators, list) else result

    cudaq.observe.side_effect = observe
    cudaq.sample.return_value = {'00': 15, '11': 8}
    compiled = _CompiledJob(object(), (), [object() for _ in range(n_obs)],
                            tuple(((i, 1.),) for i in range(n_obs)), 2,
                            sampling_groups=((object(), tuple((i, (0,)) for i in range(n_obs))),))
    with patch('pqfmlib.core.cudaq_execution._load_cudaq', return_value=cudaq), \
         patch('pqfmlib.core.cudaq_execution._compile_job', return_value=compiled), \
         patch.object(module, 'build_direct_kernel', return_value=(object(), ())):
        report = module.benchmark_heisenberg_cudaq_shots(
            qfm, X[:2], repeats=2, shots=23, per_row=per_row)
    assert len(report) == 12
    assert report.warmup.sum() == 4
    assert set(report.path) == {'pqfmlib_adapter', 'cudaq_direct'}
    assert set(report.expectation_method) == {'exact', 'shots'}
    assert (report.execute_calls == (2 if per_row else 1)).all()
    exact = report[report.expectation_method == 'exact']
    sampled = report[report.expectation_method == 'shots']
    assert (exact.observe_calls == 2).all()
    assert (exact.get_state_calls == 0).all()
    assert (exact.state_kernel_initializations == 0).all()
    direct = sampled[sampled.path == 'cudaq_direct']
    adapter = sampled[sampled.path == 'pqfmlib_adapter']
    assert (direct.observe_calls == 2 * n_obs).all()
    assert (adapter.observe_calls == 0).all()
    assert (adapter.sample_calls == 2).all()
    assert (sampled.get_state_calls == 2).all()
    assert (direct.state_kernel_initializations == 2 * n_obs).all()
    assert (adapter.state_kernel_initializations == 2).all()
    assert (sampled.shots == 23).all()
    np.testing.assert_allclose(direct.max_abs_error, .1)
    np.testing.assert_allclose(adapter.max_abs_error, 7 / 23 - .25)
    assert (exact.max_abs_error == 0).all()
    assert qfm.backend is original_backend
    assert qfm.expectation_method == original_method


def test_measurement_grouping_and_cudaq_bit_order():
    from pqfmlib.core.cudaq_execution import _group_paulis, _sample_expectations
    labels = ('IIZ', 'IZI', 'ZZI', 'IIX', 'IXI', 'IXX', 'IIY', 'IYI', 'IYY', 'IXZ')
    groups = _group_paulis(labels)
    assert groups[:3] == [('ZZZ', (0, 1, 2)), ('IXX', (3, 4, 5)), ('IYY', (6, 7, 8))]
    assert groups[3] == ('IXZ', (9,))
    # CUDA-Q q0 is the LEFT bit, unlike Qiskit labels. Parities include q2.
    supports = ((0, (0,)), (1, (2,)), (2, (0, 2)))
    assert _sample_expectations({'100': 3, '001': 1}, supports, 4, 3) == [
        (0, -.5), (1, .5), (2, -1.)]
    with pytest.raises(RuntimeError, match='number'):
        _sample_expectations({'100': 3}, supports, 4, 3)
    with pytest.raises(RuntimeError, match='bitstrings'):
        _sample_expectations({'10': 4}, supports, 4, 3)


@gpu
@pytest.mark.parametrize('width', [1, 3])
def test_grouped_mps_sampling_axes_cross_terms_weighted_sums_and_real_shots(width):
    qc = QuantumCircuit(width)
    qc.ry(.37, 0)
    qc.rx(.81, 0)
    if width > 1:
        qc.x(2)
        qc.ry(-.45, 1)
        qc.cx(0, 1)
    labels = ['I' * (width - 1) + a for a in 'XYZ']
    if width > 1:
        labels += ['ZII', 'IZI', 'IZZ', 'IXX', 'IYY', 'IXZ', 'ZYX']
    obs = [SparsePauliOp(label) for label in labels]
    obs += [SparsePauliOp.from_list([(labels[0], .4), (labels[-1], -.7), ('I' * width, .2)]),
            SparsePauliOp('I' * width)]
    prepared = PreparedProjectedFeatureJob(qc, (), [[o] for o in obs], [])
    executor = CudaqExecutor(mps=True, seed=192)
    actual, _ = executor.execute(prepared, np.empty((1, 0)), shots=32768)
    reference = np.array([[Statevector(qc).expectation_value(o).real for o in obs]])
    np.testing.assert_allclose(actual, reference, atol=.03, rtol=0)
    again, _ = executor.execute(prepared, np.empty((1, 0)), shots=32768)
    np.testing.assert_array_equal(actual, again)
    stats = executor.last_execution_stats
    assert stats['observe_calls'] == 0
    assert stats['sample_calls'] == stats['n_measurement_groups']
    assert stats['get_state_calls'] == int(width > 1)
    assert stats['n_measurement_groups'] <= len(labels)
    one, _ = executor.execute(prepared, np.empty((1, 0)), shots=1)
    assert np.isin(one[0, :len(labels)], [-1., 1.]).all()
    assert np.max(np.abs(one - reference)) > .1


@gpu
@pytest.mark.parametrize('map_type', MAP_TYPES)
def test_all_pqfms_grouped_mps_shots_match_cpu_reference(map_type):
    reference = map_type('grouped_reference', q_enc=3, expectation_method='statevector')
    expected = reference.fit(X).transform(X[:2])
    qfm = map_type('grouped', q_enc=3, framework='cudaq', mps=True, shots=32768)
    actual = qfm.fit(X).transform(X[:2])
    np.testing.assert_allclose(actual, expected, atol=.03, rtol=0)
    stats = qfm.backend.last_execution_stats
    assert stats['observe_calls'] == 0
    assert stats['sample_calls'] == 2 * stats['n_measurement_groups']


def test_sixty_three_local_terms_require_three_measurement_groups():
    from pqfmlib.core.cudaq_execution import _group_paulis
    labels = ['I' * q + axis + 'I' * (20 - q) for axis in 'XYZ' for q in range(21)]
    groups = _group_paulis(labels)
    assert len(groups) == 3
    assert [basis for basis, _ in groups] == [axis * 21 for axis in 'XYZ']
    assert [len(members) for _, members in groups] == [21, 21, 21]


@gpu
def test_grouped_shots_benchmark_against_direct_scalar_gpu(tmp_path):
    from scripts.benchmark_heisenberg_cudaq import benchmark_heisenberg_cudaq_shots
    data = np.random.default_rng(72).normal(size=(8, 8))
    qfm = HeisenbergProjectiveQFM('grouped_benchmark', q_enc=8, R=2,
                                 framework='cudaq', mps=True, shots=4096)
    qfm.fit(data)
    report = benchmark_heisenberg_cudaq_shots(qfm, data[:1], repeats=2, per_row=True)
    report.to_csv(tmp_path / 'mps_shots_benchmark.csv', index=False)
    measured = report[~report.warmup]
    sampled = measured[measured.expectation_method == 'shots']
    adapter = sampled[sampled.path == 'pqfmlib_adapter']
    direct = sampled[sampled.path == 'cudaq_direct']
    assert (adapter.observe_calls == 0).all()
    assert (adapter.sample_calls == 3).all()
    assert (adapter.state_kernel_initializations == 3).all()
    assert (direct.observe_calls == 24).all()
    assert (direct.state_kernel_initializations == 24).all()
    assert (sampled.max_abs_error < .1).all()
    print('\nGPU diagnostic (8 qubits, R=2, 4096 shots, one row):')
    print(measured.groupby(['path', 'expectation_method'])[
        ['wall_seconds', 'observe_calls', 'sample_calls', 'max_abs_error']].median().to_string())
    print('Report:', tmp_path / 'mps_shots_benchmark.csv')


def test_statevector_shots_aggregate_cache_weighted_terms_and_exact_route():
    cudaq = Mock()
    cudaq.num_available_gpus.return_value = 1
    with patch('pqfmlib.core.cudaq_execution._load_cudaq', return_value=cudaq):
        executor = CudaqExecutor()
    compiled = _CompiledJob(object(), (), [object(), object()],
                            (((0, 2.), (1, -.5), (-1, .3)), ((-1, 1.),)), 2)
    prepared = SimpleNamespace(qc_t=QuantumCircuit(2), param_order=(),
                               obs_isa_broadcast=[None, None], obs_metadata=object())
    operator = object()
    result = Mock()
    result.expectation.side_effect = lambda term: {'x': .25, 'y': -.5}[term]
    cudaq.observe.return_value = result
    with patch('pqfmlib.core.cudaq_execution._compile_job', return_value=compiled), \
         patch('pqfmlib.core.cudaq_execution._aggregate_observables',
               return_value=(operator, ('x', 'y'))) as aggregate:
        for _ in range(2):
            actual, metadata = executor.execute(prepared, np.empty((3, 0)), shots=4096)
            np.testing.assert_allclose(actual, [[1.05, 1.]] * 3)
            assert metadata is prepared.obs_metadata
        aggregate.assert_called_once_with(cudaq, compiled.observables)
        assert cudaq.observe.call_count == 6
        for call in cudaq.observe.call_args_list:
            assert call.args[1] is operator
            assert call.kwargs == {'shots_count': 4096}
        result.counts.assert_not_called()
        cudaq.observe.return_value = [SimpleNamespace(expectation=lambda: .25)] * 2
        executor.execute(prepared, np.empty((1, 0)), expectation_method='exact')
        assert cudaq.observe.call_args.args[1] is compiled.observables
        assert cudaq.observe.call_args.kwargs == {'shots_count': -1}
        cudaq.observe.side_effect = RuntimeError('test failure')
        with pytest.raises(RuntimeError, match='test failure'):
            executor.execute(prepared, np.empty((1, 0)), shots=4096)
        assert executor.last_execution_stats is None
        cudaq.set_target.assert_called_with(cudaq.get_target.return_value)


@gpu
def test_statevector_aggregate_shots_matches_list_for_weighted_and_identity_observables():
    import cudaq
    from pqfmlib.core.cudaq_execution import _compile_job
    qc = QuantumCircuit(3)
    qc.ry(.7, 0)
    qc.h(1)
    qc.cx(0, 2)
    obs = [SparsePauliOp.from_list([('III', .3), ('XIX', 2.), ('IZI', -.5)]),
           SparsePauliOp.from_list([('III', 1.)]),
           SparsePauliOp.from_list([('YZY', 1.), ('XIX', -.2)])]
    prepared = PreparedProjectedFeatureJob(qc, (), [[o] for o in obs], [])
    executor = CudaqExecutor(seed=42)
    actual, _ = executor.execute(prepared, np.empty((2, 0)), shots=4096)
    with executor._target_context(3):
        compiled = _compile_job(cudaq, prepared)
        cudaq.set_random_seed(42)
        expected = []
        for _ in range(2):
            results = cudaq.observe(compiled.kernel, compiled.observables, [], shots_count=4096)
            values = [r.expectation() for r in results]
            expected.append([sum(c * (1 if i == -1 else values[i]) for i, c in terms)
                             for terms in compiled.observable_terms])
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(executor.execute(prepared, np.empty((2, 0)), shots=4096)[0], actual)


def test_conservative_mps_bond_heisenberg_and_nonlocal_gates():
    from pqfmlib.core.cudaq_execution import _conservative_mps_bond
    qc = QuantumCircuit(30)
    for _ in range(2):
        for parity in (0, 1):
            for q in range(parity, 29, 2):
                for gate in (qc.rxx, qc.ryy, qc.rzz):
                    gate(.1, q, q + 1)
    assert _conservative_mps_bond(qc) == 64
    product = QuantumCircuit(30)
    product.h(range(30))
    assert _conservative_mps_bond(product) == 1
    product.cx(0, 29)
    assert _conservative_mps_bond(product) == 2


def test_conservative_mps_bond_dominates_intermediate_schmidt_ranks():
    from pqfmlib.core.cudaq_execution import _conservative_mps_bond
    rng = np.random.default_rng(51)
    qc = QuantumCircuit(6)
    for q in range(6):
        qc.ry(float(rng.normal()), q)
    for a, b in [(0, 5), (1, 4), (0, 2), (3, 5), (2, 4)]:
        qc.rxx(float(rng.normal()), a, b)
        bound = _conservative_mps_bond(qc)
        state = Statevector.from_instruction(qc).data
        for cut in range(1, 6):
            assert np.linalg.matrix_rank(state.reshape(2**cut, -1), tol=1e-12) <= bound
    qc.swap(0, 5)  # Generic gates must be lowered before counting.
    assert _conservative_mps_bond(qc) <= 8


@pytest.mark.parametrize('explicit', [None, 7])
def test_mps_automatic_bond_message_cache_and_explicit_override(explicit, capsys):
    cudaq = Mock()
    cudaq.num_available_gpus.return_value = 1
    qc = QuantumCircuit(30)
    qc.h(0)
    qc.cx(0, 29)
    prepared = SimpleNamespace(qc_t=qc, param_order=(), obs_isa_broadcast=[], obs_metadata=[])
    compiled = _CompiledJob(object(), (), [], (), 30, sampling_groups=[])
    with patch('pqfmlib.core.cudaq_execution._load_cudaq', return_value=cudaq):
        executor = CudaqExecutor(mps=True, mps_max_bond_dimension=explicit)
    def compile_job(*args):
        assert os.environ['CUDAQ_MPS_MAX_BOND'] == str(explicit or 2)
        return compiled
    with patch('pqfmlib.core.cudaq_execution._compile_job', side_effect=compile_job) as compile_mock:
        for _ in range(2):
            executor.execute(prepared, np.empty((1, 0)))
        assert compile_mock.call_count == 1
    output = capsys.readouterr().out
    assert output.count('Teto conservador calculado: 2') == (1 if explicit is None else 0)
    assert executor.last_execution_stats['mps_effective_max_bond'] == (explicit or 2)
