"""Precision selection, lifecycle, and numerical checks; GPU checks are opt-in."""
import inspect
import os
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from qiskit.providers.fake_provider import GenericBackendV2
from sklearn.base import clone

from pqfmlib import CDIsingProjectiveQFM, HeisenbergProjectiveQFM, XYZProjectiveQFM
from pqfmlib.core.backend import make_aer_backend
from pqfmlib.core.cudaq_execution import CudaqExecutor

MAPS = (CDIsingProjectiveQFM, HeisenbergProjectiveQFM, XYZProjectiveQFM)
X = np.random.default_rng(19).normal(size=(6, 2))
gpu = pytest.mark.skipif(os.environ.get('PQFMLIB_TEST_CUDAQ_GPU') != '1',
                        reason='Set PQFMLIB_TEST_CUDAQ_GPU=1 for GPU checks.')


@pytest.mark.parametrize('map_type', MAPS)
@pytest.mark.parametrize('framework', ['qiskit', 'cudaq'])
def test_precision_api_clone_and_invalid_values(map_type, framework):
    parameter = inspect.signature(map_type).parameters['precision']
    assert parameter.kind == inspect.Parameter.KEYWORD_ONLY
    assert parameter.default == 'double'
    qfm = map_type('precision', 23, precision='single', framework=framework)
    assert qfm.seed == 23
    assert clone(qfm).get_params() == qfm.get_params()
    for value in ('fp32', 'float32', '', None, 32):
        with pytest.raises(ValueError, match='precision'):
            map_type('invalid', precision=value, framework=framework)
    qfm.set_params(precision='invalid')
    with patch.object(qfm, '_load_real_backend') as load:
        with pytest.raises(ValueError, match='precision'):
            qfm.fit(X)
        load.assert_not_called()


@pytest.mark.parametrize('map_type', MAPS)
def test_unsupported_modes_fail_before_loading_backend(map_type):
    for kwargs in (dict(mps=True),
                   dict(ideal=False, fakebackend=True, fakebackend_method='matrix_product_state'),
                   dict(ideal=False, fakebackend=True, fakebackend_method='automatic'),
                   dict(ideal=False, simulation=False)):
        with pytest.raises(ValueError, match='precision'):
            map_type('invalid', precision='single', **kwargs)
    qfm = map_type('changed', mps=True)
    qfm.set_params(precision='single')
    with patch.object(qfm, '_make_aer_backend') as build:
        with pytest.raises(ValueError, match='precision'):
            qfm.fit(X)
        build.assert_not_called()
    # Keep legacy ideal-mode normalization and hardware defaults compatible.
    assert map_type('ideal', precision='single', simulation=False).simulation
    assert map_type('hardware', ideal=False, simulation=False).precision == 'double'


def test_low_level_validation_precedes_optional_imports():
    with patch('pqfmlib.core.cudaq_execution._load_cudaq') as load:
        with pytest.raises(ValueError, match='precision'):
            CudaqExecutor(precision='bad')
        load.assert_not_called()
    with pytest.raises(ValueError, match='precision'):
        make_aer_backend(mps=True, precision='single')


@pytest.mark.parametrize('map_type', MAPS)
@pytest.mark.parametrize('method', ['statevector', 'density_matrix'])
def test_fake_backend_precision_and_revalidation(map_type, method):
    qfm = map_type('fake', q_enc=2, ideal=False, fakebackend=True,
                   fakebackend_method=method, precision='single')
    qfm.real_backend = GenericBackendV2(num_qubits=2)
    with patch.object(qfm, '_load_real_backend'), patch('pqfmlib.core.base.make_estimator'):
        qfm.setup_backend_and_estimator()
    assert qfm.backend.options.precision == 'single'
    assert qfm.backend.options.method == method
    qfm.set_params(fakebackend_method='matrix_product_state')
    with patch.object(qfm, '_load_real_backend') as load:
        with pytest.raises(ValueError, match='precision'):
            qfm.setup_backend_and_estimator()
        load.assert_not_called()


@pytest.mark.parametrize('map_type', MAPS)
@pytest.mark.parametrize('method', ['statevector', 'shots'])
def test_aer_precision_numerics_and_refit(map_type, method):
    kwargs = dict(name_file='cpu', q_enc=2, expectation_method=method, shots=256)
    default = map_type(**kwargs).fit_transform(X)
    double = map_type(**kwargs, precision='double').fit_transform(X)
    np.testing.assert_array_equal(default, double)
    qfm = map_type(**kwargs, precision='single')
    single = qfm.fit_transform(X)
    assert qfm.backend.options.precision == 'single'
    assert single.dtype == np.float64
    assert np.isfinite(single).all()
    np.testing.assert_allclose(single, double, atol=2e-5 if method == 'statevector' else .2, rtol=0)
    qfm.set_params(precision='double')
    with pytest.raises(RuntimeError, match='configuration changed'):
        qfm.transform(X)
    np.testing.assert_array_equal(qfm.fit_transform(X), double)
    assert qfm.backend.options.precision == 'double'


@pytest.mark.parametrize('mps', [False, True])
@pytest.mark.parametrize('precision,option', [('single', 'fp32'), ('double', 'fp64')])
def test_cudaq_target_precision_and_restoration(mps, precision, option):
    cudaq = Mock()
    cudaq.num_available_gpus.return_value = 1
    with patch('pqfmlib.core.cudaq_execution._load_cudaq', return_value=cudaq):
        executor = CudaqExecutor(mps=mps, precision=precision)
    with pytest.raises(RuntimeError, match='injected'):
        with executor._target_context(2):
            cudaq.set_target.assert_called_with(executor.target, option=option)
            raise RuntimeError('injected')
    cudaq.set_target.assert_called_with(cudaq.get_target.return_value)


def check_run(map_type, tmp_path, framework, mps, precision, device=None):
    frame = pd.DataFrame(X)
    frame['y'] = np.arange(len(X)) % 2
    frame.to_csv(tmp_path / 'data.csv', index=False)
    kwargs = dict(name_file='data', q_enc=2, data_dir=str(tmp_path),
                  output_root=str(tmp_path / 'output'), framework=framework,
                  mps=mps, precision=precision, statevector_device=device,
                  expectation_method='exact' if framework == 'cudaq' else 'statevector')
    qfm = map_type(**kwargs)
    expected = qfm.fit_transform(X)
    actual = qfm.run()
    np.testing.assert_allclose(actual['Xq_all_raw'], expected, atol=2e-5, rtol=0)
    np.testing.assert_array_equal(np.load(actual['npy_path']), actual['Xq_all_raw'])
    assert pd.read_csv(actual['csv_path']).shape[0] == len(X)
    actual_precision = qfm.backend.precision if framework == 'cudaq' else qfm.backend.options.precision
    assert actual_precision == precision


@pytest.mark.parametrize('map_type', MAPS)
def test_aer_fp32_run_exports(map_type, tmp_path):
    check_run(map_type, tmp_path, 'qiskit', False, 'single')


@gpu
@pytest.mark.parametrize('map_type', MAPS)
@pytest.mark.parametrize('framework,mps', [('cudaq', False), ('cudaq', True), ('qiskit', False)])
def test_gpu_precision_switch_numerics_and_exports(map_type, framework, mps, tmp_path):
    reference = map_type('reference', q_enc=2, expectation_method='statevector').fit_transform(X)
    qfm = map_type('gpu', q_enc=2, framework=framework, mps=mps,
                   statevector_device='GPU',
                   expectation_method='exact' if framework == 'cudaq' else 'statevector')
    # Alternate settings on the same fitted object to detect cached native precision.
    for precision in ('double', 'single', 'double'):
        qfm.set_params(precision=precision)
        actual = qfm.fit_transform(X)
        tolerance = 2e-5 if precision == 'single' else 1e-10
        np.testing.assert_allclose(actual, reference, atol=tolerance, rtol=tolerance)
        np.testing.assert_allclose(qfm.transform(X[:2]), actual[:2], atol=tolerance, rtol=0)
        if framework == 'cudaq':
            assert qfm.backend.last_execution_stats['precision'] == precision
        else:
            assert qfm.backend.options.precision == precision
        qfm.set_params(precision='single' if precision == 'double' else 'double')
        with pytest.raises(RuntimeError, match='configuration changed'):
            qfm.transform(X)
    qfm.set_params(precision='single', expectation_method='shots', shots=4096)
    sampled = qfm.fit_transform(X)
    assert np.isfinite(sampled).all()
    np.testing.assert_allclose(sampled, reference, atol=.12, rtol=0)
    check_run(map_type, tmp_path, framework, mps, 'single', device='GPU')


@gpu
def test_fp32_diagnostics_preserve_selected_precision():
    from scripts.benchmark_heisenberg_cudaq import (
        benchmark_heisenberg_cudaq_direct, benchmark_heisenberg_cudaq_shots,
    )
    qfm = HeisenbergProjectiveQFM('diagnostic', q_enc=2, framework='cudaq',
                                 mps=True, precision='single', expectation_method='exact')
    qfm.fit(X)
    direct = benchmark_heisenberg_cudaq_direct(qfm, X[:1], repeats=1)
    sampled = benchmark_heisenberg_cudaq_shots(qfm, X[:1], repeats=1, shots=32)
    assert set(direct.precision) == {'single'}
    assert set(sampled.precision) == {'single'}
    assert qfm.backend.precision == 'single'
