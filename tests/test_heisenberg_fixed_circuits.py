"""QPY reuse tests without credentials or real QPU submissions."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit_aer import AerSimulator

from pqfmlib import HeisenbergProjectiveQFM
from pqfmlib.maps.heisenberg import build_heisenberg_chain_feature_circuit


def local_backend():
    return GenericBackendV2(
        num_qubits=5, basis_gates=['id', 'rz', 'sx', 'x', 'cx'], seed=10,
        coupling_map=[[i, j] for i in range(5) for j in range(5) if abs(i - j) == 1],
    )


def dump(path, circuit):
    with Path(path).open('wb') as stream:
        qpy.dump(circuit, stream)


class HeisenbergFixedCircuitTests(unittest.TestCase):
    def test_logical_qpy_both_modes_and_all_axes_full_or_chain(self):
        for mode in ('shared_feature', 'multi_axis'):
            for axes in (('x',), ('y', 'x'), ('x', 'y', 'z')):
                for topology in ('chain', 'full'):
                    with self.subTest(mode=mode, axes=axes, topology=topology), tempfile.TemporaryDirectory() as folder:
                        options = dict(name_file='test', q_enc=3, encoding_mode=mode,
                                       interaction_axes=axes, topology=topology,
                                       measure_cross_observables=True, expectation_method='statevector')
                        original = HeisenbergProjectiveQFM(**options)
                        X = np.array([[.1, -.2, .3, .4]])
                        expected = original.fit_transform(X)
                        circuit, _, _, _ = original._build_circuit_and_observables()
                        path = Path(folder) / 'logical.qpy'
                        dump(path, circuit)
                        loaded = HeisenbergProjectiveQFM(**options, fixed_circuit_file_name=str(path))
                        np.testing.assert_allclose(loaded.fit_transform(X), expected, atol=1e-12)
                        self.assertTrue(loaded._prepared_execution.fixed_circuit_loaded)

    def test_routed_qpy_preserves_layout_and_matches_logical_in_fit_and_run(self):
        backend = local_backend()
        X = np.array([[.1, -.2, .3, .4]])
        with tempfile.TemporaryDirectory() as folder:
            options = dict(name_file='routed', q_enc=3, encoding_mode='multi_axis',
                           interaction_axes=('x', 'y'), measure_cross_observables=True,
                           measure_2local_diagonal=True, expectation_method='statevector')
            original = HeisenbergProjectiveQFM(**options)
            expected = original.fit_transform(X)
            logical, _, _, _ = original._build_circuit_and_observables()
            physical = transpile(logical, backend=backend, initial_layout=[4, 2, 0],
                                 optimization_level=1, seed_transpiler=42)
            self.assertNotEqual(physical.layout.initial_index_layout(True),
                                physical.layout.final_index_layout(True))
            path = Path(folder) / 'physical.qpy'
            dump(path, physical)
            loaded = HeisenbergProjectiveQFM(**options, fixed_circuit_file_name=str(path), output_root=folder)
            with patch('pqfmlib.maps.heisenberg.transpile', side_effect=AssertionError('must preserve ISA')):
                np.testing.assert_allclose(loaded.fit_transform(X), expected, atol=1e-12)
                self.assertEqual(loaded.phys_nodes, [4, 2, 0])
                loaded.load_data = Mock()
                loaded.X_q_all, loaded.y = X, [0]
                result = loaded.run()
                np.testing.assert_allclose(result['Xq_all_raw'], expected, atol=1e-12)
            saved = Path(result['base_folder']) / 'heisenberg_circuit.qpy'
            with saved.open('rb') as stream:
                restored = qpy.load(stream)[0]
            self.assertEqual(restored, physical)
            self.assertEqual(restored.layout.final_index_layout(True), physical.layout.final_index_layout(True))

    def test_native_logical_and_physical_qpy_submit_offline_in_both_modes(self):
        backend = local_backend()
        for mode in ('shared_feature', 'multi_axis'):
            for physical in (False, True):
                with self.subTest(mode=mode, physical=physical), tempfile.TemporaryDirectory() as folder:
                    nodes = [2, 1, 3]
                    circuit, _, _ = build_heisenberg_chain_feature_circuit(
                        3, num_blocks=1, encoding_mode=mode, interaction_axes=('x', 'y'),
                        edges_act=[(0, 1), (0, 2)],
                    )
                    if physical:
                        circuit = transpile(circuit, backend=backend, initial_layout=nodes,
                                            optimization_level=1, seed_transpiler=42)
                    path = Path(folder) / 'fixed.qpy'
                    dump(path, circuit)
                    nodes_file = Path(folder) / 'nodes.json'
                    nodes_file.write_text(json.dumps(nodes))
                    qfm = HeisenbergProjectiveQFM(
                        name_file='native', q_enc=3, ideal=False, simulation=False, topology='native',
                        encoding_mode=mode, interaction_axes=('x', 'y'), measure_cross_observables=True,
                        fixed_circuit_file_name=str(path), output_root=folder,
                        use_fixed_phys_nodes=not physical, fixed_phys_nodes_file=str(nodes_file),
                    )
                    qfm.real_backend = qfm.backend = backend
                    qfm.estimator = Mock()
                    qfm.estimator.run.return_value.job_id.return_value = 'offline-job'
                    qfm.setup_backend_and_estimator = Mock()
                    qfm.load_data = Mock()
                    qfm.X_q_all = np.array([[.1, .2]])
                    qfm.num_features = 2
                    with patch('pqfmlib.maps.heisenberg.test_circuit_t'):
                        self.assertEqual(qfm.run(), 'offline-job')
                    self.assertEqual(qfm.phys_nodes, nodes)
                    self.assertEqual(qfm.edges_act, [(0, 1), (0, 2)])
                    submitted, obs, theta = qfm.estimator.run.call_args.args[0][0]
                    if physical:
                        self.assertEqual(submitted, circuit)
                    self.assertEqual(submitted.layout.initial_index_layout(False)[:3], nodes)
                    self.assertEqual(theta.shape, (1, 4 if mode == 'multi_axis' else 2))
                    np.testing.assert_allclose(theta[0, :2], 2 * np.pi * np.tanh(np.array([.1, .2]) / 3))
                    np.testing.assert_array_equal(theta[0, 2:], 0)
                    self.assertEqual(len(obs), 21)
                    final_nodes = submitted.layout.final_index_layout(True)
                    xy_label = ['I'] * submitted.num_qubits
                    xy_label[-1 - final_nodes[0]] = 'X'
                    xy_label[-1 - final_nodes[1]] = 'Y'
                    self.assertEqual(obs[9][0].paulis[0].to_label(), ''.join(xy_label))
                    if not physical:
                        saved_logical = Path(qfm.base_folder) / 'heisenberg_circuit_logical.qpy'
                        with saved_logical.open('rb') as stream:
                            self.assertEqual(qpy.load(stream)[0], circuit)
                    self.assertTrue(all(o[0].num_qubits == 5 for o in obs))
                    meta = json.loads((Path(qfm.base_folder) / 'job_meta.json').read_text())
                    self.assertEqual(meta['encoding_mode'], mode)
                    self.assertEqual(meta['phys_nodes'], nodes)
                    self.assertEqual(meta['edges_act'], [[0, 1], [0, 2]])
                    self.assertTrue(meta['fixed_circuit_loaded'])

    def test_fakebackend_accepts_saved_layout_without_retranspiling(self):
        backend = local_backend()
        circuit, _, _ = build_heisenberg_chain_feature_circuit(3, num_blocks=1, encoding_mode='multi_axis')
        circuit = transpile(circuit, backend=backend, initial_layout=[3, 2, 1], seed_transpiler=42)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixed.qpy'
            dump(path, circuit)
            qfm = HeisenbergProjectiveQFM(
                name_file='fake', q_enc=3, ideal=False, fakebackend=True, encoding_mode='multi_axis',
                fixed_circuit_file_name=str(path), measure_cross_observables=True,
            )
            qfm.real_backend = backend
            qfm.backend = AerSimulator.from_backend(backend)
            qfm.estimator = Mock()
            qfm.setup_backend_and_estimator = Mock()
            with patch('pqfmlib.maps.heisenberg.transpile', side_effect=AssertionError('must preserve ISA')):
                qfm.fit([[.1, .2]])
            self.assertEqual(qfm.phys_nodes, [3, 2, 1])
            self.assertEqual(qfm._prepared_execution.qc_t, circuit)

    def test_configuration_and_parameter_mismatches_fail_before_submission(self):
        for change in ({'encoding_mode': 'multi_axis'}, {'interaction_axes': ('y', 'x')},
                       {'alpha': .2}, {'R': 3}, {'seed': 43}, {'edges_act': [(0, 1), (0, 2)]}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                kwargs = dict(num_blocks=1, interaction_axes=('x', 'y'))
                kwargs.update(change)
                circuit, _, _ = build_heisenberg_chain_feature_circuit(3, **kwargs)
                path = Path(folder) / 'fixed.qpy'
                dump(path, circuit)
                qfm = HeisenbergProjectiveQFM(
                    name_file='bad', q_enc=3, interaction_axes=('x', 'y'),
                    fixed_circuit_file_name=str(path), expectation_method='statevector',
                )
                with self.assertRaisesRegex(ValueError, 'incompatible|configuration differs'):
                    qfm.fit([[.1, .2]])
        # Equal parameter count does not prove equal encoding semantics.
        circuit, _, _ = build_heisenberg_chain_feature_circuit(
            3, num_blocks=1, encoding_mode='multi_axis', interaction_axes=('x',),
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixed.qpy'
            dump(path, circuit)
            qfm = HeisenbergProjectiveQFM(name_file='bad', q_enc=3, interaction_axes=('x',),
                                         fixed_circuit_file_name=str(path))
            with self.assertRaisesRegex(ValueError, 'encoding_mode'):
                qfm.fit([[.1, .2]])

    def test_run_rejects_mismatch_before_calling_estimator(self):
        with tempfile.TemporaryDirectory() as folder:
            circuit, _, _ = build_heisenberg_chain_feature_circuit(
                3, num_blocks=1, encoding_mode='multi_axis', interaction_axes=('y', 'x'),
            )
            path = Path(folder) / 'fixed.qpy'
            dump(path, circuit)
            qfm = HeisenbergProjectiveQFM(
                name_file='mismatch', q_enc=3, ideal=False, simulation=False,
                encoding_mode='multi_axis', interaction_axes=('x', 'y'),
                fixed_circuit_file_name=str(path), output_root=folder,
            )
            qfm.real_backend = qfm.backend = local_backend()
            qfm.estimator = Mock()
            qfm.setup_backend_and_estimator = Mock()
            qfm.load_data = Mock()
            qfm.X_q_all, qfm.num_features = np.array([[.1, .2]]), 2
            with self.assertRaisesRegex(ValueError, 'interaction_axes'):
                qfm.run()
            qfm.estimator.run.assert_not_called()

    def test_legacy_qpy_uses_declared_configuration_with_warning(self):
        for mode in ('shared_feature', 'multi_axis'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                circuit, _, _ = build_heisenberg_chain_feature_circuit(
                    3, num_blocks=1, interaction_axes=('x', 'y'), encoding_mode=mode,
                )
                circuit.metadata = {}
                path = Path(folder) / 'legacy.qpy'
                dump(path, circuit)
                qfm = HeisenbergProjectiveQFM(name_file='old', q_enc=3, encoding_mode=mode,
                                             interaction_axes=('x', 'y'), fixed_circuit_file_name=str(path),
                                             expectation_method='statevector')
                with self.assertWarnsRegex(UserWarning, 'cannot be verified'):
                    qfm.fit_transform([[.1, .2]])

    def test_invalid_width_layout_backend_and_parameter_set_are_rejected(self):
        backend = local_backend()
        logical, _, _ = build_heisenberg_chain_feature_circuit(3, num_blocks=1)
        physical = transpile(logical, backend=backend, initial_layout=[3, 2, 1], seed_transpiler=42)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixed.qpy'
            nodes_file = Path(folder) / 'nodes.json'
            nodes_file.write_text(json.dumps([0, 1, 2]))
            qfm = HeisenbergProjectiveQFM(
                name_file='bad', q_enc=3, ideal=False, simulation=False,
                fixed_circuit_file_name=str(path), use_fixed_phys_nodes=True,
                fixed_phys_nodes_file=str(nodes_file),
            )
            qfm.real_backend = qfm.backend = backend
            dump(path, physical)
            qfm.prepare_layout()
            with self.assertRaisesRegex(ValueError, 'conflicts'):
                qfm._transpile_or_load_circuit(logical)
            qfm.use_fixed_phys_nodes = False
            unsupported = physical.copy()
            unsupported.h(0)
            dump(path, unsupported)
            qfm.prepare_layout()
            with self.assertRaisesRegex(ValueError, 'incompatible with the target'):
                qfm._transpile_or_load_circuit(logical)
            dump(path, QuantumCircuit(4))
            with self.assertRaisesRegex(ValueError, 'q_enc'):
                qfm.prepare_layout()
            dump(path, logical.assign_parameters({next(iter(logical.parameters)): 0}))
            qfm.prepare_layout()
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                qfm._transpile_or_load_circuit(logical)
            dump(path, [logical, logical])
            with self.assertRaisesRegex(ValueError, 'exactly one'):
                qfm.prepare_layout()


if __name__ == '__main__':
    unittest.main()
