"""Offline regression tests for Heisenberg interaction graphs."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, random_unitary
from sklearn.base import clone

from pqfmlib import HeisenbergProjectiveQFM
from pqfmlib.maps.heisenberg import (
    build_heisenberg_chain_feature_circuit,
    heisenberg_chain_observables,
    make_heisenberg_theta_matrix,
)


class HeisenbergTopologyTests(unittest.TestCase):
    def test_default_chain_matches_original_schedule_and_state(self):
        circuit, params, _ = build_heisenberg_chain_feature_circuit(5, num_blocks=2)
        values = np.arange(8) / 7
        reference = QuantumCircuit(5)
        for q in range(5):
            reference.unitary(random_unitary(2, seed=42 + q), [q])
        for _ in range(2):
            index = 0
            for _ in range(2):
                for i in (0, 2, 1, 3):
                    angle = 0.025 * values[index]
                    reference.rzz(angle, i, i + 1)
                    reference.rxx(angle, i, i + 1)
                    reference.ryy(angle, i, i + 1)
                    index += 1
                reference.barrier()
        np.testing.assert_allclose(
            Statevector.from_instruction(circuit.assign_parameters(dict(zip(params, values)))).data,
            Statevector.from_instruction(reference).data, atol=1e-14,
        )

    def test_graph_capacity_padding_order_and_parameter_sharing(self):
        edges = [(2, 0), (1, 0), (0, 2), (2, 1)]
        theta, info = make_heisenberg_theta_matrix(
            np.array([[1., 2., 3., 4.]]), 3, edges_act=edges, use_tanh_scaling=False,
        )
        np.testing.assert_array_equal(theta, [[1, 2, 3, 4, 0, 0]])
        self.assertEqual(info['features_per_block'], 3)
        circuit, params, _ = build_heisenberg_chain_feature_circuit(
            3, num_blocks=2, R=1, edges_act=edges,
        )
        gates = [inst for inst in circuit.data if inst.operation.name in ('rzz', 'rxx', 'ryy')]
        for slot in range(6):
            group = gates[3 * slot:3 * slot + 3]
            self.assertEqual([inst.operation.name for inst in group], ['rzz', 'rxx', 'ryy'])
            for inst in group:
                self.assertEqual(tuple(circuit.find_bit(q).index for q in inst.qubits),
                                 [(0, 1), (0, 2), (1, 2)][slot % 3])
                self.assertEqual(inst.operation.params[0].parameters, {params[slot]})
        for invalid in ([], [(0, 0)], [(0, 3)], [(-1, 0)], [(0.5, 1)]):
            with self.subTest(edges=invalid), self.assertRaises(ValueError):
                build_heisenberg_chain_feature_circuit(3, num_blocks=1, edges_act=invalid)

    def test_topology_validation_and_clone(self):
        for kwargs in ({'topology': 'other'}, {'topology': 'native'},
                       {'topology': 'full', 'ideal': False}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HeisenbergProjectiveQFM(name_file='test', **kwargs)
        qfm = HeisenbergProjectiveQFM(name_file='test', topology='full')
        self.assertEqual(clone(qfm).topology, 'full')

    def test_full_fit_transform_and_run_match_exact_reference(self):
        X = np.array([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
        with tempfile.TemporaryDirectory() as folder:
            qfm = HeisenbergProjectiveQFM(
                name_file='test', q_enc=3, topology='full', expectation_method='statevector',
                measure_2local_diagonal=True, output_root=folder,
            )
            actual = qfm.fit_transform(X)
            self.assertEqual(qfm.edges_act, [(0, 1), (0, 2), (1, 2)])
            self.assertEqual(actual.shape, (2, 18))
            self.assertEqual(qfm.theta_info['num_blocks'], 2)
            circuit, params, _ = build_heisenberg_chain_feature_circuit(
                3, num_blocks=2, edges_act=qfm.edges_act,
            )
            theta, _ = make_heisenberg_theta_matrix(X, 3, edges_act=qfm.edges_act)
            obs, _ = heisenberg_chain_observables(3, measure_2local_diagonal=True, edges_act=qfm.edges_act)
            expected = []
            for row in theta:
                state = Statevector.from_instruction(circuit.assign_parameters(dict(zip(params, row))))
                expected.append([state.expectation_value(o).real for o in obs])
            np.testing.assert_allclose(actual, expected, atol=1e-12)
            qfm.load_data = Mock()
            qfm.X_q_all, qfm.y = X, [0, 1]
            result = qfm.run()
            np.testing.assert_allclose(result['Xq_all_raw'], actual, atol=1e-12)
            self.assertTrue(Path(result['csv_path']).exists())
            saved_meta = json.loads(Path(result['metadata_path']).read_text())
            self.assertEqual(saved_meta['topology'], 'full')
            self.assertEqual(saved_meta['edges_act'], [[0, 1], [0, 2], [1, 2]])
            self.assertTrue(saved_meta['measure_2local_diagonal'])
            self.assertFalse(saved_meta['measure_cross_observables'])
            self.assertEqual(saved_meta['num_features'], 4)
            self.assertEqual(saved_meta['obs_metadata'], [list(item) for item in qfm.obs_metadata])
            self.assertEqual(len(saved_meta['parameter_names']), 6)
            self.assertTrue(result['base_folder'].endswith('_full_diagobs'))
            qfm.set_params(topology='chain')
            with self.assertRaises(RuntimeError):
                qfm.transform(X)

    def test_native_induced_subgraph_fixed_nodes_and_offline_execution(self):
        # Deliberately non-chain physical numbering, with bidirectional edges.
        coupling = [(5, 11), (11, 5), (8, 11), (11, 20)]
        costs = {edge: 0.01 for edge in coupling}
        with tempfile.TemporaryDirectory() as folder:
            nodes_file = Path(folder) / 'nodes.json'
            nodes_file.write_text(json.dumps([11, 5, 8]))
            qfm = HeisenbergProjectiveQFM(
                name_file='test', q_enc=3, ideal=False, topology='native',
                use_fixed_phys_nodes=True, fixed_phys_nodes_file=str(nodes_file),
                expectation_method='statevector', measure_2local_diagonal=True, output_root=folder,
            )
            qfm.real_backend = Mock()
            with patch('pqfmlib.maps.heisenberg.get_coupling_edges_and_costs', return_value=(coupling, costs)):
                result = qfm.fit_transform([[0.1, 0.2, 0.3, 0.4]])
                self.assertEqual(qfm.phys_nodes, [11, 5, 8])
                self.assertEqual(qfm.edges_act, [(0, 1), (0, 2)])
                self.assertEqual(result.shape, (1, 15))
                self.assertEqual(qfm.theta_info['features_per_block'], 2)
                self.assertEqual({tuple(item[1:]) for item in qfm.obs_metadata[9:]},
                                 {(0, 1), (0, 2)})
                qfm.load_data = Mock()
                qfm.X_q_all, qfm.y = np.array([[0.1, 0.2, 0.3, 0.4]]), [0]
                legacy = qfm.run()
                np.testing.assert_allclose(legacy['Xq_all_raw'], result, atol=1e-12)
                self.assertTrue(legacy['base_folder'].endswith('_native_diagobs'))
                qfm.use_fixed_phys_nodes = False
                qfm.prepare_layout()
                self.assertEqual(len(qfm.phys_nodes), 3)
                induced = {tuple(sorted((qfm.phys_nodes[i], qfm.phys_nodes[j])))
                           for i, j in qfm.edges_act}
                physical = {tuple(sorted(edge)) for edge in coupling}
                self.assertEqual(induced, {edge for edge in physical if set(edge) <= set(qfm.phys_nodes)})
                nodes_file.write_text(json.dumps([5, 8, 20]))
                qfm.use_fixed_phys_nodes = True
                with self.assertRaisesRegex(ValueError, 'connected'):
                    qfm.prepare_layout()

    def test_native_pins_automatic_layout_for_hardware_and_fakebackend(self):
        for simulation in (True, False):
            with self.subTest(simulation=simulation):
                qfm = HeisenbergProjectiveQFM(
                    name_file='test', q_enc=3, ideal=False, topology='native',
                    simulation=simulation, fakebackend=simulation,
                )
                qfm.phys_nodes = [5, 8, 11]
                qfm.backend = Mock()
                circuit = QuantumCircuit(3)
                with patch('pqfmlib.maps.heisenberg.transpile', return_value=circuit) as transpile:
                    qfm._transpile_or_load_circuit(circuit)
                layout = transpile.call_args.kwargs['initial_layout']
                self.assertEqual([layout[q] for q in circuit.qubits], [5, 8, 11])


class HeisenbergAxisEncodingTests(unittest.TestCase):
    def test_parameters_only_encode_selected_two_qubit_interactions(self):
        for axes in (("x",), ("y",), ("z",), ("x", "y"), ("z", "x"), ("y", "z", "x")):
            for mode in ("shared_feature", "multi_axis"):
                with self.subTest(axes=axes, mode=mode):
                    capacity = 2 * (len(axes) if mode == "multi_axis" else 1)
                    circuit, params, _ = build_heisenberg_chain_feature_circuit(
                        3, num_blocks=2, R=2, interaction_axes=axes, encoding_mode=mode,
                    )
                    self.assertEqual(len(params), 2 * capacity)
                    parameterized = [inst for inst in circuit.data if inst.operation.is_parameterized()]
                    self.assertEqual(len(parameterized), 8 * len(axes))
                    for inst in parameterized:
                        self.assertEqual(len(inst.qubits), 2)
                        self.assertIn(inst.operation.name, ["r" + a + a for a in axes])
                    for parameter_index, parameter in enumerate(params):
                        uses = [inst for inst in parameterized if parameter in inst.operation.params[0].parameters]
                        expected_axes = axes if mode == "shared_feature" else (axes[parameter_index % len(axes)],)
                        self.assertEqual(len(uses), 2 * len(expected_axes))
                        self.assertEqual({inst.operation.name for inst in uses},
                                         {"r" + a + a for a in expected_axes})
                        for inst in uses:
                            self.assertAlmostEqual(float(inst.operation.params[0].bind({parameter: 1})), 0.025)
                    X = np.arange(capacity + 1, dtype=float).reshape(1, -1)
                    theta, info = make_heisenberg_theta_matrix(
                        X, 3, interaction_axes=axes, encoding_mode=mode,
                    )
                    self.assertEqual(info['features_per_block'], capacity)
                    self.assertEqual(info['num_blocks'], 2)
                    np.testing.assert_allclose(theta[:, :capacity + 1], 2 * np.pi * np.tanh(X / 3))
                    np.testing.assert_array_equal(theta[:, capacity + 1:], 0)

    def test_cross_observables_are_separate_ordered_and_independent(self):
        for diagonal in (False, True):
            for cross in (False, True):
                with self.subTest(diagonal=diagonal, cross=cross):
                    obs, metadata = heisenberg_chain_observables(
                        3, edges_act=[(2, 0), (0, 2)],
                        measure_2local_diagonal=diagonal, measure_cross_observables=cross,
                    )
                    pairs = (["zz", "xx", "yy"] if diagonal else [])
                    pairs += (["xy", "yx", "xz", "zx", "yz", "zy"] if cross else [])
                    self.assertEqual(len(obs), 9 + len(pairs))
                    self.assertEqual(metadata[9:], [(pair, 0, 2) for pair in pairs])
                    self.assertEqual([o.paulis[0].to_label() for o in obs[9:]],
                                     [pair[1].upper() + "I" + pair[0].upper() for pair in pairs])
        _, chain_metadata = heisenberg_chain_observables(3, measure_cross_observables=True)
        self.assertEqual({m[1:] for m in chain_metadata[9:]}, {(0, 1), (1, 2)})

    def test_encoding_and_cross_measurements_match_independent_statevector(self):
        from qiskit.quantum_info import SparsePauliOp

        for topology in ("chain", "full", "native"):
            for mode in ("shared_feature", "multi_axis"):
                with self.subTest(topology=topology, mode=mode), tempfile.TemporaryDirectory() as folder:
                    axes = ("y", "x")
                    edges = ([(0, 1), (0, 2), (1, 2)] if topology == "full"
                             else [(0, 1), (0, 2)] if topology == "native"
                             else [(0, 1), (1, 2)])
                    capacity = len(edges) * (2 if mode == "multi_axis" else 1)
                    X = np.linspace(-0.8, 0.9, capacity + 1).reshape(1, -1)
                    nodes = Path(folder) / 'nodes.json'
                    nodes.write_text(json.dumps([5, 8, 11]))
                    qfm = HeisenbergProjectiveQFM(
                        name_file='axes', q_enc=3, topology=topology, ideal=topology != "native",
                        interaction_axes=axes, encoding_mode=mode, use_tanh_scaling=False,
                        measure_cross_observables=True, measure_2local_diagonal=True,
                        expectation_method='statevector', output_root=folder,
                        use_fixed_phys_nodes=topology == "native", fixed_phys_nodes_file=str(nodes),
                    )
                    qfm.real_backend = Mock()
                    with patch('pqfmlib.maps.heisenberg.get_coupling_edges_and_costs',
                               return_value=([(5, 8), (5, 11)], {(5, 8): .01, (5, 11): .02})), patch(
                               'pqfmlib.core.data.mutual_information_matrix') as mutual_information:
                        actual = qfm.fit_transform(X)
                        mutual_information.assert_not_called()
                        self.assertEqual(qfm.theta_info['features_per_block'], capacity)
                        self.assertEqual(actual.shape, (1, 9 + 9 * len(edges)))
                        # Build the expected evolution explicitly, without either encoding helper.
                        reference = QuantumCircuit(3)
                        for q in range(3):
                            reference.unitary(random_unitary(2, seed=42 + q), [q])
                        padded = np.pad(X[0], (0, 2 * capacity - X.shape[1]))
                        for _ in range(2):
                            for block in range(2):
                                for edge_index, (i, j) in enumerate(edges):
                                    offset = block * capacity + edge_index * (2 if mode == "multi_axis" else 1)
                                    y_value = padded[offset]
                                    x_value = padded[offset + 1] if mode == "multi_axis" else y_value
                                    reference.rxx(0.025 * x_value, i, j)
                                    reference.ryy(0.025 * y_value, i, j)
                        state = Statevector.from_instruction(reference)
                        expected = []
                        for item in qfm.obs_metadata:
                            label = ['I'] * 3
                            if len(item) == 2:
                                label[2 - item[1]] = item[0].upper()
                            else:
                                pair, i, j = item
                                label[2 - i], label[2 - j] = pair.upper()
                            expected.append(state.expectation_value(SparsePauliOp(''.join(label))).real)
                        np.testing.assert_allclose(actual, [expected], atol=1e-12)
                        np.testing.assert_allclose(qfm.transform(X), actual, atol=1e-12)
                        qfm.load_data = Mock()
                        qfm.X_q_all, qfm.y = X, [0]
                        result = qfm.run()
                        np.testing.assert_allclose(result['Xq_all_raw'], actual, atol=1e-12)
                        import pandas as pd
                        columns = pd.read_csv(result['csv_path']).columns
                        self.assertIn('q2_xy_0_1', columns)
                        self.assertIn('q2_yx_0_1', columns)

    def test_measurements_do_not_change_encoding_or_depend_on_interaction_axes(self):
        prepared = []
        for diagonal, cross in ((False, False), (True, False), (False, True), (True, True)):
            qfm = HeisenbergProjectiveQFM(
                name_file='measurements', q_enc=2, interaction_axes=("x",),
                measure_2local_diagonal=diagonal, measure_cross_observables=cross,
                expectation_method='statevector',
            )
            result = qfm.fit_transform([[.3]])
            self.assertEqual(result.shape, (1, 6 + 3 * diagonal + 6 * cross))
            circuit, _, obs, metadata = qfm._build_circuit_and_observables()
            prepared.append((circuit, result[:, :6]))
            if cross:
                self.assertIn(('yz', 0, 1), metadata)
                self.assertIn(('zy', 0, 1), metadata)
        for circuit, one_local in prepared[1:]:
            # ParameterVector identities differ; compare the same bound circuit.
            left = circuit.assign_parameters([.3])
            right = prepared[0][0].assign_parameters([.3])
            self.assertEqual(left, right)
            np.testing.assert_allclose(one_local, prepared[0][1], atol=1e-12)

    def test_validation_clone_and_frozen_configuration(self):
        for options in ({'interaction_axes': ()}, {'interaction_axes': ('x', 'x')},
                        {'interaction_axes': ('a',)}, {'interaction_axes': None},
                        {'interaction_axes': 'xy'}, {'encoding_mode': 'unknown'}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    HeisenbergProjectiveQFM(name_file='invalid', **options)
                with self.assertRaises(ValueError):
                    make_heisenberg_theta_matrix(np.ones((1, 2)), 3, **options)
                with self.assertRaises(ValueError):
                    build_heisenberg_chain_feature_circuit(3, num_blocks=1, **options)
        for change in ({'interaction_axes': ('x',)}, {'encoding_mode': 'shared_feature'},
                       {'measure_cross_observables': False}, {'measure_2local_diagonal': True}):
            qfm = HeisenbergProjectiveQFM(
                name_file='frozen', q_enc=2, interaction_axes=['Y', 'X'], encoding_mode='multi_axis',
                measure_cross_observables=True, expectation_method='statevector',
            )
            self.assertEqual(clone(qfm).interaction_axes, ('y', 'x'))
            qfm.fit([[.1, .2]])
            qfm.set_params(**change)
            with self.assertRaises(RuntimeError):
                qfm.transform([[.1, .2]])
        qfm = HeisenbergProjectiveQFM(name_file='invalid_set_params')
        qfm.set_params(interaction_axes=())
        with self.assertRaises(ValueError):
            qfm.fit([[.1]])


if __name__ == '__main__':
    unittest.main()
