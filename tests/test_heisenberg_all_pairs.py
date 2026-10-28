"""Topology-independent readout checks without credentials or real QPU calls."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from qiskit import qpy
from qiskit.primitives import BackendEstimatorV2
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp, Statevector
from sklearn.base import clone

from pqfmlib import HeisenbergProjectiveQFM, SuperexchangeHeisenbergPQFM
from pqfmlib.core.serialization import metadata_to_column_name
from pqfmlib.maps.heisenberg import heisenberg_chain_observables


class AllPairsTests(unittest.TestCase):
    def test_order_labels_capacity_and_default_compatibility(self):
        for n in (3, 25, 40):
            obs, metadata = heisenberg_chain_observables(n, measure_all_pairs=True)
            expected = [(axis, i) for i in range(n) for axis in ("z", "x", "y")]
            expected += [(axes, i, j) for i in range(n) for j in range(i + 1, n)
                         for axes in ("zz", "xx", "yy")]
            self.assertEqual(metadata, expected)
            self.assertEqual(len(obs), {3: 18, 25: 975, 40: 2460}[n])
            self.assertEqual(len({metadata_to_column_name(m) for m in metadata}), len(obs))
            for operator, item in zip(obs, metadata):
                label = ["I"] * n
                for axis, qubit in zip(item[0], item[1:]):
                    label[n - qubit - 1] = axis.upper()
                self.assertEqual(operator.to_list(), [("".join(label), 1 + 0j)])
            # Full graph diagonal readout defines the same legacy ordering.
            legacy, legacy_meta = heisenberg_chain_observables(
                n, measure_2local_diagonal=True,
                edges_act=[(i, j) for i in range(n) for j in range(i + 1, n)],
            )
            self.assertEqual(metadata, legacy_meta)
            self.assertEqual(obs, legacy)
        _, defaults = heisenberg_chain_observables(3)
        self.assertEqual(defaults, [(a, i) for i in range(3) for a in ("z", "x", "y")])
        _, edge_only = heisenberg_chain_observables(3, measure_2local_diagonal=True)
        self.assertNotIn(("zz", 0, 2), edge_only)

    def test_diagonal_flag_is_redundant_and_cross_combination_rejected(self):
        expected = heisenberg_chain_observables(4, measure_all_pairs=True)
        actual = heisenberg_chain_observables(
            4, measure_all_pairs=True, measure_2local_diagonal=True, edges_act=[(0, 1)],
        )
        self.assertEqual(expected, actual)
        with self.assertRaisesRegex(ValueError, "measure_cross_observables"):
            heisenberg_chain_observables(3, measure_all_pairs=True, measure_cross_observables=True)
        for cls in (HeisenbergProjectiveQFM, SuperexchangeHeisenbergPQFM):
            with self.assertRaisesRegex(ValueError, "measure_cross_observables"):
                cls("test", measure_all_pairs=True, measure_cross_observables=True)
            qfm = cls("test").set_params(measure_all_pairs=True, measure_cross_observables=True)
            with self.assertRaises(ValueError):
                qfm.fit(np.zeros((1, 20)))

    def test_transform_matches_exact_observables_without_changing_circuit(self):
        X = np.array([[0., .25, .75], [.1, .4, .8]])
        for cls in (HeisenbergProjectiveQFM, SuperexchangeHeisenbergPQFM):
            for topology in ("chain", "full", "native"):
                with self.subTest(cls=cls.__name__, topology=topology):
                    options = dict(q_enc=3, topology=topology, ideal=topology != "native",
                                   expectation_method="statevector")
                    local = cls("test", **options)
                    complete = cls("test", **options, measure_all_pairs=True)
                    coupling = [(5, 11), (11, 5), (8, 11)]
                    if topology == "native":
                        local.real_backend = complete.real_backend = Mock()
                    with patch("pqfmlib.maps.heisenberg.get_coupling_edges_and_costs",
                               return_value=(coupling, {e: .01 for e in coupling})):
                        local_features = local.fit_transform(X)
                        actual = complete.fit_transform(X)
                    self.assertEqual(actual.shape, (2, 18))
                    np.testing.assert_allclose(actual[:, :9], local_features, atol=1e-12)
                    local_circuit, local_params, _, _ = local._build_circuit_and_observables()
                    circuit, params, obs, metadata = complete._build_circuit_and_observables()
                    # Independent ParameterVectors have different UUIDs; align
                    # their symbols before comparing the circuit instructions.
                    self.assertEqual(circuit, local_circuit.assign_parameters(dict(zip(local_params, params))))
                    self.assertEqual(circuit.metadata, local_circuit.metadata)
                    self.assertEqual(complete.edges_act, local.edges_act)
                    self.assertEqual(complete.theta_info, local.theta_info)
                    theta, _ = complete._make_theta_matrix(X)
                    expected = []
                    for row in theta:
                        state = Statevector.from_instruction(circuit.assign_parameters(dict(zip(params, row))))
                        expected.append([state.expectation_value(o).real for o in obs])
                    np.testing.assert_allclose(actual, expected, atol=1e-12)
                    self.assertEqual(complete.obs_metadata, metadata)
                    self.assertTrue(clone(complete).get_params()["measure_all_pairs"])
                    complete.set_params(measure_all_pairs=False)
                    with self.assertRaisesRegex(RuntimeError, "changed"):
                        complete.transform(X)

    def test_run_metadata_columns_and_qpy_reuse(self):
        X = np.array([[0., .25, .75]])
        for cls in (HeisenbergProjectiveQFM, SuperexchangeHeisenbergPQFM):
            with tempfile.TemporaryDirectory() as folder:
                pd.DataFrame([[0., .25, .75, 1]], columns=["a", "b", "c", "y"]).to_csv(
                    Path(folder) / "test.csv", index=False,
                )
                options = dict(q_enc=3, expectation_method="statevector", output_root=folder, data_dir=folder)
                local = cls("test", **options).fit(X)
                circuit, _, _, _ = local._build_circuit_and_observables()
                path = Path(folder) / "local.qpy"
                with path.open("wb") as stream:
                    qpy.dump(circuit, stream)
                complete = cls("test", **options, measure_all_pairs=True, fixed_circuit_file_name=str(path))
                expected = complete.fit_transform(X)
                # fit/transform create no output artifacts, even for all-pairs.
                self.assertEqual({p.name for p in Path(folder).iterdir()}, {"test.csv", "local.qpy"})
                result = complete.run()
                np.testing.assert_allclose(result["Xq_all_raw"], expected, atol=1e-12)
                np.testing.assert_allclose(np.load(result["npy_path"]), expected)
                metadata = json.loads(Path(result["metadata_path"]).read_text())
                self.assertTrue(metadata["measure_all_pairs"])
                self.assertFalse(metadata["measure_cross_observables"])
                self.assertEqual(metadata["obs_metadata"], [list(m) for m in complete.obs_metadata])
                self.assertEqual(pd.read_csv(result["csv_path"]).columns.tolist(),
                                 [metadata_to_column_name(m) for m in complete.obs_metadata] + ["y"])
                self.assertIn("allpairs", Path(result["base_folder"]).name)

    def test_three_global_bases_and_estimator_batching(self):
        X = np.array([[0., .25, .75], [.1, .4, .8]])
        qfm = SuperexchangeHeisenbergPQFM("test", q_enc=3, measure_all_pairs=True)
        qfm.fit(X)
        obs, _ = heisenberg_chain_observables(3, measure_all_pairs=True)
        combined = SparsePauliOp.from_list([term for o in obs for term in o.to_list()])
        groups = combined.group_commuting(qubit_wise=True)
        self.assertEqual(len(groups), 3)
        self.assertEqual({frozenset("".join(g.paulis.to_labels())) - {"I"} for g in groups},
                         {frozenset("X"), frozenset("Y"), frozenset("Z")})
        # Exercise the existing executor with a local Estimator implementing
        # grouping: one PUB, three basis circuits per row, not per observable.
        qfm.estimator = BackendEstimatorV2(backend=qfm.backend, options={"default_precision": 1 / 64})
        with patch.object(qfm.estimator, "run", wraps=qfm.estimator.run) as run:
            with patch.object(qfm.backend, "run", wraps=qfm.backend.run) as backend_run:
                sampled = qfm.transform(X)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(len(run.call_args.args[0]), 1)
        circuits = [c for call in backend_run.call_args_list for c in call.args[0]]
        self.assertEqual(len(circuits), 3 * len(X))
        exact = SuperexchangeHeisenbergPQFM(
            "test", q_enc=3, measure_all_pairs=True, expectation_method="statevector",
        ).fit_transform(X)
        np.testing.assert_allclose(sampled, exact, atol=6 / 64)

    def test_qpu_submission_keeps_one_pub_and_all_observable_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            pd.DataFrame([[0., .25, .75, 1]], columns=["a", "b", "c", "y"]).to_csv(
                Path(folder) / "test.csv", index=False,
            )
            qfm = SuperexchangeHeisenbergPQFM(
                "test", q_enc=3, ideal=False, simulation=False, measure_all_pairs=True,
                output_root=folder, data_dir=folder,
            )
            qfm.backend = GenericBackendV2(3, seed=42)
            qfm.real_backend = qfm.backend
            qfm.estimator = Mock()
            qfm.estimator.run.return_value.job_id.return_value = "offline-allpairs"
            with patch.object(qfm, "setup_backend_and_estimator"), patch("pqfmlib.maps.heisenberg.test_circuit_t"):
                self.assertEqual(qfm.run(), "offline-allpairs")
            qfm.estimator.run.assert_called_once()
            pubs = qfm.estimator.run.call_args.args[0]
            self.assertEqual(len(pubs), 1)
            self.assertEqual(len(pubs[0][1]), 18)
            self.assertEqual(pubs[0][2].shape, (1, 2))
            metadata = json.loads((Path(qfm.base_folder) / "job_meta.json").read_text())
            self.assertTrue(metadata["measure_all_pairs"])
            self.assertEqual(metadata["n_obs"], 18)
            expected_metadata = heisenberg_chain_observables(3, measure_all_pairs=True)[1]
            self.assertEqual(metadata["obs_metadata"], [list(m) for m in expected_metadata])


if __name__ == "__main__":
    unittest.main()
