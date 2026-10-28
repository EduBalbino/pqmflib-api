"""Offline mathematical, physical and API checks for spatial superexchange."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from qiskit import qpy
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.quantum_info import SparsePauliOp, Statevector
from sklearn.base import clone

from pqfmlib import HeisenbergProjectiveQFM, SuperexchangeHeisenbergPQFM
from pqfmlib.maps.heisenberg import (
    _heisenberg_edge_schedule,
    build_heisenberg_chain_feature_circuit,
    heisenberg_chain_observables,
    make_superexchange_theta_matrix,
)
from pqfmlib.core.observables import pauli_string
from pqfmlib.core.cudaq_execution import CudaqExecutor


def model(**kwargs):
    return SuperexchangeHeisenbergPQFM(
        name_file="sites", q_enc=kwargs.pop("q_enc", 3),
        expectation_method="statevector", **kwargs,
    )


class SuperexchangeCouplingTests(unittest.TestCase):
    def test_formula_symmetry_invariances_and_row_independence(self):
        X = np.array([[0., .25, .75], [.2, .2, .2]])
        edges = [(0, 1), (0, 2), (1, 2)]
        theta, info = make_superexchange_theta_matrix(X, 3, edges_act=edges)
        np.testing.assert_allclose(theta[0], np.exp([-.5, -1.5, -1.]))
        np.testing.assert_array_equal(theta[1], np.ones(3))
        self.assertEqual(info["num_blocks"], 1)
        self.assertFalse(info["use_tanh_scaling"])
        for values, graph in ((X + 5, edges), (-X, edges), (X, [(j, i) for i, j in edges])):
            np.testing.assert_allclose(make_superexchange_theta_matrix(values, 3, edges_act=graph)[0], theta)
        np.testing.assert_array_equal(make_superexchange_theta_matrix(X[:1], 3, edges_act=edges)[0], theta[:1])
        actual, _ = make_superexchange_theta_matrix(X[:1], 3, J0=2, ell=.5)
        np.testing.assert_allclose(actual, 2 * np.exp([[-1., -2.]]))

    def test_chain_uses_even_odd_order_and_every_feature(self):
        X = np.array([[0., .1, .4, .9, 1.5]])
        theta, info = make_superexchange_theta_matrix(X, 5)
        np.testing.assert_allclose(theta, np.exp([[-.2, -1., -.6, -1.2]]))
        self.assertEqual(info["total_slots"], 4)
        changed = X.copy()
        changed[0, -1] += .1
        self.assertNotEqual(make_superexchange_theta_matrix(changed, 5)[0][0, -1], theta[0, -1])
        for n in (25, 40):
            for graph in (None, [(i, j) for i in range(n) for j in range(i + 1, n)]):
                actual, info = make_superexchange_theta_matrix(np.zeros((1, n)), n, edges_act=graph)
                self.assertEqual(actual.shape, (1, n - 1 if graph is None else n * (n - 1) // 2))
                self.assertEqual(info["num_blocks"], 1)

    def test_invalid_inputs_and_configuration(self):
        for field in ("J0", "ell"):
            for value in (0, -1, np.nan, np.inf):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    model(**{field: value})
                with self.assertRaises(ValueError):
                    make_superexchange_theta_matrix([[0, 1]], 2, **{field: value})
        for X in ([0, 1, 2], [[0, 1]], [[0, np.nan, 1]], [[0, np.inf, 1]], np.empty((0, 3))):
            with self.subTest(X=X), self.assertRaises(ValueError):
                model().fit(X)
        for kwargs in ({"interaction_axes": ("x",)}, {"encoding_mode": "multi_axis"},
                       {"coupling_mode": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                model(**kwargs)

    def test_finite_extreme_positions_and_scales(self):
        maximum = np.finfo(float).max
        theta, _ = make_superexchange_theta_matrix([[-maximum, maximum]], 2, ell=maximum)
        np.testing.assert_allclose(theta, [[np.exp(-4)]])
        theta, _ = make_superexchange_theta_matrix([[0, 375]], 2, J0=maximum)
        np.testing.assert_allclose(theta, [[float(np.exp(np.log(np.longdouble(maximum)) - 750))]])
        self.assertEqual(theta.dtype, np.float64)
        theta, _ = make_superexchange_theta_matrix([[0, maximum]], 2, ell=np.finfo(float).tiny)
        np.testing.assert_array_equal(theta, [[0.]])


class SuperexchangePhysicsTests(unittest.TestCase):
    def test_two_qubit_evolution_and_scale(self):
        X = np.array([[.1, .7], [.3, .3]])
        qfm = model(q_enc=2, R=3, alpha=.8)
        actual = qfm.fit_transform(X)
        obs, _ = heisenberg_chain_observables(2)
        initial, _, _ = build_heisenberg_chain_feature_circuit(2, num_blocks=1, R=0)
        psi0 = Statevector.from_instruction(initial).data
        H = SparsePauliOp.from_list([("XX", 1), ("YY", 1), ("ZZ", 1)]).to_matrix()
        energies, vectors = np.linalg.eigh(H)
        for row, features in zip(X, actual):
            J = np.exp(-2 * abs(row[0] - row[1]))
            # RPP(alpha*J/2) => exp(-i*alpha*J*PP/4), repeated R times.
            state = Statevector(vectors @ (np.exp(-1j * energies * J * .8 * 3 / 4) * (vectors.conj().T @ psi0)))
            np.testing.assert_allclose(features, [state.expectation_value(o).real for o in obs], atol=1e-12)

    def test_isotropic_hamiltonian_and_trotter_convergence(self):
        for n in (3, 4):
            X = np.linspace(0, .5, n).reshape(1, -1)
            edges = _heisenberg_edge_schedule(n)
            couplings, _ = make_superexchange_theta_matrix(X, n)
            terms = [(pauli_string(n, [(i, axis), (j, axis)]), J)
                     for (i, j), J in zip(edges, couplings[0]) for axis in "xyz"]
            H = SparsePauliOp.from_list(terms).to_matrix()
            for axis in "xyz":
                S = SparsePauliOp.from_list([(pauli_string(n, [(i, axis)]), .5) for i in range(n)]).to_matrix()
                np.testing.assert_allclose(H @ S - S @ H, 0, atol=1e-14)
            initial, _, _ = build_heisenberg_chain_feature_circuit(n, num_blocks=1, R=0)
            psi0 = Statevector.from_instruction(initial).data
            energies, vectors = np.linalg.eigh(H)
            exact = vectors @ (np.exp(-1j * .4 * energies) * (vectors.conj().T @ psi0))
            errors = []
            for repetitions in (1, 4, 16):
                circuit, params, _ = build_heisenberg_chain_feature_circuit(
                    n, num_blocks=1, R=repetitions, alpha=4 * .4 / repetitions,
                )
                actual = Statevector.from_instruction(circuit.assign_parameters(dict(zip(params, couplings[0])))).data
                errors.append(np.linalg.norm(actual - exact))
            self.assertGreater(errors[0], errors[1])
            self.assertGreater(errors[1], errors[2])


class SuperexchangeAPITests(unittest.TestCase):
    def test_transform_lifecycle_dataframe_clone_and_no_writes(self):
        X = pd.DataFrame([[0., .25, .75], [.1, .4, .8]], columns=["a", "b", "c"])
        with tempfile.TemporaryDirectory() as folder:
            qfm = model(output_root=folder)
            with self.assertRaisesRegex(RuntimeError, "not fitted"):
                qfm.transform(X)
            self.assertIs(qfm.fit(X, y=[1, 0]), qfm)
            prepared = qfm._prepared_execution
            result = qfm.transform(X)
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(result.shape, (2, 9))
            np.testing.assert_allclose(result, model().fit_transform(X), atol=1e-12)
            np.testing.assert_allclose(result[:1], qfm.transform(X.iloc[:1]), atol=1e-12)
            self.assertIs(qfm._prepared_execution, prepared)
            self.assertEqual(list(Path(folder).iterdir()), [])
            for bad in (X[["b", "a", "c"]], X.rename(columns={"a": "other"}), X.iloc[:, :2]):
                with self.assertRaises(ValueError):
                    qfm.transform(bad)
            copied = clone(qfm)
            self.assertFalse(copied._is_fitted)
            self.assertEqual(copied.get_params()["coupling_mode"], "superexchange")
            for field, value in (("ell", .5), ("J0", 2), ("coupling_mode", "direct")):
                fitted = model().fit(X)
                fitted.set_params(**{field: value})
                with self.assertRaisesRegex(RuntimeError, "changed"):
                    fitted.transform(X)

    def test_strategy_matches_binding_existing_circuit_without_tanh(self):
        X = np.array([[0., .25, .75, 1.], [.2, .3, .6, .8]])
        for topology in ("chain", "full"):
            qfm = model(q_enc=4, topology=topology, J0=2, ell=.7, use_tanh_scaling=True)
            actual = qfm.fit_transform(X)
            edges = None if topology == "chain" else qfm.edges_act
            theta, _ = make_superexchange_theta_matrix(X, 4, edges_act=edges, J0=2, ell=.7)
            # The direct map with scaling disabled is the same execution engine.
            reference = HeisenbergProjectiveQFM(
                name_file="direct", q_enc=4, topology=topology,
                use_tanh_scaling=False, expectation_method="statevector",
            ).fit_transform(theta)
            np.testing.assert_allclose(actual, reference, atol=1e-12)
            base = HeisenbergProjectiveQFM(
                name_file="mode", q_enc=4, topology=topology, coupling_mode="superexchange",
                J0=2, ell=.7, expectation_method="statevector",
            )
            np.testing.assert_allclose(actual, base.fit_transform(X), atol=1e-12)

    def test_run_csv_artifacts_and_fixed_qpy(self):
        X = np.array([[0., .25, .75], [.1, .3, .8]])
        for topology in ("chain", "full"):
            with self.subTest(topology=topology), tempfile.TemporaryDirectory() as folder:
                frame = pd.DataFrame(X, columns=["a", "b", "c"])
                frame["y"] = [0, 1]
                frame.to_csv(Path(folder) / "sites.csv", index=False)
                qfm = model(topology=topology, data_dir=folder, output_root=folder, J0=1.5, ell=.8)
                expected = qfm.fit_transform(X)
                result = qfm.run()
                np.testing.assert_allclose(result["Xq_all_raw"], expected, atol=1e-12)
                np.testing.assert_allclose(np.load(result["npy_path"]), expected)
                saved = pd.read_csv(result["csv_path"])
                np.testing.assert_array_equal(saved["y"], [0, 1])
                np.testing.assert_allclose(saved.iloc[:, :-1], expected, atol=1e-12)
                metadata = json.loads(Path(result["metadata_path"]).read_text())
                self.assertEqual(metadata["coupling_mode"], "superexchange")
                self.assertEqual(metadata["J0"], 1.5)
                self.assertEqual(metadata["ell"], .8)
                self.assertFalse(metadata["use_tanh_scaling"])
                self.assertEqual(metadata["num_blocks"], 1)
                self.assertEqual(metadata["num_features"], 3)
                np.testing.assert_array_equal(json.loads((Path(result["base_folder"]) / "phys_nodes.json").read_text()), [0, 1, 2])
                path = Path(result["base_folder"]) / "heisenberg_circuit_logical.qpy"
                with path.open("rb") as stream:
                    circuit = qpy.load(stream)[0]
                self.assertEqual(circuit.metadata["pqfmlib_heisenberg"]["coupling_mode"], "superexchange")
                loaded = model(topology=topology, J0=1.5, ell=.8, fixed_circuit_file_name=str(path))
                np.testing.assert_allclose(loaded.fit_transform(X), expected, atol=1e-12)
                with self.assertRaisesRegex(ValueError, "J0"):
                    model(topology=topology, fixed_circuit_file_name=str(path)).fit(X)
                with self.assertRaisesRegex(ValueError, "coupling_mode"):
                    HeisenbergProjectiveQFM(
                        name_file="direct", q_enc=3, topology=topology,
                        expectation_method="statevector", fixed_circuit_file_name=str(path),
                    ).fit(np.zeros((1, len(circuit.parameters))))

    def test_native_layout_offline(self):
        qfm = model(topology="native", ideal=False)
        qfm.real_backend = Mock()
        coupling = [(5, 11), (11, 5), (8, 11)]
        with patch("pqfmlib.maps.heisenberg.get_coupling_edges_and_costs", return_value=(coupling, {e: .01 for e in coupling})):
            result = qfm.fit_transform([[0., .25, .75]])
        self.assertEqual(result.shape, (1, 9))
        self.assertEqual(len(qfm.edges_act), 2)
        self.assertEqual(qfm.theta_info["num_blocks"], 1)
        self.assertEqual(qfm.theta_info["num_features"], 3)

    def test_cudaq_receives_couplings_through_existing_executor(self):
        X = np.array([[0., .25, .75]])
        for mps in (False, True):
            qfm = SuperexchangeHeisenbergPQFM(
                name_file="sites", q_enc=3, framework="cudaq", mps=mps,
                precision="single", expectation_method="exact", shots=123,
            )
            with patch.object(CudaqExecutor, "__init__", return_value=None) as init:
                with patch.object(CudaqExecutor, "execute", return_value=(np.zeros((1, 9)), [])) as execute:
                    actual = qfm.fit_transform(X)
            self.assertEqual(actual.shape, (1, 9))
            self.assertEqual(init.call_args.kwargs["mps"], mps)
            self.assertEqual(init.call_args.kwargs["precision"], "single")
            np.testing.assert_allclose(execute.call_args.args[1], make_superexchange_theta_matrix(X, 3)[0])
            self.assertEqual(execute.call_args.kwargs, {"expectation_method": "exact", "shots": 123})

    def test_qpu_run_contract_with_mock_estimator_only(self):
        X = np.array([[0., .25, .75]])
        with tempfile.TemporaryDirectory() as folder:
            pd.DataFrame([[0., .25, .75, 1]], columns=["a", "b", "c", "y"]).to_csv(
                Path(folder) / "sites.csv", index=False,
            )
            qfm = SuperexchangeHeisenbergPQFM(
                name_file="sites", q_enc=3, ideal=False, simulation=False,
                data_dir=folder, output_root=folder,
            )
            with self.assertRaises(NotImplementedError):
                qfm.fit(X)
            qfm.backend = GenericBackendV2(3, coupling_map=[[0, 1], [1, 0], [1, 2], [2, 1]], seed=42)
            qfm.real_backend = qfm.backend
            qfm.estimator = Mock()
            qfm.estimator.run.return_value.job_id.return_value = "offline-superexchange"
            with patch.object(qfm, "setup_backend_and_estimator"), patch("pqfmlib.maps.heisenberg.test_circuit_t"):
                self.assertEqual(qfm.run(), "offline-superexchange")
            pub = qfm.estimator.run.call_args.args[0][0]
            np.testing.assert_allclose(pub[2], make_superexchange_theta_matrix(X, 3)[0])
            metadata = json.loads((Path(qfm.base_folder) / "job_meta.json").read_text())
            self.assertEqual(metadata["coupling_mode"], "superexchange")
            self.assertEqual(metadata["n_param_sets"], 1)
            self.assertEqual(metadata["num_features"], 3)


if __name__ == "__main__":
    unittest.main()
