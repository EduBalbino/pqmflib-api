"""Framework API and Qiskit regressions, without CUDA-Q or GPU dependencies."""

import builtins
import inspect
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone

from pqfmlib import CDIsingProjectiveQFM, HeisenbergProjectiveQFM, XYZProjectiveQFM
from pqfmlib.core.execution import (
    execute_prepared_projected_feature_job,
    run_projected_feature_job,
)


MAP_TYPES = (CDIsingProjectiveQFM, HeisenbergProjectiveQFM, XYZProjectiveQFM)


class FrameworkSelectionTests(unittest.TestCase):
    X = np.random.default_rng(17).normal(size=(6, 2))

    def test_framework_is_keyword_only_and_clone_preserves_it(self):
        for map_type in MAP_TYPES:
            for framework in ("qiskit", "cudaq"):
                for mps in (False, True):
                    with self.subTest(map=map_type.__name__, framework=framework, mps=mps):
                        parameter = inspect.signature(map_type).parameters["framework"]
                        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
                        self.assertEqual(parameter.default, "qiskit")
                        # Existing positional arguments keep their meaning.
                        qfm = map_type("framework", 23, framework=framework, mps=mps)
                        self.assertEqual(qfm.seed, 23)
                        self.assertEqual(clone(qfm).get_params(), qfm.get_params())

    def test_invalid_framework_and_cudaq_modes_rejected_before_normalization(self):
        for map_type in MAP_TYPES:
            for framework in ("unknown", "CUDA-Q", "", None):
                with self.subTest(map=map_type.__name__, framework=framework):
                    with self.assertRaisesRegex(ValueError, "framework must be"):
                        map_type("invalid", framework=framework)
            for overrides in ({"simulation": False}, {"ideal": False}, {"fakebackend": True}):
                with self.subTest(map=map_type.__name__, overrides=overrides):
                    with self.assertRaisesRegex(ValueError, "requires simulation=True, ideal=True"):
                        map_type("invalid", framework="cudaq", **overrides)

    def test_qiskit_retains_legacy_ideal_normalization(self):
        for map_type in MAP_TYPES:
            with self.subTest(map=map_type.__name__):
                qfm = map_type("legacy", simulation=False, fakebackend=True)
                self.assertTrue(qfm.simulation)
                self.assertFalse(qfm.fakebackend)
                self.assertEqual(qfm.framework, "qiskit")
                self.assertEqual(qfm.expectation_method, "shots")

    def test_set_params_is_revalidated_before_backend_creation(self):
        for map_type in MAP_TYPES:
            for overrides in ({"simulation": False}, {"ideal": False}, {"fakebackend": True}):
                with self.subTest(map=map_type.__name__, overrides=overrides):
                    qfm = map_type("invalid")
                    qfm.set_params(framework="cudaq", **overrides)
                    with patch.object(qfm, "_load_real_backend") as load_backend:
                        with self.assertRaisesRegex(ValueError, "framework='cudaq' requires"):
                            qfm.setup_backend_and_estimator()
                    load_backend.assert_not_called()
                    self.assertIsNone(qfm.backend)
            qfm = map_type("invalid").set_params(framework="unknown")
            with self.assertRaisesRegex(ValueError, "framework must be"):
                qfm.setup_backend_and_estimator()

    def test_framework_change_requires_refit(self):
        for map_type in MAP_TYPES:
            with self.subTest(map=map_type.__name__):
                qfm = map_type("fitted", q_enc=2)
                qfm.fit(self.X)
                qfm.set_params(framework="cudaq")
                with self.assertRaisesRegex(RuntimeError, "configuration changed"):
                    qfm.transform(self.X)

    def test_qiskit_execution_does_not_import_cudaq(self):
        original_import = builtins.__import__

        def without_cudaq(name, *args, **kwargs):
            if name == "cudaq" or name.startswith("cudaq."):
                raise AssertionError("Qiskit execution must not import CUDA-Q")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=without_cudaq):
            for map_type in MAP_TYPES:
                with self.subTest(map=map_type.__name__):
                    qfm = map_type("optional", q_enc=2, expectation_method="statevector")
                    features = qfm.fit_transform(self.X)
                    self.assertEqual(features.shape[0], len(self.X))
                    self.assertTrue(np.isfinite(features).all())

    def test_explicit_qiskit_matches_default_in_both_simulators(self):
        for map_type in MAP_TYPES:
            for mps, method in ((False, "statevector"), (False, "shots"), (True, "shots")):
                with self.subTest(map=map_type.__name__, mps=mps, method=method):
                    kwargs = dict(name_file="regression", q_enc=2, mps=mps,
                                  expectation_method=method, shots=128)
                    default = map_type(**kwargs)
                    explicit = map_type(**kwargs, framework="qiskit")
                    expected = default.fit_transform(self.X)
                    actual = explicit.fit_transform(self.X)
                    np.testing.assert_array_equal(actual, expected)
                    self.assertEqual(explicit.obs_metadata, default.obs_metadata)
                    # The legacy execution path uses the same sampled/exact contract.
                    with tempfile.TemporaryDirectory() as folder:
                        explicit.X_q_all = self.X
                        explicit.base_folder = folder
                        explicit.execute_quantum_feature_map()
                    np.testing.assert_array_equal(explicit.Xq_all_raw, actual)

    def test_missing_cudaq_dependency_never_falls_back_to_aer(self):
        for map_type in MAP_TYPES:
            for mps in (False, True):
                with self.subTest(map=map_type.__name__, mps=mps):
                    qfm = map_type("pending", framework="cudaq", mps=mps)
                    with patch.object(qfm, "_make_aer_backend") as aer:
                        with patch("pqfmlib.core.cudaq_execution.import_module", side_effect=ImportError):
                            with self.assertRaisesRegex(ImportError, "optional CUDA-Q dependencies"):
                                qfm.setup_backend_and_estimator()
                    aer.assert_not_called()

    def test_execution_helpers_never_fall_back_to_qiskit(self):
        for framework, error in (("cudaq", ValueError), ("invalid", ValueError)):
            with self.subTest(framework=framework):
                with self.assertRaises(error):
                    execute_prepared_projected_feature_job(
                        None, np.empty((1, 0)), None, None, framework=framework,
                    )
                with patch("pqfmlib.core.execution.prepare_projected_feature_job") as prepare:
                    with self.assertRaises(error):
                        run_projected_feature_job(
                            None, [], np.empty((1, 0)), None, None, None, [], [],
                            framework=framework,
                        )
                prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
