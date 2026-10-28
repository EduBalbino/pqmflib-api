"""Numerical backend comparisons; CUDA-Q GPU checks are explicitly opt-in."""

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from qiskit.providers.fake_provider import GenericBackendV2

from pqfmlib import HeisenbergProjectiveQFM, SuperexchangeHeisenbergPQFM


def model(n, **overrides):
    options = dict(name_file="backend_sites", q_enc=n, seed=71, alpha=.7, R=3,
                   J0=1.3, ell=.6, measure_all_pairs=True, expectation_method="statevector")
    options.update(overrides)
    return SuperexchangeHeisenbergPQFM(**options)


class SuperexchangeBackendTests(unittest.TestCase):
    def test_aer_mps_and_sampled_statevector_against_exact(self):
        shots = 16384
        for n in (2, 4):
            X = np.random.default_rng(20 + n).uniform(-.5, 1.5, (2, n))
            for topology in ("chain", "full"):
                expected = model(n, topology=topology).fit_transform(X)
                for mps in (False, True):
                    with self.subTest(n=n, topology=topology, mps=mps):
                        qfm = model(n, topology=topology, mps=mps, shots=shots,
                                    expectation_method="shots", mps_truncation_threshold=0.)
                        sampled = qfm.fit_transform(X)
                        np.testing.assert_allclose(sampled, expected, atol=6 / np.sqrt(shots))
                        self.assertEqual(qfm.backend.options.method, "matrix_product_state" if mps else "statevector")

    def test_single_precision_statevector_against_double(self):
        X = np.random.default_rng(4).normal(size=(3, 4))
        for topology in ("chain", "full"):
            with self.subTest(topology=topology):
                expected = model(4, topology=topology).fit_transform(X)
                single = model(4, topology=topology, precision="single")
                np.testing.assert_allclose(single.fit_transform(X), expected, atol=3e-6)
                self.assertEqual(single.backend.options.precision, "single")

    def test_direct_heisenberg_all_pairs_supports_other_encodings(self):
        X = np.random.default_rng(7).normal(size=(2, 5))
        for axes in (("x",), ("z", "y")):
            for encoding in ("shared_feature", "multi_axis"):
                with self.subTest(axes=axes, encoding=encoding):
                    options = dict(name_file="direct", q_enc=3, interaction_axes=axes,
                                   encoding_mode=encoding, expectation_method="statevector")
                    local = HeisenbergProjectiveQFM(**options)
                    complete = HeisenbergProjectiveQFM(**options, measure_all_pairs=True)
                    expected_local = local.fit_transform(X)
                    actual = complete.fit_transform(X)
                    self.assertEqual(actual.shape, (2, 18))
                    np.testing.assert_allclose(actual[:, :9], expected_local, atol=1e-12)
                    self.assertEqual(complete.theta_info, local.theta_info)

    def test_fakebackend_native_layout_without_noise_against_logical(self):
        # Hardware-shaped target, entirely local: no IBM credentials or calls.
        backend = GenericBackendV2(
            5, basis_gates=["id", "rz", "sx", "x", "cx"], seed=13, noise_info=False,
            coupling_map=[[i, j] for i in range(5) for j in range(5) if abs(i - j) == 1],
        )
        X = np.array([[.1, .3, .8], [-.2, .4, .5]])
        with tempfile.TemporaryDirectory() as folder:
            nodes = Path(folder) / "nodes.json"
            nodes.write_text("[2, 1, 3]")
            options = dict(ideal=False, topology="native", ibm_qpu="ibm_kingston",
                           use_fixed_phys_nodes=True, fixed_phys_nodes_file=str(nodes))
            logical = model(3, **options)
            logical.real_backend = backend
            expected = logical.fit_transform(X)
            fake = model(3, **options, fakebackend=True, expectation_method="shots",
                         fakebackend_method="statevector", shots=16384)
            fake.real_backend = backend
            actual = fake.fit_transform(X)
            self.assertEqual(fake.phys_nodes, [2, 1, 3])
            self.assertEqual(fake.edges_act, [(0, 1), (0, 2)])
            self.assertIsNotNone(fake._prepared_execution.qc_t.layout)
            np.testing.assert_allclose(actual, expected, atol=6 / np.sqrt(fake.shots))

    def test_unsupported_backend_combinations_still_fail(self):
        for options in ({"mps": True}, {"fakebackend": True}, {"simulation": False, "ideal": False}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                model(3, **options)
        with self.assertRaisesRegex(ValueError, "single.*unsupported"):
            model(3, expectation_method="shots", mps=True, precision="single")
        for options in ({"ideal": False}, {"fakebackend": True}, {"simulation": False}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                model(3, framework="cudaq", expectation_method="exact", **options)


@unittest.skipUnless(os.environ.get("PQFMLIB_TEST_CUDAQ_GPU") == "1",
                     "Set PQFMLIB_TEST_CUDAQ_GPU=1 on an accessible NVIDIA GPU.")
class SuperexchangeCudaqGPUTests(unittest.TestCase):
    def test_exact_statevector_and_mps_against_qiskit(self):
        # Explicit opt-in makes missing dependencies/GPU an error, not a skip.
        X = np.array([[0., .25, .75, 1.], [.1, -.2, .4, .8]])
        for topology in ("chain", "full"):
            expected = model(4, topology=topology).fit_transform(X)
            for mps in (False, True):
                for precision, atol in (("double", 1e-10), ("single", 5e-5)):
                    with self.subTest(topology=topology, mps=mps, precision=precision):
                        qfm = model(4, framework="cudaq", topology=topology, mps=mps,
                                    precision=precision, expectation_method="exact",
                                    mps_max_bond_dimension=16, mps_truncation_threshold=0.)
                        np.testing.assert_allclose(qfm.fit_transform(X), expected, atol=atol)

    def test_shots_statevector_and_mps_against_qiskit(self):
        X = np.array([[0., .25, .75]])
        expected = model(3).fit_transform(X)
        for mps in (False, True):
            with self.subTest(mps=mps):
                qfm = model(3, framework="cudaq", mps=mps, expectation_method="shots", shots=16384,
                            mps_max_bond_dimension=8, mps_truncation_threshold=0.)
                np.testing.assert_allclose(qfm.fit_transform(X), expected, atol=6 / np.sqrt(qfm.shots))


if __name__ == "__main__":
    unittest.main()
