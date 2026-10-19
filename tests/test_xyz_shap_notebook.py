"""Regression tests for the manual PCA search in the historically named notebook."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

NOTEBOOK = Path(__file__).resolve().parents[1] / 'examples' / 'xyz_breast_cancer_pca_pipeline.ipynb'


def load_scope():
    scope = {}
    for cell in json.loads(NOTEBOOK.read_text())['cells']:
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        compile(source, str(NOTEBOOK), 'exec')
        if cell['id'] in {'imports', 'data', 'options', 'manual-search', 'setup'}:
            if cell['id'] == 'options':
                scope['USE_GPU_STATEVECTOR'] = False
            exec(source, scope)
    return scope


class ManualPCANotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = load_scope()

    def test_configuration_and_real_simulation_for_all_options(self):
        s = self.scope
        self.assertEqual([m.q_enc for m in s['pqfm_options'].values()], [7, 6, 6])
        xyz = s['pqfm_options']['xyz']
        self.assertEqual(xyz.encoding_mode, 'shared_feature')
        self.assertFalse(xyz.keep_diagonal_terms)
        self.assertTrue(xyz.keep_cross_terms)
        self.assertIsInstance(xyz.measure_cross_observables, bool)
        self.assertEqual(xyz.expectation_method, "statevector")
        self.assertTrue(s['pqfm_options']['cd_ising'].measure_all_zz)
        X, _, y, _ = train_test_split(s['X'], s['y'], train_size=12,
                                      stratify=s['y'], random_state=42)
        for name, qfm in s['pqfm_options'].items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                self.assertTrue(qfm.ideal)
                self.assertTrue(qfm.simulation)
                preprocessor = clone(s['quantum_preprocessor'])
                preprocessor.set_params(features__quantum=clone(qfm).set_params(output_root=tmp))
                transformed = preprocessor.fit_transform(X, y)
                self.assertEqual(transformed.shape, (12, 6))
                self.assertEqual(preprocessor.transform(s['X'].iloc[:2]).shape, (2, 6))
                self.assertTrue(np.isfinite(transformed).all())

    def test_reuses_each_fold_and_matches_reference_search(self):
        s = self.scope
        X, y = s['X'].iloc[:40], s['y'][:40]
        splits = list(StratifiedKFold(2, shuffle=True, random_state=42).split(X, y))
        grid = {'n_estimators': [2, 3], 'max_depth': [1, 2]}
        fits, transforms, pca_inputs = [], [], []
        original_pca_fit_transform = PCA.fit_transform

        def fit(qfm, values, y=None):
            fits.append(np.asarray(values).copy())
            return qfm

        def transform(qfm, values):
            transforms.append(len(values))
            return np.sin(np.asarray(values))

        def fit_pca(pca, values, y=None):
            pca_inputs.append(np.asarray(values).copy())
            return original_pca_fit_transform(pca, values, y)

        with patch.object(s['XYZProjectiveQFM'], 'fit', fit), patch.object(s['XYZProjectiveQFM'], 'transform', transform), patch.object(PCA, 'fit_transform', fit_pca):
            result = s['manual_grid_search'](s['quantum_preprocessor'], X, y, grid, splits)
            self.assertEqual([len(v) for v in fits], [20, 20, 40])
            self.assertEqual(transforms, [20, 20, 20, 20, 40])
            self.assertEqual([v.shape for v in pca_inputs], [(20, 12), (20, 12), (40, 12)])
            for fold, (train, _) in enumerate(splits):
                expected = s['StandardScaler']().fit_transform(X.iloc[train])
                np.testing.assert_allclose(fits[fold], expected)
                np.testing.assert_allclose(pca_inputs[fold], np.column_stack([expected, np.sin(expected)]))
            result.best_estimator_.predict_proba(s['X'].iloc[40:45])
            self.assertEqual(len(fits), 3)
            self.assertEqual(len(pca_inputs), 3)
            reference = GridSearchCV(Pipeline([
                ('preprocessor', clone(s['quantum_preprocessor'])),
                ('classifier', GradientBoostingClassifier(random_state=42)),
            ]), {'classifier__' + k: v for k, v in grid.items()},
                scoring='roc_auc', cv=splits, error_score='raise')
            reference.fit(X, y)
            np.testing.assert_allclose(result.cv_results_['mean_test_score'], reference.cv_results_['mean_test_score'])
            self.assertEqual(result.best_params_, {k.removeprefix('classifier__'): v for k, v in reference.best_params_.items()})
            np.testing.assert_allclose(result.best_estimator_.predict_proba(X), reference.predict_proba(X))
        self.assertEqual(result.classifier_fit_seconds_.shape, (4, 2))
        self.assertTrue((result.preparation_seconds_ >= 0).all())
        self.assertEqual(result.best_estimator_.named_steps['classifier'].n_features_in_, 6)


if __name__ == '__main__':
    unittest.main()
