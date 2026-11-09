"""Base class for projected quantum feature maps."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from qiskit_aer import AerSimulator
from sklearn.base import BaseEstimator, TransformerMixin

from pqfmlib.core.backend import load_ibm_backend, make_aer_backend, make_estimator
from pqfmlib.core.data import load_tabular_dataset
from pqfmlib.core.framework import (
    validate_framework_configuration, validate_precision, validate_aer_precision,
)


@dataclass
class BaseProjectiveQFM(TransformerMixin, BaseEstimator):
    """Shared runner state for PQFM implementations.

    Subclasses implement the physics-specific pieces: block construction,
    circuit construction, theta matrix construction, and observables.
    """

    name_file: str
    seed: int = 42
    simulation: bool = True
    fakebackend: bool = False
    ideal: bool = True
    framework: str = field(default="qiskit", kw_only=True)
    precision: str = field(default="double", kw_only=True)
    expectation_method: str = field(default="shots", kw_only=True)
    shots: int = 4096
    ibm_qpu: str = "ibm_fez"
    q_enc: int = 20
    data_dir: str = "./data"
    output_root: str = "."
    mps: bool = False
    use_gpu_statevector: bool = False
    statevector_device: Optional[str] = None
    mps_max_bond_dimension: Optional[int] = None
    mps_truncation_threshold: float = 1e-16
    fakebackend_method: str = "density_matrix"
    fakebackend_device: str = "CPU"
    resilience_level: int = 0
    qiskit_channel: str = "ibm_cloud"
    save_circuit_drawings: bool = False

    def __sklearn_clone__(self):
        """Return an unfitted estimator containing only constructor parameters.

        PQFM subclasses normalize some constructor parameters during
        ``__post_init__`` (notably XYZ ``axes``).  Reconstructing explicitly
        implements scikit-learn's clone protocol without copying fitted quantum
        state, runtime backends, credentials, or prepared circuits.
        """
        return type(self)(**self.get_params(deep=False))

    def __post_init__(self) -> None:
        self._validate_framework()
        self._validate_expectation_method()
        self._validate_resource_estimation_mode()
        if self.ideal:
            self.simulation = True
            self.fakebackend = False
        self._validate_precision()
        if self.mps and self.use_gpu_statevector:
            self.use_gpu_statevector = False
        np.random.seed(self.seed)
        self.service = None
        self.real_backend = None
        self.backend = None
        self.estimator = None
        self.df_full: pd.DataFrame | None = None
        self.X = None
        self.y = None
        self.X_q_all = None
        self.num_features = None
        self.base_folder = None
        self._is_fitted = False
        self.n_features_in_: int | None = None
        self.feature_names_in_: np.ndarray | None = None
        self._feature_names_in_signature: tuple[object, ...] | None = None
        self._fitted_config_signature: tuple | None = None

    @staticmethod
    def _coerce_feature_matrix(X) -> tuple[np.ndarray, tuple[object, ...] | None]:
        """Return a finite 2D float matrix and optional DataFrame column names."""
        feature_names = None
        if isinstance(X, pd.DataFrame):
            non_numeric = X.columns[~X.apply(lambda s: pd.api.types.is_numeric_dtype(s))]
            if len(non_numeric) > 0:
                raise ValueError(f"X must contain only numeric columns. Non-numeric columns: {list(non_numeric)}")
            feature_names = tuple(X.columns)
            values = X.to_numpy(dtype=float, copy=True)
        else:
            try:
                values = np.asarray(X, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError("X must be convertible to a numeric matrix.") from exc
            values = np.array(values, dtype=float, copy=True)

        if values.ndim != 2:
            raise ValueError(f"X must be a 2D matrix, but got {values.ndim} dimensions.")
        if values.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")
        if values.shape[1] == 0:
            raise ValueError("X must contain at least one feature.")
        if not np.isfinite(values).all():
            raise ValueError("X must contain only finite values.")
        return values, feature_names

    def _prepare_fit_input(self, X) -> np.ndarray:
        """Validate training input and reset the internal fitted lifecycle."""
        values, feature_names = self._coerce_feature_matrix(X)
        self._is_fitted = False
        self.n_features_in_ = int(values.shape[1])
        self._feature_names_in_signature = feature_names
        self.feature_names_in_ = None if feature_names is None else np.asarray(feature_names, dtype=object)
        self._fitted_config_signature = None
        return values

    def _mark_fitted(self, config_signature: tuple) -> None:
        """Mark preparation as complete and freeze its structural signature."""
        if self.n_features_in_ is None:
            raise RuntimeError("Training input must be prepared before marking the estimator as fitted.")
        self._fitted_config_signature = tuple(config_signature)
        self._is_fitted = True

    def _prepare_transform_input(self, X, config_signature: tuple) -> np.ndarray:
        """Validate transform input against the frozen training structure."""
        if not self._is_fitted or self.n_features_in_ is None:
            raise RuntimeError("This PQFM instance is not fitted yet. Call fit(X) before transform(X).")
        if tuple(config_signature) != self._fitted_config_signature:
            raise RuntimeError("Structural PQFM configuration changed after fit; call fit(X) again before transform(X).")

        values, feature_names = self._coerce_feature_matrix(X)
        if int(values.shape[1]) != self.n_features_in_:
            raise ValueError(
                f"X has {values.shape[1]} features, but this PQFM was fitted with {self.n_features_in_} features."
            )
        if self._feature_names_in_signature is not None and feature_names != self._feature_names_in_signature:
            raise ValueError("DataFrame columns must match the names and order used during fit.")
        return values

    def _validate_framework(self) -> None:
        validate_framework_configuration(
            self.framework,
            simulation=self.simulation,
            ideal=self.ideal,
            fakebackend=self.fakebackend,
        )
    def _validate_expectation_method(self) -> None:
        if self.framework in ("cudaq", "tilelang") and self.expectation_method == "exact":
            return
        if self.framework == "tilelang":
            raise ValueError("TileLang requires expectation_method='exact'.")
        if self.expectation_method not in ("shots", "statevector"):
            raise ValueError("expectation_method must be 'shots' or 'statevector'.")
        if self.expectation_method == "statevector":
            if not self.simulation or self.fakebackend:
                raise ValueError("statevector requires simulation=True and fakebackend=False.")
            if self.mps:
                raise ValueError("statevector requires mps=False (full state vector).")

    def _validate_precision(self) -> None:
        validate_precision(self.precision)
        if not self.ideal and not self.simulation:
            if self.precision == "single":
                raise ValueError("precision='single' is a simulator option and cannot configure real QPU execution.")
            return
        if self.framework == "qiskit":
            method = (self.fakebackend_method if self.fakebackend and not self.ideal
                      else ("matrix_product_state" if self.mps else "statevector"))
            validate_aer_precision(self.precision, method)

    def _validate_resource_estimation_mode(self) -> None:
        if self.ideal and bool(getattr(self, "resource_estimation", False)):
            raise ValueError("resource_estimation=True is invalid when ideal=True; ideal simulations do not use QPU resources.")

    def load_data(self) -> None:
        self.df_full, self.X, self.y = load_tabular_dataset(self.name_file, self.data_dir)
        self.X_q_all = self.X.copy()
        self.num_features = int(self.X.shape[1])

    def setup_backend_and_estimator(self) -> None:
        self._validate_framework()
        self._validate_precision()
        self._validate_expectation_method()
        self._validate_resource_estimation_mode()
        if self.framework == "tilelang":
            from pqfmlib.experimental.heisenberg_tilelang import TileLangExecutor

            if self.precision != "double" or self.mps or self.statevector_device not in (None, "GPU"):
                raise ValueError("TileLang requires double precision and GPU statevector execution.")
            self.backend = TileLangExecutor()
            self.estimator = None
            return
        if self.framework == "cudaq":
            from pqfmlib.core.cudaq_execution import CudaqExecutor

            if self.statevector_device not in (None, "GPU"):
                raise ValueError("framework='cudaq' requires a GPU; statevector_device must be None or 'GPU'.")
            self.backend = CudaqExecutor(
                precision=self.precision,
                mps=self.mps,
                seed=self.seed,
                mps_max_bond_dimension=self.mps_max_bond_dimension,
                mps_truncation_threshold=self.mps_truncation_threshold,
            )
            self.estimator = None
            return
        if self.ideal:
            self.backend = self._make_aer_backend()
            self._set_aer_max_qubits()
        else:
            self._load_real_backend()
            if self.simulation:
                if self.fakebackend:
                    self.backend = AerSimulator.from_backend(self.real_backend)
                    self.backend.set_options(
                        precision=self.precision,
                        method=self.fakebackend_method,
                        device=self.fakebackend_device,
                        seed_simulator=self.seed,
                    )
                    self._set_aer_max_qubits()
                else:
                    self.backend = self._make_aer_backend()
                    self._set_aer_max_qubits()
            else:
                self.backend = self.real_backend
                self.backend.options.seed_transpiler = self.seed
        self.estimator = (
            make_estimator(self.backend, shots=self.shots, resilience_level=self.resilience_level)
            if self.expectation_method == "shots" else None
        )

    def _load_real_backend(self) -> None:
        if self.real_backend is None:
            self.real_backend, self.service = load_ibm_backend(self.ibm_qpu, channel=self.qiskit_channel)

    def _make_aer_backend(self):
        return make_aer_backend(
            precision=self.precision,
            seed=self.seed,
            mps=self.mps,
            use_gpu_statevector=self.use_gpu_statevector,
            statevector_device=self.statevector_device,
            mps_max_bond_dimension=self.mps_max_bond_dimension,
            mps_truncation_threshold=self.mps_truncation_threshold,
        )

    def _set_aer_max_qubits(self) -> None:
        if isinstance(self.backend, AerSimulator) and hasattr(self.backend, "set_max_qubits"):
            self.backend.set_max_qubits(int(self.q_enc))

    def ensure_output_folder(self, name: str) -> str:
        if self.output_root in ("", ".", "./", ".\\"):
            self.base_folder = name
        else:
            self.base_folder = str(Path(self.output_root) / name)
        Path(self.base_folder).mkdir(parents=True, exist_ok=True)
        return self.base_folder

    def run(self):
        raise NotImplementedError("Subclasses must implement run().")
