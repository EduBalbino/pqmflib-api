"""Heisenberg Projective Quantum Feature Map."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.quantum_info import SparsePauliOp, random_unitary
from qiskit.transpiler import Layout
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

from pqfmlib.core.base import BaseProjectiveQFM
from pqfmlib.core.execution import (
    PreparedProjectedFeatureJob,
    execute_prepared_projected_feature_job,
    reorder_theta_by_parameter_names,
    submit_job_and_save_metadata,
    validate_circuit_parameter_compatibility,
)
from pqfmlib.core.observables import pauli_string
from pqfmlib.core.resources import test_circuit_t
from pqfmlib.core.serialization import metadata_to_column_name
from pqfmlib.hardware.coupling import get_coupling_edges_and_costs
from pqfmlib.hardware.feature_assignment import pick_connected_subset_greedy
from pqfmlib.hardware.layout import edges_log_from_phys_nodes, validate_phys_nodes
from pqfmlib.utils.io import load_json, save_circuit_draw, save_json


def heisenberg_chain_observables(
    num_qubits: int,
    *,
    measure_2local_diagonal: bool = False,
    measure_cross_observables: bool = False,
    measure_all_pairs: bool = False,
    edges_act: list[tuple[int, int]] | None = None,
):
    """Build Heisenberg observables.

    Always measures one-local Z, X, Y per qubit.

    Optionally measures two-local diagonal observables:
        ZZ, XX, YY
    Independently, cross observables add XY, YX, XZ, ZX, YZ, ZY per edge.
    Measurements do not depend on the axes selected for encoding.

    By default, two-local observables are measured on nearest-neighbor
    chain edges only.

    ``measure_all_pairs`` adds ZZ, XX, YY for every pair i < j, independent
    of interaction topology and measure_2local_diagonal. It cannot be combined
    with cross observables, so its output has exactly 3*N + 3*N*(N-1)/2 terms.
    """
    num_qubits = int(num_qubits)
    if measure_all_pairs and measure_cross_observables:
        raise ValueError("measure_all_pairs cannot be combined with measure_cross_observables.")

    obs = []
    metadata = []

    # One-local observables: Z, X, Y per qubit.
    for i in range(num_qubits):
        for axis in ("z", "x", "y"):
            obs.append(
                SparsePauliOp.from_list(
                    [(pauli_string(num_qubits, [(i, axis)]), 1.0)]
                )
            )
            metadata.append((axis, i))

    # Two-local diagonal observables: ZZ, XX, YY.
    if measure_all_pairs or measure_2local_diagonal or measure_cross_observables:
        if measure_all_pairs:
            edges_act = [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        elif edges_act is None:
            edges_act = [(i, i + 1) for i in range(num_qubits - 1)]

        edges = sorted({(min(i, j), max(i, j)) for i, j in edges_act})

        axis_pairs = ["zz", "xx", "yy"] if measure_2local_diagonal or measure_all_pairs else []
        if measure_cross_observables:
            axis_pairs += ["xy", "yx", "xz", "zx", "yz", "zy"]
        for i, j in edges:
            for axis_pair in axis_pairs:
                obs.append(
                    SparsePauliOp.from_list(
                        [(pauli_string(num_qubits, [(i, axis_pair[0]), (j, axis_pair[1])]), 1.0)]
                    )
                )
                metadata.append((axis_pair, i, j))

    return obs, metadata


def _heisenberg_edge_schedule(num_qubits, edges_act=None):
    """Keep the legacy even/odd chain order or canonicalize an explicit graph."""
    if edges_act is None:
        return [(i, i + 1) for parity in (0, 1)
                for i in range(parity, num_qubits - 1, 2)]
    edges = set()
    for i, j in edges_act:
        if not isinstance(i, (int, np.integer)) or not isinstance(j, (int, np.integer)):
            raise ValueError("Interaction edges must use integer qubit indices.")
        if not 0 <= i < num_qubits or not 0 <= j < num_qubits or i == j:
            raise ValueError("Interaction edges must connect distinct in-range qubits.")
        edges.add((min(i, j), max(i, j)))
    if not edges:
        raise ValueError("Heisenberg requires at least one interaction edge.")
    return sorted(edges)


def _validate_interaction_encoding(interaction_axes, encoding_mode):
    """Validate interaction-only encoding; axis order defines feature order."""
    if not isinstance(interaction_axes, (tuple, list)):
        raise ValueError("interaction_axes must be a tuple or list of x, y, z.")
    axes = tuple(str(axis).lower() for axis in interaction_axes)
    if not axes or len(set(axes)) != len(axes) or any(a not in ("x", "y", "z") for a in axes):
        raise ValueError("interaction_axes must contain 1 to 3 unique axes from x, y, z.")
    if encoding_mode not in ("shared_feature", "multi_axis"):
        raise ValueError("encoding_mode must be 'shared_feature' or 'multi_axis'.")
    return axes


def make_superexchange_theta_matrix(
    X: np.ndarray,
    num_qubits: int,
    *,
    edges_act: list[tuple[int, int]] | None = None,
    J0: float = 1.0,
    ell: float = 1.0,
):
    """Map site positions to isotropic couplings in circuit schedule order.

    Exactly one feature is required per qubit. No dataset statistics, scaling,
    padding or extra blocks are used. Very distant sites can yield numerical
    zero through floating-point underflow of the exponential.
    """
    for name, value in (("J0", J0), ("ell", ell)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    values, _ = BaseProjectiveQFM._coerce_feature_matrix(X)
    if values.shape[1] != num_qubits:
        raise ValueError("Superexchange requires exactly q_enc features (one per qubit).")
    if num_qubits < 2:
        raise ValueError("Superexchange requires at least two qubits.")
    edges = _heisenberg_edge_schedule(num_qubits, edges_act)
    left, right = np.asarray(edges, dtype=int).T
    # Extended intermediates preserve finite distances divided by a large ell
    # and avoid underflow before multiplication by a large J0. Public parameter
    # matrices remain float64, like direct encoding.
    with np.errstate(over="ignore", under="ignore"):
        positions = values.astype(np.longdouble)
        distances = np.abs(positions[:, left] - positions[:, right])
        theta = (np.longdouble(J0) * np.exp(-2 * distances / np.longdouble(ell))).astype(np.float64)
    return theta, {
        "num_features": int(num_qubits),
        "features_per_block": len(edges),
        "num_blocks": 1,
        "total_slots": len(edges),
        "use_tanh_scaling": False,
    }


def make_heisenberg_theta_matrix(
    X: np.ndarray,
    num_qubits: int,
    *,
    features_per_block: Optional[int] = None,
    use_tanh_scaling: bool = True,
    edges_act: list[tuple[int, int]] | None = None,
    interaction_axes: tuple[str, ...] = ("x", "y", "z"),
    encoding_mode: str = "shared_feature",
):
    """Build the parameter matrix used by the notebook Heisenberg map.

    Explicit ``edges_act`` select a graph; otherwise the legacy chain is used.
    Shared encoding uses one feature per edge. Multi-axis encoding uses one
    feature per selected axis per edge, in ``interaction_axes`` order. Input
    columns fill these slots sequentially; the last block is padded with zeros.
    The notebook uses ``features_per_block = num_qubits - 1`` because each block
    encodes one feature per nearest-neighbor chain edge. If the dataset has more
    features than one block can cover, additional blocks are appended.
    """
    num_qubits = int(num_qubits)
    axes = _validate_interaction_encoding(interaction_axes, encoding_mode)
    edge_capacity = len(_heisenberg_edge_schedule(num_qubits, edges_act))
    if encoding_mode == "multi_axis":
        edge_capacity *= len(axes)
    if features_per_block is None:
        features_per_block = edge_capacity
    features_per_block = int(features_per_block)

    if features_per_block <= 0:
        raise ValueError("features_per_block must be positive.")
    if features_per_block > edge_capacity:
        raise ValueError("features_per_block cannot exceed the interaction encoding capacity.")

    n_samples, num_features = X.shape
    theta_raw = X.astype(np.float64, copy=True)
    if use_tanh_scaling:
        theta_raw = 2.0 * np.pi * np.tanh(theta_raw / 3.0)

    num_blocks = int(np.ceil(num_features / features_per_block))
    total_slots = num_blocks * features_per_block

    theta_values = np.zeros((n_samples, total_slots), dtype=np.float64)
    theta_values[:, :num_features] = theta_raw

    return theta_values, {
        "num_features": int(num_features),
        "features_per_block": int(features_per_block),
        "num_blocks": int(num_blocks),
        "total_slots": int(total_slots),
        "use_tanh_scaling": bool(use_tanh_scaling),
    }


def build_heisenberg_chain_feature_circuit(
    num_qubits: int,
    *,
    num_blocks: int,
    features_per_block: Optional[int] = None,
    R: int = 2,
    alpha: float = 0.1,
    seed: int = 42,
    add_barriers: bool = True,
    edges_act: list[tuple[int, int]] | None = None,
    interaction_axes: tuple[str, ...] = ("x", "y", "z"),
    encoding_mode: str = "shared_feature",
):
    """Build the notebook Heisenberg feature-map circuit.

    Explicit ``edges_act`` use sorted undirected edges instead of the chain.
    Shared encoding uses one parameter for all selected axes on an edge.
    Multi-axis encoding assigns consecutive parameters in interaction_axes order.
    Only RXX, RYY and RZZ encode features; initial local unitaries are fixed.
    For the default chain, the even-edge sweep precedes the odd-edge sweep.
    Each complete schedule is repeated ``R`` times.
    """
    from qiskit.circuit import ParameterVector

    num_qubits = int(num_qubits)
    num_blocks = int(num_blocks)
    R = int(R)

    axes = _validate_interaction_encoding(interaction_axes, encoding_mode)
    edges = _heisenberg_edge_schedule(num_qubits, edges_act)
    features_per_edge = len(axes) if encoding_mode == "multi_axis" else 1
    capacity = len(edges) * features_per_edge
    if features_per_block is None:
        features_per_block = capacity
    features_per_block = int(features_per_block)
    if features_per_block != capacity:
        raise ValueError("features_per_block must equal the interaction encoding capacity.")

    qc = QuantumCircuit(num_qubits)

    for q in range(num_qubits):
        unitary = random_unitary(2, seed=seed + q)
        qc.unitary(unitary, [q], label=f"U_{q}")

    theta = ParameterVector("theta", num_blocks * features_per_block)
    phi_scale = float(alpha) / (2.0 * float(num_blocks))

    for _rep in range(R):
        index = 0
        for _block in range(num_blocks):
            for i, j in edges:
                # Preserve the legacy gate order; features follow interaction_axes.
                for axis, gate in (("z", qc.rzz), ("x", qc.rxx), ("y", qc.ryy)):
                    if axis in axes:
                        offset = axes.index(axis) if encoding_mode == "multi_axis" else 0
                        gate(phi_scale * theta[index + offset], i, j)
                index += features_per_edge

            if add_barriers:
                qc.barrier()

    qc.metadata = {"pqfmlib_heisenberg": {
        "schema_version": 1,
        "q_enc": num_qubits,
        "edge_schedule": [list(edge) for edge in edges],
        "interaction_axes": list(axes),
        "encoding_mode": encoding_mode,
        "num_blocks": num_blocks,
        "features_per_block": features_per_block,
        "R": R,
        "alpha": float(alpha),
        "seed": int(seed),
    }}

    return qc, list(theta), {
        "R": int(R),
        "alpha": float(alpha),
        "phi_scale": float(phi_scale),
        "num_blocks": int(num_blocks),
        "features_per_block": int(features_per_block),
    }


def _extract_initial_phys_nodes(qc_t, qc_reference: QuantumCircuit) -> list[int] | None:
    """Extract the initial physical layout chosen by the transpiler, if available."""
    try:
        layout = qc_t.layout.initial_virtual_layout(filter_ancillas=True)
        return [int(layout[q]) for q in qc_reference.qubits]
    except Exception:
        return None


@dataclass
class HeisenbergProjectiveQFM(BaseProjectiveQFM):
    """Heisenberg PQFM implementation based on the original notebook.

    Parameters
    ----------
    topology:
        "chain" preserves the original schedule. "full" requires ideal=True.
        "native" requires ideal=False and uses the induced QPU subgraph on
        q_enc selected physical nodes, pinned as the initial transpiler layout.
        Fixed QPY files support both encodings and all available topologies.
    interaction_axes:
        One to three axes from x, y, z. In multi_axis mode their order defines
        feature assignment within each edge. Independent of measurement axes.
    encoding_mode:
        "shared_feature" uses one feature per edge across the selected axes;
        "multi_axis" uses one feature per selected axis per edge.
    coupling_mode:
        "direct" preserves feature-to-edge encoding. "superexchange" requires
        exactly q_enc site features and isotropic shared interactions, using
        J0 * exp(-2 * abs(x_i - x_j) / ell) without tanh scaling. J0 and ell
        must be finite and positive. One block covers all interaction edges.
    measure_cross_observables:
        Add separate XY, YX, XZ, ZX, YZ, ZY measurements on each active edge.
        Independent of measure_2local_diagonal; does not change encoding.
    measure_all_pairs:
        Measure local Z/X/Y followed by ZZ/XX/YY for every pair i < j,
        independently of interaction topology. Cannot be combined with
        measure_cross_observables. Does not change the circuit or Hamiltonian.
    R:
        Number of repetitions of the even/odd Heisenberg chain schedule.
    alpha:
        Global interaction scale. The effective parameter scale is
        ``alpha / (2 * num_blocks)`` as in the notebook.
    use_tanh_scaling:
        If True, encode features as ``2*pi*tanh(x/3)``.
    fixed_circuit_file_name:
        Load a logical or transpiled QPY. Logical circuits are compiled for QPU
        execution; compatible transpiled circuits retain their layout. Encoding
        metadata is checked when present. For older QPYs, the supplied encoding
        configuration is used with a warning because its semantics cannot be verified.
    use_fixed_phys_nodes:
        For the chain, False lets the transpiler choose a layout. For native
        topology, False selects a connected subgraph automatically.
        If True, the given physical nodes are
        used as an initial layout, while routing remains enabled.
    """

    topology: str = field(default="chain", kw_only=True)
    interaction_axes: tuple[str, ...] = field(default=("x", "y", "z"), kw_only=True)
    encoding_mode: str = field(default="shared_feature", kw_only=True)
    measure_cross_observables: bool = field(default=False, kw_only=True)
    measure_all_pairs: bool = field(default=False, kw_only=True)
    coupling_mode: str = field(default="direct", kw_only=True)
    J0: float = field(default=1.0, kw_only=True)
    ell: float = field(default=1.0, kw_only=True)

    R: int = 2
    alpha: float = 0.1
    use_tanh_scaling: bool = True
    add_barriers: bool = True

    use_fixed_phys_nodes: bool = False
    fixed_phys_nodes_file: str = "phys_nodes.json"
    fixed_circuit_file_name: str = ""

    resource_estimation: bool = False

    # Kept for backward compatibility. The notebook measures only one-local Z/X/Y.
    axes: tuple[str, ...] = ("z", "x", "y")
    measure_2local_diagonal: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self._validate_topology()
        self._validate_encoding()
        if int(self.q_enc) < 2:
            raise ValueError("HeisenbergProjectiveQFM requires q_enc >= 2.")
        self.edges_act = None
        self.phys_nodes = None
        self.obs_metadata = None
        self.Xq_all_raw = None
        self.job_id = None
        self.theta_info = None
        self.circuit_info = None
        self._prepared_execution = None
        self._fixed_circuit = None
        self.metadata_output_path = None

    def _validate_topology(self):
        if self.topology not in ("chain", "native", "full"):
            raise ValueError("topology must be 'chain', 'native', or 'full'.")
        if self.topology == "full" and not self.ideal:
            raise ValueError("topology='full' requires ideal=True.")
        if self.topology == "native" and self.ideal:
            raise ValueError("topology='native' requires ideal=False and a target QPU.")

    def _validate_encoding(self):
        if self.measure_all_pairs and self.measure_cross_observables:
            raise ValueError("measure_all_pairs cannot be combined with measure_cross_observables.")
        self.interaction_axes = _validate_interaction_encoding(self.interaction_axes, self.encoding_mode)
        if self.coupling_mode not in ("direct", "superexchange"):
            raise ValueError("coupling_mode must be 'direct' or 'superexchange'.")
        if self.coupling_mode == "superexchange":
            if set(self.interaction_axes) != {"x", "y", "z"} or self.encoding_mode != "shared_feature":
                raise ValueError("Superexchange requires isotropic x/y/z interactions and shared_feature encoding.")
            for name in ("J0", "ell"):
                value = getattr(self, name)
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be finite and positive.")

    def _make_theta_matrix(self, X):
        """Use the selected coupling strategy for both transformer and runner."""
        if self.coupling_mode == "superexchange":
            return make_superexchange_theta_matrix(
                X, self.q_enc, edges_act=self._encoding_edges(), J0=self.J0, ell=self.ell,
            )
        return make_heisenberg_theta_matrix(
            X, self.q_enc, edges_act=self._encoding_edges(),
            interaction_axes=self.interaction_axes, encoding_mode=self.encoding_mode,
            use_tanh_scaling=self.use_tanh_scaling,
        )

    def _coupling_metadata(self):
        metadata = {"coupling_mode": self.coupling_mode}
        if self.coupling_mode == "superexchange":
            metadata.update(J0=float(self.J0), ell=float(self.ell))
        return metadata

    def _encoding_edges(self):
        # None retains the exact legacy chain parameter ordering.
        if self.topology != "chain" and self.edges_act is None:
            raise RuntimeError("Call prepare_layout() before building the interaction map.")
        return None if self.topology == "chain" else self.edges_act

    def _transform_config_signature(self) -> tuple:
        """Return configuration that must remain unchanged after fit."""
        return (
            "HeisenbergProjectiveQFM",
            self.topology,
            tuple(self.interaction_axes),
            self.encoding_mode,
            self.coupling_mode,
            float(self.J0),
            float(self.ell),
            bool(self.measure_cross_observables),
            bool(self.measure_all_pairs),
            int(self.seed),
            bool(self.simulation),
            bool(self.fakebackend),
            bool(self.ideal),
            self.framework,
            self.precision,
            self.expectation_method,
            int(self.shots),
            str(self.ibm_qpu),
            int(self.q_enc),
            bool(self.mps),
            bool(self.use_gpu_statevector),
            self.statevector_device,
            self.mps_max_bond_dimension,
            float(self.mps_truncation_threshold),
            str(self.fakebackend_method),
            str(self.fakebackend_device),
            int(self.resilience_level),
            int(self.R),
            float(self.alpha),
            bool(self.use_tanh_scaling),
            bool(self.add_barriers),
            bool(self.use_fixed_phys_nodes),
            str(self.fixed_phys_nodes_file),
            str(self.fixed_circuit_file_name),
            tuple(self.axes),
            bool(self.measure_2local_diagonal),
        )

    def _reset_fitted_map_state(self) -> None:
        self.edges_act = None
        self.phys_nodes = None
        self.obs_metadata = None
        self.Xq_all_raw = None
        self.job_id = None
        self.theta_info = None
        self.circuit_info = None
        self._prepared_execution = None
        self._fixed_circuit = None
        self.metadata_output_path = None

    def _build_circuit_and_observables(self):
        qc_param, param_order, self.circuit_info = build_heisenberg_chain_feature_circuit(
            self.q_enc,
            num_blocks=self.theta_info["num_blocks"],
            features_per_block=self.theta_info["features_per_block"],
            R=self.R,
            alpha=self.alpha,
            seed=self.seed,
            add_barriers=self.add_barriers,
            edges_act=self._encoding_edges(),
            interaction_axes=self.interaction_axes,
            encoding_mode=self.encoding_mode,
        )
        qc_param.metadata["pqfmlib_heisenberg"].update(
            self._coupling_metadata(), use_tanh_scaling=self.theta_info["use_tanh_scaling"],
        )
        obs_list, obs_metadata = heisenberg_chain_observables(
            self.q_enc,
            measure_2local_diagonal=self.measure_2local_diagonal,
            measure_cross_observables=self.measure_cross_observables,
            measure_all_pairs=self.measure_all_pairs,
            edges_act=self.edges_act,
        )
        return qc_param, param_order, obs_list, obs_metadata

    def _prepare_fitted_execution(self) -> None:
        qc_param, param_order, obs_list, obs_metadata = self._build_circuit_and_observables()
        qc_t = self._transpile_or_load_circuit(qc_param)
        validate_circuit_parameter_compatibility(qc_t, param_order)
        if (self.fakebackend or self.fixed_circuit_file_name) and self.phys_nodes is None:
            extracted = _extract_initial_phys_nodes(qc_t, qc_param)
            if extracted is not None:
                self.phys_nodes = extracted
        has_fixed_layout = bool(self.fixed_circuit_file_name) and getattr(qc_t, "layout", None) is not None
        if self.fakebackend or has_fixed_layout:
            obs_isa = [obs.apply_layout(qc_t.layout) for obs in obs_list]
        else:
            obs_isa = obs_list
        self._prepared_execution = PreparedProjectedFeatureJob(
            qc_t=qc_t,
            param_order=tuple(param_order),
            obs_isa_broadcast=[[obs] for obs in obs_isa],
            obs_metadata=obs_metadata,
            fixed_circuit_loaded=bool(self.fixed_circuit_file_name),
        )
        self.obs_metadata = obs_metadata

    def fit(self, X, y=None):
        """Prepare the train-dimensional Heisenberg simulation state."""
        if not self.simulation:
            raise NotImplementedError("fit()/transform() currently support only simulation=True.")

        self._validate_topology()
        self._validate_encoding()
        self._prepare_fit_input(X)
        if self.coupling_mode == "superexchange" and self.n_features_in_ != self.q_enc:
            raise ValueError("Superexchange requires exactly q_enc features (one per qubit).")
        self._reset_fitted_map_state()
        self.base_folder = None
        self.num_features = self.n_features_in_
        self.setup_backend_and_estimator()
        self.prepare_layout()
        _unused_theta, self.theta_info = self._make_theta_matrix(
            np.zeros((1, self.n_features_in_), dtype=float))
        self._prepare_fitted_execution()
        self._mark_fitted(self._transform_config_signature())
        return self

    def transform(self, X) -> np.ndarray:
        """Apply the fitted Heisenberg map without changing its structure."""
        X_transform = self._prepare_transform_input(X, self._transform_config_signature())
        theta_values_all, theta_info = self._make_theta_matrix(X_transform)
        structural_keys = (
            "num_features",
            "features_per_block",
            "num_blocks",
            "total_slots",
            "use_tanh_scaling",
        )
        if any(theta_info[key] != self.theta_info[key] for key in structural_keys):
            raise RuntimeError("Heisenberg transform structure differs from the fitted state.")
        Xq, _obs_metadata = execute_prepared_projected_feature_job(
            self._prepared_execution,
            theta_values_all,
            self.backend,
            self.estimator,
            simulation=True,
            framework=self.framework,
            expectation_method=self.expectation_method,
            shots=self.shots,
            base_folder=None,
        )
        return Xq

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit the Heisenberg map and transform the training matrix."""
        return self.fit(X, y=y).transform(X)

    def prepare_output_folder(self) -> None:
        features_per_block = (int(self.q_enc) - 1 if self.topology == "chain"
                              else len(self.edges_act))
        if self.encoding_mode == "multi_axis":
            features_per_block *= len(self.interaction_axes)
        parts = [
            f"heisenberg__R{int(self.R)}",
            "quantum_features",
            self.name_file,
            str(self.num_features),
            str(features_per_block),
        ]
        if self.ideal:
            parts.append("ideal")
        else:
            parts.append(self.ibm_qpu)
            if self.simulation:
                parts.append("fakebackend" if self.fakebackend else "simulacao")
        if self.topology != "chain":
            parts.append(self.topology)
        if self.encoding_mode != "shared_feature" or self.interaction_axes != ("x", "y", "z"):
            parts.extend([self.encoding_mode, "axes" + "".join(self.interaction_axes)])
        if self.measure_2local_diagonal:
            parts.append("diagobs")
        if self.measure_cross_observables:
            parts.append("crossobs")
        if self.measure_all_pairs:
            parts.append("allpairs")
        if self.coupling_mode == "superexchange":
            parts.extend(["superexchange", f"J0{self.J0}", f"ell{self.ell}"])
        self.ensure_output_folder("_".join(parts))

    def prepare_layout(self) -> None:
        self._validate_topology()
        self._validate_encoding()
        self._fixed_circuit = None
        saved_nodes = None
        if self.fixed_circuit_file_name:
            saved_nodes = self._fixed_initial_nodes(self._read_fixed_circuit())
        self.edges_act = [(i, i + 1) for i in range(self.q_enc - 1)]
        if self.ideal:
            if self.topology == "full":
                self.edges_act = [(i, j) for i in range(self.q_enc)
                                  for j in range(i + 1, self.q_enc)]
            self.phys_nodes = list(range(self.q_enc))
            return

        self._load_real_backend()
        if self.topology == "native":
            coupling_edges, edge_cost = get_coupling_edges_and_costs(self.real_backend)
            self.phys_nodes = (
                load_json(self.fixed_phys_nodes_file) if self.use_fixed_phys_nodes
                else saved_nodes if saved_nodes is not None
                else pick_connected_subset_greedy(self.q_enc, coupling_edges, edge_cost)
            )
            validate_phys_nodes(self.phys_nodes, self.q_enc, coupling_edges, self.ibm_qpu)
            self.edges_act, _ = edges_log_from_phys_nodes(
                coupling_edges, self.phys_nodes, edge_cost,
            )
            return
        if self.use_fixed_phys_nodes:
            coupling_edges, _edge_cost = get_coupling_edges_and_costs(self.real_backend)
            self.phys_nodes = load_json(self.fixed_phys_nodes_file)
            validate_phys_nodes(self.phys_nodes, self.q_enc, coupling_edges, self.ibm_qpu)
        else:
            # In notebook mode, the transpiler chooses the layout. The selected
            # physical nodes are extracted and saved after transpilation.
            self.phys_nodes = saved_nodes

    def _read_fixed_circuit(self):
        if self._fixed_circuit is None:
            path = Path(self.fixed_circuit_file_name)
            if path.suffix != ".qpy":
                path = path.with_suffix(".qpy")
            with path.open("rb") as stream:
                circuits = qpy.load(stream)
            if len(circuits) != 1:
                raise ValueError("The fixed QPY file must contain exactly one circuit.")
            self._fixed_circuit = circuits[0]
        return self._fixed_circuit

    def _fixed_initial_nodes(self, circuit):
        if circuit.layout is None:
            if circuit.num_qubits != self.q_enc:
                raise ValueError("A fixed logical circuit must have q_enc qubits. "
                                 "Physical circuits require a saved transpiler layout.")
            return None
        try:
            # QPY may reorder Layout's dictionary entries. Use input indices,
            # not dictionary iteration order (older Qiskit initial_index_layout).
            virtual = circuit.layout.initial_virtual_layout(filter_ancillas=True).get_virtual_bits()
            nodes = [virtual[q] for q in sorted(virtual, key=circuit.layout.input_qubit_mapping.get)]
            final_nodes = circuit.layout.final_index_layout(filter_ancillas=True)
        except Exception as exc:
            raise ValueError("Cannot recover the fixed circuit's logical-to-physical layout.") from exc
        for mapping in (nodes, final_nodes):
            if (len(mapping) != self.q_enc or len(set(mapping)) != self.q_enc
                    or any(q < 0 or q >= circuit.num_qubits for q in mapping)):
                raise ValueError("Fixed circuit layout is incompatible with q_enc or circuit width.")
        return list(nodes)

    def _validate_fixed_circuit(self, circuit, reference):
        validate_circuit_parameter_compatibility(circuit, list(reference.parameters))
        stored = (circuit.metadata or {}).get("pqfmlib_heisenberg")
        expected = reference.metadata["pqfmlib_heisenberg"]
        if stored is None:
            warnings.warn(
                "Fixed QPY has no Heisenberg encoding metadata. Using the supplied configuration; "
                "axis assignments, edge schedule and interaction scales cannot be verified.",
                UserWarning, stacklevel=2,
            )
        else:
            if not isinstance(stored, dict) or stored.get("schema_version") != 1:
                raise ValueError("Unsupported Heisenberg circuit metadata schema.")
            mismatches = [key for key, value in expected.items()
                          if (key != "use_tanh_scaling" or key in stored)
                          and stored.get(key, "direct" if key == "coupling_mode" else None) != value]
            if mismatches:
                raise ValueError("Fixed circuit encoding configuration differs: " + ", ".join(mismatches))
        nodes = self._fixed_initial_nodes(circuit)
        if nodes is not None:
            if (not self.ideal and self.use_fixed_phys_nodes and self.phys_nodes != nodes):
                raise ValueError("Fixed circuit layout conflicts with fixed_phys_nodes_file.")
            self.phys_nodes = nodes
        if isinstance(self.backend, AerSimulator) and circuit.num_qubits > self.backend.num_qubits:
            # A saved ISA circuit includes physical ancillas beyond q_enc.
            self.backend.set_max_qubits(circuit.num_qubits)
        if circuit.num_clbits or any(inst.operation.name in ("measure", "reset") for inst in circuit.data):
            raise ValueError("Fixed Heisenberg circuits must be unitary and have no classical bits.")
        if nodes is not None and (not self.simulation or self.fakebackend):
            target_backend = self.real_backend if self.real_backend is not None else self.backend
            if circuit.num_qubits > target_backend.num_qubits:
                raise ValueError("Fixed circuit is wider than the target backend.")
            for inst in circuit.data:
                if inst.operation.name == "barrier":
                    continue
                qargs = tuple(circuit.find_bit(q).index for q in inst.qubits)
                if not target_backend.target.instruction_supported(
                    operation_name=inst.operation.name, qargs=qargs,
                    parameters=list(inst.operation.params),
                ):
                    raise ValueError(
                        f"Fixed transpiled circuit instruction {inst.operation.name}{qargs} "
                        "is incompatible with the target backend. Load the logical QPY to transpile again."
                    )

    def _transpile_or_load_circuit(self, qc_param: QuantumCircuit, *, persist_artifacts: bool = False):
        """Compile logical QPYs or preserve compatible physical QPYs and their layout."""
        source = qc_param
        if self.fixed_circuit_file_name:
            source = self._read_fixed_circuit()
            self._validate_fixed_circuit(source, qc_param)

        if self.fixed_circuit_file_name and source.layout is not None:
            qc_t = source
        elif not self.simulation or self.fakebackend:
            initial_layout = None
            if (self.use_fixed_phys_nodes or self.topology == "native") and self.phys_nodes is not None:
                initial_layout = Layout({source.qubits[i]: self.phys_nodes[i] for i in range(self.q_enc)})
            qc_t = transpile(
                source, backend=self.backend, initial_layout=initial_layout,
                optimization_level=3, seed_transpiler=self.seed,
            )
            extracted = _extract_initial_phys_nodes(qc_t, source)
            if extracted is not None:
                self.phys_nodes = extracted
        elif self.fixed_circuit_file_name or self.framework == "cudaq":
            qc_t = source.copy()
        else:
            pm = generate_preset_pass_manager(
                optimization_level=3, backend=self.backend, seed_transpiler=self.seed,
            )
            qc_t = pm.run(source)

        if persist_artifacts:
            folder = Path(self.base_folder)
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / "heisenberg_circuit.qpy").open("wb") as stream:
                qpy.dump(qc_t, stream)
            if source.layout is None:
                with (folder / "heisenberg_circuit_logical.qpy").open("wb") as stream:
                    qpy.dump(source, stream)
            if self.phys_nodes is not None:
                save_json(self.phys_nodes, folder / "phys_nodes.json")
            if self.save_circuit_drawings:
                save_circuit_draw(qc_t, self.base_folder, "heisenberg_circuit_transpiled")
                save_circuit_draw(source, self.base_folder, "heisenberg_circuit_logical")
        return qc_t

    def execute_quantum_feature_map(self):
        theta_values_all, self.theta_info = self._make_theta_matrix(self.X_q_all)

        qc_param, param_order, obs_list, obs_metadata = self._build_circuit_and_observables()

        qc_t = self._transpile_or_load_circuit(qc_param, persist_artifacts=self.base_folder is not None)
        validate_circuit_parameter_compatibility(qc_t, param_order)

        if not self.simulation:
            test_circuit_t(
                qc_t,
                self.backend,
                self.resource_estimation,
                csv_path=str(Path(self.base_folder) / "resource_estimation.csv"),
                shots=self.shots,
                num_param_sets=int(theta_values_all.shape[0]),
            )
            if self.resource_estimation:
                return None

        if qc_t.layout is not None:
            obs_isa = [obs.apply_layout(qc_t.layout) for obs in obs_list]
        else:
            obs_isa = obs_list
        obs_isa_broadcast = [[obs] for obs in obs_isa]

        metadata_extra = {
            **self._coupling_metadata(),
            "q_enc": int(self.q_enc),
            "topology": self.topology,
            "interaction_axes": list(self.interaction_axes),
            "encoding_mode": self.encoding_mode,
            "edges_act": [list(edge) for edge in self.edges_act],
            "measure_2local_diagonal": bool(self.measure_2local_diagonal),
            "measure_cross_observables": bool(self.measure_cross_observables),
            "measure_all_pairs": bool(self.measure_all_pairs),
            "num_features": int(self.theta_info["num_features"]),
            "parameter_names": [p.name for p in param_order],
            "circuit_parameter_names": [p.name for p in qc_t.parameters],
            "fixed_circuit_loaded": bool(self.fixed_circuit_file_name),
            "circuit_encoding": (qc_t.metadata or {}).get("pqfmlib_heisenberg"),
            "R": int(self.R),
            "alpha": float(self.alpha),
            "phi_scale": float(self.circuit_info["phi_scale"]),
            "features_per_block": int(self.theta_info["features_per_block"]),
            "num_blocks": int(self.theta_info["num_blocks"]),
            "total_slots": int(self.theta_info["total_slots"]),
            "use_tanh_scaling": self.theta_info["use_tanh_scaling"],
        }
        if self.phys_nodes is not None:
            metadata_extra["phys_nodes"] = [int(x) for x in self.phys_nodes]

        if qc_t.layout is not None:
            metadata_extra["final_phys_nodes"] = qc_t.layout.final_index_layout(filter_ancillas=True)
        if self.base_folder is not None:
            self.metadata_output_path = str(Path(self.base_folder) / "heisenberg_metadata.json")
            save_json({**metadata_extra, "obs_metadata": [list(item) for item in obs_metadata]},
                      self.metadata_output_path)

        if self.simulation:
            prepared = PreparedProjectedFeatureJob(
                qc_t, tuple(param_order), obs_isa_broadcast, obs_metadata,
            )
            self.Xq_all_raw, self.obs_metadata = execute_prepared_projected_feature_job(
                prepared, theta_values_all, self.backend, self.estimator,
                framework=self.framework,
                simulation=True,
                expectation_method=self.expectation_method,
                shots=self.shots,
                base_folder=self.base_folder,
            )
            return None

        theta_reordered = reorder_theta_by_parameter_names(qc_t, param_order, theta_values_all)
        self.job_id = submit_job_and_save_metadata(
            self.estimator,
            qc_t,
            obs_isa_broadcast,
            theta_reordered,
            obs_metadata,
            base_folder=self.base_folder,
            **metadata_extra,
        )
        return self.job_id

    def save_quantum_features(self) -> None:
        q_col_names = [metadata_to_column_name(m) for m in self.obs_metadata]
        df_q_all = pd.DataFrame(self.Xq_all_raw, columns=q_col_names)
        df_q_all["y"] = self.y
        self.csv_output_path = str(Path(self.base_folder) / f"qfeatures_all_{self.name_file}_heisenberg.csv")
        self.npy_output_path = str(Path(self.base_folder) / f"qfeatures_all_{self.name_file}_heisenberg.npy")
        df_q_all.to_csv(self.csv_output_path, index=False)
        np.save(self.npy_output_path, self.Xq_all_raw)

    def run(self):
        self._validate_topology()
        self._validate_encoding()
        self.load_data()
        if self.coupling_mode == "superexchange":
            values, _ = self._coerce_feature_matrix(self.X_q_all)
            if values.shape[1] != self.q_enc:
                raise ValueError("Superexchange requires exactly q_enc features (one per qubit).")
        self.setup_backend_and_estimator()
        if self.topology == "chain":
            self.prepare_output_folder()
            self.prepare_layout()
        else:
            self.prepare_layout()
            self.prepare_output_folder()
        result = self.execute_quantum_feature_map()
        if not self.simulation:
            return result
        self.save_quantum_features()
        print("Heisenberg quantum features generated successfully.")
        print("Features saved to:", self.csv_output_path)
        return {
            "Xq_all_raw": self.Xq_all_raw,
            "obs_metadata": self.obs_metadata,
            "csv_path": self.csv_output_path,
            "npy_path": self.npy_output_path,
            "base_folder": self.base_folder,
            "metadata_path": self.metadata_output_path,
            "theta_info": self.theta_info,
            "circuit_info": self.circuit_info,
        }


@dataclass
class SuperexchangeHeisenbergPQFM(HeisenbergProjectiveQFM):
    """Heisenberg map with one position per qubit and spatial superexchange.

    Uses ``J0 * exp(-2 * abs(x_i - x_j) / ell)`` in the inherited edge
    schedule. Inputs must have exactly ``q_enc`` finite columns. Normalization
    belongs to the caller; ``use_tanh_scaling`` is ignored in superexchange
    mode. The inherited R/alpha convention is unchanged: each RPP angle is
    ``alpha * Jij / 2`` (one block), i.e. delta = alpha/4 per repetition.
    """

    coupling_mode: str = field(default="superexchange", kw_only=True)
    use_tanh_scaling: bool = False
