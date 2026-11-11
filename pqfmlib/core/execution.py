"""Shared circuit preparation and framework dispatch for projected feature maps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from qiskit import qpy, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit.transpiler import Layout
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from pqfmlib.core.framework import validate_framework
from pqfmlib.core.resources import test_circuit_t
from pqfmlib.core.serialization import metadata_to_jsonable
from pqfmlib.utils.io import save_circuit_draw


@dataclass(frozen=True)
class PreparedProjectedFeatureJob:
    """Circuit and observables prepared independently from a transform batch."""

    qc_t: object
    param_order: tuple
    obs_isa_broadcast: object
    obs_metadata: object
    fixed_circuit_loaded: bool = False


def validate_circuit_parameter_compatibility(qc_t, param_order) -> None:
    """Require a prepared circuit to expose exactly the expected parameters."""
    expected_names = [parameter.name for parameter in param_order]
    circuit_names = [parameter.name for parameter in qc_t.parameters]
    if len(expected_names) != len(set(expected_names)):
        raise ValueError("Parameter names in param_order are not unique.")
    if len(circuit_names) != len(set(circuit_names)):
        raise ValueError("Parameter names in the prepared circuit are not unique.")
    missing = sorted(set(expected_names) - set(circuit_names))
    unexpected = sorted(set(circuit_names) - set(expected_names))
    if missing or unexpected:
        raise ValueError(
            "Prepared circuit parameters are incompatible with the fitted map. "
            f"Missing from circuit: {missing[:10]}; unexpected in circuit: {unexpected[:10]}."
        )


def reorder_theta_by_parameter_names(qc_t, param_order, theta_values_all):
    """Reorder a theta matrix to match the transpiled circuit parameter order."""
    param_t = list(qc_t.parameters)
    param_order_names = [p.name for p in param_order]
    param_t_names = [p.name for p in param_t]
    if len(param_order_names) != len(set(param_order_names)):
        raise ValueError("Parameter names in param_order are not unique.")
    pos = {name: k for k, name in enumerate(param_order_names)}
    try:
        idx = [pos[name] for name in param_t_names]
    except KeyError as exc:
        missing = [name for name in param_t_names if name not in pos]
        raise KeyError(f"Some transpiled parameters are missing from param_order: {missing[:10]}") from exc
    return theta_values_all[:, idx]


def submit_job_and_save_metadata(
    estimator,
    qc_t,
    obs_isa_broadcast,
    theta_values_all,
    obs_metadata,
    base_folder="example",
    **extra_meta,
):
    """Submit one Estimator job and write job metadata to disk."""
    job = estimator.run([(qc_t, obs_isa_broadcast, theta_values_all)])
    job_id = job.job_id()
    out = {
        "job_id": job_id,
        "base_folder": base_folder,
        "n_param_sets": int(theta_values_all.shape[0]),
        "n_obs": int(len(obs_isa_broadcast)),
        "obs_metadata": metadata_to_jsonable(obs_metadata),
    }
    out.update(extra_meta)
    Path(base_folder).mkdir(parents=True, exist_ok=True)
    Path(base_folder, "job_meta.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Job submitted. job_id={job_id}")
    print(f"Metadata saved to {base_folder}/job_meta.json")
    return job_id


def prepare_projected_feature_job(
    qc_param,
    param_order,
    phys_nodes,
    backend,
    obs_list,
    obs_metadata,
    *,
    simulation: bool = False,
    fakebackend: bool = False,
    base_folder: str | None = "example",
    fixed_circuit: str = "",
    seed_transpiler: int = 42,
    qpy_filename: str = "qfm_circuit.qpy",
    save_circuit_drawings: bool = False,
    use_fixed_circuit_in_simulation: bool = False,
    validate_parameter_compatibility: bool = False,
):
    """Prepare the circuit, parameter order, and laid-out observables once."""
    n = int(qc_param.num_qubits)
    if phys_nodes is None:
        phys_nodes = list(range(n))
    if len(phys_nodes) != n:
        raise ValueError(f"phys_nodes must have length {n}, but got {len(phys_nodes)}")

    if base_folder is not None:
        Path(base_folder).mkdir(parents=True, exist_ok=True)
    elif not simulation or save_circuit_drawings:
        raise ValueError("base_folder is required when circuit artifacts may be written.")

    fixed_circuit_loaded = bool(fixed_circuit) and (not simulation or use_fixed_circuit_in_simulation)
    if fixed_circuit_loaded:
        fixed_path = Path(fixed_circuit)
        if fixed_path.suffix != ".qpy":
            fixed_path = fixed_path.with_suffix(".qpy")
        with open(fixed_path, "rb") as f:
            qc_t = qpy.load(f)[0]
    elif not simulation:
        initial_layout = Layout({qc_param.qubits[i]: phys_nodes[i] for i in range(n)})
        qc_t = transpile(
            qc_param,
            backend=backend,
            initial_layout=initial_layout,
            optimization_level=3,
            seed_transpiler=seed_transpiler,
            routing_method="none",
        )
        with open(Path(base_folder) / qpy_filename, "wb") as f:
            qpy.dump(qc_t, f)
    elif fakebackend:
        initial_layout = Layout({qc_param.qubits[i]: phys_nodes[i] for i in range(n)})
        qc_t = transpile(
            qc_param,
            backend=backend,
            initial_layout=initial_layout,
            optimization_level=3,
            seed_transpiler=seed_transpiler,
        )
    else:
        # Preserve the legacy pure-simulation behavior: fixed_circuit is ignored
        # and the logical circuit is used without backend transpilation.
        qc_t = qc_param.copy()

    if validate_parameter_compatibility:
        validate_circuit_parameter_compatibility(qc_t, param_order)

    if save_circuit_drawings:
        save_circuit_draw(qc_t, base_folder, "circuit_transpiled")
        save_circuit_draw(qc_param, base_folder, "circuit_logical")

    if fixed_circuit_loaded and simulation:
        layout = getattr(qc_t, "layout", None)
        obs_isa = [obs.apply_layout(layout) for obs in obs_list] if layout is not None else list(obs_list)
    elif not simulation or fakebackend:
        obs_isa = [obs.apply_layout(qc_t.layout) for obs in obs_list]
    else:
        obs_isa = list(obs_list)

    return PreparedProjectedFeatureJob(
        qc_t=qc_t,
        param_order=tuple(param_order),
        obs_isa_broadcast=[[obs] for obs in obs_isa],
        obs_metadata=obs_metadata,
        fixed_circuit_loaded=fixed_circuit_loaded,
    )


def execute_statevector_features(prepared, theta_values_all, backend):
    """Simulate each sample once, then reuse its state for every observable.

    Process one state at a time so full state vectors do not accumulate with
    dataset size. Aer uses the configured CPU/GPU; expectations run on CPU.
    """
    if not isinstance(backend, AerSimulator):
        raise ValueError("statevector execution requires AerSimulator.")
    if backend.options.method != "statevector" or backend.options.noise_model is not None:
        raise ValueError("statevector execution requires a noiseless statevector backend.")
    circuit = prepared.qc_t.copy()
    if circuit.num_clbits or any(
        instruction.operation.name in {"measure", "reset", "initialize", "kraus", "superop"}
        for instruction in circuit.data
    ):
        raise ValueError("statevector execution requires a unitary circuit without classical bits.")
    # Compile once per transform, not once per observable or sample.
    circuit = transpile(circuit, backend, optimization_level=0)
    if any(instruction.operation.name in {"measure", "reset", "initialize", "kraus", "superop"}
           for instruction in circuit.data):
        raise ValueError("statevector execution requires unitary operations.")
    theta = reorder_theta_by_parameter_names(circuit, prepared.param_order, theta_values_all)
    circuit.save_statevector(label="pqfm_state")
    observables = [entry[0] for entry in prepared.obs_isa_broadcast]
    features = np.empty((len(theta), len(observables)), dtype=float)
    for row, values in enumerate(theta):
        bound = circuit.assign_parameters(dict(zip(circuit.parameters, values)))
        # One deterministic evolution; this is not sampling observables.
        result = backend.run(bound, shots=1).result()
        if not result.success:
            raise RuntimeError(f"Statevector simulation failed: {result.status}")
        state = Statevector(result.data(0)["pqfm_state"])
        for column, observable in enumerate(observables):
            features[row, column] = float(np.real(state.expectation_value(observable)))
    return features, prepared.obs_metadata


def execute_prepared_projected_feature_job(
    prepared,
    theta_values_all,
    backend,
    estimator,
    *,
    framework: str = "qiskit",
    simulation: bool = False,
    resource_estimation: bool = False,
    expectation_method: str = "shots",
    shots: int = 1024,
    base_folder: str | None = "example",
    metadata_extra: dict | None = None,
):
    """Execute one parameter batch using an already prepared circuit."""
    validate_framework(framework)
    theta_values_all = np.asarray(theta_values_all)
    if theta_values_all.ndim != 2:
        raise ValueError("theta_values_all must be a 2D matrix.")
    if framework == "tilelang":
        if not simulation or resource_estimation or expectation_method != "exact":
            raise ValueError("TileLang requires exact simulation without resource estimation.")
        from pqfmlib.experimental.heisenberg_tilelang import TileLangExecutor

        if not isinstance(backend, TileLangExecutor):
            raise ValueError("framework='tilelang' requires a TileLangExecutor backend.")
        return backend.execute(prepared, theta_values_all)
    if framework == "cudaq":
        if not simulation or resource_estimation:
            raise ValueError("CUDA-Q execution requires simulation=True and resource_estimation=False.")
        from pqfmlib.core.cudaq_execution import CudaqExecutor

        if not isinstance(backend, CudaqExecutor):
            raise ValueError("framework='cudaq' requires a CudaqExecutor backend.")
        return backend.execute(prepared, theta_values_all, expectation_method=expectation_method, shots=shots)
    if expectation_method not in ("shots", "statevector"):
        raise ValueError("Unknown expectation_method.")
    if expectation_method == "statevector":
        if not simulation:
            raise ValueError("statevector requires simulation=True.")
        return execute_statevector_features(prepared, theta_values_all, backend)
    N = int(theta_values_all.shape[0])

    if not simulation:
        if base_folder is None:
            raise ValueError("base_folder is required for non-simulation execution.")
        test_circuit_t(
            prepared.qc_t,
            backend,
            resource_estimation,
            csv_path=str(Path(base_folder) / "resource_estimation.csv"),
            shots=shots,
            num_param_sets=N,
        )
        if resource_estimation:
            return None

    theta_reordered = reorder_theta_by_parameter_names(
        prepared.qc_t,
        prepared.param_order,
        theta_values_all,
    )

    if simulation:
        job = estimator.run([(prepared.qc_t, prepared.obs_isa_broadcast, theta_reordered)])
        result = job.result()[0]
        evs = np.asarray(result.data.evs, dtype=float)
        return evs.T, prepared.obs_metadata

    return submit_job_and_save_metadata(
        estimator,
        prepared.qc_t,
        prepared.obs_isa_broadcast,
        theta_reordered,
        prepared.obs_metadata,
        base_folder=base_folder,
        **(metadata_extra or {}),
    )


def run_projected_feature_job(
    qc_param,
    param_order,
    theta_values_all,
    phys_nodes,
    backend,
    estimator,
    obs_list,
    obs_metadata,
    *,
    framework: str = "qiskit",
    simulation: bool = False,
    fakebackend: bool = False,
    resource_estimation: bool = False,
    expectation_method: str = "shots",
    shots: int = 1024,
    base_folder: str = "example",
    fixed_circuit: str = "",
    metadata_extra: dict | None = None,
    seed_transpiler: int = 42,
    qpy_filename: str = "qfm_circuit.qpy",
    save_circuit_drawings: bool = False,
):
    """Run or submit a projected feature-map Estimator job."""
    validate_framework(framework)
    if framework == "cudaq" and (not simulation or fakebackend or resource_estimation):
        raise ValueError("CUDA-Q requires simulation=True, fakebackend=False and resource_estimation=False.")
    if np.asarray(theta_values_all).ndim != 2:
        raise ValueError("theta_values_all must be a 2D matrix.")
    if expectation_method == "statevector" and (not simulation or fakebackend):
        raise ValueError("statevector requires simulation=True and fakebackend=False.")
    prepared = prepare_projected_feature_job(
        qc_param,
        param_order,
        phys_nodes,
        backend,
        obs_list,
        obs_metadata,
        simulation=simulation,
        fakebackend=fakebackend,
        base_folder=base_folder,
        fixed_circuit=fixed_circuit,
        seed_transpiler=seed_transpiler,
        qpy_filename=qpy_filename,
        save_circuit_drawings=save_circuit_drawings,
        use_fixed_circuit_in_simulation=bool(fixed_circuit),
        validate_parameter_compatibility=bool(fixed_circuit),
    )
    return execute_prepared_projected_feature_job(
        prepared,
        theta_values_all,
        backend,
        estimator,
        framework=framework,
        simulation=simulation,
        resource_estimation=resource_estimation,
        expectation_method=expectation_method,
        shots=shots,
        base_folder=base_folder,
        metadata_extra=metadata_extra,
    )
