"""Approximation log.

The default SIA input preserves all 23 numeric and 27 fixed UF one-hot columns;
there is no PCA or feature truncation. Numeric medians and standardization are
fitted only on the selected training window (default 2016-02 through 2020-12).
Quantum floating-point and sampling limits are documented in the GPU module.
"""

from pathlib import Path

NUMERIC_COLUMNS = (
    "apacs_atual",
    "apacs_lag_1",
    "apacs_lag_2",
    "apacs_lag_3",
    "apacs_lag_6",
    "apacs_lag_12",
    "apacs_media_3m",
    "apacs_media_6m",
    "apacs_media_12m",
    "apacs_desvio_3m",
    "apacs_desvio_6m",
    "apacs_desvio_12m",
    "pacientes_atual_log",
    "valor_atual_log",
    "ticket_medio_atual",
    "pct_iniciais",
    "pct_continuidades",
    "pct_fora_municipio",
    "pct_fora_uf",
    "limiar_pressao",
    "mes_alvo_sin",
    "mes_alvo_cos",
    "tendencia_meses",
)

SIA_DATASET = (
    Path(__file__).resolve().parents[2]
    / "experiments/sia/base_modelagem_congelada.parquet"
)

UF_CODES = (11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27,
            28, 29, 31, 32, 33, 35, 41, 42, 43, 50, 51, 52, 53)
UF_NAMES = ("RO", "AC", "AM", "RR", "PA", "AP", "TO", "MA", "PI", "CE",
            "RN", "PB", "PE", "AL", "SE", "BA", "MG", "ES", "RJ", "SP",
            "PR", "SC", "RS", "MS", "MT", "GO", "DF")
FULL_FEATURE_COLUMNS = NUMERIC_COLUMNS + tuple(f"uf_{code}" for code in UF_CODES)


def load_sia_input(dataset, *, train_start="2016-02-01", train_end="2020-12-01"):
    """Preserve all SIA features; fit numeric transforms only on training rows."""
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    frame = (pd.read_parquet(dataset, columns=list(NUMERIC_COLUMNS) +
                             ["uf", "cnes", "mes_alvo"])
             .sort_values(["mes_alvo", "cnes"], kind="stable").reset_index(drop=True))
    months = pd.to_datetime(frame.mes_alvo)
    train_mask = months.between(pd.Timestamp(train_start), pd.Timestamp(train_end))
    uf_text = frame.uf.astype("string").str.strip().str.upper()
    codes = uf_text.map(dict(zip(UF_NAMES, UF_CODES))).fillna(pd.to_numeric(uf_text, errors="coerce"))
    numeric = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
    numeric.fit(frame.loc[train_mask, list(NUMERIC_COLUMNS)])
    encoder = OneHotEncoder(categories=[list(UF_CODES)], sparse_output=False, handle_unknown="error")
    encoder.fit(codes.loc[train_mask].to_frame())
    features = pd.concat([
        pd.DataFrame(numeric.transform(frame[list(NUMERIC_COLUMNS)]), columns=NUMERIC_COLUMNS),
        pd.DataFrame(encoder.transform(codes.to_frame()), columns=FULL_FEATURE_COLUMNS[len(NUMERIC_COLUMNS):]),
    ], axis=1)
    return features.to_numpy()


def chain_circuit():
    """Build the shared 18-qubit SIA circuit: 50 features and one zero angle."""
    from pqfmlib.maps.heisenberg import build_heisenberg_chain_feature_circuit

    logical, parameters, _ = build_heisenberg_chain_feature_circuit(
        18, num_blocks=1, R=2, alpha=0.1, seed=42, add_barriers=False,
        encoding_mode="multi_axis", interaction_axes=("x", "y", "z"))
    feature_count = len(FULL_FEATURE_COLUMNS)
    logical = logical.assign_parameters(dict.fromkeys(parameters[feature_count:], 0.0))
    parameters = list(parameters[:feature_count])
    return logical, parameters
