"""Generate Heisenberg quantum features with CUDA-Q on an NVIDIA GPU.

From the repository root, install with ``pip install -e ".[cudaq]"`` and run
``python examples/heisenberg_cudaq.py --precision single`` for FP32, or omit
the option for FP64. No external dataset is required.
"""

import argparse

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pqfmlib import HeisenbergProjectiveQFM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--precision", choices=("single", "double"), default="double",
        help="Simulator precision: single (FP32) or double (FP64, default).",
    )
    args = parser.parse_args()
    X, y = make_classification(
        n_samples=24, n_features=4, n_informative=3, n_redundant=0,
        random_state=42,
    )
    X_train, X_test = train_test_split(
        X, test_size=0.25, stratify=y, random_state=42,
    )
    # Learn preprocessing only from training data, then reuse it on test data.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    qfm = HeisenbergProjectiveQFM(
        name_file="heisenberg_cudaq_demo",
        framework="cudaq",
        precision=args.precision,
        simulation=True,
        ideal=True,
        fakebackend=False,
        q_enc=3,
        R=2,
        alpha=0.1,
        mps=False,
        expectation_method="exact",
        seed=42,
        save_circuit_drawings=False,
    )
    # Fit the map and generate training features; reuse the map for test data.
    Xq_train = qfm.fit_transform(X_train)
    Xq_test = qfm.transform(X_test)

    print(f"Simulator precision: {qfm.precision}")
    print(f"Training: {X_train.shape} -> {Xq_train.shape}")
    print(f"Test:     {X_test.shape} -> {Xq_test.shape}")
    print("First training quantum feature row:")
    print(np.array2string(Xq_train[0], precision=6))


if __name__ == "__main__":
    main()
