"""Example: Heisenberg PQFM with CUDA-Q on an NVIDIA GPU."""

from pqfmlib import HeisenbergProjectiveQFM


if __name__ == "__main__":
    qfm = HeisenbergProjectiveQFM(
        name_file="Toxicity_preprocessed_shuffled",
        seed=42,
        ideal=True,
        simulation=True,
        fakebackend=False,
        framework="cudaq",
        precision="double",  # Use "single" for FP32.
        expectation_method="exact",
        q_enc=8,
        R=2,
        alpha=0.1,
        mps=False,
        use_tanh_scaling=True,
    )
    print(qfm.run())
