"""Example: Heisenberg PQFM with selectable topology and interaction axes."""

from pqfmlib import HeisenbergProjectiveQFM


if __name__ == "__main__":
    qfm = HeisenbergProjectiveQFM(
        name_file="Toxicity_preprocessed_shuffled",
        seed=42,
        ideal=True,  # Use False with topology="native".
        simulation=True,
        fakebackend=False,
        ibm_qpu="ibm_kingston",
        q_enc=8,
        topology="full",  # "chain", "full" (ideal=True), or "native" (ideal=False).
        interaction_axes=("x", "y"),
        encoding_mode="multi_axis",  # Or "shared_feature" for J(XX + YY).
        R=2,
        alpha=0.1,
        use_tanh_scaling=True,
        expectation_method="statevector",
        measure_2local_diagonal=True,
        measure_cross_observables=True,
    )
    print(qfm.run())
