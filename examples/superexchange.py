"""Example: superexchange Heisenberg PQFM with one feature per qubit."""

from pqfmlib import SuperexchangeHeisenbergPQFM


if __name__ == "__main__":
    qfm = SuperexchangeHeisenbergPQFM(
        name_file="Toxicity_original_MI_top20_standardized_0_1",
        data_dir="./data",
        output_root="./results",
        seed=42,
        ideal=True,  # Use False with topology="native".
        simulation=True,
        fakebackend=False,
        framework="cudaq",  # Ideal simulation on an NVIDIA GPU.
        precision="double",
        ibm_qpu="ibm_kingston",
        q_enc=20,  # Must equal the number of input features, excluding y.
        topology="full",  # "chain", "full" (ideal=True), or "native" (ideal=False).
        J0=1.0,  # Coupling: J_ij = J0 * exp(-2 * abs(x_i - x_j) / ell).
        ell=1.0,
        R=2,
        alpha=0.1,
        expectation_method="exact",  # Use "exact" with CUDA-Q.
        shots=4096,
        measure_all_pairs=True,  # Local Z/X/Y and ZZ/XX/YY for every pair.
    )
    print(qfm.run())
