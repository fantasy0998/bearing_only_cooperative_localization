from src.exp_random_data import compute_simulation_data, gen_simulation_data


def main() -> None:
    """Run the legacy AOA simulation experiment when invoked directly."""
    folder = (
        "D:/3-1cooperativeLocalization/code/bearing_only_hu/"
        "taes-bearing-only-cooperative-localization/data/"
    )
    batch_sizes = [6, 12]
    noise_standard_deviations = [0.1, 1]
    generate_data = False

    if generate_data:
        for batch_size in batch_sizes:
            for noise_standard_deviation in noise_standard_deviations:
                gen_simulation_data(
                    folder,
                    50,
                    noise_standard_deviation,
                    batch_size,
                )
    else:
        compute_simulation_data(folder)


if __name__ == "__main__":
    main()
