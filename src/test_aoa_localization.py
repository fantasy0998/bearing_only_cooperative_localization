from exp_random_data import gen_simulation_data
from exp_random_data import compute_simulation_data

folder = "D:/3-1cooperativeLocalization/code/bearing_only_hu/taes-bearing-only-cooperative-localization/data/"
batch_size = [6, 12]
noise_std = [0.1, 1]
mode_set = 0
if mode_set == 1:
    for batch in batch_size:
        for noise in noise_std:
            gen_simulation_data(folder, 50, noise, batch)
else:
    compute_simulation_data(folder)