import time
from unicodedata import name
import numpy as np

import fem_solve as cv
import fem_vis as viz
import mcmc

#PARAMS_MM = np.array([43.15946064, 128.69404921, 4.6586204, 10.00004685, 6.95143154,
#                      6.73395772, 124.86872764, 123.77079161])

TUNING_STEPS = 16
PATH = "results/08_03_2026_mcmc_results"

initial_params = np.array([43.15946064, 128.69404921, 4.6586204, 10.00004685, 6.95143154,
                           6.73395772, 124.86872764, 123.77079161])

if __name__ == "__main__":
    best_params, best_value, chains_params, chains_values= mcmc.mcmc_minimize(
        initial_params, save_path=PATH, steps=10, n_walkers=8, tuning_steps=TUNING_STEPS, proposal_std=0.1
    )

    print(best_params, best_value)