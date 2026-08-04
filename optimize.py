import time
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

def generate_seeds(n):
    seeds = []
    for i in range(n):
        found = False
        while not found:
            # empirical bounds
            theta = np.random.uniform(0,70) # 40, 50
            ch = np.random.uniform(80, 160) # 115, 160
            dh = np.random.uniform(0.9 * ch, 1.1 * ch) # 10% from ch
            dw = np.random.uniform(3, 20) # 3, 7
            g1 = np.random.uniform(9, 11) #9.95, 10.05
            cw = np.random.uniform(3, 20) # 3, 7
            sw = np.random.uniform(max(3,0.8*cw), 1.2 * cw) # 10% from cw 
            sh = np.random.uniform(0.8 * ch, 1.2 * ch) # 10% from ch
            
            proposal = np.array([theta, dh, dw, g1, cw, sw, ch, sh])
            found = mcmc.proposed_params_within_limits(proposal)
                
        seeds.append(proposal)
    return seeds

if __name__ == "__main__":
    n_walkers = 8
    seeds = generate_seeds(n_walkers)
    seeds.append([0, 125, 5, 10, 6, 6, 125, 125])
    seeds.append([0, 150, 5, 10, 6, 6, 150, 150])
    best_params, best_value, chains_params, chains_values= mcmc.mcmc_minimize(
        seeds, save_path=PATH, steps=1200, n_walkers=n_walkers+2, tuning_steps=TUNING_STEPS, proposal_std=0.1
    )

    print(best_params, best_value)