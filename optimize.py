import time
import numpy as np

from scripts import fem_solve as cv
from scripts import fem_vis as viz
from scripts import mcmc

#PARAMS_MM = np.array([43.15946064, 128.69404921, 4.6586204, 10.00004685, 6.95143154,
#                      6.73395772, 124.86872764, 123.77079161])

TUNING_STEPS = 16
PATH = "results/08_10_2026_mcmc_results"

#initial_params = np.array([43.15946064, 128.69404921, 4.6586204, 10.00004685, 6.95143154,
#                           6.73395772, 124.86872764, 123.77079161])

def generate_seeds(n):
    seeds = []
    for i in range(n):
        found = False
        while not found:
            # empirical bounds
            theta = np.random.uniform(0,20) # 40, 50
            ch = np.random.uniform(90, 145) # 115, 160
            dh = np.random.uniform(max(0.8 * ch, 90), min(145, 1.2 * ch)) # 10% from ch
            dw = np.random.uniform(3, 10) # 3, 7
            cw = np.random.uniform(3, 20) # 3, 7
            sw = np.random.uniform(max(3,0.8*cw), min(20, 1.2 * cw)) # 10% from cw 
            sh = np.random.uniform(max(90, 0.8 * ch), min(145, 1.2 * ch)) # 10% from ch
            
            proposal = np.array([theta, dh, dw, cw, sw, ch, sh])
            found = mcmc.proposed_params_within_limits(proposal)
                
        seeds.append(proposal)
    return seeds
    
if __name__ == "__main__":
    #rounded_params = np.array([0, 128.69, 4.6586, 10, 6.9514, 6.734, 124.87, 123.77])
    
    n_walkers = 10

    best_params, best_value, chains_params, chains_values = mcmc.continue_mcmc( 
        steps=1690, save_path=PATH, n_walkers=n_walkers, tuning_steps=TUNING_STEPS, proposal_std=0.1, use_surrogate=True 
    )
    #print(mcmc.fom(np.array([  0., 125.,   5.,  10.,   6.,   6., 125., 125.])))

    #print(best_params, best_value)