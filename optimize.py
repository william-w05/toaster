import time
import numpy as np

from scripts import fem_solve as cv
from scripts import fem_vis as viz
from scripts import mcmc
from scripts import noisy_mcmc

#PARAMS_MM = np.array([43.15946064, 128.69404921, 4.6586204, 10.00004685, 6.95143154,
#                      6.73395772, 124.86872764, 123.77079161])

TUNING_STEPS = 16
PATH = "results/08_13_2026_mcmc_results"

#initial_params = np.array([43.15946064, 128.69404921, 4.6586204, 10.00004685, 6.95143154,
#                           6.73395772, 124.86872764, 123.77079161])
    
if __name__ == "__main__":

    params_mm = [9.71851258, 127.23060036, 11.02606533, 14.4991172,
               15.65917039, 124.90989748, 123.52196735]
    
    specs,results = cv.run_sweep(
        lambda dx, dy, i: mcmc.make_spec(params_mm, toast_dx=dx, toast_dy=dy, tag=f"x={dx*1e3:.2f}mm"),
        mcmc.tuning_positions(params_mm, n=4),        # yields (dx, dy, f_guess) per step
        n_modes=6,
        n_workers=None,            # every core
        timeout=600,               # bad geometry fails instead of stalling
        plot_all=True,          # needed for plot_best_modes
    )
    entries = [(s, r, f"x={x*1e3:.2f} mm")
                for (x, _y, _f), s, r in zip(mcmc.tuning_positions(params_mm), specs, results)
                if r["ok"]]
    viz.plot_best_modes_magnitude_square(entries, save=f'results/best_design.png', ncol=2, suptitle='Best Result')

    #rounded_params = np.array([0, 128.69, 4.6586, 10, 6.9514, 6.734, 124.87, 123.77])
    
    #n_walkers = 10
    #seeds = generate_seeds(n_walkers)
    #print(np.asarray(seeds))

    #best_params, best_value, chains_params, chains_values = mcmc.mcmc_minimize(initial_params = seeds,
    #    steps=2500, save_path=PATH, n_walkers=n_walkers, tuning_steps=TUNING_STEPS, proposal_std=0.1, use_surrogate=True 
    #)
    #params_mm = [ 10.34560212,126.9786174 , 10.65199221, 14.03542231, 15.58424328,124.88448306,123.3904198 ]
    #params_mm = [45, 143, 8, 9, 9, 141, 141]
    #print(mcmc.fom(np.array([43.15946064, 128.69404921,   4.6586204,   6.95143154, 6.73395772 ,124.86872764, 123.77079161])))
    #print(mcmc.fom(np.array([  6.06726873,126.21761759,  9.86724139, 16.83646899, 18.0793698 ,124.89429278,125.45751509])))
    #params_m = mcmc._params_to_m(params_mm)
    #spec = viz.toaster_spec(params_m)
    #viz.plot_spec(spec, save=f'results/08_13_2026_mcmc_results/plots/temp_opt.png')
    #result = cv.solve_cavity(spec, keep_fields=True, f_target=15.084e9)
    #viz.plot_modes_square_magnitude(spec, result, save='results/initial_design.png')
    #specs,results = cv.run_sweep(
    #    lambda dx, dy, i: mcmc.make_spec(params_mm, toast_dx=dx, toast_dy=dy, tag=f"x={dx*1e3:.2f}mm"),
    #    mcmc.tuning_positions(params_mm, n=4),        # yields (dx, dy, f_guess) per step
    #    n_modes=6,
    #    n_workers=None,            # every core
    #    timeout=600,               # bad geometry fails instead of stalling
    #    plot_all=True,          # needed for plot_best_modes
    #)
    #entries = [(s, r, f"x={x*1e3:.2f} mm")
    #            for (x, _y, _f), s, r in zip(mcmc.tuning_positions(params_mm), specs, results)
    #            if r["ok"]]
    #viz.plot_best_modes_magnitude_square(entries, save=f'results/initial_design_range.png', ncol=2)

    #x0 = np.array([9.71851258, 127.23060036, 11.02606533, 14.4991172,
    #           15.65917039, 124.90989748, 123.52196735])
    #noisy_mcmc.noisy_plots(x0, "results/08_17_2026_NM_results/nm_results_3/stability.csv",
    #    save_prefix="results/08_17_2026_NM_results/nm_results_3/sweep")

    #print(best_params, best_value)