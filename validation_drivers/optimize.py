import time
import numpy as np

from scripts import fem_solve as cv
from scripts import fem_vis as viz
from scripts import mcmc
from scripts import noisy_mcmc

from scripts_3d import fem_solve_3d as cv3
from scripts_3d import fem_vis_3d as viz3

#PARAMS_MM = np.array([43.15946064, 128.69404921, 4.6586204, 10.00004685, 6.95143154,
#                      6.73395772, 124.86872764, 123.77079161])

TUNING_STEPS = 16
PATH = "results/08_13_2026_mcmc_results"

#initial_params = np.array([43.15946064, 128.69404921, 4.6586204, 10.00004685, 6.95143154,
#                           6.73395772, 124.86872764, 123.77079161])
    
if __name__ == "__main__":

    '''params_mm = [9.71851258, 127.23060036, 11.02606533, 14.4991172,
               15.65917039, 124.90989748, 123.52196735]

    params_m = mcmc._params_to_m(params_mm)
    spec = viz.toaster_spec(params_m, gap1=mcmc.GAP1_M)
    result = cv.solve_cavity(spec, keep_fields=True, f_target=15e9)
    viz.plot_modes_square_magnitude(spec, result, save='TEMP/gap1_edge_test/gap1_edge_test_11mm_center_pos.png')
    
    specs,results = cv.run_sweep(
        lambda dx, dy, i: mcmc.make_spec(params_mm, toast_dx=dx, toast_dy=dy, tag=f"x={dx*1e3:.2f}mm"),
        mcmc.tuning_positions(params_mm, n=16),        # yields (dx, dy, f_guess) per step
        n_modes=6,
        n_workers=None,            # every core
        timeout=600,               # bad geometry fails instead of stalling
        plot_all=True,          # needed for plot_best_modes
    )
    entries = [(s, r, f"x={x*1e3:.2f} mm")
                for (x, _y, _f), s, r in zip(mcmc.tuning_positions(params_mm), specs, results)
                if r["ok"]]
    viz.plot_best_modes_magnitude_square(entries, save=f'TEMP/gap1_edge_test/gap1_edge_test_11mm.png', suptitle='Best Result')

    final_spec = entries[-1][0]
    result_final = cv.solve_cavity(final_spec, keep_fields=True, f_target=8e9)
    viz.plot_modes_square_magnitude(final_spec, result_final, save='TEMP/gap1_edge_test/gap1_edge_test_11mm_extreme_pos.png')'''

    # test 3d

