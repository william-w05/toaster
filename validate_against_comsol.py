from scripts import fem_vis as viz
from scripts import fem_solve as fem
from scripts import mcmc

import numpy as np

# check against COMSOL
if __name__ == "__main__":
    rounded_params = np.array([0, 128.69, 4.6586, 10, 6.9514, 6.734, 124.87, 123.77]) # these are params that COMSOL automatically rounds
    params_m = mcmc._params_to_m(rounded_params)
    spec = viz.toaster_spec(params_m, mesh_size=0.001, mesh_uniform=True)
    viz.plot_mesh(spec, title="Mesh (compare against COMSOL)", save="test_results/test_mesh_comsol.png")

    mcmc.sim_sweep(rounded_params, plot_all=True, mesh_uniform=True)