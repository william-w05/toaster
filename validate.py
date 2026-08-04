import fem_solve as fem
import fem_vis as viz
import matplotlib.pyplot as plt
import numpy as np
import scipy

C = 3e8 # speed of light in m/s

def get_TM_modes_rect(num_modes, width, height):
    modes = []
    mn = []
    temp = num_modes * 2 # to be safe, we calculate more modes than needed and then sort them
    for m in range(1, temp):
        for n in range(1, temp):
            omega_mn = C * np.pi * np.sqrt((m/width)**2 + (n/height)**2)
            modes.append(omega_mn)
            mn.append((m,n))
    sorted_args = np.argsort(np.array(modes))[:num_modes]
    return [modes[i] for i in sorted_args], [mn[i] for i in sorted_args]

def get_TM_modes_cyl(num_modes, radius):
    modes = []
    mn = []
    temp = num_modes * 2 # to be safe, we calculate more modes than needed and then sort them
    for m in range(0, temp):
        for n in range(1, temp):
            alpha_mn = scipy.special.jn_zeros(m, n)[-1] # get the nth zero of the mth Bessel function
            k_mn = alpha_mn / radius
            omega_mn = C * k_mn # see https://www.classe.cornell.edu/~liepe/webpage/docs/P4456L19.pdf
            modes.append(omega_mn)
            mn.append((m,n))
    sorted_args = np.argsort(np.array(modes))[:num_modes]
    return [modes[i] for i in sorted_args], [mn[i] for i in sorted_args]

def get_degeneracies_cyl(mn_list, modes_list, num_modes):
    degeneracies = []
    degen_modes = []
    for i, (m, n) in enumerate(mn_list):
        if m == 0:
            degeneracies.append((m,n))
            degen_modes.append(modes_list[i])
        else:
            degeneracies.append((m,n)) 
            degeneracies.append((-m,n)) # m and -m are degenerate
            degen_modes.append(modes_list[i])
            degen_modes.append(modes_list[i])
    return degeneracies[:num_modes], degen_modes[:num_modes]

# test solver on rectangular configuration
rect_spec = viz.rect_spec(width=0.25, height=0.5, mesh_size=0.005) # dimensions in meters
viz.plot_spec(rect_spec, title="Rectangular Specimen", save="test_results/test_rect_spec.png")
viz.plot_mesh(rect_spec, title="Rectangular Specimen Mesh", save="test_results/test_rect_mesh.png")

out = fem.solve_cavity(rect_spec, keep_fields=True)
viz.plot_modes(rect_spec, out, save="test_results/test_rect_modes.png")

# ground truth for rectangular configuration
fig, ax = plt.subplots(2, 3, figsize=(6,4))
x = np.linspace(-0.125, 0.125, 100)
y = np.linspace(-0.25, 0.25, 100)
X, Y = np.meshgrid(x, y)

# calculating lowest 6 modes for rectangular cavity:
a = 0.25
b = 0.5
lowest_modes, lowest_mn = get_TM_modes_rect(6, a, b)

ax[0,0].contourf(X, Y, np.sin(lowest_mn[0][0] * np.pi * (X-0.125) / 0.25) * np.sin(lowest_mn[0][1] * np.pi * (Y-0.25) / 0.5), cmap="RdBu_r", levels=30) # TM_11
ax[0,1].contourf(X, Y, np.sin(lowest_mn[1][0] * np.pi * (X-0.125) / 0.25) * np.sin(lowest_mn[1][1] * np.pi * (Y-0.25) / 0.5), cmap="RdBu_r", levels=30) # TM_21
ax[0,2].contourf(X, Y, np.sin(lowest_mn[2][0] * np.pi * (X-0.125) / 0.25) * np.sin(lowest_mn[2][1] * np.pi * (Y-0.25) / 0.5), cmap="RdBu_r", levels=30) # TM_12
ax[1,0].contourf(X, Y, np.sin(lowest_mn[3][0] * np.pi * (X-0.125) / 0.25) * np.sin(lowest_mn[3][1] * np.pi * (Y-0.25) / 0.5), cmap="RdBu_r", levels=30) # TM_22
ax[1,1].contourf(X, Y, np.sin(lowest_mn[4][0] * np.pi * (X-0.125) / 0.25) * np.sin(lowest_mn[4][1] * np.pi * (Y-0.25) / 0.5), cmap="RdBu_r", levels=30) # TM_31
ax[1,2].contourf(X, Y, np.sin(lowest_mn[5][0] * np.pi * (X-0.125) / 0.25) * np.sin(lowest_mn[5][1] * np.pi * (Y-0.25) / 0.5), cmap="RdBu_r", levels=30) # TM_13
plt.tight_layout()
plt.savefig("test_results/test_rect_modes_ground_truth.png", dpi=140); plt.close(fig)

# test solver on cylindrical configuration
cyl_spec = viz.cyl_spec(radius=0.125, mesh_size=0.0025) # dimensions in meters
viz.plot_spec(cyl_spec, title="Cylindrical Specimen", save="test_results/test_cyl_spec.png")
viz.plot_mesh(cyl_spec, title="Cylindrical Specimen Mesh", save="test_results/test_cyl_mesh.png")

out_cyl = fem.solve_cavity(cyl_spec, n_modes=12, keep_fields=True)
viz.plot_modes(cyl_spec, out_cyl, save="test_results/test_cyl_modes.png")

r_max = 0.125

theta = np.linspace(0, 2 * np.pi, 100)
r = np.linspace(0, r_max, 100)
THETA, R = np.meshgrid(theta, r)

lowest_modes_cyl, lowest_mn_cyl = get_TM_modes_cyl(12, r_max)
degeneracies, degen_modes = get_degeneracies_cyl(lowest_mn_cyl, lowest_modes_cyl, 12)
#print('degeneracies:', degeneracies)
#print('degenerate modes:', degen_modes)

fig1, ax1 = plt.subplots(4, 3, figsize=(10, 8), subplot_kw={'projection': 'polar'})
axes_flat = ax1.flatten()

for i in range(12):
    m = degeneracies[i][0]
    k_mn = degen_modes[i]/C   

    # E_z(r, theta) \propto J_m(k_mn r)exp(i*m*theta), see https://www.classe.cornell.edu/~liepe/webpage/docs/P4456L19.pdf
    Z = np.real(scipy.special.jv(m, k_mn * R) * np.exp(1j * m * THETA))
    axes_flat[i].contourf(THETA, R, Z, cmap="RdBu_r", levels=30)
    axes_flat[i].set_xticklabels([]) 
    axes_flat[i].set_yticklabels([])

plt.tight_layout()
plt.savefig("test_results/test_cyl_modes_ground_truth.png", dpi=140); plt.close(fig1)