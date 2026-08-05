# COMSOL clone for faster parameter search for future ADMX DM search config

Custom-built FEM solver for cavity simulations, and MCMC + simulated annealing for optimization. Also has plotting utility.

# Scalar problem

For propagation of EM waves in a hollow waveguide or cavity in the $z$-direction, it can be shown that (Eq. 8.24) in Jackson

$$\mathbf{B}_\perp=\frac{1}{\mu_0\varepsilon_0\frac{\omega^2}{c^2}-k^2}\left[\nabla_\perp\left(\frac{\partial B_z}{\partial z}\right)+i\mu_0\varepsilon_0\frac{\omega}{c}\hat{\mathbf{z}}\times\nabla_\perp E_z\right]$$

and

$$\mathbf{E}_\perp=\frac1{\mu_0\varepsilon_0\frac{\omega^2}{c^2}-k^2}\left[\nabla_\perp\left(\frac{\partial E_z}{\partial z}\right)-i\frac{\omega}{c}\hat{\mathbf{z}}\times\nabla_\perp B_z\right]$$

i.e. the transverse components of the electric and magnetic fields depend only on the $z$-components of the electric and magnetic fields. For TM waves, we also have the boundary conditions $E_z\equiv0$ on the surface and $B_z\equiv0$ everywhere. The above equations therefore reduce to

$$\mathbf{B}_\perp=\frac{\mu_0\varepsilon_0\omega}{ck}\hat{\mathbf{z}}\times\mathbf{E}_\perp$$

and

$$\mathbf{E}_\perp=\frac{ik}{\gamma^2}\nabla_\perp\psi$$

where $\psi=E_z$ and $\gamma^2=\mu_0\varepsilon_0\frac{\omega^2}{c^2}-k^2.$ Thus, this reduces the 3D wave equation to an appropriate scalar problem.