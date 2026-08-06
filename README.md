# COMSOL clone for faster parameter search for future ADMX DM search config

Custom-built FEM solver for cavity simulations, and MCMC + simulated annealing for optimization. Also has plotting utility.

# Scalar problem

For propagation of EM waves in a hollow waveguide or cavity in the $z$-direction with homogeneous media, it can be shown that (Eq. 8.24 in Jackson, 2nd ed.)

$$\mathbf{B}_\perp=\frac{1}{\mu\varepsilon\frac{\omega^2}{c^2}-k^2}\left[\nabla_\perp\left(\frac{\partial B_z}{\partial z}\right)+i\mu\varepsilon\frac{\omega}{c}\hat{\mathbf{z}}\times\nabla_\perp E_z\right]$$

and

$$\mathbf{E}_\perp=\frac1{\mu\varepsilon\frac{\omega^2}{c^2}-k^2}\left[\nabla_\perp\left(\frac{\partial E_z}{\partial z}\right)-i\frac{\omega}{c}\hat{\mathbf{z}}\times\nabla_\perp B_z\right]$$

i.e. the transverse components of the electric and magnetic fields depend only on the $z$-components of the electric and magnetic fields. For TM waves, we also have $B_z\equiv0$ everywhere, as well as the boundary condition $E_z\equiv0$ on the surface $S$. The above equations therefore reduce to

$$\mathbf{B}_\perp=\frac{\mu\varepsilon\omega}{ck}\hat{\mathbf{z}}\times\mathbf{E}_\perp$$

and

$$\mathbf{E}_\perp=\frac{ik}{\gamma^2}\nabla_\perp\psi$$

where $\psi=E_z$ and $\gamma^2=\mu\varepsilon\frac{\omega^2}{c^2}-k^2.$ Thus, this reduces the 3D wave equation to an appropriate scalar problem, since $\psi$ satisfies the equation

$$(\nabla_\perp^2+\gamma^2)\psi=0$$

subject to 
$$\psi|_{S}=0.$$

We can rewrite this as follows: write 
$$\psi(x,y,z)=u(x,y)e^{ikz}$$, so that $\nabla_\perp^2\psi=\left(\nabla^2u\right)e^{ikz}$. Then

$$(\nabla^2+\gamma^2)u=0$$

after dividing by $e^{ikz}.$ If $\Omega$ now denotes the 2D domain of definition of $u$, then multiplying both sides of the above equation by $v\in H_0^1(\Omega)$ and integrating gives the weak formulation

$$\int_\Omega\nabla u\cdot\nabla v\mathrm dx=\gamma^2\int_\Omega uv\mathrm dx.$$

This is the form solved by `scipy.sparse.linalg.eigsh`. (Note that we set $k=0$ because we only care for the modes with no $z$-variation. It follows then that $\gamma^2=\mu\varepsilon\frac{\omega^2}{c^2}$ give the eigenfrequencies we are solving for.)