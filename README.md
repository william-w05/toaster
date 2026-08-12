# COMSOL clone for faster parameter search for future ADMX DM search config

Custom-built FEM solver for cavity simulations, and MCMC + simulated annealing for optimization. Also has plotting utility.

# Theory of solving the PDE

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

# Optimization details

Our optimization happened in 3 stages: a non-surrogate boosted, "wide" parameter space annealing search with 10 walkers, a surrogate-boosted annealing search over a restricted parameter space (10 walkers), and then a Nelder-Mead search on the top 10 results from the restricted search.

Over the course of optimization, we had two objectives, one mapping $\mathbb{R}^8\to\mathbb{R}$ and one mapping $\mathbb{R}^7\to\mathbb{R}.$ The first objective had parameters 

$$[\text{theta, divider height, divider width, side gaps, center toast width, side toast width, center toast height, side toast height}].$$

After running 3500 iterations (first stage), we set the side gaps to be $10\ \mathrm{mm}$ and restricted the parameter space, giving rise to the second objective. The second objective was used for both the second and third optimization stages.

# Optimization technique

The primary optimization technique used to find the optimal geometry is an MCMC-type Metropolis-Hastings simulated annealing strategy. In typical Metropolis-Hastings for sampling from a distribution $P(x)$, we assume that we have access to a function $f(x)$ which is proportional to $P(x).$ We first choose a (symmetric) proposal function $g(y\mid x)$, and at each step of the walk, the next step $x'$ is chosen by sampling from $g(x'\mid x)$. The acceptance probability is then calculated as 

$$\alpha=\min\left(1,\frac{f(x')}{f(x)}\right).$$

The stationary distribution of this can be shown to match up with $P(x)$.

When annealing, we apply a similar strategy, except we now have a temperature function $T(n)$, where $n$ is the current step number. The acceptance probability is then calculated as

$$\alpha_{\text{anneal}}=\min\left(1,\exp\left(-\frac{f(x')-f(x)}{T(n)}\right)\right).$$

Note that this no longer gives a stationary distribution $P(x)$. The rationale now, however, is that at lower temperatures, the walker will naturally find its way toward lower objective values.

Our proposal function is a $\chi^2$ distribution with 3 degrees of freedom. Our temperature function is $T(n)=0.999^n$ (can both be changed).

## Surrogate boost

An issue with MCMC/annealing is that steps may be "bad" in the sense that they don't walk towards lower objective values. One fix is to use a surrogate function which is supposed to approximate the objective sufficiently well. Our surrogate was a multi-layer perceptron (MLP) trained on the log-objective values *only on the buffer*. That is, if we have run 250 iterations, then the surrogate only sees the previous 250 iterations. This was done because we frequently restricted the parameter space. 

For a given set of parameters $x$, the surrogate would only train on $(x, f(x))$ if the minimum form factor $C_\text{min}(x)$ along the tuning range in the calculation of $f(x)$ satisfied $C_\text{min(x)}\geq0.05.$ This was done because we set a penalty of 1e+33 whenever $C_\text{min(x)}<0.05$ to discount low-form factor geometries that would also yield relatively low figures of merit. 

The MLP was first trained after 100 initial iterations (1000 individual evaluations across 10 walkers); subsequently, the MLP was trained after every 50 iterations (500 evaluations across 10 walkers). (At times, we would prematurely stop/restart the MCMC, and at each restart, we would retrain the MLP, so the interval between each retraining could have been less than 50 iterations.) 

We use the model to *rank feasible points.* At each step, $n=64$ points are proposed by the algorithm by drawing from the proposal distribution $g$ ($\chi^2$ distribution). The MLP predicts the values at each of the $n$ proposed points and chooses the point $\bar{x}$ for which its predicted value is lowest. The objective is then evaluated at $\bar{x}$, and the annealing procedure described above is applied to determine whether or not to accept $\bar{x}$ as the next step in the MCMC.

### MLP architecture

On the training buffer, we first normalized all observations because the scale of the parameters varied. Let $(x_i,y_i)$ denote an observation ($x\in\mathbb{R}^7$ or $\mathbb{R}^8$, $y\in\mathbb{R}$) in log space, i.e. $y_i=\log f(x_i).$ Set

$$\tilde{x}_i=\frac{x_i-\mu_{x}}{\sigma_{x}+\varepsilon}$$

and

$$\tilde{y}_i=\frac{y_i-\mu_y}{\sigma_{y}+\varepsilon}$$

with $\varepsilon=$1e-8. We used 4 fully-connected layers with $(\tilde{x}_i,\tilde{y}_i)$:

$$\tilde{x}\xrightarrow{\text{layer 1}}\varphi(W_1\tilde{x}+b_1)=h_1\xrightarrow{\text{layer 2}}\varphi(W_2h_1+b_2)=h_2\xrightarrow{\text{layer 3}}\varphi(W_3h_2+b_3)=h_3\xrightarrow{\text{layer 4}}W_4^Th_3+b_3=\tilde{y}$$

with $W_1\in\mathbb{R}^{128\times 7}$ or $\mathbb{R}^{128\times 8}$, $W_2\in\mathbb{R}^{128\times128}$, $W_3\in\mathbb{R}^{64\times128}$, $W_4\in\mathbb{R}^{64}$. Activation is SiLU:

$$\varphi(x)=\frac{x}{1+e^{-x}}.$$ 

We use the Adam update rule. 

## Efficacy of the MLP

(WIP) During stage 2 of the optimization procedure, we computed the standard deviation of the log-objective (not including the points where the minimal form factor was less than 0.05) to be approximately 0.26. At the same time, at the 149th iteration and every 50th iteration after that, we computed the RMSE of the MLP on the 49 iterations not used in training the MLP up to that point (This RMSE should be read as an indicator of how well the MLP knows where it wants to go.) The RMSE was initially ~0.5 but dropped to ~0.14 after several hundred iterations. Since the MLP was trained on the (R)MSE of the z-scores of the observations within the training buffer, the "test set" RMSE of approximately 0.14 (less than 0.26) suggests that the MLP is somewhat better than just predicting the mean (i.e. better than random), at least along the parameters for which it chooses to guide the MCMC along.