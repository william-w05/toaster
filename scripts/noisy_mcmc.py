"""
Robust ("noisy") objective for the toaster optimisation -- 33-dimensional
perturbation model with explicit LEFT / RIGHT parts and per-part rotation.

    F_rob(x) = E_delta[ FOM(x + delta) ]  ~  (1/n) sum_j FOM(x_j),
    x_j ~ N(mu(x), Sigma)

install() points mcmc.fom here, so mcmc_minimize / continue_mcmc / NM_opt run
unchanged on the robust objective.

TWO VECTORS
    DESIGN (7, optimised by the MCMC):
        angle, div_h, div_w, ctr_w, side_w, ctr_h, side_h
        These set the LEFT divider, the CENTRE toast and the LEFT side toast.
    EXTENDED (33, perturbed): the 7 above plus 26 extras --
        6 gaps      gap1L_out, gap1L_in, gap0L, gap0R, gap1R_in, gap1R_out
        1 cavity    cav_h
        5 rotations sideL_theta, divL_theta, ctr_theta, divR_theta, sideR_theta
        4 right dims divR_w, divR_h, sideR_w, sideR_h
        10 offsets  sideL_dx/dy, divL_dx/dy, ctr_dx/dy, divR_dx/dy, sideR_dx/dy

    Layout left to right, which is what the gap names track:
        wall | gap1L_out | sideL | gap1L_in | divL | gap0L | ctr |
                                gap0R | divR | gap1R_in | sideR | gap1R_out | wall
    so  cav_w = sum of those six gaps and five widths. Cavity WIDTH is therefore
    derived, never perturbed directly; cavity HEIGHT is its own dimension.

LEFT vs RIGHT
    The design vector carries only one divider and one side toast, because the
    NOMINAL cavity is mirror-symmetric. The extended vector gives the right-hand
    parts their own width/height/rotation/offset, so a sample is asymmetric. The
    left parts are named explicitly (sideL_*, divL_*) rather than being the
    "unlabelled" ones; ALIASES below accepts the shorter spellings when you set
    sigmas, so `default_cov(div_theta=0.2)` still works.

    This is the point of the whole model: stability.py showed that the dominant
    degradation comes from SYMMETRY BREAKING -- independent per-part errors
    detuning the cells -- which a symmetric perturbation cannot produce.

OFFSETS, NOT ABSOLUTE POSITIONS
    The ten position dimensions are DISPLACEMENTS from the gap-chain position,
    with nominal 0. This matters: the gaps and the absolute positions describe
    the same 11 layout degrees of freedom, so treating both as free would
    double-count. Here the gaps set where each part nominally sits (a stack-up:
    an error in gap1L_out shifts everything to its right), and the offsets are an
    additional placement error on top. Both sigmas are then independently
    meaningful.

ROTATIONS
    Each bar may tilt about its own centre, sideL_theta etc., in DEGREES,
    counter-clockwise positive, nominal 0 -- i.e. a rotation away from the +y
    axis. fem_solve builds the rectangle axis-aligned and rotates it in gmsh, so
    the tilt is genuinely in the mesh.

COMMON RANDOM NUMBERS (default)
    The same z-bank is reused for every design point. At n = 5 the standard error
    of the mean is ~0.13 log units, comparable to the difference between competing
    designs; fresh draws per evaluation mis-rank two designs differing by 0.10 log
    units about 30% of the time. Sharing the draws cancels the common part in the
    DIFFERENCE, which is all Metropolis uses, and makes F_rob deterministic.

COST
    n full tuning sweeps per objective evaluation: n * tuning_steps FEM solves.
"""

from __future__ import annotations

import numpy as np

from . import mcmc
from . import fem_solve as fem
from . import fem_vis as viz


# ═════════════════════════════════════════════════════════════════════════════
# the extended (noisy) parameter vector -- 33 dimensions
# ═════════════════════════════════════════════════════════════════════════════

DESIGN_NAMES = list(mcmc.PARAM_NAMES)          # 7
NOMINAL_TILT = 0.0                             # degrees; bars nominally upright

GAP_NAMES = ["gap1L_out", "gap1L_in", "gap0L", "gap0R", "gap1R_in", "gap1R_out"]
TILT_NAMES = ["sideL_theta", "divL_theta", "ctr_theta", "divR_theta", "sideR_theta"]
RIGHT_DIM_NAMES = ["divR_w", "divR_h", "sideR_w", "sideR_h"]
PART_NAMES = ["sideL", "divL", "ctr", "divR", "sideR"]          # left to right
OFFSET_NAMES = [f"{p}_{c}" for p in PART_NAMES for c in ("dx", "dy")]

EXTRA_NAMES = GAP_NAMES + ["cav_h"] + TILT_NAMES + RIGHT_DIM_NAMES + OFFSET_NAMES
NOISY_NAMES = DESIGN_NAMES + EXTRA_NAMES
D_NOISY = len(NOISY_NAMES)                     # 33
N_DESIGN = len(DESIGN_NAMES)
I_EXT = {nm: i for i, nm in enumerate(NOISY_NAMES)}

# short spellings accepted when setting sigmas
ALIASES = {"gap0": "gap0L", "gap1": "gap1L_out", "gap1_1": "gap1L_in",
           "gap1_2": "gap1R_in", "gap1_3": "gap1R_out",
           "side_theta": "sideL_theta", "div_theta": "divL_theta",
           "side_x": "sideL_dx", "side_y": "sideL_dy",
           "div_x": "divL_dx", "div_y": "divL_dy",
           "ctr_x": "ctr_dx", "ctr_y": "ctr_dy",
           "divR_x": "divR_dx", "divR_y": "divR_dy",
           "sideR_x": "sideR_dx", "sideR_y": "sideR_dy"}


def embed(params):
    """
    DESIGN vector (7, or legacy 8 with gap1 inside) -> EXTENDED mean vector (33).

    Right-hand dimensions mirror their left-hand counterparts; gaps take their
    nominal values; tilts and offsets are nominally zero.

    Indices come from mcmc.I_* rather than literals. The design order is
    (angle, div_h, div_w, ctr_w, side_w, ctr_h, side_h) -- note ctr_w PRECEDES
    side_w and ctr_h PRECEDES side_h, which is easy to get backwards.
    """
    p = np.asarray(params, dtype=np.float64).ravel()
    gap1 = None
    if p.size == 8:
        gap1 = float(p[3])
        p = np.delete(p, 3)
    if p.size != N_DESIGN:
        raise ValueError(f"expected {N_DESIGN} design parameters, got {p.size}")

    mu = np.zeros(D_NOISY, dtype=np.float64)      # zeros: offsets/tilts default 0
    mu[:N_DESIGN] = p
    for nm in GAP_NAMES:
        mu[I_EXT[nm]] = mcmc.GAP0 if nm.startswith("gap0") else mcmc.GAP1
    mu[I_EXT["cav_h"]] = mcmc.CAVITY_HEIGHT
    for nm in TILT_NAMES:
        mu[I_EXT[nm]] = NOMINAL_TILT
    mu[I_EXT["divR_w"]]  = p[mcmc.I_DIVW]
    mu[I_EXT["divR_h"]]  = p[mcmc.I_DIVH]
    mu[I_EXT["sideR_w"]] = p[mcmc.I_SIDEW]
    mu[I_EXT["sideR_h"]] = p[mcmc.I_SIDEH]
    if gap1 is not None:                          # legacy 8-vector
        for nm in ("gap1L_out", "gap1L_in", "gap1R_in", "gap1R_out"):
            mu[I_EXT[nm]] = gap1
    return mu


def split(x_ext):
    """EXTENDED vector -> (design 7-vector, dict of the 26 extras)."""
    x = np.asarray(x_ext, dtype=np.float64).ravel()
    if x.size != D_NOISY:
        raise ValueError(f"expected {D_NOISY} extended parameters, got {x.size}")
    return x[:N_DESIGN].copy(), {nm: float(x[I_EXT[nm]]) for nm in EXTRA_NAMES}


def parts_of(x_ext):
    """
    EXTENDED vector -> per-part geometry, in MILLIMETRES:
        [{name, w, h, cx, cy, theta, moves}, ...]  plus (cav_w, cav_h).

    cx comes from the GAP CHAIN (so a gap error shifts everything outboard of it)
    plus that part's own offset. `moves` marks the three toasts that the tuning
    stage translates; the dividers stay put.
    """
    d, e = split(x_ext)
    w = {"sideL": d[mcmc.I_SIDEW], "divL": d[mcmc.I_DIVW], "ctr": d[mcmc.I_CTRW],
         "divR": e["divR_w"], "sideR": e["sideR_w"]}
    h = {"sideL": d[mcmc.I_SIDEH], "divL": d[mcmc.I_DIVH], "ctr": d[mcmc.I_CTRH],
         "divR": e["divR_h"], "sideR": e["sideR_h"]}
    gaps = [e[g] for g in GAP_NAMES]                       # left -> right
    cav_w = sum(gaps) + sum(w[p] for p in PART_NAMES)

    out, cur = [], -0.5 * cav_w
    for i, name in enumerate(PART_NAMES):
        cur += gaps[i]                                     # gap before this part
        cx = cur + 0.5 * w[name]
        cur += w[name]
        out.append({"name": name, "w": w[name], "h": h[name],
                    "cx": cx + e[f"{name}_dx"], "cy": e[f"{name}_dy"],
                    "theta": e[f"{name}_theta"],
                    "moves": name in ("sideL", "ctr", "sideR")})
    return out, cav_w, e["cav_h"]


def default_cov(sigmas=None, **overrides):
    """
    Diagonal covariance from per-parameter 1-sigma values. MILLIMETRES for
    lengths, DEGREES for `angle` and the five tilts. Short aliases are accepted.

    Defaults follow the machining split in stability.py: a machined bar's own
    dimensions hold tighter than where it ends up once assembled, so gaps and
    placement offsets get the looser number. Tilt: 0.1 deg over a ~130 mm bar is
    ~0.23 mm of tip displacement, which is already large next to a 10 mm gap.
    """
    base = {"angle": 0.10,                                    # deg, stage
            "div_h": 0.03, "div_w": 0.03, "ctr_w": 0.03,
            "side_w": 0.03, "ctr_h": 0.03, "side_h": 0.03,
            "divR_w": 0.03, "divR_h": 0.03,
            "sideR_w": 0.03, "sideR_h": 0.03,
            "cav_h": 0.03}
    for g in GAP_NAMES:
        base[g] = 0.03
    for t in TILT_NAMES:
        base[t] = 0.10                                        # degrees
    for o in OFFSET_NAMES:
        base[o] = 0.03
    if sigmas:
        base.update(sigmas)
    base.update(overrides)
    resolved = {ALIASES.get(k, k): v for k, v in base.items()}
    unknown = set(resolved) - set(NOISY_NAMES)
    if unknown:
        raise ValueError(f"unknown parameter name(s): {sorted(unknown)}")
    s = np.array([resolved.get(nm, 0.0) for nm in NOISY_NAMES], dtype=np.float64)
    return np.diag(s ** 2)


# ═════════════════════════════════════════════════════════════════════════════
# sampling
# ═════════════════════════════════════════════════════════════════════════════

_Z_BANK: dict = {}          # (D, n, seed) -> the shared standardised draws


def get_z_bank(n, D=D_NOISY, clip=3.0, seed=0, antithetic=False,
               match_moments=True, whiten=None):
    """
    The shared standardised draws used by every design point (CRN).
    Cached, so the same arguments always return the same array.

    WHAT ACTUALLY MATTERS, measured on a generic quadratic response in D = 33:

    whiten (the big one) : once n > D the sample covariance of the centred draws
        is invertible, so the bank can be transformed to have sample covariance
        EXACTLY I. The second-order error of the average then vanishes: at n = 40
        the RMS error against the true E[F] falls from 2.39 to 0.077, a factor of
        31. This is a THRESHOLD, not a 1/sqrt(n) improvement -- n = 33 cannot do
        it (the centred sample has rank 32), n = 34 can. If you are paying for
        tens of samples anyway, land above D + 1 rather than just below it.
        Default: on whenever n > D, otherwise fall back to match_moments.

    match_moments : rescale each column to unit sample sd. All that is available
        when n <= D, since the full covariance is then singular. Fixes the
        per-parameter variance but not the off-diagonal structure.

    antithetic : pairs (z, -z), which makes the sample mean exactly zero and so
        kills the FIRST-order (gradient) error. DEFAULT OFF, which reverses my
        earlier advice, because it is counterproductive here: F(x + d) = F(x - d)
        for a purely even response, so an antithetic pair evaluates essentially
        the same geometry twice and halves the effective sample for the EVEN part.
        The tolerance study showed this response IS curvature-dominated -- every
        perturbed sample came out worse than nominal, which is the signature of an
        even, one-sided degradation. Measured: antithetic loses to plain sampling
        at every gradient strength tested except the most gradient-dominated, and
        by ~sqrt(2) once the quadratic term dominates. Turn it on only if you know
        you are far from an optimum.

    clip : np.clip CENSORS rather than truncates, so it shrinks the perturbation:
        at clip = 2 the effective sigma is 0.96 and var(z) = 0.918, understating
        the robustness penalty by ~8% (that penalty is exactly the curvature term
        scaled by var(z)). At clip = 3 the loss is 0.7%. Whitening or moment
        matching restores the variance afterwards in any case.

    HONEST CAVEAT: this is a SAMPLE-AVERAGE APPROXIMATION over n fixed scenarios,
    not E[F]. With whitening it matches the true expectation to second order in
    the perturbation; beyond that it is only as good as a quadratic model of the
    response.
    """
    if whiten is None:
        whiten = int(n) > int(D)
    key = (int(n), int(D), float(clip), int(seed), bool(antithetic),
           bool(match_moments), bool(whiten))
    if key not in _Z_BANK:
        rng = np.random.default_rng(seed)
        m, Dm = int(n), int(D)
        if antithetic:
            half = (m + 1) // 2
            z = rng.standard_normal((half, Dm))
            z = np.vstack([z, -z])[:m]
        else:
            z = rng.standard_normal((m, Dm))
        if np.isfinite(clip):
            z = np.clip(z, -clip, clip)
        z = z - z.mean(axis=0, keepdims=True)          # exact zero mean, always
        if whiten and m > Dm:
            # POPULATION second moment (divide by m, not m-1). The SAA's
            # second-order term is 0.5*tr(H * (1/m) sum_j d_j d_j^T), so it is the
            # 1/m moment that must equal Sigma. np.cov's default ddof=1 would leave
            # the population variance at (m-1)/m -- 2.5% low at m=40 -- which
            # understates the robustness penalty by the same 2.5% and would also
            # disagree with the ddof=0 used on the match_moments path below.
            S = (z.T @ z) / m
            w, V = np.linalg.eigh(S)
            z = z @ (V @ np.diag(np.maximum(w, 1e-12) ** -0.5) @ V.T)
        elif match_moments and m > 1:
            sd = z.std(axis=0, ddof=0, keepdims=True)
            z = np.divide(z, np.where(sd > 1e-12, sd, 1.0))
        _Z_BANK[key] = z
    return _Z_BANK[key]


def _sample_proposal(params, cov, n=5, clip=3.0, rng=None, common=True, seed=0,
                     antithetic=False, match_moments=True, whiten=None):
    """
    n perturbed EXTENDED parameter vectors around `params`.

    params : DESIGN vector (7, or legacy 8). Embedded via embed() so gap0/gap1
             enter the mean at their nominal values.
    cov    : (D, D) covariance over the extended vector, D = len(NOISY_NAMES).
             A 1-D array of length D is accepted and read as per-parameter
             VARIANCES; a scalar as a common variance.
    clip   : truncation in STANDARDISED units, applied to z before the Cholesky
             rotation, so correlations survive.
    common : reuse the shared z-bank (common random numbers). Turn it off only if
             you deliberately want independent draws per evaluation -- see the
             module docstring for why that is usually the wrong choice here.

    Returns (n, D).
    """
    mu = embed(params)
    D = mu.size
    cov = np.asarray(cov, dtype=np.float64)
    if cov.ndim == 0:
        cov = np.eye(D) * float(cov)
    elif cov.ndim == 1:
        if cov.size != D:
            raise ValueError(f"cov has length {cov.size}, expected {D}")
        cov = np.diag(cov)
    elif cov.shape != (D, D):
        raise ValueError(f"cov has shape {cov.shape}, expected ({D}, {D})")

    if common:
        z = get_z_bank(n, D, clip, seed, antithetic, match_moments, whiten)
    else:
        r = rng or np.random.default_rng()
        z = np.clip(r.standard_normal((int(n), D)), -clip, clip)

    # Cholesky of a PSD (possibly singular, e.g. a zero-variance row) matrix
    try:
        L = np.linalg.cholesky(cov + 1e-18 * np.eye(D))
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(cov)
        L = V @ np.diag(np.sqrt(np.maximum(w, 0.0)))
    return mu[None, :] + z @ L.T


# ═════════════════════════════════════════════════════════════════════════════
# evaluating one perturbed sample
# ═════════════════════════════════════════════════════════════════════════════

def within_limits_ext(x_ext, min_gap=0.5):
    """
    Constraints for an EXTENDED sample. Every gap and both right-hand parts are
    checked from the SAMPLE, not from the module constants -- with independent
    per-part errors a design can be feasible nominally and not as built.

    The geometric checks (positive gaps, parts inside the cavity, no overlap) use
    the ROTATED footprints, since a tilted bar reaches further than its width.
    """
    d, e = split(x_ext)
    parts, cav_w, cav_h = parts_of(x_ext)

    theta = d[mcmc.I_ANGLE]
    if np.any(d[1:] <= 0) or np.any(d[:N_DESIGN] >= 160):
        return False
    if theta < mcmc.ANGLE_MIN or theta > mcmc.ANGLE_MAX:
        return False
    if cav_h <= 0 or cav_w <= 0 or cav_w >= mcmc.TOTAL_W_MAX:
        return False
    if any(e[g] < min_gap for g in GAP_NAMES):
        return False

    # design-level limits, applied to BOTH sides
    for p in parts:
        if p["h"] < mcmc.H_MIN or p["h"] > mcmc.H_MAX:
            return False
    ctr_h = d[mcmc.I_CTRH]
    for nm, hh in (("divL", d[mcmc.I_DIVH]), ("divR", e["divR_h"]),
                   ("sideL", d[mcmc.I_SIDEH]), ("sideR", e["sideR_h"])):
        if hh <= (1 - mcmc.H_TOL) * ctr_h or hh >= (1 + mcmc.H_TOL) * ctr_h:
            return False
    ctr_w = d[mcmc.I_CTRW]
    if ctr_w < mcmc.CTR_W_MIN or ctr_w > mcmc.CTR_W_MAX:
        return False
    for sw in (d[mcmc.I_SIDEW], e["sideR_w"]):
        if sw < mcmc.SIDE_W_MIN or sw > mcmc.SIDE_W_MAX:
            return False
        if sw >= (1 + mcmc.SIDE_W_TOL) * ctr_w or sw < (1 - mcmc.SIDE_W_TOL) * ctr_w:
            return False
    for dw, g in ((d[mcmc.I_DIVW], e["gap0L"]), (e["divR_w"], e["gap0R"])):
        if dw < mcmc.DIV_W_MIN or dw >= g:
            return False
    if ctr_h > cav_h - 2 * e["gap0L"] * np.abs(np.tan(np.radians(theta))):
        return False

    # rotated footprints: inside the cavity, and not overlapping each other
    boxes = []
    for p in parts:
        r = fem.Rect.from_center(p["cx"], p["cy"], p["w"], p["h"], p["name"],
                                 p["theta"])
        b = r.bounds
        if (b[0] <= -0.5 * cav_w or b[2] >= 0.5 * cav_w
                or b[1] <= -0.5 * cav_h or b[3] >= 0.5 * cav_h):
            return False
        boxes.append(b)
    for a in range(len(boxes)):
        for b in range(a + 1, len(boxes)):
            A, B = boxes[a], boxes[b]
            if (A[0] < B[2] and B[0] < A[2] and A[1] < B[3] and B[1] < A[3]):
                return False
    return True


def make_spec_ext(x_ext, toast_dx=0.0, toast_dy=0.0, mesh_size=None,
                  tag="toaster", mesh_uniform=False):
    """
    CavitySpec for one perturbed sample: five INDEPENDENT bars, each with its own
    width, height, centre and rotation, inside a cavity whose width comes from the
    gap chain and whose height is its own perturbed dimension.

    Built directly rather than through viz.toaster_spec, which assumes the
    mirror-symmetric nominal layout and cannot express per-part asymmetry.
    toast_dx/dy (METRES) translate the three toasts; the dividers stay fixed.
    """
    parts, cav_w, cav_h = parts_of(x_ext)
    metal = []
    for p in parts:
        cx = p["cx"] * mcmc.MM + (toast_dx if p["moves"] else 0.0)
        cy = p["cy"] * mcmc.MM + (toast_dy if p["moves"] else 0.0)
        metal.append(fem.Rect.from_center(cx, cy, p["w"] * mcmc.MM,
                                          p["h"] * mcmc.MM, p["name"],
                                          p["theta"]))
    return fem.CavitySpec(
        outer=fem.Rect.from_center(0.0, 0.0, cav_w * mcmc.MM, cav_h * mcmc.MM,
                                   "cavity"),
        metal=metal,
        mesh_size=mcmc.MESH_SIZE if mesh_size is None else mesh_size,
        mesh_uniform=mesh_uniform, tag=tag,
        wall_material=mcmc.ALUMINIUM, metal_material=mcmc.ALUMINIUM)


def tuning_positions_ext(x_ext, n=16):
    """(dx, dy, f_guess) for a perturbed sample. The frequency target uses the
    sample's OWN gap0L, so the shift-invert lands on the right mode."""
    design, extra = split(x_ext)
    g0m = extra["gap0L"] * mcmc.MM
    t = np.tan(np.radians(float(design[mcmc.I_ANGLE])))
    for x in -np.linspace(0.0, mcmc.X_MAX_M, n):
        yield float(x), float(abs(x) * t), 3e8 / (2.0 * (g0m + abs(x)))


def fom_single(x_ext, tuning_steps=16, mesh_size=None, c_cutoff=True,
               verbose=False, return_details=False, check_limits=True):
    """Nominal FOM of ONE perturbed sample. Same definition as mcmc.fom."""
    if check_limits and not within_limits_ext(x_ext):
        d = {"C": np.array([]), "Q": np.array([]), "f": np.array([]),
             "V": np.array([]), "loc": np.array([]), "n_failed": 0,
             "n_steps": tuning_steps, "infeasible": True}
        return (mcmc.PENALTY, d) if return_details else mcmc.PENALTY

    positions = list(tuning_positions_ext(x_ext, n=tuning_steps))
    specs, results = fem.run_sweep(
        lambda dx, dy, i: make_spec_ext(x_ext, toast_dx=dx, toast_dy=dy,
                                        mesh_size=mesh_size,
                                        tag=f"x={dx*1e3:.2f}mm"),
        positions, n_modes=mcmc.N_MODES, n_workers=mcmc.SWEEP_WORKERS,
        timeout=mcmc.STEP_TIMEOUT, verbose=False)

    C, Q, f, V, loc = [], [], [], [], []
    n_failed = 0
    for r in results:
        if not r["ok"] or not r.get("modes"):
            n_failed += 1; continue
        m = fem.best_mode(r)
        if m is None:
            n_failed += 1; continue
        C.append(m["C"]); Q.append(m["Q"]); f.append(m["f"])
        V.append(m["area"]); loc.append(m["localisation"])
    C, Q, f, V, loc = map(np.asarray, (C, Q, f, V, loc))
    d = {"C": C, "Q": Q, "f": f, "V": V, "loc": loc, "n_failed": n_failed,
         "n_steps": tuning_steps, "infeasible": False}

    if f.size < 2 or n_failed or (c_cutoff and C.size and C.min() < mcmc.C_FLOOR):
        return (mcmc.PENALTY, d) if return_details else mcmc.PENALTY
    fm, Cm = 0.5 * (f[:-1] + f[1:]), 0.5 * (C[:-1] + C[1:])
    Qm, Vm = 0.5 * (Q[:-1] + Q[1:]), 0.5 * (V[:-1] + V[1:])
    val = float(np.sum(fm ** 2 / (Vm ** 2 * Cm ** 2 * Qm) * np.abs(np.diff(f))))
    if (not np.isfinite(val)) or val <= 0.0:
        val = mcmc.PENALTY
    return (val, d) if return_details else val


# ═════════════════════════════════════════════════════════════════════════════
# the robust objective
# ═════════════════════════════════════════════════════════════════════════════

# module state, so the installed mcmc.fom knows what to sample
_CFG = {"cov": None, "n": 6, "clip": 3.0, "common": True, "seed": 0,
        "aggregate": "mean", "tuning_steps": 16, "mesh_size": None}


def fom_mean(params, tuning_steps=None, c_cutoff=True, mesh_size=None,
             verbose=False, return_details=False,
             cov=None, n=None, clip=None, common=None, seed=None,
             aggregate=None, check_limits=True):
    """
    Robust objective: the MEAN nominal FOM over n perturbed samples.

    Signature-compatible with mcmc.fom, so install() can swap it in.

    aggregate :
      "mean" (default) -- the arithmetic mean of the scan times. This is the
          physically meaningful choice: build many cavities to these tolerances
          and the AVERAGE scan time is the arithmetic mean, not the geometric one.
          It is dominated by the worst samples, which is the intended behaviour --
          a design with a 20% chance of landing in a bad configuration should be
          scored accordingly.
      "geometric" -- exp(mean(log FOM)). More stable, less sensitive to a single
          penalty sample, but no longer an expected scan time.
      "median", "p90" -- robust / pessimistic alternatives.

    Samples that violate the constraints, fail to solve, or trip the form-factor
    floor return PENALTY and are INCLUDED in the average. That is deliberate:
    a nominal design sitting close to the feasibility boundary is genuinely
    fragile, and averaging the penalty in is what expresses that.
    """
    cov = _CFG["cov"] if cov is None else cov
    if cov is None:
        raise RuntimeError("no covariance configured -- call install(cov=...) or "
                           "pass cov= explicitly.")
    n = _CFG["n"] if n is None else int(n)
    clip = _CFG["clip"] if clip is None else float(clip)
    common = _CFG["common"] if common is None else bool(common)
    seed = _CFG["seed"] if seed is None else int(seed)
    aggregate = _CFG["aggregate"] if aggregate is None else aggregate
    tuning_steps = _CFG["tuning_steps"] if tuning_steps is None else tuning_steps
    mesh_size = _CFG["mesh_size"] if mesh_size is None else mesh_size

    X = _sample_proposal(params, cov, n=n, clip=clip, common=common, seed=seed)

    vals, dets = [], []
    for j, x_ext in enumerate(X):
        v, d = fom_single(x_ext, tuning_steps=tuning_steps, mesh_size=mesh_size,
                          c_cutoff=c_cutoff, return_details=True, check_limits=check_limits)
        vals.append(float(v)); dets.append(d)
        if verbose:
            print(f"    sample {j+1}/{n}: FOM={v:.4g}"
                  + ("  [infeasible]" if d.get("infeasible") else
                     f"  minC={d['C'].min():.4f}" if d["C"].size else ""),
                  flush=True)
    vals = np.asarray(vals, dtype=np.float64)

    if aggregate == "mean":
        value = float(np.mean(vals))
    elif aggregate == "geometric":
        value = float(np.exp(np.mean(np.log(np.maximum(vals, 1e-300)))))
    elif aggregate == "median":
        value = float(np.median(vals))
    elif aggregate == "p90":
        value = float(np.percentile(vals, 90))
    else:
        raise ValueError(f"unknown aggregate {aggregate!r}")
    if (not np.isfinite(value)) or value <= 0.0:
        value = mcmc.PENALTY

    if not return_details:
        return value

    # pool the per-sample diagnostics so the CSV columns stay meaningful:
    # MinC becomes the worst form factor over the whole tolerance ball.
    def cat(key):
        a = [d[key] for d in dets if np.size(d[key])]
        return np.concatenate(a) if a else np.array([])
    details = {k: cat(k) for k in ("C", "Q", "f", "V", "loc")}
    details["n_failed"] = int(sum(d["n_failed"] for d in dets))
    details["n_steps"] = tuning_steps
    details["samples"] = vals
    details["n_penalty"] = int(np.sum(vals >= mcmc.PENALTY * (1 - 1e-12)))
    details["nominal"] = float(vals[0]) if len(vals) else np.nan
    details["spread_log"] = (float(np.std(np.log(np.maximum(vals, 1e-300)), ddof=1))
                             if len(vals) > 1 else 0.0)
    return value, details


# ═════════════════════════════════════════════════════════════════════════════
# wiring
# ═════════════════════════════════════════════════════════════════════════════

_ORIGINAL_FOM = None


def install(cov=None, n=6, clip=3.0, common=True, seed=0, aggregate="mean",
            tuning_steps=16, mesh_size=None, verbose=True):
    """
    Point mcmc.fom at the robust objective. After this, mcmc.mcmc_minimize,
    mcmc.continue_mcmc and mcmc.NM_opt all optimise E[FOM] with no other changes.

    Returns the config dict. Call restore() to put the nominal objective back.
    """
    global _ORIGINAL_FOM
    if _ORIGINAL_FOM is None:
        _ORIGINAL_FOM = mcmc.fom
    _CFG.update(cov=(default_cov() if cov is None else np.asarray(cov, float)),
                n=int(n), clip=float(clip), common=bool(common), seed=int(seed),
                aggregate=aggregate, tuning_steps=int(tuning_steps),
                mesh_size=mesh_size)
    mcmc.fom = fom_mean
    if verbose:
        sd = np.sqrt(np.diag(_CFG["cov"]))
        print(f"[noisy] robust objective installed: aggregate={aggregate}, "
              f"n={n}, clip={clip}, common_random_numbers={common}")
        print(f"[noisy] perturbed dimensions (1 sigma):")
        for nm, s in zip(NOISY_NAMES, sd):
            print(f"          {nm:<8} {s:.4g}" + (" deg" if nm == "angle" else " mm"))
        print(f"[noisy] cost: {n} x {tuning_steps} = {n*tuning_steps} FEM solves "
              f"per objective evaluation")
    return dict(_CFG)


def restore(verbose=True):
    """Put the nominal (single-point) objective back."""
    global _ORIGINAL_FOM
    if _ORIGINAL_FOM is not None:
        mcmc.fom = _ORIGINAL_FOM
        _ORIGINAL_FOM = None
        if verbose:
            print("[noisy] nominal objective restored")


# ═════════════════════════════════════════════════════════════════════════════
# visualising the perturbations
# ═════════════════════════════════════════════════════════════════════════════

def plot_samples(params, cov=None, n=None, clip=None, common=None, seed=None,
                 save=None, show_nominal=True, max_overlay=40, dpi=160):
    """
    Three panels showing what the perturbation actually does.

      (a) every sampled geometry drawn on top of the nominal, so tilts, offsets
          and asymmetry are visible directly;
      (b) the realised GAPS, which is where the frequency lives -- the nominal is
          six equal gaps, and the scatter here is the symmetry breaking;
      (c) each perturbed dimension in units of its own sigma, which shows whether
          the clip is biting and which dimensions actually move.

    Draws only the geometry: no FEM is run, so it is instant.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    cov = _CFG["cov"] if cov is None else cov
    if cov is None:
        cov = default_cov()
    n = int(_CFG["n"] if n is None else n)
    clip = _CFG["clip"] if clip is None else clip
    common = _CFG["common"] if common is None else common
    seed = _CFG["seed"] if seed is None else seed

    X = _sample_proposal(params, cov, n=n, clip=clip, common=common, seed=seed)
    mu = embed(params)
    sd = np.sqrt(np.diag(np.asarray(cov, dtype=np.float64)))

    fig = plt.figure(figsize=(11.5, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], hspace=0.45, wspace=0.28)
    axg = fig.add_subplot(gs[:, 0])
    axk = fig.add_subplot(gs[0, 1])
    axz = fig.add_subplot(gs[1, 1])

    # ---- (a) overlaid geometries -------------------------------------------
    def draw(x_ext, color, alpha, lw, z):
        parts, cav_w, cav_h = parts_of(x_ext)
        axg.add_patch(plt.Rectangle((-cav_w / 2, -cav_h / 2), cav_w, cav_h,
                                    fill=False, ec=color, lw=lw, alpha=alpha,
                                    zorder=z))
        for p in parts:
            r = fem.Rect.from_center(p["cx"], p["cy"], p["w"], p["h"],
                                     p["name"], p["theta"])
            axg.add_patch(Polygon(r.corners(), closed=True, fill=False,
                                  ec=color, lw=lw, alpha=alpha, zorder=z))
        return cav_w, cav_h

    for x in X[:max_overlay]:
        draw(x, "#b03a2e", 0.45, 0.8, 2)
    cav_w, cav_h = draw(mu, "#1f4e79", 1.0, 1.8, 3) if show_nominal else (None, None)
    axg.set_aspect("equal")
    axg.set_xlim(-0.56 * cav_w, 0.56 * cav_w)
    axg.set_ylim(-0.56 * cav_h, 0.56 * cav_h)
    axg.set_xlabel("x (mm)"); axg.set_ylabel("y (mm)")
    axg.set_title(f"(a) {min(n, max_overlay)} perturbed geometries "
                  f"(red) vs nominal (blue)", fontsize=9)
    axg.grid(alpha=0.25, ls=":")

    # ---- (b) realised gaps --------------------------------------------------
    G = np.array([[split(x)[1][g] for g in GAP_NAMES] for x in X])
    pos = np.arange(len(GAP_NAMES))
    for row in G:
        axk.plot(pos, row, "o", color="#b03a2e", ms=3.5, alpha=0.5)
    axk.plot(pos, [split(mu)[1][g] for g in GAP_NAMES], "s-", color="#1f4e79",
             ms=5, lw=1.4, label="nominal")
    axk.set_xticks(pos)
    axk.set_xticklabels([g.replace("gap", "") for g in GAP_NAMES],
                        rotation=30, fontsize=7)
    axk.set_ylabel("gap (mm)")
    axk.set_title("(b) realised gaps -- the frequency lives here", fontsize=9)
    axk.grid(alpha=0.25, ls=":"); axk.legend(fontsize=7, frameon=False)

    # ---- (c) per-dimension excursion, in sigma ------------------------------
    live = np.where(sd > 0)[0]
    Z = (X[:, live] - mu[live]) / sd[live]
    axz.axhspan(-clip, clip, color="0.85", zorder=0, label=f"clip = {clip}$\\sigma$")
    for k, idx in enumerate(live):
        axz.plot(np.full(len(Z), k), Z[:, k], "o", color="#b03a2e", ms=3, alpha=0.55)
    axz.axhline(0, color="#1f4e79", lw=1.0)
    axz.set_xticks(range(len(live)))
    axz.set_xticklabels([NOISY_NAMES[i] for i in live], rotation=90, fontsize=5.5)
    axz.set_ylabel(r"excursion / $\sigma$")
    axz.set_title("(c) perturbation per dimension", fontsize=9)
    axz.grid(alpha=0.25, ls=":"); axz.legend(fontsize=7, frameon=False, loc="upper right")

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig


def describe_samples(params, cov=None, n=None, **kw):
    """Text companion to plot_samples: the realised spread of the derived
    quantities that actually matter, printed rather than plotted."""
    cov = _CFG["cov"] if cov is None else cov
    if cov is None:
        cov = default_cov()
    n = int(_CFG["n"] if n is None else n)
    X = _sample_proposal(params, cov, n=n, **kw)
    rows = [parts_of(x) for x in X]
    cav_w = np.array([r[1] for r in rows]); cav_h = np.array([r[2] for r in rows])
    G = np.array([[split(x)[1][g] for g in GAP_NAMES] for x in X])
    print(f"[noisy] {n} samples, {D_NOISY} perturbed dimensions")
    print(f"  cavity width  {cav_w.mean():8.4f} +/- {cav_w.std(ddof=1):.4f} mm")
    print(f"  cavity height {cav_h.mean():8.4f} +/- {cav_h.std(ddof=1):.4f} mm")
    print(f"  {'gap':>10} {'mean':>9} {'sd':>8} {'min':>9} {'max':>9}")
    for k, g in enumerate(GAP_NAMES):
        c = G[:, k]
        print(f"  {g:>10} {c.mean():>9.4f} {c.std(ddof=1):>8.4f} "
              f"{c.min():>9.4f} {c.max():>9.4f}")
    asym = np.abs(G[:, 0] - G[:, 5]) + np.abs(G[:, 1] - G[:, 4]) + np.abs(G[:, 2] - G[:, 3])
    print(f"  left-right gap asymmetry: {asym.mean():.4f} mm mean "
          f"(0 would be a mirror-symmetric sample)")
    n_ok = sum(within_limits_ext(x) for x in X)
    print(f"  feasible samples: {n_ok}/{n}")
    return {"cav_w": cav_w, "cav_h": cav_h, "gaps": G, "asymmetry": asym}