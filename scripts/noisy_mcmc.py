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

    CONSEQUENCE FOR THE REALISED SPREAD (this surprises people, see
    describe_samples): the bank is centred and moment-matched, so the sample mean
    of every perturbed dimension is EXACTLY its nominal value and the sample sd is
    EXACTLY its input sigma -- as a POPULATION (ddof = 0) sd. A ddof = 1 sd of the
    same draws is larger by sqrt(n / (n - 1)), which at n = 8 is 6.9%. That is a
    property of the estimator, not a bug in the sampler.

CONSTRAINTS ARE NOT ALL THE SAME KIND OF PROBLEM
    classify_limits() sorts a violation into one of three severities, because
    lumping them together is what made an out-of-box design indistinguishable
    from a broken one:

      "fatal"     the spec cannot be BUILT: a non-positive width, height or
                  cavity. There is nothing to solve, so it is always PENALTY.
      "geometry"  it builds and meshes, but it is not the cavity you meant --
                  bars overlapping, a bar through the wall, a sliver gap, at rest
                  or at full tuning displacement. gmsh does NOT complain about
                  any of these (overlapping cut tools just merge into one hole),
                  so you get a number and the number is meaningless.
      "bounds"    a perfectly good cavity that sits outside the DESIGN box:
                  heights outside [H_MIN, H_MAX], div_w >= gap0, angle out of
                  range. Nothing physical is wrong with it.

    check_limits chooses how much of that is fatal to the OBJECTIVE:

      "warn" (default)   fatal -> PENALTY; geometry and bounds warn and are
                         EVALUATED ANYWAY. This matches mcmc.fom, which never
                         consulted the limits at all, so the robust and nominal
                         objectives now agree on which points get a real number.
      "geometry"         fatal and geometry -> PENALTY; bounds warn and evaluate.
                         The sane middle ground for an actual optimisation run:
                         a meaningless number never enters the chain, but a
                         design-box excursion still gets scored.
      "strict" / True    any violation -> PENALTY (the behaviour before).
      "off" / False      no checks, no warnings.

    Warnings are deduplicated (noisy_mcmc.WARN_MODE = "once" | "always" |
    "never") and tallied in noisy_mcmc.VIOLATION_TALLY, so a long run reports
    what it hit without flooding the log.

COST
    n full tuning sweeps per objective evaluation: n * tuning_steps FEM solves.
    NOTE that in "warn" mode an out-of-box sample costs a full sweep where it
    used to cost nothing, so a design sitting outside the box is now n times
    more expensive than it was.

SEEING THE EFFECT ON THE FIELD (not just the geometry)
    plot_samples / describe_samples show the perturbed GEOMETRY, which for
    0.03 mm errors is visually indistinguishable from nominal. The functions in
    the "field-level comparison" section below run the actual FEM solve at one
    tuning position for the nominal geometry and every sample, and show what the
    perturbation does to E_z:

        solve_field_samples   solve nominal + samples, keeping the fields
        plot_field_samples    panel grid of |E_z|^2 via fem_vis
        plot_field_difference |E_z|^2 MINUS the nominal, on a common grid

    plot_field_samples is the shape view: solve_cavity peak-normalises every
    eigenvector, so those panels compare mode SHAPE, not amplitude.
    plot_field_difference is the quantitative one: it rescales each mode to a
    fixed stored energy (fem_solve.field_scale) and interpolates onto a shared
    grid, because each perturbed geometry has its OWN mesh and a nodal difference
    is not defined between two different meshes.
"""

from __future__ import annotations

import numpy as np

from . import mcmc
from . import fem_solve as fem
from . import fem_vis as viz

import csv
import os
import re as _re

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

# dimensions measured in DEGREES rather than millimetres -- used only for
# labelling, but getting it wrong in a printout is how sigmas get misread
DEGREE_NAMES = frozenset(["angle"] + TILT_NAMES)

# short spellings accepted when setting sigmas
# NOTE these are one-to-one, not one-to-many: `default_cov(gap0=0.05)` sets
# gap0L ONLY, and `gap1=0.05` sets gap1L_out ONLY. To set a whole family, pass
# the canonical names, e.g. default_cov(**{g: 0.05 for g in GAP_NAMES}).
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
    base = {"angle": 0.05,                                    # deg, stage
            "div_h": 0.05, "div_w": 0.05, "ctr_w": 0.05,
            "side_w": 0.05, "ctr_h": 0.05, "side_h": 0.05,
            "divR_w": 0.05, "divR_h": 0.05,
            "sideR_w": 0.05, "sideR_h": 0.05,
            "cav_h": 0.10}
    for g in GAP_NAMES:
        base[g] = 0.05
    for t in TILT_NAMES:
        base[t] = 0.05                                      # degrees
    for o in OFFSET_NAMES:
        base[o] = 0.05
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

        BECAUSE the rescale happens AFTER the clip, individual entries of the
        returned bank may exceed `clip` slightly. The shaded band in
        plot_samples panel (c) is therefore the requested clip, not a hard bound
        on the realised draws.

    NOTE ON THE REALISED MOMENTS: after centring (always) and moment matching or
    whitening, each column has sample mean exactly 0 and sample sd exactly 1 in
    the POPULATION (ddof = 0) convention. A ddof = 1 estimate of the same column
    reads sqrt(n / (n - 1)) instead. See describe_samples.

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

# ─────────────────────────────────────────────────────────────────────────────
# constraints -- three severities; see the module docstring for why
# ─────────────────────────────────────────────────────────────────────────────

SEVERITIES = ("fatal", "geometry", "bounds")
LIMIT_MODES = ("strict", "geometry", "warn", "off")
MIN_GAP = 0.5              # mm; a gap below this is a sliver, not a cell

WARN_MODE = "once"          # "once" | "always" | "never"
VIOLATION_TALLY: dict = {}  # RULE -> how many times it has been hit
_VIOLATION_EXAMPLE: dict = {}   # RULE -> the first message that produced it
_WARN_SEEN: set = set()

# Every violation message carries the offending NUMBERS, which is what you want
# to read but useless as an identity: each perturbed sample produces a distinct
# string, so dedup would never fire and the tally would grow one entry per
# evaluation. Blanking the numbers gives the RULE, which is the thing there are
# finitely many of.
_NUMERIC = _re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _rule_key(msg):
    """The message with every number blanked -- i.e. which RULE was broken."""
    return _NUMERIC.sub("#", msg)


def reset_violations():
    """Clear the warning-dedup memory and the tally. Call between runs."""
    VIOLATION_TALLY.clear()
    _VIOLATION_EXAMPLE.clear()
    _WARN_SEEN.clear()


def report_violations(top=20):
    """
    Print the accumulated tally, commonest first, one line per RULE with an
    example of the numbers that tripped it. Returns the sorted list.
    """
    items = sorted(VIOLATION_TALLY.items(), key=lambda kv: -kv[1])
    if not items:
        print("[noisy] no limit violations recorded")
        return items
    print(f"[noisy] limit violations recorded ({len(items)} distinct rule(s)):")
    for rule, k in items[:top]:
        print(f"    {k:>7} x  {_VIOLATION_EXAMPLE.get(rule, rule)}")
    if len(items) > top:
        print(f"    ... and {len(items) - top} more")
    return items


def _norm_limits_mode(check_limits):
    """
    True/False/None/str -> one of LIMIT_MODES.

    None means "use the configured default", which is _CFG["check_limits"].
    True and False are still accepted so old call sites keep working, and mean
    "strict" and "off" exactly as they used to.
    """
    if check_limits is None:
        check_limits = _CFG.get("check_limits", "warn")
    if check_limits is True:
        return "strict"
    if check_limits is False:
        return "off"
    m = str(check_limits).lower()
    if m not in LIMIT_MODES:
        raise ValueError(f"check_limits must be one of {LIMIT_MODES} (or "
                         f"True/False), got {check_limits!r}")
    return m


def _boxes_at(x_ext, dx_mm=0.0, dy_mm=0.0):
    """
    Rotated axis-aligned footprints of the five bars, in mm, with the three
    TOASTS displaced by (dx_mm, dy_mm) and the dividers left where they are --
    i.e. exactly what make_spec_ext builds at that tuning position.
    """
    parts, cav_w, cav_h = parts_of(x_ext)
    boxes = []
    for p in parts:
        cx = p["cx"] + (dx_mm if p["moves"] else 0.0)
        cy = p["cy"] + (dy_mm if p["moves"] else 0.0)
        r = fem.Rect.from_center(cx, cy, p["w"], p["h"], p["name"], p["theta"])
        boxes.append((p["name"], r.bounds))
    return boxes, cav_w, cav_h


def _placement_violations(x_ext, dx_mm, dy_mm, where):
    """Containment and pairwise overlap of the ROTATED footprints at one tuning
    position. Bounding boxes, so it is conservative for a tilted bar."""
    boxes, cav_w, cav_h = _boxes_at(x_ext, dx_mm, dy_mm)
    out = []
    for nm, b in boxes:
        if b[0] <= -0.5 * cav_w or b[2] >= 0.5 * cav_w:
            out.append(f"{nm} spans x = [{b[0]:.3f}, {b[2]:.3f}] mm, past the "
                       f"cavity half-width {0.5 * cav_w:.3f} mm ({where})")
        if b[1] <= -0.5 * cav_h or b[3] >= 0.5 * cav_h:
            out.append(f"{nm} spans y = [{b[1]:.3f}, {b[3]:.3f}] mm, past the "
                       f"cavity half-height {0.5 * cav_h:.3f} mm ({where})")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (na, A), (nb, B) = boxes[i], boxes[j]
            if A[0] < B[2] and B[0] < A[2] and A[1] < B[3] and B[1] < A[3]:
                out.append(f"{na} and {nb} overlap ({where})")
    return out


def classify_limits(x_ext_or_design, min_gap=MIN_GAP, tuning_check=True):
    """
    -> {"fatal": [...], "geometry": [...], "bounds": [...]} of human-readable
    violation strings. All three empty means feasible.

    Accepts an EXTENDED sample (33) or a DESIGN vector (7 / legacy 8), which is
    embedded first. Every gap and both right-hand parts are read from the SAMPLE,
    not from the module constants -- with independent per-part errors a design
    can be feasible nominally and not as built.

    tuning_check : also test the placement at FULL tuning displacement,
        dx = -X_MAX_FREQ and dy = |dx| tan(theta). The sweep really does build
        those geometries, and a design whose side toast reaches the wall only at
        the end of the scan used to pass every test here and then quietly hand
        gmsh a bar sticking through the cavity. Cheap, so on by default. It is
        strictly additional: for any design satisfying the ctr_h clearance rule
        (which uses 2*gap0 = 20 mm against an actual travel of 8.75 mm) the
        vertical part of it can never bind.
    """
    x = np.asarray(x_ext_or_design, dtype=np.float64).ravel()
    if x.size in (N_DESIGN, N_DESIGN + 1):
        x = embed(x)
    d, e = split(x)
    parts, cav_w, cav_h = parts_of(x)
    v = {k: [] for k in SEVERITIES}

    # ---- fatal: gmsh cannot build this at all -------------------------------
    for p in parts:
        if p["w"] <= 0:
            v["fatal"].append(f"{p['name']} width = {p['w']:.4g} mm <= 0")
        if p["h"] <= 0:
            v["fatal"].append(f"{p['name']} height = {p['h']:.4g} mm <= 0")
    if cav_w <= 0:
        v["fatal"].append(f"cavity width = {cav_w:.4g} mm <= 0")
    if cav_h <= 0:
        v["fatal"].append(f"cavity height = {cav_h:.4g} mm <= 0")
    if v["fatal"]:
        # every coordinate below is meaningless once a dimension is negative,
        # so stop here rather than emit a cascade of derived nonsense
        return v

    # ---- geometry: builds, but is not the cavity you meant -------------------
    for g in GAP_NAMES:
        if e[g] <= 0:
            v["geometry"].append(f"{g} = {e[g]:.4g} mm <= 0 (parts have swapped "
                                 f"order)")
        elif e[g] < min_gap:
            v["geometry"].append(f"{g} = {e[g]:.4g} mm < min_gap = {min_gap} "
                                 f"(sliver: the mesh cannot resolve it)")
    v["geometry"] += _placement_violations(x, 0.0, 0.0, "at rest")
    if tuning_check:
        theta = float(d[mcmc.I_ANGLE])
        dx_mm = -float(mcmc.X_MAX_FREQ)
        dy_mm = abs(dx_mm) * np.tan(np.radians(theta))
        v["geometry"] += _placement_violations(x, dx_mm, dy_mm,
                                               "at full tuning displacement")

    # ---- bounds: a good cavity, outside the design box -----------------------
    theta = d[mcmc.I_ANGLE]
    ctr_h, ctr_w = d[mcmc.I_CTRH], d[mcmc.I_CTRW]
    for nm, val in zip(DESIGN_NAMES, d):
        if val >= 160:
            v["bounds"].append(f"{nm} = {val:.4g} >= 160")
    if not (mcmc.ANGLE_MIN <= theta <= mcmc.ANGLE_MAX):
        v["bounds"].append(f"angle = {theta:.4g} outside "
                           f"[{mcmc.ANGLE_MIN}, {mcmc.ANGLE_MAX}] deg")
    if cav_w >= mcmc.TOTAL_W_MAX:
        v["bounds"].append(f"cav_w = {cav_w:.4g} >= TOTAL_W_MAX = "
                           f"{mcmc.TOTAL_W_MAX:.4g} mm (will not fit the bore)")
    for p in parts:
        if not (mcmc.H_MIN <= p["h"] <= mcmc.H_MAX):
            v["bounds"].append(f"{p['name']} height = {p['h']:.4g} outside "
                               f"[{mcmc.H_MIN}, {mcmc.H_MAX}]")
    for nm, hh in (("divL", d[mcmc.I_DIVH]), ("divR", e["divR_h"]),
                   ("sideL", d[mcmc.I_SIDEH]), ("sideR", e["sideR_h"])):
        if hh <= (1 - mcmc.H_TOL) * ctr_h or hh >= (1 + mcmc.H_TOL) * ctr_h:
            v["bounds"].append(f"{nm} height = {hh:.4g} not within "
                               f"+/-{mcmc.H_TOL:.0%} of ctr_h = {ctr_h:.4g}")
    if not (mcmc.CTR_W_MIN <= ctr_w <= mcmc.CTR_W_MAX):
        v["bounds"].append(f"ctr_w = {ctr_w:.4g} outside "
                           f"[{mcmc.CTR_W_MIN}, {mcmc.CTR_W_MAX}]")
    for nm, sw in (("sideL", d[mcmc.I_SIDEW]), ("sideR", e["sideR_w"])):
        if not (mcmc.SIDE_W_MIN <= sw <= mcmc.SIDE_W_MAX):
            v["bounds"].append(f"{nm}_w = {sw:.4g} outside "
                               f"[{mcmc.SIDE_W_MIN}, {mcmc.SIDE_W_MAX}]")
        if sw >= (1 + mcmc.SIDE_W_TOL) * ctr_w or sw < (1 - mcmc.SIDE_W_TOL) * ctr_w:
            v["bounds"].append(f"{nm}_w = {sw:.4g} not within "
                               f"+/-{mcmc.SIDE_W_TOL:.0%} of ctr_w = {ctr_w:.4g}")
    for nm, dw, gnm in (("divL", d[mcmc.I_DIVW], "gap0L"),
                        ("divR", e["divR_w"], "gap0R")):
        if dw < mcmc.DIV_W_MIN:
            v["bounds"].append(f"{nm}_w = {dw:.4g} < DIV_W_MIN = "
                               f"{mcmc.DIV_W_MIN}")
        if dw >= e[gnm]:
            # NOTE this is gap0 (the gap flanking the CENTRE toast), not gap1.
            # Both are 10 mm nominally, which makes it easy to misread. It is a
            # DESIGN rule, not a buildability one: the divider sits between the
            # centre and side toasts and never touches gap0, so a wider divider
            # is perfectly constructible -- see mcmc.DIV_W_MIN, "div_w in [3, gap0)".
            v["bounds"].append(f"{nm}_w = {dw:.4g} >= {gnm} = {e[gnm]:.4g} "
                               f"(design rule div_w < gap0 -- note gap0, the gap "
                               f"flanking the CENTRE toast, not gap1; this is not "
                               f"an overlap test, see the geometry severity for "
                               f"those)")
    clear = cav_h - 2 * e["gap0L"] * np.abs(np.tan(np.radians(theta)))
    if ctr_h > clear:
        v["bounds"].append(f"ctr_h = {ctr_h:.4g} > wall clearance {clear:.4g} "
                           f"at angle = {theta:.4g} deg")
    return v


def within_limits_ext(x_ext, min_gap=MIN_GAP, tuning_check=True):
    """
    True only when NOTHING is violated, at any severity. Kept as the single
    boolean used for the `feasible` columns and the sample counts; the objective
    itself goes through classify_limits so it can treat the severities
    differently.
    """
    v = classify_limits(x_ext, min_gap=min_gap, tuning_check=tuning_check)
    return not (v["fatal"] or v["geometry"] or v["bounds"])


def why_infeasible(x_ext_or_design, min_gap=MIN_GAP, tuning_check=True):
    """
    Flat, severity-tagged list of what an EXTENDED sample (33) or a DESIGN vector
    (7 / legacy 8) violates. Empty means feasible.
    """
    v = classify_limits(x_ext_or_design, min_gap=min_gap,
                        tuning_check=tuning_check)
    return [f"[{k}] {m}" for k in SEVERITIES for m in v[k]]


def _should_penalise(v, mode):
    """Given a classification and a mode, does the objective return PENALTY?"""
    if mode == "off":
        return False
    if v["fatal"]:
        return True                      # nothing to solve, in every mode
    if mode == "strict":
        return bool(v["geometry"] or v["bounds"])
    if mode == "geometry":
        return bool(v["geometry"])
    return False                         # "warn": evaluate it anyway


def _warn_violations(kind, msgs, label, penalised):
    """Tally every violation by RULE; print according to WARN_MODE."""
    rules = []
    for m in msgs:
        r = _rule_key(m)
        rules.append(r)
        VIOLATION_TALLY[r] = VIOLATION_TALLY.get(r, 0) + 1
        _VIOLATION_EXAMPLE.setdefault(r, m)
    if WARN_MODE == "never" or not msgs:
        return
    # dedup on the RULES, not the message text, and NOT on the label: otherwise
    # every sample index re-arms the warning and an MCMC run floods the log
    key = (kind, tuple(rules), bool(penalised))
    if WARN_MODE == "once":
        if key in _WARN_SEEN:
            return
        _WARN_SEEN.add(key)
    where = f" [{label}]" if label else ""
    tail = " -> PENALTY" if penalised else " -> EVALUATED ANYWAY"
    note = ("  (repeats suppressed; noisy_mcmc.WARN_MODE = 'always' to see them "
            "all, report_violations() for the tally)"
            if WARN_MODE == "once" else "")
    print(f"[noisy] {kind.upper()}{where}: " + "; ".join(msgs) + tail + note,
          flush=True)


def _limits_verdict(x_ext, mode, label="", min_gap=MIN_GAP, tuning_check=True):
    """-> (penalise, classification). Warns as a side effect."""
    empty = {k: [] for k in SEVERITIES}
    if mode == "off":
        return False, empty
    v = classify_limits(x_ext, min_gap=min_gap, tuning_check=tuning_check)
    penalise = _should_penalise(v, mode)
    for kind in SEVERITIES:
        if v[kind]:
            # a category is only "the reason" for the penalty if this mode
            # actually acts on it
            acts = (kind == "fatal" or mode == "strict"
                    or (mode == "geometry" and kind == "geometry"))
            _warn_violations(kind, v[kind], label, penalise and acts)
    return penalise, v


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
               verbose=False, return_details=False, check_limits=None,
               label="", min_gap=MIN_GAP):
    """
    Nominal FOM of ONE perturbed sample. Same definition as mcmc.fom.

    check_limits : None -> _CFG["check_limits"] (default "warn"). See the module
        docstring. In "warn" mode an out-of-box geometry is WARNED ABOUT and then
        solved normally, so it gets a real number exactly as mcmc.fom would give
        it one; only a fatal (unbuildable) spec short-circuits to PENALTY.

    The details dict carries:
        infeasible  -- violates something, whether or not it was penalised
        penalised   -- short-circuited to PENALTY by the limit check
        violations  -- the classify_limits dict
    A sample can now be infeasible AND scored, which is the whole point.
    """
    mode = _norm_limits_mode(check_limits)
    penalise, viol = _limits_verdict(x_ext, mode, label=label, min_gap=min_gap)
    out_of_box = bool(viol["fatal"] or viol["geometry"] or viol["bounds"])
    if penalise:
        d = {"C": np.array([]), "Q": np.array([]), "f": np.array([]),
             "V": np.array([]), "loc": np.array([]), "n_failed": 0,
             "n_steps": tuning_steps, "infeasible": True, "penalised": True,
             "violations": viol}
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
        if not r or not r.get("ok") or not r.get("modes"):
            n_failed += 1; continue
        m = fem.best_mode(r)
        if m is None:
            n_failed += 1; continue
        C.append(m["C"]); Q.append(m["Q"]); f.append(m["f"])
        V.append(m["area"]); loc.append(m["localisation"])
    C, Q, f, V, loc = map(np.asarray, (C, Q, f, V, loc))
    d = {"C": C, "Q": Q, "f": f, "V": V, "loc": loc, "n_failed": n_failed,
         "n_steps": tuning_steps, "infeasible": out_of_box, "penalised": False,
         "violations": viol}

    if f.size < 2 or n_failed or (c_cutoff and C.size and C.min() < mcmc.C_FLOOR):
        if verbose:
            print(f"    [fom_single] penalty: n_steps_ok={f.size} "
                  f"n_failed={n_failed} "
                  f"minC={(C.min() if C.size else float('nan')):.4f}", flush=True)
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

# Module state, so the installed mcmc.fom knows what to sample.
# clip MUST agree with install()'s default: the z-bank is cached on (n, D, clip,
# seed, ...), so a different clip here would hand the diagnostics
# (plot_samples / describe_samples / the field plots) a DIFFERENT set of samples
# from the ones the objective is actually scoring.
_CFG = {"cov": None, "n": 6, "clip": 3.0, "common": True, "seed": 0,
        "aggregate": "mean", "tuning_steps": 16, "mesh_size": None,
        # "warn" by default: an out-of-box sample is warned about and SOLVED,
        # which is what mcmc.fom does (it never consulted the limits at all).
        # Use "geometry" for a production MCMC run -- see the module docstring.
        "check_limits": "warn"}


def _resolve_sampling(cov=None, n=None, clip=None, common=None, seed=None):
    """
    THE single place the sampling settings are resolved for the DIAGNOSTICS.

    Every viewer (plot_samples, describe_samples, solve_field_samples) goes
    through this so they all draw the same bank as fom_mean. They used to
    disagree: describe_samples fell through to _sample_proposal's own clip
    default while plot_samples read _CFG, so the two showed different geometries.

    Unlike fom_mean, a missing covariance here falls back to default_cov()
    rather than raising -- a viewer with no install() should still draw something.
    """
    cov = _CFG["cov"] if cov is None else cov
    if cov is None:
        cov = default_cov()
    n = int(_CFG["n"] if n is None else n)
    clip = float(_CFG["clip"] if clip is None else clip)
    common = bool(_CFG["common"] if common is None else common)
    seed = int(_CFG["seed"] if seed is None else seed)
    return np.asarray(cov, dtype=np.float64), n, clip, common, seed


def fom_mean(params, tuning_steps=None, c_cutoff=True, mesh_size=None,
             verbose=False, return_details=False,
             cov=None, n=None, clip=None, common=None, seed=None,
             aggregate=None, check_limits=None):
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

    check_limits : None -> _CFG["check_limits"], default "warn". Under "warn" an
        out-of-box sample is warned about and then SOLVED, so it contributes its
        real scan time to the average instead of a penalty. Under "strict" (the
        old behaviour) it contributes PENALTY, which is what made a design sitting
        just outside the box indistinguishable from a broken one.

        Samples that FAIL TO SOLVE or trip the form-factor floor still return
        PENALTY in every mode and are INCLUDED in the average. That part is
        deliberate: a design whose sweep falls apart under a 0.03 mm error is
        genuinely fragile, and averaging the penalty in is what expresses that.

    NOTE check_limits here is NOT wired to mcmc's check_limits, which screens
    PROPOSALS rather than samples. The two are independent knobs: mcmc's decides
    which designs are ever offered, this one decides how samples of an offered
    design are scored.
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
    mode = _norm_limits_mode(check_limits)

    X = _sample_proposal(params, cov, n=n, clip=clip, common=common, seed=seed)

    vals, dets = [], []
    for j, x_ext in enumerate(X):
        v, d = fom_single(x_ext, tuning_steps=tuning_steps, mesh_size=mesh_size,
                          c_cutoff=c_cutoff, return_details=True,
                          check_limits=mode, label=f"sample {j+1}/{n}")
        vals.append(float(v)); dets.append(d)
        if verbose:
            flag = ("  [penalised: out of limits]" if d.get("penalised") else
                    "  [out of limits, scored]" if d.get("infeasible") else "")
            print(f"    sample {j+1}/{n}: FOM={v:.4g}" + flag
                  + (f"  minC={d['C'].min():.4f}" if d["C"].size else ""),
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
    # RENAMED from "nominal", which it never was: every row of X is perturbed
    # (the z-bank is centred, so no sample sits AT the mean), so vals[0] is the
    # first PERTURBED sample. The unperturbed FOM would cost another sweep.
    details["first_sample"] = float(vals[0]) if len(vals) else np.nan
    # OUT OF BOUNDS and PENALISED are now different numbers: in "warn" mode a
    # sample can be the first without being the second
    details["n_out_of_bounds"] = int(sum(bool(d.get("infeasible")) for d in dets))
    details["n_penalised_limits"] = int(sum(bool(d.get("penalised")) for d in dets))
    details["limits_mode"] = mode
    details["violations"] = {f"sample {j+1}": d.get("violations")
                             for j, d in enumerate(dets) if d.get("infeasible")}
    details["spread_log"] = (float(np.std(np.log(np.maximum(vals, 1e-300)), ddof=1))
                             if len(vals) > 1 else 0.0)
    return value, details


# ═════════════════════════════════════════════════════════════════════════════
# wiring
# ═════════════════════════════════════════════════════════════════════════════

_ORIGINAL_FOM = None


def install(cov=None, n=6, clip=3.0, common=True, seed=0, aggregate="mean",
            tuning_steps=16, mesh_size=None, check_limits="warn", verbose=True):
    """
    Point mcmc.fom at the robust objective. After this, mcmc.mcmc_minimize,
    mcmc.continue_mcmc and mcmc.NM_opt all optimise E[FOM] with no other changes.

    THE DESIGN VECTOR DOES NOT CHANGE. install() rebinds one name; it does not
    touch the parameterisation, the bounds or the CSV format. The same 7-vector
    (or legacy 8-vector) feeds mcmc.fom and fom_mean, embed() supplies the fixed
    gap0/gap1 and the mirrored right-hand parts, and restore() puts the nominal
    objective back. nominal_fom() calls the unperturbed objective even while the
    robust one is installed, and describe_objective() dry-runs both.

    check_limits : "warn" (default), "geometry", "strict"/True, "off"/False.
        See the module docstring. "warn" makes fom_mean agree with mcmc.fom about
        which geometries get a real number.

    CAVEAT on tuning_steps: mcmc._evaluate and NM_opt always pass tuning_steps
    EXPLICITLY (their own default is 16), and an explicit argument beats _CFG. So
    install(tuning_steps=8) changes nothing for a run driven through
    mcmc_minimize -- set it there instead. It does apply when you call fom_mean
    or fom_single directly.

    Returns the config dict.
    """
    global _ORIGINAL_FOM
    if _ORIGINAL_FOM is None:
        _ORIGINAL_FOM = mcmc.fom
    mode = _norm_limits_mode(check_limits)
    _CFG.update(cov=(default_cov() if cov is None else np.asarray(cov, float)),
                n=int(n), clip=float(clip), common=bool(common), seed=int(seed),
                aggregate=aggregate, tuning_steps=int(tuning_steps),
                mesh_size=mesh_size, check_limits=mode)
    mcmc.fom = fom_mean
    if verbose:
        sd = np.sqrt(np.diag(_CFG["cov"]))
        print(f"[noisy] robust objective installed: aggregate={aggregate}, "
              f"n={n}, clip={clip}, common_random_numbers={common}, "
              f"check_limits={mode!r}")
        if mode == "warn":
            print("[noisy] check_limits='warn': out-of-box samples are WARNED "
                  "ABOUT and SOLVED, not penalised.")
            print("[noisy] they therefore cost a full sweep each; use "
                  "check_limits='geometry' to keep")
            print("[noisy] unbuildable/overlapping geometries out while still "
                  "scoring design-box excursions.")
        print(f"[noisy] perturbed dimensions (1 sigma):")
        for nm, s in zip(NOISY_NAMES, sd):
            unit = " deg" if nm in DEGREE_NAMES else " mm"
            print(f"          {nm:<12} {s:.4g}{unit}")
        print(f"[noisy] cost: {n} x {tuning_steps} = {n*tuning_steps} FEM solves "
              f"per objective evaluation")
        print(f"[noisy] note: mcmc_minimize/NM_opt pass tuning_steps explicitly, "
              f"so set it there, not here")
    return dict(_CFG)


def nominal_fom(params, **kw):
    """
    The UNPERTURBED objective, callable even while the robust one is installed.

    mcmc.fom is a module-level name that install() rebinds, so once installed
    there is no way to reach the original through it. This hands back the
    function install() displaced, so the same design vector can be scored both
    ways in one session:

        f_rob = mcmc.fom(x0, tuning_steps=16)     # robust, installed
        f_nom = nz.nominal_fom(x0, tuning_steps=16)   # single-point
    """
    f = _ORIGINAL_FOM if _ORIGINAL_FOM is not None else mcmc.fom
    if f is fom_mean:
        raise RuntimeError("the nominal objective is not recoverable: mcmc.fom "
                           "was already fom_mean when install() ran")
    return f(params, **kw)


def restore(verbose=True):
    """Put the nominal (single-point) objective back."""
    global _ORIGINAL_FOM
    if _ORIGINAL_FOM is not None:
        mcmc.fom = _ORIGINAL_FOM
        _ORIGINAL_FOM = None
        if verbose:
            print("[noisy] nominal objective restored")


def describe_objective(params, cov=None, n=None, clip=None, common=None,
                       seed=None, check_limits=None, tuning_steps=None,
                       verbose=True):
    """
    DRY RUN. What will happen when this design vector is evaluated -- by the
    robust objective and by the nominal one -- WITHOUT running a single FEM
    solve. Use it before committing hours of sweeps.

    Reports:
      * which objective mcmc.fom currently points at, and whether fom_mean can
        accept the call mcmc._evaluate / NM_opt will make (checked with
        inspect.signature, not guessed);
      * whether the DESIGN vector passes mcmc.proposed_params_within_limits, and
        if not, exactly which rules and at what severity;
      * for the configured sample bank, how many samples are clean / out of the
        design box / geometrically broken, and how many would be PENALISED under
        the current mode;
      * the FEM cost.

    Returns the summary dict.
    """
    import inspect

    mode = _norm_limits_mode(check_limits)
    cov, n, clip, common, seed = _resolve_sampling(cov, n, clip, common, seed)
    ts = int(_CFG["tuning_steps"] if tuning_steps is None else tuning_steps)

    p = np.asarray(params, dtype=np.float64).ravel()
    if p.size == 8:
        p = np.delete(p, 3)
    if p.size != N_DESIGN:
        raise ValueError(f"expected {N_DESIGN} design parameters, got {p.size}")

    installed = mcmc.fom is fom_mean
    try:
        # exactly the call mcmc._evaluate and NM_opt.objective make
        inspect.signature(fom_mean).bind(p, tuning_steps=ts, return_details=True)
        sig_ok = True
    except TypeError:
        sig_ok = False

    design_ok = bool(mcmc.proposed_params_within_limits(p))
    nom_v = classify_limits(embed(p))
    X = _sample_proposal(p, cov, n=n, clip=clip, common=common, seed=seed)

    rows, n_pen, n_clean = [], 0, 0
    for j, x in enumerate(X):
        v = classify_limits(x)
        pen = _should_penalise(v, mode)
        n_pen += bool(pen)
        n_clean += not (v["fatal"] or v["geometry"] or v["bounds"])
        rows.append({"sample": j + 1, "penalised": bool(pen),
                     **{k: len(v[k]) for k in SEVERITIES}})

    out = {"installed": installed, "signature_ok": sig_ok, "mode": mode,
           "design_within_limits": design_ok, "nominal_violations": nom_v,
           "samples": rows, "n": n, "n_penalised": n_pen, "n_clean": n_clean,
           "tuning_steps": ts, "fem_solves": n * ts}
    if not verbose:
        return out

    print(f"[noisy] mcmc.fom -> {'fom_mean (robust)' if installed else 'nominal'}"
          f"   |   check_limits = {mode!r}")
    print(f"[noisy] signature compatible with mcmc._evaluate / NM_opt: "
          f"{'yes' if sig_ok else 'NO'}")
    print(f"[noisy] design vector ({N_DESIGN} entries; gap0 = {mcmc.GAP0} and "
          f"gap1 = {mcmc.GAP1} are fixed, not optimised)")
    print(f"          mcmc.proposed_params_within_limits: {design_ok}")
    for k in SEVERITIES:
        for m in nom_v[k]:
            print(f"          [{k}] {m}")
    if not any(nom_v[k] for k in SEVERITIES):
        print("          no violations at any severity")
    print(f"[noisy] sample bank: n = {n}, {n_clean} clean, "
          f"{n - n_clean} out of limits, {n_pen} would be PENALISED under "
          f"{mode!r}")
    if n_pen and mode == "strict":
        print("          -> under 'strict' those samples never reach the solver; "
              "the average is dominated by PENALTY")
    if (n - n_clean) and mode == "warn":
        print("          -> under 'warn' every one of them is solved anyway, so "
              "the value is a real scan time")
    nom_would = "a real FOM (mcmc.fom never consults the limits)"
    print(f"[noisy] the SAME vector under the nominal objective: {nom_would}")
    print(f"[noisy] cost of one robust evaluation: {n} x {ts} = {n * ts} "
          f"FEM solves")
    return out


# ═════════════════════════════════════════════════════════════════════════════
# visualising the perturbations -- GEOMETRY
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

    Draws only the geometry: no FEM is run, so it is instant. At 0.03 mm sigmas
    panel (a) will look identical to nominal at this scale -- that is expected,
    and is why plot_field_difference exists.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    cov, n, clip, common, seed = _resolve_sampling(cov, n, clip, common, seed)

    X = _sample_proposal(params, cov, n=n, clip=clip, common=common, seed=seed)
    mu = embed(params)
    sd = np.sqrt(np.diag(cov))

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
    # the axis limits come from the NOMINAL cavity whether or not it is drawn:
    # the old code unpacked (None, None) here and crashed on show_nominal=False
    _, cav_w, cav_h = parts_of(mu)
    if show_nominal:
        draw(mu, "#1f4e79", 1.0, 1.8, 3)
    axg.set_aspect("equal")
    axg.set_xlim(-0.56 * cav_w, 0.56 * cav_w)
    axg.set_ylim(-0.56 * cav_h, 0.56 * cav_h)
    axg.set_xlabel("x (mm)"); axg.set_ylabel("y (mm)")
    axg.set_title(f"(a) {min(n, max_overlay)} perturbed geometries "
                  f"(red) vs nominal (blue)", fontsize=9)
    axg.grid(alpha=0.3, ls=":")

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
    axk.set_title("(b) realised gaps", fontsize=9)
    axk.grid(alpha=0.3, ls=":"); axk.legend(fontsize=7)

    # ---- (c) per-dimension excursion, in sigma ------------------------------
    live = np.where(sd > 0)[0]
    Z = (X[:, live] - mu[live]) / sd[live]
    lim = float(np.max(np.abs(Z))) if Z.size else 1.0
    # an infinite clip used to be handed straight to axhspan, which destroys the
    # y-limits; only shade a FINITE band, and only then add a legend entry
    if np.isfinite(clip):
        axz.axhspan(-clip, clip, color="0.85", zorder=0,
                    label=f"clip = {clip:g}$\\sigma$ (pre-rescale)")
        lim = max(lim, float(clip))
    for k, idx in enumerate(live):
        axz.plot(np.full(len(Z), k), Z[:, k], "o", color="#b03a2e", ms=3, alpha=0.55)
    axz.axhline(0, color="#1f4e79", lw=1.0)
    axz.set_ylim(-1.15 * lim, 1.15 * lim)
    axz.set_xticks(range(len(live)))
    axz.set_xticklabels([NOISY_NAMES[i] for i in live], rotation=90, fontsize=5.5)
    axz.set_ylabel(r"excursion / $\sigma$")
    axz.set_title("(c) perturbation per dimension", fontsize=9)
    axz.grid(alpha=0.3, ls=":")
    if np.isfinite(clip):
        axz.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig


def describe_samples(params, cov=None, n=None, clip=None, common=None, seed=None,
                     **kw):
    """
    Text companion to plot_samples: the realised spread of the derived quantities
    that actually matter, printed rather than plotted.

    WHY EVERY SD COMES OUT IDENTICAL (and slightly above your input sigma):
    the z-bank is centred and moment-matched, so each perturbed dimension has, by
    construction, sample mean exactly its nominal value and POPULATION sample sd
    exactly its input sigma. The realised spread is therefore DETERMINISTIC, not
    a random draw, and identical for every dimension sharing a sigma. Reporting a
    ddof = 1 sd of those same draws inflates it by sqrt(n / (n - 1)) -- 6.9% at
    n = 8, i.e. 0.03 -> 0.0321 -- which is the estimator, not the sampler.

    Both conventions are printed, next to the sigma you asked for, so the
    comparison is unambiguous. cav_w and cav_h are different: cav_h is a single
    perturbed dimension (so it is pinned exactly), while cav_w is a SUM of eleven
    of them and its spread is not constrained by moment matching.
    """
    cov, n, clip, common, seed = _resolve_sampling(cov, n, clip, common, seed)
    X = _sample_proposal(params, cov, n=n, clip=clip, common=common, seed=seed,
                         **kw)
    sd_in = np.sqrt(np.diag(cov))
    rows = [parts_of(x) for x in X]
    cav_w = np.array([r[1] for r in rows]); cav_h = np.array([r[2] for r in rows])
    G = np.array([[split(x)[1][g] for g in GAP_NAMES] for x in X])

    infl = np.sqrt(n / (n - 1)) if n > 1 else np.nan
    print(f"[noisy] {n} samples, {D_NOISY} perturbed dimensions "
          f"(common_random_numbers={common}, clip={clip:g}, seed={seed})")
    if common:
        print(f"  the z-bank is centred and moment-matched: sd(ddof=0) reproduces "
              f"the input sigma EXACTLY,")
        print(f"  and sd(ddof=1) is larger by sqrt(n/(n-1)) = {infl:.4f}. "
              f"Neither is an error.")
    print(f"  cavity width  {cav_w.mean():8.4f}  sd0 {cav_w.std(ddof=0):.4f}  "
          f"sd1 {cav_w.std(ddof=1):.4f} mm   (derived: sum of 6 gaps + 5 widths)")
    print(f"  cavity height {cav_h.mean():8.4f}  sd0 {cav_h.std(ddof=0):.4f}  "
          f"sd1 {cav_h.std(ddof=1):.4f} mm   "
          f"(sigma_in {sd_in[I_EXT['cav_h']]:.4f})")
    print(f"  {'gap':>10} {'sigma_in':>9} {'mean':>9} {'sd(ddof=0)':>11} "
          f"{'sd(ddof=1)':>11} {'min':>9} {'max':>9}")
    for k, g in enumerate(GAP_NAMES):
        c = G[:, k]
        print(f"  {g:>10} {sd_in[I_EXT[g]]:>9.4f} {c.mean():>9.4f} "
              f"{c.std(ddof=0):>11.4f} {c.std(ddof=1):>11.4f} "
              f"{c.min():>9.4f} {c.max():>9.4f}")
    asym = (np.abs(G[:, 0] - G[:, 5]) + np.abs(G[:, 1] - G[:, 4])
            + np.abs(G[:, 2] - G[:, 3]))
    print(f"  left-right gap asymmetry: {asym.mean():.4f} mm mean "
          f"(0 would be a mirror-symmetric sample)")

    cls = [classify_limits(x) for x in X]
    n_ok = sum(not (c["fatal"] or c["geometry"] or c["bounds"]) for c in cls)
    n_geo = sum(bool(c["fatal"] or c["geometry"]) for c in cls)
    print(f"  feasible samples: {n_ok}/{n}"
          + (f"   ({n_geo} geometrically broken, "
             f"{n - n_ok - n_geo} only outside the design box)"
             if n_ok < n else ""))
    if n_ok < n:
        # a bare count is useless when the answer is 0/n, which is what a design
        # point that is itself out of the box produces
        bad = why_infeasible(embed(params))
        if bad:
            print(f"  the NOMINAL design already violates {len(bad)} "
                  f"rule(s) -- every sample inherits them:")
            for b in bad:
                print(f"      - {b}")
            if all(b.startswith("[bounds]") for b in bad):
                print("  all of these are DESIGN-BOX rules: the geometry itself "
                      "is sound (nothing overlaps,")
                print("  nothing leaves the cavity), so with check_limits='warn' "
                      "it will still be solved.")
        else:
            first = next((why_infeasible(x) for x, c in zip(X, cls)
                          if (c["fatal"] or c["geometry"] or c["bounds"])), [])
            print("  nominal is feasible; the first out-of-limits SAMPLE fails:")
            for b in first:
                print(f"      - {b}")
    return {"cav_w": cav_w, "cav_h": cav_h, "gaps": G, "asymmetry": asym,
            "n_feasible": int(n_ok)}


# ═════════════════════════════════════════════════════════════════════════════
# visualising the perturbations -- FIELD
#
# The geometry views above are useless at 0.03 mm: the perturbation is a tenth
# of a line width. What is NOT small is the effect on E_z, because the operating
# mode lives in the degenerate trio of gaps and per-part errors detune those
# cells against each other. These functions run the real solve and show it.
# ═════════════════════════════════════════════════════════════════════════════

def _best_index(result, min_localisation=0.0):
    """
    Index of the highest-C mode, optionally refusing localised ones.
    Mirrors fem_vis._pick_best so the panels and the numbers agree; kept local
    rather than importing a private name across modules.
    """
    idx = [i for i, m in enumerate(result["modes"])
           if m["localisation"] >= min_localisation]
    if not idx:
        idx = list(range(len(result["modes"])))
    return max(idx, key=lambda i: result["modes"][i]["C"])


def solve_field_samples(params, cov=None, n=None, clip=None, common=None,
                        seed=None, tuning_index=0, tuning_steps=16,
                        mesh_size=None, n_modes=None, n_workers=None,
                        timeout=None, include_nominal=True, verbose=True):
    """
    Solve ONE tuning position for the nominal geometry and every perturbed
    sample, KEEPING THE FIELDS. This is the expensive step the plotters share.

    params        : DESIGN vector (7, or legacy 8), same as everywhere else.
    tuning_index  : which step of the tuning sweep to solve, 0 = untuned
                    (|x| = 0, the highest frequency), tuning_steps-1 = fully
                    tuned. Each sample uses its OWN tuning_positions_ext, so the
                    shift-invert target tracks that sample's gap0L exactly as it
                    does inside fom_single.
    include_nominal : prepend the unperturbed geometry, so it is entries[0] and
                    the difference plot has something to subtract.
    timeout       : passed to fem.run_batch. Default None on purpose --
                    run_batch's timeout is a budget for the WHOLE batch (it goes
                    to as_completed), not per solve, and overrunning it raises
                    rather than marking failures.

    Returns (entries, info):
      entries : [(spec, result, label), ...] exactly as fem_vis.plot_best_modes*
                wants, with failed solves DROPPED.
      info    : {"meta": [...], "dropped": [...], "X": ..., "labels": ...},
                where meta[i] matches entries[i] and carries x_ext, dx, dy and
                the frequency target.

    Cost: (n + 1) FEM solves, run as one parallel batch.
    """
    cov, n, clip, common, seed = _resolve_sampling(cov, n, clip, common, seed)
    mesh_size = _CFG["mesh_size"] if mesh_size is None else mesh_size
    n_modes = mcmc.N_MODES if n_modes is None else int(n_modes)

    X = _sample_proposal(params, cov, n=n, clip=clip, common=common, seed=seed)
    labels = [f"sample {j + 1}" for j in range(len(X))]
    if include_nominal:
        X = np.vstack([embed(params)[None, :], X])
        labels = ["nominal"] + labels

    specs, targets, meta = [], [], []
    for lab, x in zip(labels, X):
        pos = list(tuning_positions_ext(x, n=tuning_steps))
        if not 0 <= int(tuning_index) < len(pos):
            raise IndexError(f"tuning_index {tuning_index} outside "
                             f"[0, {len(pos) - 1}] for tuning_steps={tuning_steps}")
        dx, dy, fg = pos[int(tuning_index)]
        specs.append(make_spec_ext(x, toast_dx=dx, toast_dy=dy,
                                   mesh_size=mesh_size, tag=lab))
        targets.append(fg)
        meta.append({"label": lab, "x_ext": np.asarray(x, dtype=np.float64),
                     "dx": dx, "dy": dy, "f_target": fg,
                     "feasible": bool(within_limits_ext(x))})

    if verbose:
        dx_mm = meta[0]["dx"] * 1e3
        print(f"[field] {len(specs)} geometries at tuning step {tuning_index} "
              f"(|x| = {abs(dx_mm):.2f} mm, f_guess ~ "
              f"{meta[0]['f_target']/1e9:.3f} GHz), one batch", flush=True)

    results = fem.run_batch(specs, n_modes=n_modes, f_target=targets,
                            n_workers=n_workers, timeout=timeout,
                            verbose=False, keep_fields=True)

    entries, kept_meta, dropped = [], [], []
    for spec, r, m in zip(specs, results, meta):
        if not r or not r.get("ok") or not r.get("modes") or "fields" not in r:
            dropped.append((m["label"], (r or {}).get("error", "no modes")))
            continue
        entries.append((spec, r, m["label"]))
        kept_meta.append(m)
    if verbose and dropped:
        print(f"[field] {len(dropped)} geometry/geometries failed to solve:",
              flush=True)
        for lab, err in dropped:
            print(f"          {lab}: {err}", flush=True)
    return entries, {"meta": kept_meta, "dropped": dropped, "X": X,
                     "labels": labels, "tuning_index": int(tuning_index),
                     "tuning_steps": int(tuning_steps)}


def field_table(entries, min_localisation=0.0, verbose=True):
    """
    The scalar companion to the field pictures: f, C, Q and localisation of the
    operating mode for each entry, plus the fractional change against the FIRST
    entry (the nominal, if solve_field_samples was called with include_nominal).
    """
    rows = []
    for spec, r, lab in entries:
        i = _best_index(r, min_localisation)
        md = r["modes"][i]
        rows.append({"label": lab, "mode": i, "f": float(r["freqs"][i]),
                     "C": float(md["C"]), "Q": float(md["Q"]),
                     "loc": float(md["localisation"])})
    if rows:
        ref = rows[0]
        for rw in rows:
            rw["df_MHz"] = (rw["f"] - ref["f"]) / 1e6
            rw["dC_pct"] = 100.0 * (rw["C"] - ref["C"]) / ref["C"] if ref["C"] else np.nan
            rw["dQ_pct"] = 100.0 * (rw["Q"] - ref["Q"]) / ref["Q"] if ref["Q"] else np.nan
    if verbose and rows:
        print(f"  {'label':>10} {'f (GHz)':>9} {'df (MHz)':>9} {'C':>8} "
              f"{'dC (%)':>8} {'Q':>10} {'dQ (%)':>8} {'loc':>6}")
        for rw in rows:
            print(f"  {rw['label']:>10} {rw['f']/1e9:>9.4f} {rw['df_MHz']:>9.2f} "
                  f"{rw['C']:>8.4f} {rw['dC_pct']:>8.3f} {rw['Q']:>10.4g} "
                  f"{rw['dQ_pct']:>8.3f} {rw['loc']:>6.3f}")
    return rows


def plot_field_samples(params, solved=None, save=None, ncol=4, cmap="RdBu_r",
                       min_localisation=0.0, share_scale=False, suptitle=None,
                       print_table=True, **solve_kw):
    """
    Panel grid of |E_z|^2 for the nominal geometry and every perturbed sample at
    one tuning position, via fem_vis.plot_best_modes_magnitude_square.

    This is the SHAPE view. solve_cavity normalises every eigenvector to peak
    |E| = 1, so the panels are all on the same peak scale whatever share_scale
    says, and what you are comparing is where the field sits, not how much of it
    there is. For the amplitude-level picture use plot_field_difference, which
    rescales to a fixed stored energy.

    cmap defaults to a SEQUENTIAL map: |E|^2 >= 0, and fem_vis's own default of
    RdBu_r puts the neutral colour in the middle of a one-sided range, which
    reads as if there were negative values.

    solved : reuse the (entries, info) tuple from solve_field_samples instead of
             re-solving -- worth doing when you want both plots.

    Returns (fig, entries, info).
    """
    entries, info = (solved if solved is not None
                     else solve_field_samples(params, **solve_kw))
    if not entries:
        raise RuntimeError("no geometry solved successfully; see info['dropped']")
    if print_table:
        print("[field] operating mode per geometry:")
        field_table(entries, min_localisation=min_localisation)
    ti = info.get("tuning_index", 0)
    fig = viz.plot_best_modes_magnitude_square(
        entries, save=save, cmap=cmap, ncol=ncol,
        min_localisation=min_localisation,
        suptitle=(suptitle or f"$|E_z|^2$ of the operating mode, tuning step "
                              f"{ti} -- nominal vs perturbed "
                              f"(each panel peak-normalised)"),
        share_scale=share_scale)
    return fig, entries, info


def field_on_grid(result, XX, YY, i=None, normalize="energy", energy=1.0,
                  min_localisation=0.0):
    """
    |E_z|^2 of one mode sampled on a REGULAR GRID, NaN outside the mesh (i.e.
    inside the metal bars and beyond the cavity wall).

    This exists because every perturbed geometry gets its own mesh, so two
    results have no common set of nodes and a nodal difference is undefined.
    Interpolating both onto a shared grid is the cheapest honest way to subtract
    them.

    normalize : passed to fem_solve.field_scale.
        "energy" (default) puts every mode at the same stored energy (J per metre
        of cavity length), so E is in V/m and the panels are physically
        comparable. "peak" leaves the eigenvector's arbitrary peak = 1
        normalisation alone and compares shape only.

    The field is interpolated LINEARLY from the P2 vertex values (vertex dofs
    come first in skfem's ordering) and squared afterwards, which is a little
    smoother than interpolating the square.
    """
    from matplotlib.tri import Triangulation, LinearTriInterpolator
    if i is None:
        i = _best_index(result, min_localisation)
    m = result["mesh"]
    nv = m.p.shape[1]
    u = np.asarray(result["fields"][i][:nv], dtype=np.float64)
    scale, unit = fem.field_scale(result, i, normalize=normalize, energy=energy)
    tri = Triangulation(m.p[0], m.p[1], m.t.T)
    interp = LinearTriInterpolator(tri, scale * u)
    ui = np.ma.filled(interp(XX, YY), np.nan)
    return ui ** 2, unit, i


def plot_field_difference(params, solved=None, save=None, ncol=4, nx=420,
                          normalize="energy", energy=1.0, cmap="RdBu_r",
                          min_localisation=0.0, pct=99.5, dpi=160,
                          show_outline=True, print_table=True, **solve_kw):
    """
    Delta|E_z|^2 = (perturbed) - (nominal), one panel per sample, on a grid
    shared by every geometry.

    This is the plot that actually shows a 0.03 mm error: the geometries are
    indistinguishable, the fields are not. Red/blue is where the mode has moved
    energy INTO / OUT OF a cell relative to nominal, and a systematic left-right
    imbalance is the symmetry breaking the whole 33-dimensional model exists to
    capture.

    normalize="energy" (default) fixes the stored energy of every mode, so the
    difference is physical rather than an artefact of eigenvector scaling. Note
    the two modes are then compared at equal stored energy, not at equal drive.

    pct   : colour limits are +/- the `pct` percentile of |Delta|, so one hot
            pixel at a re-entrant corner does not flatten the rest.
    nx    : grid columns; rows follow from the aspect ratio.

    White regions are where the two geometries do not overlap (metal in one and
    vacuum in the other) plus the metal itself -- the difference is genuinely
    undefined there, not zero.

    Returns (fig, stats) with stats carrying per-sample RMS and peak changes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    entries, info = (solved if solved is not None
                     else solve_field_samples(params, **solve_kw))
    if not entries:
        raise RuntimeError("no geometry solved successfully; see info['dropped']")
    if entries[0][2] != "nominal":
        raise RuntimeError("the nominal geometry is not entries[0] -- either it "
                           "failed to solve or include_nominal was False; there "
                           "is nothing to difference against.")
    if print_table:
        print("[field] operating mode per geometry:")
        field_table(entries, min_localisation=min_localisation)

    # grid over the NOMINAL cavity, in metres
    nom_spec, nom_res, _ = entries[0]
    x0, y0, x1, y1 = nom_spec.extent
    ny = max(8, int(round(nx * (y1 - y0) / (x1 - x0))))
    xs = np.linspace(x0, x1, int(nx)); ys = np.linspace(y0, y1, int(ny))
    XX, YY = np.meshgrid(xs, ys)
    ext_mm = [x0 * 1e3, x1 * 1e3, y0 * 1e3, y1 * 1e3]

    base, unit, i_nom = field_on_grid(nom_res, XX, YY, normalize=normalize,
                                      energy=energy,
                                      min_localisation=min_localisation)

    diffs, stats = [], []
    for (spec, r, lab), meta in zip(entries[1:], info["meta"][1:]):
        val, _u, i_mode = field_on_grid(r, XX, YY, normalize=normalize,
                                        energy=energy,
                                        min_localisation=min_localisation)
        d = val - base
        diffs.append((lab, d, spec, meta))
        finite = np.isfinite(d)
        peak = float(np.nanmax(base)) if np.isfinite(base).any() else np.nan
        stats.append({"label": lab, "mode": i_mode,
                      "rms": float(np.sqrt(np.nanmean(d[finite] ** 2)))
                      if finite.any() else np.nan,
                      "max_abs": float(np.nanmax(np.abs(d))) if finite.any() else np.nan,
                      "rms_rel_peak": (float(np.sqrt(np.nanmean(d[finite] ** 2)) / peak)
                                       if finite.any() and peak else np.nan),
                      "overlap_frac": float(finite.mean())})
    if not diffs:
        raise RuntimeError("only the nominal geometry solved; nothing to compare")

    allpos = np.abs(np.concatenate([d[np.isfinite(d)].ravel() for _l, d, _s, _m in diffs]))
    lim = float(np.percentile(allpos, pct)) if allpos.size else 1.0
    if not np.isfinite(lim) or lim <= 0:
        lim = 1.0

    ncol = min(int(ncol), len(diffs))
    nrow = int(np.ceil(len(diffs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.7 * nrow),
                             squeeze=False)

    def outline(ax, x_ext, dx, dy, color, lw):
        parts, cw, ch = parts_of(x_ext)
        ax.add_patch(plt.Rectangle((-cw / 2, -ch / 2), cw, ch, fill=False,
                                   ec=color, lw=lw, zorder=4))
        for p in parts:
            cx = p["cx"] + (dx * 1e3 if p["moves"] else 0.0)
            cy = p["cy"] + (dy * 1e3 if p["moves"] else 0.0)
            r = fem.Rect.from_center(cx, cy, p["w"], p["h"], p["name"], p["theta"])
            ax.add_patch(Polygon(r.corners(), closed=True, fill=False, ec=color,
                                 lw=lw, zorder=4))

    nom_meta = info["meta"][0]
    for k, (lab, d, spec, meta) in enumerate(diffs):
        ax = axes[k // ncol][k % ncol]
        im = ax.imshow(d, origin="lower", extent=ext_mm, cmap=cmap,
                       vmin=-lim, vmax=lim, interpolation="nearest", zorder=1)
        if show_outline:
            outline(ax, nom_meta["x_ext"], nom_meta["dx"], nom_meta["dy"],
                    "#111111", 0.8)
        st = stats[k]
        ax.set_title(f"{lab}\nRMS $\\Delta$ = {st['rms']:.3g}, "
                     f"peak {st['max_abs']:.3g}", fontsize=8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=0.04)
        cb.set_label(r"$\Delta$" + unit, fontsize=7)
        cb.ax.tick_params(labelsize=6)
    for j in range(len(diffs), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    ti = info.get("tuning_index", 0)
    fig.suptitle(r"$|E_z|^2$ change from the nominal geometry, tuning step "
                 f"{ti}  (normalise = {normalize})", fontsize=13, y=1.0)
    fig.tight_layout()
    if print_table:
        print(f"[field] colour limits +/- {lim:.4g} ({pct} percentile of "
              f"|Delta|); mode {i_nom} chosen for the nominal")
        for st in stats:
            print(f"          {st['label']:>10}: RMS {st['rms']:.4g}  "
                  f"peak {st['max_abs']:.4g}  "
                  f"RMS/peak(nominal) {st['rms_rel_peak']:.3%}  "
                  f"overlap {st['overlap_frac']:.1%}")
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig, {"stats": stats, "lim": lim, "unit": unit, "info": info,
                 "entries": entries}


def field_report(params, save_prefix=None, tuning_index=0, print_table=True,
                 **solve_kw):
    """
    One call, both field views (and the scalar table), from a SINGLE batch of
    solves -- the intended entry point.

        nz.field_report(x0, n=5, save_prefix="TEMP/field")

    writes <prefix>_panels.png and <prefix>_diff.png.
    """
    solved = solve_field_samples(params, tuning_index=tuning_index, **solve_kw)
    fig_p, _e, info = plot_field_samples(
        params, solved=solved, print_table=print_table,
        save=(f"{save_prefix}_panels.png" if save_prefix else None))
    fig_d, stats = plot_field_difference(
        params, solved=solved, print_table=print_table,
        save=(f"{save_prefix}_diff.png" if save_prefix else None))
    return {"panels": fig_p, "difference": fig_d, "stats": stats,
            "entries": solved[0], "info": info}


# ═════════════════════════════════════════════════════════════════════════════
# reading
# ═════════════════════════════════════════════════════════════════════════════

def read_stability_csv(path, expect_names=None, max_rows=None,
                       feasible_only=False, drop_penalty=True):
    """
    -> (X, meta) with X of shape (n_rows, 33) holding the extended vectors.

    feasible_only : keep only rows whose `feasible` column is True.
    drop_penalty  : discard rows at the PENALTY value; their sweep is not a
                    meaningful curve (the geometry was rejected outright).
    """
    names = list(NOISY_NAMES) if expect_names is None else list(expect_names)
    D = len(names)
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError(f"{path} is empty")
    header, body = rows[0], rows[1:]
    tail = header[-D:]
    if tail != names:
        raise ValueError(
            f"the last {D} columns of {path} are not the expected parameter "
            f"vector.\n  found:    {tail}\n  expected: {names}")
    idx = {nm: i for i, nm in enumerate(header)}      # first occurrence wins

    # a MISSING feasible column used to fall through idx.get("feasible", -1) to
    # column -1, which is the last PARAMETER, silently marking everything
    # infeasible
    i_feas = idx.get("feasible")
    if feasible_only and i_feas is None:
        raise ValueError(f"feasible_only=True but {path} has no 'feasible' column")

    X, meta = [], []
    for r in body:
        if not r or len(r) < len(header):
            continue
        feas = (True if i_feas is None
                else str(r[i_feas]).strip().lower() in ("true", "1"))
        fom = float(r[idx["fom"]]) if "fom" in idx else np.nan
        if feasible_only and not feas:
            continue
        if drop_penalty and np.isfinite(fom) and fom >= mcmc.PENALTY * (1 - 1e-12):
            continue
        X.append([float(v) for v in r[-D:]])
        meta.append({"sample": r[idx["sample"]] if "sample" in idx else len(meta),
                     "fom": fom, "feasible": feas})
        if max_rows and len(X) >= max_rows:
            break
    return np.asarray(X, dtype=np.float64), meta


# ═════════════════════════════════════════════════════════════════════════════
# sweeping
# ═════════════════════════════════════════════════════════════════════════════

def sweep_many(X_ext, tuning_steps=16, mesh_size=None, n_workers=None,
               min_localisation=0.0, timeout=600, verbose=True):
    """
    Tuning sweep of every geometry in X_ext (n_geom, 33).

    Returns a list of dicts with x_mm, C, Q, f (all length tuning_steps, NaN at a
    position that failed to solve). NaNs rather than dropped points, so the curves
    stay aligned with the tuning position on the x-axis.
    """
    X_ext = np.atleast_2d(np.asarray(X_ext, dtype=np.float64))
    n_geom = X_ext.shape[0]

    specs, targets, owner = [], [], []
    xs = np.zeros((n_geom, tuning_steps))
    for gi, x in enumerate(X_ext):
        for si, (dx, dy, fg) in enumerate(tuning_positions_ext(x, n=tuning_steps)):
            specs.append(make_spec_ext(x, toast_dx=dx, toast_dy=dy,
                                       mesh_size=mesh_size,
                                       tag=f"g{gi}s{si}"))
            targets.append(fg)
            owner.append((gi, si))
            xs[gi, si] = abs(dx) * 1e3            # |x| in mm
    if verbose:
        print(f"[sweep] {n_geom} geometries x {tuning_steps} positions "
              f"= {len(specs)} FEM solves, submitted as one batch", flush=True)

    results = fem.run_batch(specs, n_modes=mcmc.N_MODES, f_target=targets,
                            n_workers=n_workers, timeout=timeout, verbose=False)

    out = [{"x_mm": xs[g], "C": np.full(tuning_steps, np.nan),
            "Q": np.full(tuning_steps, np.nan),
            "f": np.full(tuning_steps, np.nan), "n_failed": 0}
           for g in range(n_geom)]
    for (gi, si), r in zip(owner, results):
        if not r or not r.get("ok") or not r.get("modes"):
            out[gi]["n_failed"] += 1
            continue
        m = fem.best_mode(r, min_localisation=min_localisation)
        if m is None:
            out[gi]["n_failed"] += 1
            continue
        out[gi]["C"][si] = m["C"]; out[gi]["Q"][si] = m["Q"]; out[gi]["f"][si] = m["f"]
    if verbose:
        bad = sum(o["n_failed"] for o in out)
        print(f"[sweep] done; {bad} position(s) failed to solve", flush=True)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# plotting
# ═════════════════════════════════════════════════════════════════════════════

_QUANTITIES = (("C", "Form factor $C$", 1.0),
               ("Q", "Quality factor $Q$", 1.0),
               ("f", r"Frequency $\nu$ (GHz)", 1e-9))


def plot_sweeps(nominal, perturbed, save_prefix=None, dpi=200,
                alpha=0.35, lw_pert=0.9, combined=True, title=None):
    """
    One overlaid plot per quantity: every geometry on the same axes.

    nominal  : a single sweep dict (drawn bold, on top).
    perturbed: list of sweep dicts (drawn faded, behind).

    Writes <prefix>_C.png, <prefix>_Q.png, <prefix>_nu.png, and if combined,
    <prefix>_all.png with the three stacked and sharing the x-axis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_NOM, C_PERT = "#1f4e79", "#b03a2e"
    figs = {}

    def draw(ax, key, label, scale):
        for k, s in enumerate(perturbed):
            ax.plot(s["x_mm"], s[key] * scale, "-", color=C_PERT, lw=lw_pert,
                    alpha=alpha, zorder=1,
                    label=(f"perturbed (n={len(perturbed)})" if k == 0 else None))
        ax.plot(nominal["x_mm"], nominal[key] * scale, "o-", color=C_NOM,
                lw=2.4, ms=5, zorder=3, label="nominal")
        ax.set_ylabel(label, fontsize='large')
        ax.grid(alpha=0.3, ls=":")

    for key, label, scale in _QUANTITIES:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        draw(ax, key, label, scale)
        if key == "C":
            ax.axhline(mcmc.C_FLOOR, color="k", ls=":", lw=1.2,
                       label=f"C floor = {mcmc.C_FLOOR}")
        ax.set_xlabel("Tuning position $|x|$ (mm)", fontsize='large')
        ax.legend(fontsize=8)
        if title:
            ax.set_title(title, fontsize=10)
        fig.tight_layout()
        figs[key] = fig
        if save_prefix:
            nm = "nu" if key == "f" else key
            fig.savefig(f"{save_prefix}_{nm}.png", dpi=dpi, bbox_inches="tight")

    if combined:
        fig, axes = plt.subplots(3, 1, figsize=(6.6, 9.2), sharex=True)
        for ax, (key, label, scale) in zip(axes, _QUANTITIES):
            draw(ax, key, label, scale)
        axes[0].legend(fontsize=8)
        axes[-1].set_xlabel("Tuning position $|x|$ (mm)", fontsize='large')
        fig.suptitle(title or "Geometric parameters vs. tuning position $|x|$",
                     fontsize=20)
        fig.tight_layout()
        figs["all"] = fig
        if save_prefix:
            fig.savefig(f"{save_prefix}_all.png", dpi=dpi, bbox_inches="tight")
    if save_prefix:
        for f in figs.values():
            plt.close(f)
    return figs


# ═════════════════════════════════════════════════════════════════════════════
# driver
# ═════════════════════════════════════════════════════════════════════════════

def noisy_plots(x0, csv_path, tuning_steps=16, mesh_size=None, n_workers=None,
                save_prefix=None, max_rows=None, feasible_only=False,
                alpha=0.35, verbose=True):
    """
    x0 : the NOMINAL design vector (7 entries, or legacy 8). Embedded via
         noisy_mcmc.embed, so the nominal curve is the unperturbed geometry.
    """
    X, meta = read_stability_csv(csv_path, max_rows=max_rows,
                                 feasible_only=feasible_only)
    if verbose:
        print(f"[main] {len(X)} perturbed geometries from {csv_path}")
    x_nom = embed(x0)
    allX = np.vstack([x_nom[None, :], X])
    sweeps = sweep_many(allX, tuning_steps=tuning_steps, mesh_size=mesh_size,
                        n_workers=n_workers, verbose=verbose)
    nominal, perturbed = sweeps[0], sweeps[1:]

    if save_prefix:
        os.makedirs(os.path.dirname(os.path.abspath(save_prefix)) or ".",
                    exist_ok=True)
    plot_sweeps(nominal, perturbed, save_prefix=save_prefix, alpha=alpha)
    if verbose and save_prefix:
        print(f"[main] wrote {save_prefix}_C.png, _Q.png, _nu.png, _all.png")
        C = np.array([np.nanmean(s["C"]) for s in perturbed])
        print(f"[main] mean C: nominal {np.nanmean(nominal['C']):.4f}  "
              f"perturbed {np.nanmean(C):.4f} +/- {np.nanstd(C):.4f}")
    return {"nominal": nominal, "perturbed": perturbed, "meta": meta}