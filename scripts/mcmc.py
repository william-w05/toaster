"""
Multi-walker simulated-annealing MCMC driving the FEM cavity solver.

UNITS -- the one thing to keep straight
    The MCMC (parameters, proposals, constraints, CSVs) works entirely in
    MILLIMETERS, matching proposed_params_within_limits (gap0 = gap1 = 10, cy = 160).
    fem_solve works entirely in METERS. The conversion happens in exactly one
    place, _params_to_m(), at the boundary. Nothing else converts.
    params[0] is an ANGLE in degrees and is never scaled.

PARAMETER VECTOR -- 7 entries (gap1 is fixed, no longer optimised)
    0 angle | 1 div_h | 2 div_w | 3 ctr_w | 4 side_w | 5 ctr_h | 6 side_h
    Index them through the I_* constants, never with literals: dropping gap1
    shifted every index above 2, and a literal 6 that used to mean ctr_h now
    means side_h. Legacy 8-vectors are accepted anywhere and the gap1 entry
    stripped, so old CSVs and starting points still load.

CSV ATOMICITY
    Rows are committed one COMPLETE MCMC iteration at a time. Interrupting
    mid-iteration (some walkers stepped, others not) discards that iteration
    entirely rather than leaving a ragged partial group in fem_evals.csv.
    This now covers ANY exception, not just Ctrl-C: the iteration is rolled
    back, everything complete is flushed, and the exception is re-raised.

check_limits
    Every entry point takes check_limits (default True). With it OFF the
    proposal screen no longer calls proposed_params_within_limits, so the chain
    may visit geometries that are not buildable. Two consequences worth knowing:
      * an infeasible geometry can mesh and solve perfectly well and return an
        excellent FOM, and the chain will happily converge onto it. Every
        fem_evals row carries a WithinLimits column so the best FEASIBLE point
        visited is always recoverable afterwards.
      * the Surrogate.fit rationale for exclude_penalty changes -- see there.
"""

import os
import csv
import time
import re as _re
from collections import deque

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from . import fem_solve as fem
from . import fem_vis as viz

_NUM = _re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


# ═════════════════════════════════════════════════════════════════════════════
# constants
# ═════════════════════════════════════════════════════════════════════════════

# ── geometry, MILLIMETERS ───────────────────────────────────────────────────
GAP0          = 10.0     # fixed gap flanking the centre toast
GAP1          = 10.0     # FIXED: no longer a free parameter
CAVITY_HEIGHT = 160.0
X_MAX_FREQ    = 8.75     # |x| that tunes 15 GHz -> 8 GHz
F_MAX         = 3e11 / (2.0 * GAP0)          # 15 GHz at x=0 (c = 3e11 mm/s)

MM = 1e-3                # millimeters -> meters
GAP0_M   = GAP0 * MM
GAP1_M   = GAP1 * MM
CAV_H_M  = CAVITY_HEIGHT * MM
X_MAX_M  = X_MAX_FREQ * MM

# ── the parameter vector ────────────────────────────────────────────────────
PARAM_NAMES = ["angle", "div_h", "div_w", "ctr_w", "side_w", "ctr_h", "side_h"]
N_PARAMS    = 7
I_ANGLE, I_DIVH, I_DIVW, I_CTRW, I_SIDEW, I_CTRH, I_SIDEH = range(N_PARAMS)

# ── bounds (MILLIMETERS / degrees) ──────────────────────────────────────────
ANGLE_MIN, ANGLE_MAX   = 0.0, 50.0     # was [0, 70]
H_MIN, H_MAX           = 90.0, 145.0   # ALL heights: div_h, ctr_h, side_h
H_TOL                  = 0.2           # div_h, side_h within +/-H_TOL of ctr_h
CTR_W_MIN, CTR_W_MAX   = 3.0, 20.0
SIDE_W_MIN, SIDE_W_MAX = 3.0, 20.0     # side width capped at 20 mm
SIDE_W_TOL             = 0.4           # side_w within +/-SIDE_W_TOL of ctr_w
DIV_W_MIN              = 3.0           # div_w in [3, gap0)
# NOTE: TOTAL_W_MAX ~ 282.8 mm, but the widest geometry the other bounds allow is
# 20 + 2*10 + 2*10 + 4*10 + 2*20 = 140 mm, so this test can never fire. Kept as
# documentation of the physical envelope; it would bind again if the width caps
# were relaxed.
TOTAL_W_MAX            = 400.0 / np.sqrt(2.0)

# ── solver settings ─────────────────────────────────────────────────────────
MESH_SIZE     = 0.001    # METERS
N_MODES       = 6
SWEEP_WORKERS = None     # None -> every core (the sweep is the parallel part)
STEP_TIMEOUT  = 600      # s per tuning position

ALUMINIUM = fem.Material("aluminium", sigma=fem.SIGMA_AL_COMSOL)   # 3.774e7 S/m

# ── annealing / objective ───────────────────────────────────────────────────
TEMP0    = 1.0
COOLING  = 0.999
TEMP_MIN = 1e-3
PENALTY  = 1e33
C_FLOOR  = 0.05          # reject a geometry whose worst-step form factor is below this

# ── surrogate defaults ──────────────────────────────────────────────────────
# The schedule is counted in MCMC STEPS, not evaluations: one step is n_walkers
# evaluations, so with n walkers the first fit sees ~100n evaluations and each
# refit adds ~50n. Keying on steps keeps the cadence identical however many
# walkers you run. (Exactly: the baseline evaluations are also observed, so the
# first fit sees min_steps*n + n = (min_steps + 1)*n points.)
SURROGATE_MIN_STEPS     = 100    # MCMC steps before the FIRST fit
SURROGATE_RETRAIN_EVERY = 50     # refit every this many MCMC steps
SURROGATE_EPOCHS        = 1000   # epochs per fit
SURROGATE_PROGRESS_EVERY = 100   # print training RMSE every this many epochs
SURROGATE_MIN_SAMPLES   = 20     # hard floor: never fit on fewer points than this
SURROGATE_HIDDEN        = 128
SURROGATE_LR            = 3e-4
SURROGATE_BUFFER        = 20000

# hard stop on the proposal retry loop, so an unrecoverable walker raises a clear
# error instead of spinning forever
MAX_BATCH_RETRIES = 1000

_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ═════════════════════════════════════════════════════════════════════════════
# helpers
# ═════════════════════════════════════════════════════════════════════════════

def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _fmt_params(p) -> str:
    return ", ".join(f"{n}={v:.4g}" for n, v in zip(PARAM_NAMES, np.ravel(p)))


def _fmt_arr(p):
    """One-line array repr for the CSVs. np.array2string wraps at 75 characters
    by default, which puts newlines INSIDE a csv field -- legal, but awkward to
    read and grep. The parser copes with either form, so old files still load."""
    return np.array2string(np.asarray(p, dtype=np.float64), precision=8,
                           separator=",", max_line_width=10**6)


def _to_obj(value) -> float:
    """Physical objective -> log objective used by Metropolis and the surrogate."""
    v = float(value)
    if (not np.isfinite(v)) or v <= 0.0:
        return float(np.log(PENALTY))
    return float(np.log(v))


def _as_7(params_mm):
    """Coerce a 7- or legacy 8-vector to the 7-vector, in MILLIMETERS."""
    p = np.asarray(params_mm, dtype=np.float64).ravel()
    if p.size == 8:                       # legacy: strip the gap1 entry
        p = np.delete(p, 3)
    if p.size != N_PARAMS:
        raise ValueError(f"expected {N_PARAMS} parameters, got {p.size}")
    return p


def _params_to_m(params_mm):
    """THE unit boundary: mm -> m for lengths, angle untouched."""
    p = _as_7(params_mm)
    return np.concatenate([[p[0]], p[1:] * MM])


def _safe_log(p):
    """
    log for the proposal transform. ANGLE_MIN is 0, so params[0] can approach
    zero and log(0) = -inf would poison the whole proposal batch.

    LOSSY BY CONSTRUCTION: any value <= 0 is clamped to 1e-12, so exp(_safe_log(x))
    does NOT round-trip for non-positive x. Callers in "linear" mode must pass the
    physical point through _batch_proposals(params=...) rather than relying on the
    round-trip -- see the note there.
    """
    return np.log(np.maximum(np.asarray(p, dtype=np.float64), 1e-12))


# ═════════════════════════════════════════════════════════════════════════════
# geometry / sweep
# ═════════════════════════════════════════════════════════════════════════════

def make_spec(params_mm, toast_dx=0.0, toast_dy=0.0, mesh_size=MESH_SIZE,
              tag="toaster", mesh_uniform=False):
    """
    CavitySpec at one tuning position. toast_dx/dy are in METERS and move ALL
    THREE TOASTS (the dividers stay fixed). gap1 is FIXED at GAP1_M and passed
    explicitly, since it is no longer carried in the parameter vector.
    """
    return viz.toaster_spec(
        _params_to_m(params_mm),          # <- conversion happens here, once
        gap0=GAP0_M, gap1=GAP1_M, cavity_h=CAV_H_M,
        toast_dx=toast_dx, toast_dy=toast_dy,
        mesh_size=mesh_size, tag=tag,
        wall_material=ALUMINIUM,
        metal_material=ALUMINIUM,
        mesh_uniform=mesh_uniform,
    )


def tuning_positions(params_mm, n=16):
    """
    Yields (dx, dy, f_guess) with dx/dy in METERS and f_guess in Hz.
    |x| sweeps 0 -> X_MAX_M and y = |x|*tan(theta); the frequency depends only on
    |x|, so f = c / (2*(gap0 + |x|)) is the shift-invert target for that step.
    """
    p = _as_7(params_mm)
    t = np.tan(np.radians(float(p[I_ANGLE])))
    for x in -np.linspace(0.0, X_MAX_M, n):
        yield float(x), float(abs(x) * t), 3e8 / (2.0 * (GAP0_M + abs(x)))


def sim_sweep(params_mm, tuning_steps=16, mesh_size=MESH_SIZE, verbose=False,
              plot_all=False, mesh_uniform=False):
    """
    Solve the full tuning sweep in parallel and return the operating mode at each
    position.

    Returns dict with arrays C, Q, f, V (length = number of SUCCESSFUL steps)
    plus n_failed. V is the cavity cross-sectional area in m^2 (the 2D stand-in
    for mode volume, per unit length).
    """
    positions = list(tuning_positions(params_mm, n=tuning_steps))
    specs, results = fem.run_sweep(
        lambda dx, dy, i: make_spec(params_mm, toast_dx=dx, toast_dy=dy,
                                    mesh_size=mesh_size,
                                    tag=f"x={dx*1e3:.2f}mm",
                                    mesh_uniform=mesh_uniform),
        positions,
        n_modes=N_MODES,
        n_workers=SWEEP_WORKERS,
        timeout=STEP_TIMEOUT,
        plot_all=plot_all,          # fields are only needed for plotting
        verbose=False,
    )

    C, Q, f, V, loc = [], [], [], [], []
    n_failed = 0
    for (x, y, fg), r in zip(positions, results):
        if not r["ok"] or not r.get("modes"):
            n_failed += 1
            continue
        m = fem.best_mode(r)
        if m is None:
            n_failed += 1
            continue
        C.append(m["C"]); Q.append(m["Q"]); f.append(m["f"])
        V.append(m["area"]); loc.append(m["localisation"])
        if verbose:
            print(f"  x={x*1e3:6.2f} y={y*1e3:6.2f}  guess={fg/1e9:7.3f}  "
                  f"f={m['f']/1e9:7.3f} GHz  C={m['C']:.4f}  Q={m['Q']:8.4g}  "
                  f"loc={m['localisation']:.3f}", flush=True)

    return {"C": np.asarray(C), "Q": np.asarray(Q), "f": np.asarray(f),
            "V": np.asarray(V), "loc": np.asarray(loc), "n_failed": n_failed,
            "n_steps": tuning_steps}


# ═════════════════════════════════════════════════════════════════════════════
# figure of merit
# ═════════════════════════════════════════════════════════════════════════════

def fom(params_mm, tuning_steps=16, c_cutoff=True, mesh_size=MESH_SIZE,
        verbose=False, return_details=False):
    """
    Scan time (lower is better):  T propto integral f^2 / (V^2 C^2 Q) df

    Trapezoid over the tuning band. NOTE abs(df): f DECREASES along the sweep
    (15 -> 8 GHz), so a raw f[1:]-f[:-1] is negative and the integral comes out
    negative -- which the Metropolis test reads as non-physical and rejects every
    single proposal.

    FOUR ROUTES TO PENALTY, only ONE of which c_cutoff controls:
        1. fewer than 2 usable tuning steps          (unconditional)
        2. any tuning step failed                    (unconditional)
        3. min form factor C below C_FLOOR           (c_cutoff only)
        4. non-finite or non-positive integral       (unconditional)
    So re-running a penalised point with c_cutoff=False recovers the smooth
    surface ONLY for route 3; routes 1, 2 and 4 return PENALTY either way.

    NOTE this function does NOT consult proposed_params_within_limits, by design:
    NM_opt needs the raw objective everywhere so it never sees a plateau of tied
    PENALTY values. Feasibility is the caller's business.
    """
    d = sim_sweep(params_mm, tuning_steps=tuning_steps, mesh_size=mesh_size,
                  verbose=verbose)
    C, Q, f, V = d["C"], d["Q"], d["f"], d["V"]

    if f.size < 2:
        if verbose:
            print(f"  [fom] only {f.size} usable steps -> penalty", flush=True)
        return (PENALTY, d) if return_details else PENALTY
    if d["n_failed"]:
        if verbose:
            print(f"  [fom] {d['n_failed']} step(s) failed -> penalty", flush=True)
        return (PENALTY, d) if return_details else PENALTY
    if c_cutoff and np.any(C < C_FLOOR):
        if verbose:
            print(f"  [fom] min C={C.min():.4f} < {C_FLOOR} -> penalty", flush=True)
        return (PENALTY, d) if return_details else PENALTY

    f_mid = 0.5 * (f[:-1] + f[1:])
    C_mid = 0.5 * (C[:-1] + C[1:])
    Q_mid = 0.5 * (Q[:-1] + Q[1:])
    V_mid = 0.5 * (V[:-1] + V[1:])
    df    = np.abs(np.diff(f))                     # <- abs: see docstring

    value = float(np.sum(f_mid**2 / (V_mid**2 * C_mid**2 * Q_mid) * df))
    if (not np.isfinite(value)) or value <= 0.0:
        value = PENALTY
    return (value, d) if return_details else value


# ═════════════════════════════════════════════════════════════════════════════
# constraints (MILLIMETERS)
# ═════════════════════════════════════════════════════════════════════════════

def proposed_params_within_limits(proposal):
    """
    All lengths in MILLIMETERS, angle in degrees. Indices come from the I_*
    constants, NOT literals -- dropping gap1 shifted every index above 2.
    """
    try:
        p = _as_7(proposal)
    except ValueError:
        return False

    theta  = p[I_ANGLE]
    div_h  = p[I_DIVH];  div_w  = p[I_DIVW]
    ctr_w  = p[I_CTRW];  side_w = p[I_SIDEW]
    ctr_h  = p[I_CTRH];  side_h = p[I_SIDEH]

    # the tuning angle may legitimately be 0; every length must be > 0
    if np.any(p[1:] <= 0) or np.any(p >= 160):
        return False

    # TUNING ANGLE
    if theta < ANGLE_MIN or theta > ANGLE_MAX:
        return False

    # ALL HEIGHTS in [H_MIN, H_MAX]
    for h in (div_h, ctr_h, side_h):
        if h < H_MIN or h > H_MAX:
            return False

    # CENTRE TOAST HEIGHT also clears the wall at maximum displacement.
    # This BINDS: the cap 160 - 20*tan(theta) drops below H_MAX (145) once
    # tan(theta) > 0.75, i.e. theta > 36.87 deg, reaching 136.2 mm at ANGLE_MAX
    # = 50. (The old comment claiming it never fires assumed ANGLE_MAX = 20.)
    if ctr_h > CAVITY_HEIGHT - 2 * GAP0 * np.abs(np.tan(np.radians(theta))):
        return False

    # DIVIDER / SIDE HEIGHTS within +/-H_TOL of the centre toast
    if div_h <= (1 - H_TOL) * ctr_h or div_h >= (1 + H_TOL) * ctr_h:
        return False
    if side_h <= (1 - H_TOL) * ctr_h or side_h >= (1 + H_TOL) * ctr_h:
        return False

    # WIDTHS
    if ctr_w < CTR_W_MIN or ctr_w > CTR_W_MAX:
        return False
    if side_w < SIDE_W_MIN or side_w > SIDE_W_MAX:
        return False
    if side_w >= (1 + SIDE_W_TOL) * ctr_w or side_w < (1 - SIDE_W_TOL) * ctr_w:
        return False
    if div_w < DIV_W_MIN or div_w >= GAP0:
        return False

    # TOTAL WIDTH: ctr_w + 2*gap0 + 2*div_w + 4*gap1 + 2*side_w
    # (unreachable under the current caps -- see the TOTAL_W_MAX note)
    if (ctr_w + 2 * GAP0 + 2 * div_w + 4 * GAP1 + 2 * side_w) >= TOTAL_W_MAX:
        return False

    return True


# ═════════════════════════════════════════════════════════════════════════════
# proposals / walker init
# ═════════════════════════════════════════════════════════════════════════════

# ── proposal geometry ────────────────────────────────────────────────────────
# "log"  : x' = x * exp(eps)  -- a MULTIPLICATIVE step, i.e. a fixed FRACTION of
#          the current value.
# "linear": x' = x + eps      -- an ADDITIVE step of fixed absolute size.
#
# Multiplicative was the right default when the box spanned decades. With the
# current bounds it is not:
#   * the absolute step scales with the value, so at proposal_std = 0.1 the angle
#     moves by 1e-4 deg near zero and 2 deg near 20 -- a factor of 2e4 across its
#     own range (the heights, spanning only 1.6x, barely notice);
#   * ANGLE_MIN is now 0, and a log-space walk on a parameter whose lower bound is
#     zero is unbounded BELOW. Simulated walkers drift to ~1e-3 deg and cannot
#     climb back: escaping to 20 deg needs ~50 consecutive same-sign steps. Since
#     low angle is the good branch, walkers migrate there and then stop exploring.
#   * an additive step is symmetric in the parameter itself, so the Metropolis
#     ratio needs no Jacobian. The multiplicative version is symmetric only in log
#     space and formally requires an x'/x correction that is not applied. For an
#     annealer this biases exploration rather than breaking anything, but the
#     additive form removes the issue.
# "linear" is therefore the default. Set PROPOSAL_MODE = "log" to restore the old
# behaviour; proposal_std is then read as a fraction rather than an absolute step.
PROPOSAL_MODE = "linear"


def param_ranges():
    """(lo, hi) box width per design parameter, for scaling additive steps."""
    return np.array([ANGLE_MAX - ANGLE_MIN,          # angle
                     H_MAX - H_MIN,                  # div_h
                     GAP0 - DIV_W_MIN,               # div_w
                     CTR_W_MAX - CTR_W_MIN,          # ctr_w
                     SIDE_W_MAX - SIDE_W_MIN,        # side_w
                     H_MAX - H_MIN,                  # ctr_h
                     H_MAX - H_MIN], dtype=np.float64)   # side_h


def range_step(frac=0.05):
    """
    Per-parameter additive step: `frac` of each parameter's own range. This makes
    proposal_std mean the same thing for every parameter, which a single scalar
    fraction of the VALUE does not.
    """
    return float(frac) * param_ranges()


def _batch_proposals(log_params, proposal_std, n=64, df=3, clip=2.0,
                     mode=None, params=None):
    """
    n heavy-tailed (Student-t, df=3) proposals.

    Called as _batch_proposals(_safe_log(x), std): the first argument is LOG
    parameters for backward compatibility. In "linear" mode it is exponentiated
    back immediately, so callers need not change.

    PASS params= IN LINEAR MODE. _safe_log clamps at 1e-12, so a non-positive
    coordinate does not survive the log/exp round trip: a walker sitting at
    div_w = -2 would silently generate its whole batch around div_w = 1e-12,
    with nothing raised and the stored state still reading -2. Unreachable while
    proposed_params_within_limits is enforced (every length is > 0 there), but
    live the moment check_limits is turned off. Passing the physical point
    bypasses the round trip entirely.

    clip is in units of proposal_std, applied to the standardised step.
    """
    mode = PROPOSAL_MODE if mode is None else mode
    d = log_params.shape[0]
    lp = torch.tensor(log_params, dtype=torch.float64, device=_dev)
    std = torch.tensor(np.asarray(proposal_std, dtype=np.float64),
                       dtype=torch.float64, device=_dev)
    z = torch.randn(n, d, device=_dev)
    chi2 = torch.distributions.Chi2(float(df)).sample((n, d)).to(_dev)
    t = (z / torch.sqrt(chi2 / df)).clamp(-clip, clip)      # standardised t-step

    if mode == "log":
        return torch.exp(lp.unsqueeze(0) + t * std.unsqueeze(0)).cpu().numpy()
    if mode == "linear":
        x = torch.exp(lp) if params is None else torch.tensor(
            np.asarray(params, dtype=np.float64), dtype=torch.float64, device=_dev)
        return (x.unsqueeze(0) + t * std.unsqueeze(0)).cpu().numpy()
    raise ValueError(f"unknown PROPOSAL_MODE {mode!r}")


def _resolve_std(proposal_std, n_p):
    """
    Turn `proposal_std` into a per-parameter vector.

    A SCALAR means different things in the two modes, so it is resolved here
    rather than at the call site:
      "log"    -> a fraction of the current VALUE, the same number for each.
      "linear" -> a fraction of each parameter's own RANGE, so one number gives
                  sensible absolute steps for a 20-degree angle and a 55 mm height
                  at the same time.
    Pass an explicit vector to override either.

    EVERY entry point must go through this. continue_mcmc used to build the
    vector itself with np.full(n_p, std), which in linear mode reads the scalar
    as an ABSOLUTE step: proposal_std = 0.1 then meant 0.1 mm rather than 0.1 of
    each range, a 7x to 55x reduction applied silently on resume.
    """
    if np.ndim(proposal_std) > 0:
        return np.asarray(proposal_std, dtype=np.float64)
    if PROPOSAL_MODE == "linear" and n_p == N_PARAMS:
        return range_step(float(proposal_std))
    return np.full(n_p, float(proposal_std))


def _jitter_within_limits(base, init_jitter, max_tries=200, check_limits=True,
                          warn=True):
    """
    A perturbed copy of `base`, retried until it satisfies the limits.

    The jitter follows PROPOSAL_MODE for the same reason the proposals do: the
    old log-space-only version multiplied by exp(N(0, j)), which collapses to
    ~1e-12 for any coordinate at or below zero. With ANGLE_MIN = 0 a start point
    at angle = 0 produced walkers whose angles were all 1e-12.

    Returns base.copy() if no trial satisfies the limits within max_tries, which
    makes every walker identical -- warned about loudly, because silently
    collapsing a 10-walker ensemble to one point is very hard to spot downstream.
    """
    base = _as_7(base)
    if PROPOSAL_MODE == "linear" and base.size == N_PARAMS:
        scale = float(init_jitter) * param_ranges()
        draw = lambda: base + np.random.normal(0.0, scale)          # noqa: E731
    else:
        lp = _safe_log(base)
        draw = lambda: np.exp(lp + np.random.normal(0.0, init_jitter,   # noqa: E731
                                                    size=base.shape))
    for _ in range(max_tries):
        trial = draw()
        if not check_limits or proposed_params_within_limits(trial):
            return trial
    if warn:
        print(f"[MCMC] WARNING: no feasible jitter of {_fmt_params(base)} in "
              f"{max_tries} tries; returning the base point UNCHANGED. Walkers "
              f"seeded this way are identical and the ensemble is not "
              f"independent. Lower init_jitter or start from a feasible point.",
              flush=True)
    return base.copy()


def _init_walker_params(initial_params, n_walkers, init_jitter=0.05,
                        check_limits=True):
    ip = np.asarray(initial_params, dtype=np.float64)
    if ip.ndim == 2:
        if ip.shape[0] != n_walkers:
            raise ValueError(f"initial_params has {ip.shape[0]} rows but "
                             f"n_walkers={n_walkers}")
        return [_as_7(row) for row in ip]
    if ip.ndim != 1:
        raise ValueError("initial_params must be shape (d,) or (n_walkers, d)")
    ip = _as_7(ip)
    if not proposed_params_within_limits(ip):
        print(f"[MCMC] {'WARNING' if check_limits else 'note'}: initial_params "
              f"violates the limits"
              + ("." if check_limits else " (check_limits is off, so this is "
                                          "allowed)."), flush=True)
    walkers = [ip.copy()]
    for _ in range(1, n_walkers):
        walkers.append(_jitter_within_limits(ip, init_jitter,
                                             check_limits=check_limits)
                       if init_jitter > 0 else ip.copy())
    return walkers


# ═════════════════════════════════════════════════════════════════════════════
# CSVs  -- rows are built in memory and committed one COMPLETE iteration at a time
# ═════════════════════════════════════════════════════════════════════════════

# AcceptProb is the Metropolis acceptance probability for this proposal; Accepted
# is the outcome of the coin flip. Both are blank for the baseline evaluations,
# which are not proposals and get no Metropolis test. WithinLimits records
# feasibility of the evaluated point -- always True when check_limits is on, and
# the only way to recover the best BUILDABLE geometry when it is off.
_EVALS_HEADER = ["Walker", "Parameters", "Value", "Time", "MinC", "MeanQ",
                 "FreqLo", "FreqHi", "Temp", "AcceptProb", "Accepted",
                 "WithinLimits"]


_RMSE_HEADER = ["Iteration", "NEvals", "NTest", "RMSE", "Spearman", "Kendall"]


def _ensure_csvs(save_path):
    """Also reports the evals row width, so an existing CSV written by an older
    version keeps its own column count instead of being silently misaligned."""
    all_path   = os.path.join(save_path, "all_params_all_values.csv")
    best_path  = os.path.join(save_path, "best_params_best_values.csv")
    evals_path = os.path.join(save_path, "fem_evals.csv")
    rmse_path  = os.path.join(save_path, "surrogate_rmse.csv")
    os.makedirs(save_path, exist_ok=True)
    if not os.path.exists(rmse_path):
        with open(rmse_path, "w", newline="") as fh:
            csv.writer(fh).writerow(_RMSE_HEADER)
    if not os.path.exists(all_path):
        with open(all_path, "w", newline="") as fh:
            csv.writer(fh).writerow(["Walker", "Parameters", "Value"])
    if not os.path.exists(evals_path):
        with open(evals_path, "w", newline="") as fh:
            csv.writer(fh).writerow(_EVALS_HEADER)
    with open(evals_path, newline="") as fh:
        hdr = next(csv.reader(fh), None)
    evals_width = len(hdr) if hdr else len(_EVALS_HEADER)
    if evals_width < len(_EVALS_HEADER):
        missing = _EVALS_HEADER[evals_width:]
        print(f"[MCMC] note: {evals_path} has {evals_width} columns (older "
              f"format); {', '.join(missing)} will not be recorded there. Delete "
              f"or rename it to get the full header.", flush=True)
    return all_path, best_path, evals_path, rmse_path, evals_width


def _eval_row(w, params, value, elapsed, details, temp,
              accept_prob=None, accepted=None, within_limits=None):
    """Build (do not write) one fem_evals.csv row."""
    C = details["C"]; Q = details["Q"]; f = details["f"]
    if within_limits is None:
        within_limits = proposed_params_within_limits(params)
    return [w, _fmt_arr(params), float(value), elapsed,
            (float(C.min()) if C.size else ""), (float(Q.mean()) if Q.size else ""),
            (float(f.min()) if f.size else ""), (float(f.max()) if f.size else ""),
            float(temp),
            ("" if accept_prob is None else float(accept_prob)),
            ("" if accepted is None else bool(accepted)),
            bool(within_limits)]


def _flush_rows(path, rows, width=None):
    """Append a list of already-built rows, then clear the list."""
    if not rows:
        return
    with open(path, "a", newline="") as fh:
        wtr = csv.writer(fh)
        for r in rows:
            wtr.writerow(r if width is None else r[:width])
    rows.clear()


# ═════════════════════════════════════════════════════════════════════════════
# surrogate MLP
# ═════════════════════════════════════════════════════════════════════════════

class _SurrogateNet(nn.Module):
    """params -> predicted LOG objective."""

    def __init__(self, d, hidden=SURROGATE_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Surrogate:
    """
    Replay buffer of FOM evaluations plus an MLP approximating them, used to screen
    candidate proposals so only the most promising one costs a real sweep.

    Trained on the LOG objective (see _to_obj): raw FOM values span ~1e30-1e33, and
    normalising those directly makes the target scale meaningless.

    DETAILED BALANCE: picking argmin over screened candidates is greedy, not a
    symmetric Metropolis proposal, so with the surrogate ON the chain is a
    stochastic optimiser rather than a sampler. The temperature keeps its
    optimisation role (it is the only thing that accepts uphill moves and escapes
    a local minimum) but stops being a thermodynamic temperature: heat capacity,
    relaxation time and constant-thermodynamic-speed cooling all require
    use_surrogate=False. Screening also inflates the acceptance rate, so do not
    tune the temperature against it.
    """

    def __init__(self, input_dim, min_samples=SURROGATE_MIN_SAMPLES,
                 buffer_size=SURROGATE_BUFFER, hidden=SURROGATE_HIDDEN,
                 lr=SURROGATE_LR):
        self.d = int(input_dim)
        self.min_samples = int(min_samples)
        self._buf = deque(maxlen=buffer_size)     # training set
        self._hist_X, self._hist_y = [], []       # full history, in arrival order
        # pin the dtype at construction and derive it everywhere else: nn.Linear
        # takes its dtype from the AMBIENT torch default at build time, so a module
        # built under a different default silently mismatches its own inputs.
        self.net = _SurrogateNet(self.d, hidden).to(device=_dev, dtype=torch.float64)
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.trained = False
        self._Xmu = np.zeros(self.d); self._Xsi = np.ones(self.d)
        self._ymu = 0.0; self._ysi = 1.0
        self.n_trained_on = 0                     # history length at the last fit
        self.rmse_log = []                        # (n_obs, n_test, rmse, rho, tau)
        self.train_rmse_log = []                  # (n_obs, first-epoch, final)

    # ---- data ---------------------------------------------------------------
    def observe(self, params, log_value):
        p = _as_7(params)
        self._buf.append((p, float(log_value)))
        self._hist_X.append(p); self._hist_y.append(float(log_value))

    @property
    def n_obs(self):
        return len(self._hist_y)

    @property
    def ready(self):
        return self.trained

    # ---- fit ----------------------------------------------------------------
    def _split_penalty(self, X, y):
        """Separate PENALTY observations from real ones."""
        thr = np.log(PENALTY) - 1e-6
        bad = y >= thr
        return X[~bad], y[~bad], int(bad.sum())

    def fit(self, epochs=SURROGATE_EPOCHS, batch_size=64, verbose=False,
            progress_every=SURROGATE_PROGRESS_EVERY, exclude_penalty=True):
        """
        Refit from scratch on the replay buffer.

        exclude_penalty : drop observations sitting at log(PENALTY) before fitting.
            KEEP THIS ON. Penalty points are a cliff in parameter space, not a
            smooth function of the geometry, and an MSE regressor cannot fit one.
            Because log(PENALTY) ~ 76 while real geometries are ~70, even a 5%
            contamination inflates the target std ~4x and the network collapses to
            predicting the mean -- which shows up as a training RMSE frozen at
            exactly _ysi, learning nothing. The surrogate only has to RANK feasible
            candidates.

            The cost is that the model has no representation of the cliff and can
            rank a doomed candidate first, wasting the very sweep screening exists
            to save. With check_limits ON, geometric infeasibility is screened by
            proposed_params_within_limits before a proposal is ever evaluated, so
            the only cliff left is the C < C_FLOOR / solver-failure boundary
            inside fom. With check_limits OFF, nothing is screened and the model
            is blind to both. If the penalty rate among proposals climbs, the fix
            is a separate feasibility classifier trained on the FULL history
            (penalties as labels, not targets) -- not folding penalties back into
            this regressor.

        progress_every : print the running TRAINING RMSE every this many epochs,
            converted back to LOG-OBJECTIVE units (the loss is computed on
            z-scored targets, so the raw MSE is unitless and unreadable). This is
            a training score -- it tells you the optimiser is descending, NOT how
            well the model generalises; holdout_rmse() is the honest number.
            Set to 0 to silence.
        """
        if len(self._buf) < self.min_samples:
            return False
        X = np.stack([p for p, _ in self._buf])
        y = np.array([v for _, v in self._buf], dtype=np.float64)
        n_all = len(y)
        if exclude_penalty:
            X, y, n_bad = self._split_penalty(X, y)
            if verbose and n_bad:
                print(f"[{_ts()}] [surrogate] dropped {n_bad}/{n_all} PENALTY "
                      f"observations ({100*n_bad/max(1,n_all):.1f}%) before fitting",
                      flush=True)
            if len(y) < self.min_samples:
                if verbose:
                    print(f"[{_ts()}] [surrogate] only {len(y)} non-penalty samples "
                          f"(< {self.min_samples}); skipping this fit", flush=True)
                return False
        self._Xmu, self._Xsi = X.mean(0), X.std(0) + 1e-8
        self._ymu, self._ysi = float(y.mean()), float(y.std()) + 1e-8

        dt = next(self.net.parameters()).dtype    # follow the weights, never hardcode
        Xt = torch.tensor((X - self._Xmu) / self._Xsi, dtype=dt, device=_dev)
        yt = torch.tensor((y - self._ymu) / self._ysi, dtype=dt, device=_dev)
        dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xt, yt),
            batch_size=batch_size, shuffle=True)

        epochs = int(epochs)
        if verbose:
            print(f"[{_ts()}] [surrogate] fitting on {len(y)} samples "
                  f"(target sd = {self._ysi:.4f}), {epochs} epochs. RMSE below is "
                  f"in log-objective units; a value stuck at the target sd means "
                  f"the net is predicting the mean and learning nothing.",
                  flush=True)
        self.net.train()
        first = last = float("nan")
        t_fit = time.perf_counter()
        for ep in range(1, epochs + 1):
            tot = 0.0
            for xb, yb in dl:
                self.opt.zero_grad()
                loss = nn.functional.mse_loss(self.net(xb), yb)
                loss.backward(); self.opt.step()
                tot += float(loss.detach()) * xb.shape[0]
            mse = tot / max(1, Xt.shape[0])
            # the loss is on z-scored targets; multiply by ysi to get log units
            last = float(np.sqrt(max(mse, 0.0)) * self._ysi)
            if ep == 1:
                first = last
            if progress_every and (ep % progress_every == 0 or ep == epochs):
                print(f"[{_ts()}] [surrogate]   epoch {ep:>5}/{epochs} | "
                      f"train RMSE = {last:.4f}", flush=True)
        self.trained = True
        self.n_trained_on = self.n_obs
        self.train_rmse_log.append((self.n_obs, first, last))
        if verbose:
            print(f"[{_ts()}] [surrogate] done in {time.perf_counter()-t_fit:.1f}s | "
                  f"train RMSE {first:.4f} -> {last:.4f}", flush=True)
        return True

    # ---- predict ------------------------------------------------------------
    def batch_predict(self, proposals):
        """(N, d) -> predicted LOG objective (N,)."""
        X = np.atleast_2d(np.asarray(proposals, dtype=np.float64))
        Xn = (X - self._Xmu) / self._Xsi
        dt = next(self.net.parameters()).dtype
        self.net.eval()
        with torch.no_grad():
            out = self.net(torch.tensor(Xn, dtype=dt, device=_dev)).cpu().numpy()
        return np.atleast_1d(out) * self._ysi + self._ymu

    # ---- held-out diagnostic ------------------------------------------------
    def holdout_rmse(self, n_test):
        """
        RMSE (in LOG-objective units) on the newest `n_test` evaluations.

        Called immediately BEFORE a refit, so every one of those points arrived
        after the last fit and is genuinely unseen -- a real generalisation
        estimate, not a training score. Returns None if not yet fitted or there
        are not that many unseen points.

        CAVEAT on interpretation: these points are all argmin winners of their
        own screening batches, and no two of them ever competed against each
        other. The rank correlation here is therefore a FLOOR on screening
        quality, measured under heavy range restriction, not an estimate of it.
        """
        if not self.trained:
            return None
        n = int(min(n_test, self.n_obs - self.n_trained_on))
        if n < 1:
            return None
        X = np.stack(self._hist_X[-n:])
        y = np.asarray(self._hist_y[-n:], dtype=np.float64)
        # score only the points the model was ever asked to represent
        X, y, _n_bad = self._split_penalty(X, y)
        if y.size < 1:
            return None
        n = int(y.size)
        pred = self.batch_predict(X)
        rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
        # Rank correlation on the SAME held-out set, with the SAME model, at the
        # same moment -- this is what screening actually depends on, and unlike
        # RMSE it is blind to a constant bias. Note RMSE ~ sigma(y) is NOT
        # evidence of mean-prediction: at correlation r the best achievable RMSE
        # is sigma*sqrt(1-r^2), which is within 15% of sigma for r below 0.5.
        rho = tau = float("nan")
        if n >= 3 and not (np.allclose(y, y[0]) or np.allclose(pred, pred[0])):
            try:
                from scipy.stats import spearmanr, kendalltau
                rho = float(spearmanr(pred, y).statistic)
                tau = float(kendalltau(pred, y).statistic)
            except Exception:
                pass
        self.rmse_log.append((self.n_obs, n, rmse, rho, tau))
        return rmse

    # ---- persistence --------------------------------------------------------
    def save_checkpoint(self, path):
        """
        Save everything needed to REPRODUCE this model exactly: weights, the Adam
        moment estimates, the normalisation constants, the observation history and
        the RNG states. Without this the model is unrecoverable -- the weights live
        only in memory, and both the initialisation and the mini-batch shuffle
        order are unseeded, so a past model cannot be rebuilt bit-for-bit.
        """
        torch.save({
            "net": self.net.state_dict(),
            "opt": self.opt.state_dict(),
            "d": self.d, "min_samples": self.min_samples,
            "trained": self.trained, "n_trained_on": self.n_trained_on,
            "Xmu": self._Xmu, "Xsi": self._Xsi,
            "ymu": self._ymu, "ysi": self._ysi,
            "hist_X": np.asarray(self._hist_X), "hist_y": np.asarray(self._hist_y),
            "rmse_log": self.rmse_log, "train_rmse_log": self.train_rmse_log,
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": np.random.get_state(),
        }, path)
        return path

    @classmethod
    def load_checkpoint(cls, path, restore_rng=False):
        """Rebuild a Surrogate saved by save_checkpoint()."""
        ck = torch.load(path, map_location=_dev, weights_only=False)
        s = cls(ck["d"], min_samples=ck["min_samples"])
        s.net.load_state_dict(ck["net"]); s.opt.load_state_dict(ck["opt"])
        s.trained = ck["trained"]; s.n_trained_on = ck["n_trained_on"]
        s._Xmu, s._Xsi = ck["Xmu"], ck["Xsi"]
        s._ymu, s._ysi = ck["ymu"], ck["ysi"]
        s._hist_X = [np.asarray(r) for r in ck["hist_X"]]
        s._hist_y = list(ck["hist_y"])
        s._buf.clear()
        for p, v in zip(s._hist_X, s._hist_y):
            s._buf.append((p, float(v)))
        s.rmse_log = list(ck["rmse_log"]); s.train_rmse_log = list(ck["train_rmse_log"])
        if restore_rng:
            torch.set_rng_state(ck["torch_rng"]); np.random.set_state(ck["numpy_rng"])
        return s

    def get_rmses(self):
        return list(self.rmse_log)


def _seed_surrogate_from_csv(surrogate, evals_path, n_params=N_PARAMS,
                             verbose=True, rows=None):
    """Replay every past evaluation into the surrogate buffer (physical -> log)."""
    n = 0
    for _w, p, v in (rows if rows is not None else _read_rows(evals_path, n_params)):
        if p is None or not np.isfinite(v):
            continue
        surrogate.observe(p, _to_obj(v))
        n += 1
    if verbose:
        print(f"[surrogate] seeded {n} prior evaluations from {evals_path}",
              flush=True)
    return n


# ═════════════════════════════════════════════════════════════════════════════
# MCMC
# ═════════════════════════════════════════════════════════════════════════════

def _new_stats(check_limits=True):
    return {"proposals": 0, "accepted": 0, "improved": 0, "skipped_noop": 0,
            "limits_checked": bool(check_limits),
            # stays at 0 when limits_checked is False -- read the flag first, or
            # "not checked" is indistinguishable from "checked, none rejected"
            "rejected_limits": 0, "rejected_form_factor": 0, "failed_eval": 0,
            "failed_steps": 0, "evaluations": 0, "discarded_partial_evals": 0,
            "evaluated_outside_limits": 0,
            "surrogate_screens": 0, "surrogate_fits": 0}


def _evaluate(w, params, tuning_steps, stats, log_each_solve, temp, tag="eval",
              surrogate=None):
    """
    One FOM evaluation (= one full parallel tuning sweep).

    Does NOT write a CSV row: the acceptance probability is only known after the
    Metropolis test, and rows are committed a whole iteration at a time.
    Returns (value, details, elapsed, within_limits).
    """
    if log_each_solve:
        print(f"[{_ts()}] {tag} w{w} | eval START | {_fmt_params(params)}", flush=True)
    t0 = time.perf_counter()
    value, details = fom(params, tuning_steps=tuning_steps, return_details=True)
    elapsed = time.perf_counter() - t0
    stats["evaluations"] += 1
    if surrogate is not None:                     # every real eval trains the model
        surrogate.observe(params, _to_obj(value))
    if details["n_failed"]:
        stats["failed_steps"] += details["n_failed"]
    if details["C"].size and details["C"].min() < C_FLOOR:
        stats["rejected_form_factor"] += 1
    feasible = proposed_params_within_limits(params)
    if not feasible:
        stats["evaluated_outside_limits"] += 1
    if log_each_solve:
        minC = float(details["C"].min()) if details["C"].size else float("nan")
        print(f"[{_ts()}] {tag} w{w} | eval DONE  | {elapsed:6.1f}s | "
              f"FOM={float(value):.4g} | minC={minC:.3f} | temp={temp:.4g}"
              + ("" if feasible else " | OUTSIDE LIMITS"), flush=True)
    return value, details, elapsed, feasible


def _maybe_train_surrogate(surrogate, stats, tstate, surrogate_epochs,
                           retrain_every, step, n_walkers, log=True,
                           rmse_rows=None, progress_every=SURROGATE_PROGRESS_EVERY):
    """
    Held-out RMSE, then refit. Called ONCE PER COMPLETE MCMC STEP, and the
    schedule is counted in steps: first fit at `min_steps`, refit every
    `retrain_every` steps. With n walkers that is n evaluations per step, so the
    model sees ~min_steps*n and then ~retrain_every*n new evaluations per fit.

    The baseline evaluations are observed too but do NOT advance `step`, so at
    the first fit the buffer holds (min_steps + 1)*n_walkers points and a
    checkpoint saved at step S carries n_obs = (S + 1)*n_walkers.

    Ordering matters: the RMSE is taken on the newest (retrain_every - 1)*n_walkers
    evaluations BEFORE the refit that absorbs them, so every scored point arrived
    after the last fit and is genuinely unseen -- a generalisation estimate, not a
    training score.

    tstate = {"last_fit_step", "min_steps", "rmse_done", "ckpt_dir"}.
    """
    if surrogate is None:
        return
    since = step - tstate["last_fit_step"]

    if surrogate.trained and not tstate["rmse_done"] and since >= retrain_every - 1:
        n_test = (retrain_every - 1) * max(1, n_walkers)
        r = surrogate.holdout_rmse(n_test)
        if r is not None:
            tstate["rmse_done"] = True
            stats["surrogate_rmse_last"] = round(r, 5)
            if rmse_rows is not None:
                _, n_used, _r, rho, tau = surrogate.rmse_log[-1]
                # buffered, not written here: committed with the rest of the
                # iteration so an interrupted iteration leaves no orphan row
                rmse_rows.append([int(step), int(surrogate.n_obs), int(n_used),
                                  float(r), float(rho), float(tau)])
            if log:
                print(f"[{_ts()}] [surrogate] step {step} | held-out RMSE "
                      f"= {r:.4f} | Spearman rho = {rho:+.4f} | "
                      f"Kendall tau = {tau:+.4f}", flush=True)

    need_first = (not surrogate.trained) and step >= tstate["min_steps"]
    need_refit = surrogate.trained and since >= retrain_every
    if need_first or need_refit:
        if surrogate.fit(epochs=surrogate_epochs, verbose=log,
                         progress_every=progress_every):
            tstate["last_fit_step"] = step
            tstate["rmse_done"] = False
            stats["surrogate_fits"] += 1
            ckdir = tstate.get("ckpt_dir")
            if ckdir:
                try:
                    os.makedirs(ckdir, exist_ok=True)
                    surrogate.save_checkpoint(
                        os.path.join(ckdir, f"surrogate_step{step:06d}.pt"))
                except Exception as e:
                    print(f"[surrogate] checkpoint failed: {e}", flush=True)


def _walker_step(w, i, current_params, current_value, proposal_std, n_candidates,
                 temp, tuning_steps, stats, log_each_solve, stuck_warn_every,
                 surrogate=None, check_limits=True,
                 max_batch_retries=MAX_BATCH_RETRIES):
    """
    One Metropolis step for walker w (exactly one successful FOM evaluation).
    Returns (params, value, accepted, eval_row) -- the row is handed back rather
    than written, so the caller can commit a whole iteration atomically.
    """
    batch_retries = 0
    while True:
        batch_retries += 1
        if batch_retries > max_batch_retries:
            raise RuntimeError(
                f"walker {w} produced no usable proposal in {max_batch_retries} "
                f"batches of {n_candidates} at {_fmt_params(current_params)}. "
                f"Either the walker is stranded outside the feasible region or "
                f"every evaluation is failing; check the last EVAL ERROR above.")

        # params= is REQUIRED here: _safe_log clamps non-positive coordinates to
        # 1e-12, so without it a walker holding a negative value would generate
        # its entire batch around 1e-12 instead. Unreachable with check_limits on,
        # silent corruption with it off.
        raw = _batch_proposals(_safe_log(current_params), proposal_std,
                               n=n_candidates, params=current_params)
        stats["proposals"] += n_candidates

        noop = np.array([np.allclose(p, current_params, rtol=1e-3, atol=1e-12)
                         for p in raw])
        stats["skipped_noop"] += int(noop.sum())

        if check_limits:
            within = np.array([proposed_params_within_limits(p) for p in raw])
            stats["rejected_limits"] += int((~within & ~noop).sum())
            valid = ~noop & within
        else:
            valid = ~noop
        if not valid.any():
            if batch_retries % stuck_warn_every == 0:
                print(f"[{_ts()}] step {i:>4} w{w} | {batch_retries} batches, still "
                      f"no valid proposal | current: {_fmt_params(current_params)}",
                      flush=True)
            continue

        candidates = raw[valid]
        if surrogate is not None and surrogate.ready:
            # greedy screen: only the most promising candidate costs a real sweep
            proposal = candidates[int(np.argmin(surrogate.batch_predict(candidates)))]
            stats["surrogate_screens"] += 1
        else:
            proposal = candidates[np.random.randint(len(candidates))]

        try:
            proposal_value, details, elapsed, feasible = _evaluate(
                w, proposal, tuning_steps, stats, log_each_solve, temp,
                tag=f"step {i:>4}", surrogate=surrogate)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            stats["failed_eval"] += 1
            print(f"[{_ts()}] step {i:>4} w{w} | EVAL ERROR | err={e} | "
                  f"retrying batch", flush=True)
            continue

        # Metropolis on the LOG objective
        if (not np.isfinite(proposal_value)) or proposal_value <= 0.0:
            accept_prob = 0.0
        else:
            d_obj = _to_obj(proposal_value) - _to_obj(current_value)
            accept_prob = 1.0 if d_obj < 0 else min(1.0, float(np.exp(-d_obj / temp)))

        accepted = bool(np.random.rand() < accept_prob)
        row = _eval_row(w, proposal, proposal_value, elapsed, details, temp,
                        accept_prob=accept_prob, accepted=accepted,
                        within_limits=feasible)
        if log_each_solve:
            print(f"[{_ts()}] step {i:>4} w{w} | p_accept={accept_prob:.4g} | "
                  f"accepted={accepted}", flush=True)

        if accepted:
            stats["accepted"] += 1
            current_params = proposal
            current_value = proposal_value
        return current_params, current_value, accepted, row


def _run_chains(walker_params, walker_values, best_params, best_value, temp,
                steps, proposal_std, tuning_steps, all_path, best_path,
                evals_path, rmse_path, evals_width, save_interval, n_candidates,
                log_each_solve, stuck_warn_every, stats, desc="MCMC",
                seed_chain_csv=False, surrogate=None,
                surrogate_epochs=SURROGATE_EPOCHS,
                surrogate_retrain_every=SURROGATE_RETRAIN_EVERY,
                surrogate_min_steps=SURROGATE_MIN_STEPS,
                surrogate_last_fit_step=0,
                surrogate_progress_every=SURROGATE_PROGRESS_EVERY,
                step_offset=0, check_limits=True):
    """
    ATOMIC ITERATIONS. Every walker's CSV rows are buffered until the whole
    iteration finishes. An interruption part-way through an iteration discards
    that iteration's rows and rolls the in-memory walker state back to the
    iteration boundary, so fem_evals.csv never contains a ragged partial group
    and the files stay consistent with what continue_mcmc will read back.

    This applies to ANY exception, not only Ctrl-C: an unexpected failure rolls
    back the partial iteration, flushes every COMPLETE one, writes the best-so-far
    file, and only then re-raises. Losing hours of sweeps to a stray TypeError is
    not an acceptable failure mode.
    """
    n_walkers = len(walker_params)
    chains_params = [[p.copy()] for p in walker_params]
    chains_values = [[v] for v in walker_values]
    pending_chain = ([[w, _fmt_arr(walker_params[w]), walker_values[w]]
                      for w in range(n_walkers)] if seed_chain_csv else [])
    pending_evals, pending_rmse = [], []
    accepted_per_walker = [0] * n_walkers
    evals_at_start = stats["evaluations"]
    tstate = {"last_fit_step": int(surrogate_last_fit_step),
              "min_steps": int(surrogate_min_steps), "rmse_done": False,
              "ckpt_dir": os.path.join(os.path.dirname(rmse_path), "surrogate_ckpt")}
    interrupted = False
    fatal = None

    pbar = tqdm(range(steps), desc=desc)
    for i in pbar:
        # snapshot the iteration boundary so a partial iteration can be undone
        snap = ([p.copy() for p in walker_params], list(walker_values),
                best_params.copy(), best_value, list(accepted_per_walker),
                stats["accepted"])
        iter_evals, iter_chain, iter_rmse = [], [], []
        iter_improved = 0
        try:
            temp = max(TEMP_MIN, temp * COOLING)
            for w in range(n_walkers):
                # NOTE: named cur_val, NOT cv -- `fem` is the solver module and
                # shadowing a module alias here breaks any later module call.
                cur_par, cur_val, accepted, row = _walker_step(
                    w, i, walker_params[w], walker_values[w], proposal_std,
                    n_candidates, temp, tuning_steps, stats, log_each_solve,
                    stuck_warn_every, surrogate=surrogate,
                    check_limits=check_limits)
                iter_evals.append(row)
                walker_params[w], walker_values[w] = cur_par, cur_val
                if accepted:
                    accepted_per_walker[w] += 1
                    if float(cur_val) < float(best_value):
                        iter_improved += 1
                        best_params = cur_par.copy()
                        best_value = cur_val
                iter_chain.append([w, _fmt_arr(cur_par), cur_val])
            # ONCE PER COMPLETE STEP -- the schedule counts steps, not evaluations
            _maybe_train_surrogate(surrogate, stats, tstate, surrogate_epochs,
                                   surrogate_retrain_every,
                                   step=step_offset + i + 1, n_walkers=n_walkers,
                                   log=log_each_solve, rmse_rows=iter_rmse,
                                   progress_every=surrogate_progress_every)
        except BaseException as exc:                 # noqa: BLE001 -- see docstring
            # roll the iteration back; its rows are never written
            (walker_params, walker_values, best_params, best_value,
             accepted_per_walker) = snap[0], snap[1], snap[2], snap[3], snap[4]
            # roll the accept counter back too, or the reported acceptance rate
            # keeps a numerator from an iteration whose denominator was discarded
            stats["accepted"] = snap[5]
            stats["discarded_partial_evals"] += len(iter_evals)
            interrupted = True
            if not isinstance(exc, KeyboardInterrupt):
                fatal = exc
                print(f"\n[{_ts()}] {type(exc).__name__} during iteration {i}: "
                      f"{exc}", flush=True)
            print(f"[{_ts()}] discarding {len(iter_evals)}/{n_walkers} partial "
                  f"walker row(s); the CSVs end at the last COMPLETE iteration.",
                  flush=True)
            break

        # ---- the iteration completed: commit it ----------------------------
        stats["improved"] += iter_improved
        pending_evals.extend(iter_evals)
        pending_chain.extend(iter_chain)
        pending_rmse.extend(iter_rmse)
        for w in range(n_walkers):
            chains_params[w].append(walker_params[w].copy())
            chains_values[w].append(walker_values[w])

        post = {"best": f"{float(best_value):.4g}",
                "evals": stats["evaluations"], "T": f"{temp:.3g}"}
        if surrogate is not None:
            post["sur"] = "on" if surrogate.ready else "cold"
            if surrogate.rmse_log:
                post["rmse"] = f"{surrogate.rmse_log[-1][2]:.3f}"
                if len(surrogate.rmse_log[-1]) > 3:
                    post["rho"] = f"{surrogate.rmse_log[-1][3]:+.2f}"
        pbar.set_postfix(post); pbar.refresh()

        if save_interval and (i % save_interval == 0):
            _flush_rows(evals_path, pending_evals, evals_width)
            _flush_rows(all_path, pending_chain)
            _flush_rows(rmse_path, pending_rmse)

    # final flush (only ever whole iterations)
    _flush_rows(evals_path, pending_evals, evals_width)
    _flush_rows(all_path, pending_chain)
    _flush_rows(rmse_path, pending_rmse)
    with open(best_path, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["Parameters", "Value"])
        wtr.writerow([_fmt_arr(best_params), best_value])

    done = len(chains_values[0]) - 1
    session = stats["evaluations"] - evals_at_start
    if interrupted:
        print(f"[{_ts()}] stopped after {done} complete iteration(s) of {steps}.",
              flush=True)
    print(f"\nFOM evaluations this session: {session} "
          f"({done} complete iterations x {n_walkers} walkers + retries)")
    print("\nMCMC diagnostics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    if not stats.get("limits_checked", True):
        print("  NOTE: check_limits was OFF -- rejected_limits is 0 because the "
              "test was never run, not because nothing was rejected. See the "
              "WithinLimits column of fem_evals.csv.")
    print(f"  acceptance rate (per walker-step): "
          f"{stats['accepted'] / max(1, done * n_walkers):.3f}")
    for w in range(n_walkers):
        print(f"    walker {w}: {accepted_per_walker[w] / max(1, done):.3f} "
              f"| final FOM={float(walker_values[w]):.4g}")
    if surrogate is not None and surrogate.rmse_log:
        print("\n  surrogate held-out RMSE (log-objective units):")
        for rec in surrogate.rmse_log:
            n_obs, n_test, r = rec[0], rec[1], rec[2]
            rho = rec[3] if len(rec) > 3 else float("nan")
            print(f"    after {n_obs:>5} evals | test n={n_test:>3} | "
                  f"RMSE={r:.4f} | rho={rho:+.4f}")
        print(f"    -> appended to {rmse_path}")

    if fatal is not None:
        raise fatal
    return best_params, best_value, chains_params, chains_values


def mcmc_minimize(initial_params, steps=10, proposal_std=0.1, tuning_steps=16,
                  save_path="./csvs/", save_interval=1, n_candidates=64,
                  log_each_solve=True, stuck_warn_every=25, n_walkers=1,
                  init_jitter=0.05,
                  use_surrogate=False,
                  surrogate_min_steps=SURROGATE_MIN_STEPS,
                  surrogate_retrain_every=SURROGATE_RETRAIN_EVERY,
                  surrogate_epochs=SURROGATE_EPOCHS,
                  surrogate_progress_every=SURROGATE_PROGRESS_EVERY,
                  check_limits=True):
    """
    Multi-walker simulated-annealing MCMC minimising the FEM scan-time FOM.

    initial_params : (7,) in MILLIMETERS -> walker 0 starts there, others at
        jittered copies; or (n_walkers, 7) to set every start explicitly.
        A legacy 8-vector is accepted and its gap1 entry dropped.

    proposal_std : scalar or (7,) vector. A scalar goes through _resolve_std, so
        in "linear" mode it is a FRACTION OF EACH PARAMETER'S RANGE (0.1 -> 5 deg
        of angle, 5.5 mm of height, 1.7 mm of width), not an absolute step.

    check_limits : screen proposals with proposed_params_within_limits. Turning it
        OFF lets the chain visit geometries that are not buildable; those still
        mesh and solve and can score well, so read the WithinLimits column of
        fem_evals.csv before trusting the final point.

    use_surrogate : train an MLP on the log objective and use it to screen the
        candidate batch, so only the most promising proposal costs a real sweep.
        The schedule counts MCMC STEPS: first fit after `surrogate_min_steps`,
        refit every `surrogate_retrain_every` steps, `surrogate_epochs` epochs
        each. One step is n_walkers evaluations; the baseline evaluations are also
        observed, so the first fit sees (surrogate_min_steps + 1)*n_walkers
        points. A held-out RMSE on the newest (retrain_every-1)*n_walkers unseen
        evaluations is reported just before every refit.

        OFF by default: argmin screening is greedy and breaks detailed balance.
        See the Surrogate docstring for what that costs.

    Ctrl-C is safe: the run stops at the last COMPLETE iteration and the CSVs are
    left consistent with it. So is an unexpected exception -- everything complete
    is flushed before it propagates.

    Returns (best_params, best_value, chains_params, chains_values).
    """
    stats = _new_stats(check_limits)
    all_path, best_path, evals_path, rmse_path, evals_width = _ensure_csvs(save_path)
    walker_params = _init_walker_params(initial_params, n_walkers, init_jitter,
                                        check_limits=check_limits)
    n_p = walker_params[0].size
    proposal_std = _resolve_std(proposal_std, n_p)
    print(f"[MCMC] proposal sd per parameter: "
          f"{', '.join(f'{n}={s:.4g}' for n, s in zip(PARAM_NAMES, proposal_std))}",
          flush=True)
    if not check_limits:
        print("[MCMC] check_limits is OFF: proposals are NOT screened against "
              "proposed_params_within_limits.", flush=True)

    surrogate = None
    if use_surrogate:
        surrogate = Surrogate(n_p)
        print(f"[surrogate] ON  | first fit at step {surrogate_min_steps} "
              f"({(surrogate_min_steps + 1) * n_walkers} evals), refit every "
              f"{surrogate_retrain_every} steps ({surrogate_retrain_every*n_walkers} "
              f"evals), {surrogate_epochs} epochs", flush=True)

    # baseline evaluations are not proposals -> AcceptProb/Accepted blank.
    # They are their own complete group, so they commit immediately.
    walker_values, init_rows = [], []
    for w in range(n_walkers):
        v, det, el, feas = _evaluate(w, walker_params[w], tuning_steps, stats,
                                     log_each_solve, TEMP0, tag="init",
                                     surrogate=surrogate)
        init_rows.append(_eval_row(w, walker_params[w], v, el, det, TEMP0,
                                   within_limits=feas))
        walker_values.append(v)
    _flush_rows(evals_path, init_rows, evals_width)

    best_w = int(np.argmin([float(v) for v in walker_values]))
    best_params = walker_params[best_w].copy()
    best_value = walker_values[best_w]

    return _run_chains(walker_params, walker_values, best_params, best_value,
                       TEMP0, steps, proposal_std, tuning_steps, all_path,
                       best_path, evals_path, rmse_path, evals_width,
                       save_interval, n_candidates, log_each_solve,
                       stuck_warn_every, stats,
                       desc="MCMC", seed_chain_csv=True, surrogate=surrogate,
                       surrogate_epochs=surrogate_epochs,
                       surrogate_retrain_every=surrogate_retrain_every,
                       surrogate_min_steps=surrogate_min_steps,
                       surrogate_last_fit_step=0,
                       surrogate_progress_every=surrogate_progress_every,
                       step_offset=0, check_limits=check_limits)


# ═════════════════════════════════════════════════════════════════════════════
# resume
# ═════════════════════════════════════════════════════════════════════════════

def _parse_params(cell, n_params=N_PARAMS):
    nums = _NUM.findall(str(cell))
    if len(nums) < n_params:
        return None
    vals = [float(x) for x in nums[:max(n_params, 8)][:8]]
    if len(nums) >= 8 and n_params == N_PARAMS:
        # an 8-number cell is a legacy row: drop the gap1 entry
        return np.delete(np.array(vals[:8], dtype=np.float64), 3)
    return np.array([float(x) for x in nums[:n_params]], dtype=np.float64)


def _parse_value(cell):
    try:
        return float(cell)
    except (TypeError, ValueError):
        m = _NUM.findall(str(cell))
        return float(m[0]) if m else np.nan


def _read_rows(path, n_params=N_PARAMS):
    """Read a Walker|Parameters|Value CSV -> list of (walker, params, value)."""
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        has_walker = bool(header) and header[0].strip().lower() == "walker"
        for row in reader:
            if not row or len(row) < 2:
                continue
            if has_walker:
                if len(row) < 3:
                    continue
                try:
                    wid = int(float(row[0]))
                except (TypeError, ValueError):
                    continue
                p, v = _parse_params(row[1], n_params), _parse_value(row[2])
            else:
                wid = 0
                p, v = _parse_params(row[0], n_params), _parse_value(row[1])
            if p is None:
                continue
            rows.append((wid, p, v))
    return rows


def load_mcmc_state(save_path="./csvs/", n_params=N_PARAMS, n_walkers=1,
                    init_jitter=0.05, check_limits=True):
    """
    Rebuild everything needed to resume, from the CSVs written by mcmc_minimize.

    Walker positions come from all_params_all_values.csv, which is the CHAIN (one
    row per walker per step, repeats included when a proposal is rejected).
    fem_evals.csv is NOT used for this: its last row is the last PROPOSAL, which
    is usually not the accepted state. The evals file is only a fallback if the
    chain file is missing, plus a count of prior evaluations.

    Returns dict: walkers [{params, value}], best_params, best_value,
    n_steps_done, n_prior_evals, eval_rows, source.
    """
    all_path   = os.path.join(save_path, "all_params_all_values.csv")
    best_path  = os.path.join(save_path, "best_params_best_values.csv")
    evals_path = os.path.join(save_path, "fem_evals.csv")

    eval_rows = _read_rows(evals_path, n_params)     # read once, reused below
    chain = _read_rows(all_path, n_params)
    source = "all_params_all_values.csv (chain states)"
    if not chain:
        chain = eval_rows
        source = "fem_evals.csv (FALLBACK: proposals, not chain states)"
    if not chain:
        raise FileNotFoundError(
            f"no resumable rows in {all_path} or {evals_path}. Run mcmc_minimize "
            f"first, or check save_path.")

    last_state, per_walker = {}, {}
    for wid, p, v in chain:
        last_state[wid] = (p, v)
        per_walker[wid] = per_walker.get(wid, 0) + 1
    # the chain file is seeded with one row per walker before stepping, so the
    # number of completed steps is (rows - 1)
    n_steps_done = max(0, max(per_walker.values()) - 1)

    best_params = best_value = None
    best_rows = _read_rows(best_path, n_params)
    if best_rows:
        _, best_params, best_value = best_rows[0]
    if best_params is None:
        pool = [(p, v) for _w, p, v in chain if np.isfinite(v) and v > 0]
        if not pool:
            pool = [(p, v) for _w, p, v in chain]
        best_params, best_value = min(pool, key=lambda pv: pv[1])

    walkers = []
    for w in range(n_walkers):
        if w in last_state:
            p, v = last_state[w]
            walkers.append({"params": np.asarray(p, dtype=np.float64),
                            "value": float(v)})
        else:
            walkers.append({"params": _jitter_within_limits(
                np.asarray(best_params, dtype=np.float64), init_jitter,
                check_limits=check_limits),
                "value": None})          # unknown -> evaluated by continue_mcmc

    extra = sorted(set(last_state) - set(range(n_walkers)))
    if extra:
        print(f"[resume] NOTE: the file also contains walkers {extra}; they are "
              f"ignored because n_walkers={n_walkers}.", flush=True)

    return {"walkers": walkers,
            "best_params": np.asarray(best_params, dtype=np.float64),
            "best_value": float(best_value),
            "n_steps_done": int(n_steps_done),
            "n_prior_evals": int(len(eval_rows)),
            "eval_rows": eval_rows,
            "source": source}


def continue_mcmc(steps, save_path="./csvs/", proposal_std=0.1, tuning_steps=16,
                  save_interval=1, n_candidates=64, log_each_solve=True,
                  stuck_warn_every=25, start_from="last", start_temp=None,
                  n_walkers=1, init_jitter=0.05, n_params=N_PARAMS,
                  use_surrogate=False,
                  surrogate_min_steps=SURROGATE_MIN_STEPS,
                  surrogate_retrain_every=SURROGATE_RETRAIN_EVERY,
                  surrogate_epochs=SURROGATE_EPOCHS,
                  surrogate_progress_every=SURROGATE_PROGRESS_EVERY,
                  check_limits=True):
    """
    Resume a run from its CSVs. See mcmc_minimize for use_surrogate, proposal_std
    and check_limits semantics -- all three are resolved identically here, which
    they were not before: proposal_std used to bypass _resolve_std and be read as
    an absolute step on resume.

    start_from : "last" -> every walker resumes from its own last chain state
                 "best" -> walker 0 restarts at the global best, the rest at
                           jittered copies of it
    start_temp : override the annealing temperature. Otherwise rebuilt from the
                 schedule, TEMP0 * COOLING**n_steps_done (floored at TEMP_MIN),
                 exact here because the cooling is geometric -- an adaptive
                 schedule would have to be persisted to a file instead.

    On resume the surrogate is SEEDED from fem_evals.csv and fitted immediately if
    there is enough data, so a continued run does not spend another
    `surrogate_min_steps` steps relearning what the last session paid for. The
    refit schedule then continues from the resumed step count.
    """
    state = load_mcmc_state(save_path, n_params, n_walkers, init_jitter,
                            check_limits=check_limits)
    print(f"[resume] source        : {state['source']}", flush=True)
    print(f"[resume] prior steps   : {state['n_steps_done']}  "
          f"({state['n_prior_evals']} prior evaluations)", flush=True)
    print(f"[resume] best so far   : {state['best_value']:.6g}", flush=True)
    print(f"[resume] best params   : {_fmt_params(state['best_params'])}", flush=True)

    stats = _new_stats(check_limits)
    stats["evaluations"] = state["n_prior_evals"]
    all_path, best_path, evals_path, rmse_path, evals_width = _ensure_csvs(save_path)

    if start_from == "best":
        walker_params = [state["best_params"].copy()]
        walker_values = [state["best_value"]]
        for _ in range(1, n_walkers):
            walker_params.append(_jitter_within_limits(
                state["best_params"], init_jitter, check_limits=check_limits))
            walker_values.append(None)
    elif start_from == "last":
        walker_params = [wk["params"].copy() for wk in state["walkers"]]
        walker_values = [wk["value"] for wk in state["walkers"]]
    else:
        raise ValueError("start_from must be 'last' or 'best'")

    best_params = state["best_params"].copy()
    best_value = state["best_value"]

    n_p = walker_params[0].size
    # THE fix: identical resolution to mcmc_minimize. The old np.full(n_p, std)
    # read a scalar as an absolute step, shrinking every proposal by 7x-55x.
    proposal_std = _resolve_std(proposal_std, n_p)
    print(f"[resume] proposal sd   : "
          f"{', '.join(f'{n}={s:.4g}' for n, s in zip(PARAM_NAMES, proposal_std))}",
          flush=True)
    if not check_limits:
        print("[resume] check_limits is OFF: proposals are NOT screened against "
              "proposed_params_within_limits.", flush=True)

    surrogate = None
    last_fit_step = 0
    if use_surrogate:
        surrogate = Surrogate(n_p)
        _seed_surrogate_from_csv(surrogate, evals_path, n_params,
                                 rows=state["eval_rows"])
        if surrogate.n_obs >= max(SURROGATE_MIN_SAMPLES,
                                  surrogate_min_steps * n_walkers):
            surrogate.fit(epochs=surrogate_epochs, verbose=True,
                          progress_every=surrogate_progress_every)
            last_fit_step = state["n_steps_done"]
            stats["surrogate_fits"] += 1
        print(f"[surrogate] ON  | seeded {surrogate.n_obs} evals | "
              f"ready={surrogate.ready} | next refit at step "
              f"{last_fit_step + surrogate_retrain_every}", flush=True)

    if start_temp is not None:
        temp, src = float(start_temp), "start_temp argument"
    else:
        temp = max(TEMP_MIN, TEMP0 * (COOLING ** state["n_steps_done"]))
        src = f"schedule TEMP0*{COOLING}^{state['n_steps_done']}"
    print(f"[resume] starting temp : {temp:.6g}  ({src})", flush=True)

    seed_rows = []
    for w in range(n_walkers):
        if walker_values[w] is None:
            v, det, el, feas = _evaluate(w, walker_params[w], tuning_steps, stats,
                                         log_each_solve, temp, tag="seed",
                                         surrogate=surrogate)
            seed_rows.append(_eval_row(w, walker_params[w], v, el, det, temp,
                                       within_limits=feas))
            walker_values[w] = v
            if float(v) < float(best_value):
                best_params, best_value = walker_params[w].copy(), v
    _flush_rows(evals_path, seed_rows, evals_width)

    return _run_chains(walker_params, walker_values, best_params, best_value,
                       temp, steps, proposal_std, tuning_steps, all_path,
                       best_path, evals_path, rmse_path, evals_width,
                       save_interval, n_candidates, log_each_solve,
                       stuck_warn_every, stats,
                       desc="MCMC(resume)",
                       # the resumed states are already the last rows of the chain
                       # file; re-seeding would duplicate them and inflate the step
                       # count on the NEXT resume
                       seed_chain_csv=False, surrogate=surrogate,
                       surrogate_epochs=surrogate_epochs,
                       surrogate_retrain_every=surrogate_retrain_every,
                       surrogate_min_steps=surrogate_min_steps,
                       surrogate_last_fit_step=last_fit_step,
                       surrogate_progress_every=surrogate_progress_every,
                       # cumulative iteration index, so surrogate_rmse.csv is a
                       # continuous series across resumes rather than restarting
                       step_offset=state["n_steps_done"],
                       check_limits=check_limits)


# ═════════════════════════════════════════════════════════════════════════════
# Nelder-Mead refinement
# ═════════════════════════════════════════════════════════════════════════════

_NM_HEADER = ["Eval", "Iter", "Parameters", "Value", "Time", "WithinLimits",
              "MinC", "MeanQ", "FreqLo", "FreqHi", "BestSoFar"]


def NM_opt(x0, max_iters, tuning_steps=16, save_path="./csvs/",
           csv_name="nm_evals.csv", append=False, log_each=True,
           return_history=False):
    """
    Local refinement of a single geometry, on the RAW objective -- no constraint
    penalty. Nelder-Mead therefore never sees a plateau of tied PENALTY values and
    keeps full ordering information everywhere it steps.

    The trade: NM will actually evaluate geometries outside the limits, which costs
    real FEM sweeps, and -- the part that matters -- an infeasible geometry can mesh
    and solve perfectly well and return an excellent FOM. NM will happily converge
    onto one.

    EVERY evaluation is written to save_path/csv_name as it happens (flushed per
    row, so an interrupted run keeps everything up to the interruption). The
    WithinLimits column is what makes that log worth having: if NM converges to an
    infeasible point, the CSV still contains the best FEASIBLE geometry it visited,
    which is usually what you actually want. That point is reported at the end and
    returned in the history.

    Note `Eval` counts objective calls while `Iter` counts NM iterations, which are
    NOT the same -- a simplex iteration costs one evaluation for a reflection and
    several for a shrink, so Iter lags Eval and repeats.

    Feed it an MCMC optimum; do NOT feed its trajectory back into the MCMC or the
    surrogate -- hundreds of near-identical points in one basin distort the
    surrogate fit and add nothing to the chain.

    Returns res.x, or (res.x, history) if return_history=True.
    """
    os.makedirs(save_path, exist_ok=True)
    path = os.path.join(save_path, csv_name)
    # NOTE: `fresh` is True when we are APPENDING to an existing file (the name is
    # historical); the header is written only when starting a new one.
    fresh = append and os.path.exists(path)
    with open(path, "a" if fresh else "w", newline="") as fh:
        if not fresh:
            csv.writer(fh).writerow(_NM_HEADER)

    state = {"n": 0, "it": 0, "best": np.inf, "best_x": None,
             "best_feas": np.inf, "best_feas_x": None, "rows": []}

    def objective(x):
        x = np.asarray(x, dtype=np.float64).ravel()
        t0 = time.perf_counter()
        value, details = fom(x, tuning_steps=tuning_steps, return_details=True)
        elapsed = time.perf_counter() - t0
        value = float(value)
        state["n"] += 1
        feas = bool(proposed_params_within_limits(x))
        if value < state["best"]:
            state["best"], state["best_x"] = value, x.copy()
        if feas and value < state["best_feas"]:
            state["best_feas"], state["best_feas_x"] = value, x.copy()

        C, Q, f = details.get("C"), details.get("Q"), details.get("f")
        row = [state["n"], state["it"], _fmt_arr(x), value, elapsed, feas,
               (float(np.min(C)) if np.size(C) else ""),
               (float(np.mean(Q)) if np.size(Q) else ""),
               (float(np.min(f)) if np.size(f) else ""),
               (float(np.max(f)) if np.size(f) else ""),
               float(state["best"])]
        state["rows"].append(row)
        with open(path, "a", newline="") as fh:      # flush per row: interrupt-safe
            csv.writer(fh).writerow(row)
        if log_each:
            print(f"[{_ts()}] NM eval {state['n']:>5} (iter {state['it']:>4}) | "
                  f"{elapsed:6.1f}s | FOM={value:.6g} | "
                  f"{'feasible' if feas else 'OUTSIDE LIMITS'} | "
                  f"best={state['best']:.6g}", flush=True)
        return value

    def callback(xk):
        state["it"] += 1

    try:
        res = scipy.optimize.minimize(
            objective, _as_7(x0), method="Nelder-Mead", callback=callback,
            options={"disp": True, "maxiter": max_iters})
        x_final, f_final = res.x, float(res.fun)
    except KeyboardInterrupt:
        print(f"\n[NM_opt] interrupted after {state['n']} evaluations; "
              f"{path} holds everything so far.", flush=True)
        x_final = state["best_x"] if state["best_x"] is not None else _as_7(x0)
        f_final = state["best"]

    print(f"\n[NM_opt] {state['n']} evaluations over {state['it']} iterations "
          f"-> {path}")
    print(f"[NM_opt] final point    : FOM={f_final:.6g}  {_fmt_params(x_final)}")
    if not proposed_params_within_limits(x_final):
        print(f"[NM_opt] WARNING: the converged point VIOLATES the limits and is "
              f"not a buildable geometry. Re-run from a different start, or clamp "
              f"the offending parameter and re-refine.", flush=True)
        if state["best_feas_x"] is not None:
            print(f"[NM_opt] best FEASIBLE point visited: "
                  f"FOM={state['best_feas']:.6g}  "
                  f"{_fmt_params(state['best_feas_x'])}", flush=True)
        else:
            print("[NM_opt] no feasible point was visited at all.", flush=True)
    elif state["best_feas_x"] is not None and state["best_feas"] < f_final - 1e-12:
        print(f"[NM_opt] note: a BETTER feasible point was visited earlier: "
              f"FOM={state['best_feas']:.6g}  "
              f"{_fmt_params(state['best_feas_x'])}", flush=True)

    if return_history:
        return x_final, {"csv": path, "n_evals": state["n"],
                         "n_iters": state["it"], "rows": state["rows"],
                         "best": state["best"], "best_x": state["best_x"],
                         "best_feasible": state["best_feas"],
                         "best_feasible_x": state["best_feas_x"]}
    return x_final


# ═════════════════════════════════════════════════════════════════════════════
# seeding
# ═════════════════════════════════════════════════════════════════════════════

def generate_seeds(n, rng=None, max_draws=None):
    """
    n feasible starting geometries, drawn by constructive conditional sampling
    plus a rejection check.

    Each dependent coordinate is drawn from the interval its already-drawn parents
    allow, which gets acceptance to ~98% rather than the ~4% a uniform draw over
    the box would give. The price is that the result is NOT uniform over the
    feasible set -- the conditional interval widths are not accounted for -- so
    this is fine for seeding walkers and wrong for estimating feasible volume or
    for any Monte-Carlo integral over the design space.

    rng : optional np.random.Generator for reproducible seeds.
    """
    rng = np.random.default_rng() if rng is None else rng
    max_draws = int(max_draws if max_draws is not None else 1000 * max(1, n))
    seeds, draws = [], 0
    while len(seeds) < n:
        draws += 1
        if draws > max_draws:
            raise RuntimeError(
                f"only {len(seeds)}/{n} feasible seeds in {max_draws} draws; the "
                f"bounds are probably inconsistent (check H_TOL, SIDE_W_TOL and "
                f"the angle/ctr_h clearance).")
        ch = rng.uniform(H_MIN, H_MAX)
        cw = rng.uniform(CTR_W_MIN, CTR_W_MAX)
        proposal = np.array([
            rng.uniform(ANGLE_MIN, ANGLE_MAX),                                # angle
            rng.uniform(max((1 - H_TOL) * ch, H_MIN),
                        min(H_MAX, (1 + H_TOL) * ch)),                        # div_h
            rng.uniform(DIV_W_MIN, GAP0),                                     # div_w
            cw,                                                               # ctr_w
            rng.uniform(max(SIDE_W_MIN, (1 - SIDE_W_TOL) * cw),
                        min(SIDE_W_MAX, (1 + SIDE_W_TOL) * cw)),              # side_w
            ch,                                                               # ctr_h
            rng.uniform(max(H_MIN, (1 - H_TOL) * ch),
                        min(H_MAX, (1 + H_TOL) * ch)),                        # side_h
        ])
        if proposed_params_within_limits(proposal):
            seeds.append(proposal)
    return np.asarray(seeds)