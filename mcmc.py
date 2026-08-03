"""
Multi-walker simulated-annealing MCMC driving the FEM cavity solver.

UNITS -- the one thing to keep straight
    The MCMC (parameters, proposals, constraints, CSVs) works entirely in
    MILLIMETERS, matching proposed_params_within_limits (gap0=10, cy=160).
    fem_solve works entirely in METRES. The conversion happens in exactly one
    place, _params_to_m(), at the boundary. Nothing else converts.
    params[0] is an ANGLE in degrees and is never scaled.
"""

import os
import csv
import time

import numpy as np
import torch
from tqdm import tqdm

import fem_solve as fem
import fem_vis as viz

# ── geometry constants, MILLIMETRES ─────────────────────────────────────────
GAP0          = 10.0     # fixed gap flanking the centre toast
CAVITY_HEIGHT = 160.0
X_MAX_FREQ    = 8.75     # |x| that tunes 15 GHz -> 8 GHz
F_MAX         = 3e11 / (2.0 * GAP0)          # 15 GHz at x=0 (c = 3e11 mm/s)

MM = 1e-3                # millimetres -> metres
GAP0_M   = GAP0 * MM
CAV_H_M  = CAVITY_HEIGHT * MM
X_MAX_M  = X_MAX_FREQ * MM

# ── solver settings ─────────────────────────────────────────────────────────
MESH_SIZE   = 0.001     # METRES. Q converged to 0.6%, C to 2.7%; ~2.5x faster
                         # than 0.001. Tighten when ranking geometries whose C
                         # differs by only a few percent.
N_MODES     = 6
SWEEP_WORKERS = None     # None -> every core (the sweep is the parallel part)
STEP_TIMEOUT  = 600      # s per tuning position

ALUMINIUM = fem.Material("aluminium", sigma=fem.SIGMA_AL_COMSOL)   # 3.774e7 S/m

# ── annealing / objective ───────────────────────────────────────────────────
TEMP0    = 1.0
COOLING  = 0.999
TEMP_MIN = 1e-3
PENALTY  = 1e33
C_FLOOR  = 0.05          # reject a geometry whose worst-step form factor is below this

PARAM_NAMES = ["angle", "div_h", "div_w", "gap1",
               "ctr_w", "side_w", "ctr_h", "side_h"]

_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""_DEVICE_BANNER_SHOWN = False


def _print_device_banner() -> None:
    global _DEVICE_BANNER_SHOWN
    if _DEVICE_BANNER_SHOWN:
        return

    print(f"[MCMC] device: {_dev}")
    if _dev.type == "cuda":
        print(f"[MCMC] GPU: {torch.cuda.get_device_name(0)}")
    _DEVICE_BANNER_SHOWN = True


_print_device_banner()"""


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _fmt_params(p) -> str:
    return ", ".join(f"{n}={v:.4g}" for n, v in zip(PARAM_NAMES, p))


def _to_obj(value) -> float:
    """Physical objective -> log objective used by Metropolis."""
    v = float(value)
    if (not np.isfinite(v)) or v <= 0.0:
        return float(np.log(PENALTY))
    return float(np.log(v))


def _params_to_m(params_mm):
    """
    THE unit boundary: mm -> m for lengths, angle untouched.
    Everything upstream of this is mm; everything downstream is metres.
    """
    p = np.asarray(params_mm, dtype=np.float64).ravel()
    return np.concatenate([[p[0]], p[1:] * MM])


def _safe_log(p):
    """log for the proposal transform. min_angle is 0, so params[0] can approach
    zero and log(0) = -inf would poison the whole proposal batch."""
    return np.log(np.maximum(np.asarray(p, dtype=np.float64), 1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# geometry / sweep
# ─────────────────────────────────────────────────────────────────────────────

def make_spec(params_mm, toast_dx=0.0, toast_dy=0.0, mesh_size=MESH_SIZE,
              tag="toaster"):
    """
    CavitySpec at one tuning position. toast_dx/dy are in METRES and move ALL
    THREE TOASTS (the dividers stay fixed).
    """
    return viz.toaster_spec(
        _params_to_m(params_mm),          # <- conversion happens here, once
        gap0=GAP0_M, cavity_h=CAV_H_M,
        toast_dx=toast_dx, toast_dy=toast_dy,
        mesh_size=mesh_size, tag=tag,
        wall_material=ALUMINIUM,
        metal_material=ALUMINIUM,
    )


def tuning_positions(params_mm, n=16):
    """
    Yields (dx, dy, f_guess) with dx/dy in METRES and f_guess in Hz.
    |x| sweeps 0 -> X_MAX_M and y = |x|*tan(theta); the frequency depends only on
    |x|, so f = c / (2*(gap0 + |x|)) is the shift-invert target for that step.
    """
    t = np.tan(np.radians(float(params_mm[0])))
    for x in -np.linspace(0.0, X_MAX_M, n):
        yield float(x), float(abs(x) * t), 3e8 / (2.0 * (GAP0_M + abs(x)))


def sim_sweep(params_mm, tuning_steps=16, mesh_size=MESH_SIZE, verbose=False):
    """
    Solve the full tuning sweep in parallel and return the operating mode at each
    position.

    Returns dict with arrays C, Q, f, V (all length = number of SUCCESSFUL steps)
    plus n_failed. V is the cavity cross-sectional area in m^2 (the 2D stand-in
    for mode volume, per unit length).
    """
    positions = list(tuning_positions(params_mm, n=tuning_steps))
    specs, results = fem.run_sweep(
        lambda dx, dy, i: make_spec(params_mm, toast_dx=dx, toast_dy=dy,
                                    tag=f"x={dx*1e3:.2f}mm"),
        positions,
        n_modes=N_MODES,
        n_workers=SWEEP_WORKERS,
        timeout=STEP_TIMEOUT,
        keep_fields=False,          # fields are only needed for plotting
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


# ─────────────────────────────────────────────────────────────────────────────
# figure of merit
# ─────────────────────────────────────────────────────────────────────────────

def fom(params_mm, tuning_steps=16, c_cutoff=True, mesh_size=MESH_SIZE,
        verbose=False, return_details=False):
    """
    Scan time (lower is better):   T = integral f^2 / (V^2 C^2 Q) df

    Trapezoid over the tuning band. NOTE abs(df): f DECREASES along the sweep
    (15 -> 8 GHz), so a raw f[1:]-f[:-1] is negative and the integral comes out
    negative -- which the Metropolis test reads as non-physical and rejects every
    single proposal.
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

    integrand = f_mid**2 / (V_mid**2 * C_mid**2 * Q_mid)
    value = float(np.sum(integrand * df))
    if (not np.isfinite(value)) or value <= 0.0:
        value = PENALTY
    return (value, d) if return_details else value


# ─────────────────────────────────────────────────────────────────────────────
# constraints (MILLIMETRES)
# ─────────────────────────────────────────────────────────────────────────────

def proposed_params_within_limits(proposal):
    okay = True
    gap0, cy = GAP0, CAVITY_HEIGHT
    min_angle, max_angle = 0.0, 70.0
    theta = proposal[0]

    # the tuning angle may legitimately be 0, so only params 1..7 must be > 0
    if np.any(np.asarray(proposal)[1:] <= 0):
        okay = False
    elif np.any(np.asarray(proposal) >= 160):
        okay = False

    if theta < min_angle or theta > max_angle:
        okay = False

    # SIDE GAPS: within 10% of the centre gap
    if proposal[3] > 1.1 * gap0 or proposal[3] < 0.9 * gap0:
        okay = False

    # CENTRE TOAST HEIGHT: clears the wall at maximum displacement
    if proposal[6] > cy - 2 * gap0 * np.abs(np.tan(np.radians(theta))):
        okay = False

    # CENTRE TOAST WIDTH
    if proposal[4] < 3:
        okay = False

    # SIDE TOAST HEIGHT / WIDTH: within 20% of the centre toast
    if proposal[7] <= 0.8 * proposal[6] or proposal[7] >= 1.2 * proposal[6]:
        okay = False
    if proposal[5] >= 1.2 * proposal[4] or proposal[5] < 0.8 * proposal[4] or proposal[5] < 3:
        okay = False

    # DIVIDER HEIGHT / WIDTH
    if proposal[1] <= 0.8 * proposal[6] or proposal[1] >= 1.2 * proposal[6]:
        okay = False
    if proposal[2] >= gap0 or proposal[2] < 3:
        okay = False

    # TOTAL WIDTH: centre + 2*gap0 + 2*div + 4*gap1 + 2*side  (the +20 is 2*gap0,
    # which the original expression omitted)
    if (4 * proposal[3] + 2 * proposal[5] + 2 * proposal[2] + proposal[4]
            + 2 * gap0 >= 400 / np.sqrt(2)):
        okay = False

    return okay


# ─────────────────────────────────────────────────────────────────────────────
# proposals / walker init
# ─────────────────────────────────────────────────────────────────────────────

def _batch_proposals(log_params, proposal_std, n=64, df=3, clip=2.0):
    """n heavy-tailed Student-t proposals in log-space."""
    d = log_params.shape[0]
    lp = torch.tensor(log_params, dtype=torch.float64, device=_dev)
    std = torch.tensor(proposal_std, dtype=torch.float64, device=_dev)
    z = torch.randn(n, d, device=_dev)
    chi2 = torch.distributions.Chi2(float(df)).sample((n, d)).to(_dev)
    step = ((z / torch.sqrt(chi2 / df)) * std.unsqueeze(0)).clamp(-clip, clip)
    return torch.exp(lp.unsqueeze(0) + step).cpu().numpy()


def _jitter_within_limits(base, init_jitter, max_tries=200):
    lp = _safe_log(base)
    for _ in range(max_tries):
        trial = np.exp(lp + np.random.normal(0.0, init_jitter, size=np.shape(base)))
        if proposed_params_within_limits(trial):
            return trial
    return np.asarray(base, dtype=np.float64).copy()


def _init_walker_params(initial_params, n_walkers, init_jitter=0.05):
    ip = np.asarray(initial_params, dtype=np.float64)
    if ip.ndim == 2:
        if ip.shape[0] != n_walkers:
            raise ValueError(f"initial_params has {ip.shape[0]} rows but "
                             f"n_walkers={n_walkers}")
        return [row.copy() for row in ip]
    if ip.ndim != 1:
        raise ValueError("initial_params must be shape (d,) or (n_walkers, d)")
    if not proposed_params_within_limits(ip):
        print("[MCMC] WARNING: initial_params violates the limits.", flush=True)
    walkers = [ip.copy()]
    for _ in range(1, n_walkers):
        walkers.append(_jitter_within_limits(ip, init_jitter) if init_jitter > 0
                       else ip.copy())
    return walkers


# ─────────────────────────────────────────────────────────────────────────────
# CSVs
# ─────────────────────────────────────────────────────────────────────────────

# AcceptProb is the Metropolis acceptance probability for this proposal; Accepted
# is the outcome of the coin flip. Both are blank for the baseline evaluations,
# which are not proposals and get no Metropolis test.
_EVALS_HEADER = ["Walker", "Parameters", "Value", "Time", "MinC", "MeanQ",
                 "FreqLo", "FreqHi", "Temp", "AcceptProb", "Accepted"]


def _ensure_csvs(save_path):
    """Also reports the evals row width, so an existing CSV written by an older
    version keeps its own column count instead of being silently misaligned."""
    all_path   = os.path.join(save_path, "all_params_all_values.csv")
    best_path  = os.path.join(save_path, "best_params_best_values.csv")
    evals_path = os.path.join(save_path, "fem_evals.csv")
    os.makedirs(save_path, exist_ok=True)
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
        print(f"[MCMC] note: {evals_path} has {evals_width} columns (older "
              f"format); AcceptProb/Accepted will not be recorded there. Delete "
              f"or rename it to get the full header.", flush=True)
    return all_path, best_path, evals_path, evals_width


def _log_eval(evals_path, evals_width, w, params, value, elapsed, details, temp,
              accept_prob=None, accepted=None):
    C = details["C"]; Q = details["Q"]; f = details["f"]
    row = [w, np.array2string(np.asarray(params), precision=8, separator=","),
           float(value), elapsed,
           (float(C.min()) if C.size else ""), (float(Q.mean()) if Q.size else ""),
           (float(f.min()) if f.size else ""), (float(f.max()) if f.size else ""),
           float(temp),
           ("" if accept_prob is None else float(accept_prob)),
           ("" if accepted is None else bool(accepted))]
    with open(evals_path, "a", newline="") as fh:
        csv.writer(fh).writerow(row[:evals_width])


# ─────────────────────────────────────────────────────────────────────────────
# MCMC
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate(w, params, tuning_steps, stats, log_each_solve, temp, tag="eval"):
    """
    One FOM evaluation (= one full parallel tuning sweep).

    Deliberately does NOT write the CSV row: the acceptance probability is only
    known after the Metropolis test, so the caller logs once it has both.
    Returns (value, details, elapsed).
    """
    if log_each_solve:
        print(f"[{_ts()}] {tag} w{w} | eval START | {_fmt_params(params)}", flush=True)
    t0 = time.perf_counter()
    value, details = fom(params, tuning_steps=tuning_steps, return_details=True)
    elapsed = time.perf_counter() - t0
    stats["evaluations"] += 1
    if details["n_failed"]:
        stats["failed_steps"] += details["n_failed"]
    if details["C"].size and details["C"].min() < C_FLOOR:
        stats["rejected_form_factor"] += 1
    if log_each_solve:
        minC = float(details["C"].min()) if details["C"].size else float("nan")
        print(f"[{_ts()}] {tag} w{w} | eval DONE  | {elapsed:6.1f}s | "
              f"FOM={float(value):.4g} | minC={minC:.3f} | temp={temp:.4g}",
              flush=True)
    return value, details, elapsed


def _walker_step(w, i, current_params, current_value, proposal_std, n_candidates,
                 temp, tuning_steps, evals_path, evals_width, stats,
                 log_each_solve, stuck_warn_every):
    """One Metropolis step for walker w (exactly one successful FOM evaluation)."""
    batch_retries = 0
    while True:
        batch_retries += 1
        raw = _batch_proposals(_safe_log(current_params), proposal_std,
                               n=n_candidates)
        stats["proposals"] += n_candidates

        noop = np.array([np.allclose(p, current_params, rtol=1e-3, atol=1e-12)
                         for p in raw])
        stats["skipped_noop"] += int(noop.sum())
        within = np.array([proposed_params_within_limits(p) for p in raw])
        stats["rejected_limits"] += int((~within & ~noop).sum())

        valid = ~noop & within
        if not valid.any():
            if batch_retries % stuck_warn_every == 0:
                print(f"[{_ts()}] step {i:>4} w{w} | {batch_retries} batches, still "
                      f"no valid proposal | current: {_fmt_params(current_params)}",
                      flush=True)
            continue

        candidates = raw[valid]
        proposal = candidates[np.random.randint(len(candidates))]

        try:
            proposal_value, details, elapsed = _evaluate(
                w, proposal, tuning_steps, stats, log_each_solve, temp,
                tag=f"step {i:>4}")
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
        _log_eval(evals_path, evals_width, w, proposal, proposal_value, elapsed,
                  details, temp, accept_prob=accept_prob, accepted=accepted)
        if log_each_solve:
            print(f"[{_ts()}] step {i:>4} w{w} | p_accept={accept_prob:.4g} | "
                  f"accepted={accepted}", flush=True)

        if accepted:
            stats["accepted"] += 1
            current_params = proposal
            current_value = proposal_value
        return current_params, current_value, accepted


def _run_chains(walker_params, walker_values, best_params, best_value, temp,
                steps, proposal_std, tuning_steps, all_path, best_path,
                evals_path, evals_width, save_interval, n_candidates,
                log_each_solve, stuck_warn_every, stats, desc="MCMC",
                seed_chain_csv=False):
    n_walkers = len(walker_params)
    chains_params = [[p.copy()] for p in walker_params]
    chains_values = [[v] for v in walker_values]
    pending_rows = ([(w, walker_params[w].copy(), walker_values[w])
                     for w in range(n_walkers)] if seed_chain_csv else [])
    accepted_per_walker = [0] * n_walkers
    evals_at_start = stats["evaluations"]

    pbar = tqdm(range(steps), desc=desc)
    for i in pbar:
        temp = max(TEMP_MIN, temp * COOLING)

        for w in range(n_walkers):
            # NOTE: the returned value is named cur_val, NOT cv -- `cv`/`fem` is
            # the solver module and shadowing it here silently breaks any later
            # module call inside this function.
            cur_par, cur_val, accepted = _walker_step(
                w, i, walker_params[w], walker_values[w], proposal_std,
                n_candidates, temp, tuning_steps, evals_path, evals_width,
                stats, log_each_solve, stuck_warn_every)
            walker_params[w], walker_values[w] = cur_par, cur_val
            if accepted:
                accepted_per_walker[w] += 1
                if float(cur_val) < float(best_value):
                    stats["improved"] += 1
                    best_params = cur_par.copy()
                    best_value = cur_val
            chains_params[w].append(cur_par.copy())
            chains_values[w].append(cur_val)
            pending_rows.append((w, cur_par.copy(), cur_val))

        pbar.set_postfix({"best": f"{float(best_value):.4g}",
                          "evals": stats["evaluations"],
                          "T": f"{temp:.3g}"})
        pbar.refresh()

        if save_interval and i % save_interval == 0:
            with open(all_path, "a", newline="") as fh:
                wtr = csv.writer(fh)
                for row_w, p, v in pending_rows:
                    wtr.writerow([row_w, np.array2string(p, precision=8,
                                                         separator=","), v])
            pending_rows = []

    with open(all_path, "a", newline="") as fh:
        wtr = csv.writer(fh)
        for row_w, p, v in pending_rows:
            wtr.writerow([row_w, np.array2string(p, precision=8, separator=","), v])
    with open(best_path, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["Parameters", "Value"])
        wtr.writerow([np.array2string(np.asarray(best_params), precision=8,
                                      separator=","), best_value])

    session = stats["evaluations"] - evals_at_start
    print(f"\nFOM evaluations this session: {session} "
          f"({steps} steps x {n_walkers} walkers + retries)")
    print("\nMCMC diagnostics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  acceptance rate (per walker-step): "
          f"{stats['accepted'] / max(1, steps * n_walkers):.3f}")
    for w in range(n_walkers):
        print(f"    walker {w}: {accepted_per_walker[w] / max(1, steps):.3f} "
              f"| final FOM={float(walker_values[w]):.4g}")
    return best_params, best_value, chains_params, chains_values


def mcmc_minimize(initial_params, steps=10, proposal_std=0.1, tuning_steps=16,
                  save_path="./csvs/", save_interval=10, n_candidates=64,
                  log_each_solve=True, stuck_warn_every=25, n_walkers=1,
                  init_jitter=0.05):
    """
    Multi-walker simulated-annealing MCMC minimising the FEM scan-time FOM.

    initial_params : (d,) in MILLIMETRES -> walker 0 starts there, others at
        log-jittered copies; or (n_walkers, d) to set every start explicitly.

    Each FOM evaluation runs a full tuning sweep, and THAT sweep is what uses all
    the cores (fem.run_sweep). Walkers are advanced sequentially so the parallel
    work is not oversubscribed.

    Returns (best_params, best_value, chains_params, chains_values).
    """
    stats = {"proposals": 0, "accepted": 0, "improved": 0, "skipped_noop": 0,
             "rejected_limits": 0, "rejected_form_factor": 0, "failed_eval": 0,
             "failed_steps": 0, "evaluations": 0}

    all_path, best_path, evals_path, evals_width = _ensure_csvs(save_path)
    walker_params = _init_walker_params(initial_params, n_walkers, init_jitter)
    n_params = walker_params[0].size
    proposal_std = (np.asarray(proposal_std, dtype=np.float64)
                    if np.ndim(proposal_std) > 0
                    else np.full(n_params, float(proposal_std)))

    walker_values = []
    for w in range(n_walkers):
        v, det, el = _evaluate(w, walker_params[w], tuning_steps, stats,
                               log_each_solve, TEMP0, tag="init")
        # baseline evaluations are not proposals -> AcceptProb/Accepted blank
        _log_eval(evals_path, evals_width, w, walker_params[w], v, el, det, TEMP0)
        walker_values.append(v)

    best_w = int(np.argmin([float(v) for v in walker_values]))
    best_params = walker_params[best_w].copy()
    best_value = walker_values[best_w]

    return _run_chains(walker_params, walker_values, best_params, best_value,
                       TEMP0, steps, proposal_std, tuning_steps, all_path,
                       best_path, evals_path, evals_width, save_interval,
                       n_candidates, log_each_solve, stuck_warn_every, stats,
                       desc="MCMC", seed_chain_csv=True)