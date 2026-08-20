"""
Manufacturing-tolerance study for the toaster cavity.

WHAT CHANGED
    The perturbation model now lives in ONE place, noisy_mcmc, and this module
    imports it. Previously the two files described the geometry differently --
    stability.py had its own five-part model with per-part width/height/position
    and a separate stage-error model, while noisy_mcmc had the 33-dimensional
    extended vector. Keeping both would have guaranteed they drifted apart.
    Everything here is therefore expressed in the SAME 33 dimensions:

        7 design      angle, div_h, div_w, ctr_w, side_w, ctr_h, side_h
        6 gaps        gap1L_out, gap1L_in, gap0L, gap0R, gap1R_in, gap1R_out
        1 cavity      cav_h
        5 rotations   sideL_theta, divL_theta, ctr_theta, divR_theta, sideR_theta
        4 right dims  divR_w, divR_h, sideR_w, sideR_h
       10 offsets     sideL_dx/dy, divL_dx/dy, ctr_dx/dy, divR_dx/dy, sideR_dx/dy

NO TUNING-POSITION ERROR
    The stage/leadscrew model (jitter, backlash, lead error, straightness,
    once-per-revolution, yaw) has been REMOVED. Every sample now walks the exact
    nominal trajectory: |x| sweeps 0 -> X_MAX on a clean linspace and the per-part
    offsets are fixed assembly errors applied identically at every tuning step, so
    nothing accumulates along the sweep. This matches noisy_mcmc exactly, so the
    robust objective the optimiser minimises and the tolerance study that audits
    it now describe the same physical assumptions.

    The one residual travel-dependent effect is the ANGLE: the trajectory is
    dy = |dx| tan(angle), so a perturbed angle tilts the whole path (zero
    deviation at x = 0, ~47 um at full travel for sigma = 0.3 deg). It is a fixed
    misalignment, not stage noise. Set default_cov(angle=0.0) to pin it.

IID SAMPLING, NOT THE MOMENT-MATCHED BANK
    noisy_mcmc's z-bank is whitened/moment-matched, which makes the SAMPLE moments
    exact so that a small n estimates E[F] efficiently. That is the wrong tool
    here: this module reports a DISTRIBUTION (p5, p95, worst case), and whitened
    points are no longer independent draws from N(0, Sigma). sample_geometries()
    therefore draws plain iid samples.
"""

from __future__ import annotations

import os
import csv
import time

import numpy as np

from . import mcmc
from . import noisy_mcmc as nz
from . import fem_solve as fem


# the perturbation model is noisy_mcmc's; re-exported for convenience
NOISY_NAMES = nz.NOISY_NAMES
D_NOISY = nz.D_NOISY
GAP_NAMES = nz.GAP_NAMES
PART_NAMES = nz.PART_NAMES
default_cov = nz.default_cov
embed = nz.embed
split = nz.split
parts_of = nz.parts_of

PENALTY = mcmc.PENALTY


# ═════════════════════════════════════════════════════════════════════════════
# sampling
# ═════════════════════════════════════════════════════════════════════════════

def sample_geometries(x0, cov=None, n=100, seed=None, clip=None):
    """
    n INDEPENDENT perturbed extended vectors around the design point x0.

    Deliberately not routed through noisy_mcmc._sample_proposal: that applies
    common random numbers and whitening, which are right for estimating a mean
    with few samples and wrong for characterising a distribution. Here every
    sample is an honest iid draw, so percentiles mean what they say.

    clip : in units of sigma. None (default) means no truncation -- a tolerance
        study should not discard the tail outcomes it exists to quantify.
    """
    cov = default_cov() if cov is None else np.asarray(cov, dtype=np.float64)
    mu = embed(x0)
    D = mu.size
    if cov.ndim == 1:
        cov = np.diag(cov)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((int(n), D))
    if clip is not None and np.isfinite(clip):
        z = np.clip(z, -clip, clip)
    try:
        L = np.linalg.cholesky(cov + 1e-18 * np.eye(D))
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(cov)
        L = V @ np.diag(np.sqrt(np.maximum(w, 0.0)))
    return mu[None, :] + z @ L.T


# ═════════════════════════════════════════════════════════════════════════════
# evaluating one geometry
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_geometry(x_ext, tuning_steps=16, mesh_size=None, c_cutoff=True, check_limits=True):
    """
    Full tuning sweep of one perturbed geometry.

    Delegates to noisy_mcmc.fom_single, so the geometry builder, the tuning
    trajectory and the FOM definition are identical to the ones the optimiser
    uses -- including the absence of any tuning-position error.
    """
    value, details = nz.fom_single(x_ext, tuning_steps=tuning_steps,
                                   mesh_size=mesh_size, c_cutoff=c_cutoff,
                                   return_details=True, check_limits=check_limits)
    d = dict(details)
    d["fom"] = float(value)
    d["feasible"] = not d.get("infeasible", False)
    _, cav_w, cav_h = parts_of(x_ext)
    d["cav_w"], d["cav_h"] = float(cav_w), float(cav_h)
    d["gaps"] = np.array([split(x_ext)[1][g] for g in GAP_NAMES])
    return d


def gap_asymmetry(x_ext):
    """
    Sum |left gap - mirrored right gap| over the three pairs. Zero for a
    mirror-symmetric sample. This is the quantity that matters: the cells are
    degenerate only while the layout is symmetric, and it is that degeneracy
    which delocalises the mode and gives a high form factor.
    """
    g = split(x_ext)[1]
    return float(abs(g["gap1L_out"] - g["gap1R_out"])
                 + abs(g["gap1L_in"] - g["gap1R_in"])
                 + abs(g["gap0L"] - g["gap0R"]))


# ═════════════════════════════════════════════════════════════════════════════
# the study
# ═════════════════════════════════════════════════════════════════════════════

def check_stability(x0, n_samples=100, cov=None, tuning_steps=16,
                    mesh_size=None, seed=None, clip=None,
                    save_path=None, csv_name="stability.csv", verbose=True, check_limits=True):
    """
    Monte-Carlo the 33-dimensional tolerance model around a nominal design.

    x0 : the nominal DESIGN vector (7 entries, or legacy 8).
    cov : (33, 33) covariance; default_cov() supplies sensible 1-sigma values and
        accepts short aliases, e.g. default_cov(gap0=0.02, ctr_theta=0.05).

    Returns (nominal, samples, summary).
    """
    cov = default_cov() if cov is None else cov
    x_nom = embed(x0)

    if verbose:
        sd = np.sqrt(np.diag(np.asarray(cov, dtype=np.float64)))
        live = [(NOISY_NAMES[i], sd[i]) for i in range(len(sd)) if sd[i] > 0]
        print(f"[stability] {D_NOISY} dimensions, {len(live)} with non-zero sigma")
        print(f"[stability] {n_samples} iid samples x {tuning_steps} tuning steps "
              f"= {n_samples*tuning_steps} FEM solves")
        print(f"[stability] no tuning-position error: every sample walks the exact "
              f"nominal trajectory", flush=True)

    t0 = time.perf_counter()
    nominal = evaluate_geometry(x_nom, tuning_steps, mesh_size, c_cutoff=False, check_limits=check_limits)
    if verbose:
        print(f"[stability] nominal: FOM={nominal['fom']:.6g}  "
              f"meanC={nominal['C'].mean() if nominal['C'].size else float('nan'):.4f}  "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)

    X = sample_geometries(x0, cov, n_samples, seed=seed, clip=clip)
    samples = []
    for i, x in enumerate(X):
        d = evaluate_geometry(x, tuning_steps, mesh_size, c_cutoff=False, check_limits=check_limits)
        d["x_ext"] = x
        d["asymmetry"] = gap_asymmetry(x)
        samples.append(d)
        if verbose:
            print(f"  [{i+1}/{n_samples}] FOM={d['fom']:.4g}  "
                  f"minC={d['C'].min() if d['C'].size else float('nan'):.4f}  "
                  f"gap asym={d['asymmetry']:.4f} mm"
                  + ("" if d["feasible"] else "  [infeasible]"), flush=True)

    summary = summarise(nominal, samples, verbose=verbose)
    if save_path:
        write_csv(os.path.join(save_path, csv_name), nominal, samples)
        if verbose:
            print(f"[stability] wrote {os.path.join(save_path, csv_name)}")
    return nominal, samples, summary


def summarise(nominal, samples, verbose=True):
    """Spread of every quantity, plus the failure fraction."""
    ok = [s for s in samples if s["fom"] < PENALTY]
    out = {"n_samples": len(samples), "n_penalty": len(samples) - len(ok),
           "frac_penalty": (len(samples) - len(ok)) / max(1, len(samples)),
           "n_infeasible": sum(1 for s in samples if not s["feasible"])}

    def col(key, red):
        return np.array([red(s[key]) for s in ok if np.size(s[key])])

    fields = (("fom",   np.array([s["fom"] for s in ok]),  nominal["fom"]),
              ("meanC", col("C", np.mean), _safe(nominal["C"], np.mean)),
              ("minC",  col("C", np.min),  _safe(nominal["C"], np.min)),
              ("meanQ", col("Q", np.mean), _safe(nominal["Q"], np.mean)),
              ("fmin",  col("f", np.min),  _safe(nominal["f"], np.min)),
              ("fmax",  col("f", np.max),  _safe(nominal["f"], np.max)),
              ("asym",  np.array([s["asymmetry"] for s in ok]), 0.0))
    for name, arr, nom in fields:
        if arr.size == 0:
            continue
        out[name] = {"nominal": float(nom), "mean": float(arr.mean()),
                     "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                     "min": float(arr.min()), "max": float(arr.max()),
                     "p5": float(np.percentile(arr, 5)),
                     "p95": float(np.percentile(arr, 95))}
        m = out[name]["mean"]
        out[name]["rel_std"] = out[name]["std"] / abs(m) if m else np.nan

    if verbose:
        print(f"\n[stability] {out['n_samples']} samples | "
              f"{out['n_penalty']} hit the penalty ({100*out['frac_penalty']:.1f}%) | "
              f"{out['n_infeasible']} geometrically infeasible")
        print(f"  {'quantity':>8} {'nominal':>12} {'mean':>12} {'rel sd':>8} "
              f"{'p5':>12} {'p95':>12}")
        for k in ("fom", "meanC", "minC", "meanQ", "fmin", "fmax", "asym"):
            if k not in out:
                continue
            d = out[k]
            print(f"  {k:>8} {d['nominal']:>12.5g} {d['mean']:>12.5g} "
                  f"{d['rel_std']:>7.1%} {d['p5']:>12.5g} {d['p95']:>12.5g}")
        if "fom" in out and out["fom"]["nominal"] > 0:
            r = out["fom"]["mean"] / out["fom"]["nominal"]
            print(f"\n  E[FOM] / FOM(nominal) = {r:.3f}x"
                  + ("   <- perturbation can only make it worse; the nominal is a"
                     " maximum" if r > 1.05 else ""))
    return out


def _safe(a, red):
    return float(red(a)) if np.size(a) else float("nan")


def write_csv(path, nominal, samples):
    """One row per sample: the FOM, the observables, and all 33 dimensions."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    hdr = (["sample", "fom", "feasible", "meanC", "minC", "meanQ", "fmin", "fmax",
            "n_failed", "cav_w", "cav_h", "asymmetry"]
           + list(GAP_NAMES) + list(NOISY_NAMES))
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(hdr)
        for i, s in enumerate(samples):
            C, Q, f = s["C"], s["Q"], s["f"]
            w.writerow([i, s["fom"], s["feasible"],
                        _safe(C, np.mean), _safe(C, np.min), _safe(Q, np.mean),
                        _safe(f, np.min), _safe(f, np.max),
                        s["n_failed"], s["cav_w"], s["cav_h"], s["asymmetry"]]
                       + list(s["gaps"]) + list(s["x_ext"]))
    return path


# ═════════════════════════════════════════════════════════════════════════════
# plotting
# ═════════════════════════════════════════════════════════════════════════════

def plot_stability(nominal, samples, save=None, dpi=160):
    """
    Four panels: the FOM distribution against nominal, form factor, the driver
    (FOM vs gap asymmetry) and the frequency band.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [s for s in samples if s["fom"] < PENALTY]
    fom = np.array([s["fom"] for s in ok])
    asym = np.array([s["asymmetry"] for s in ok])
    minC = np.array([s["C"].min() for s in ok if s["C"].size])
    fig, ax = plt.subplots(2, 2, figsize=(10, 7))

    a = ax[0][0]
    a.hist(np.log10(fom), bins=25, color="#b03a2e", alpha=0.75)
    a.axvline(np.log10(nominal["fom"]), color="#1f4e79", lw=2, label="nominal")
    a.axvline(np.log10(fom.mean()), color="k", ls="--", lw=1.4, label="E[FOM]")
    a.set_xlabel(r"$\log_{10}$ FOM"); a.set_ylabel("count")
    a.set_title("(a) scan-time distribution", fontsize=9)
    a.legend(fontsize=7, frameon=False)

    a = ax[0][1]
    if minC.size:
        a.hist(minC, bins=25, color="#1f4e79", alpha=0.75)
        if nominal["C"].size:
            a.axvline(nominal["C"].min(), color="#b03a2e", lw=2, label="nominal")
        a.axvline(mcmc.C_FLOOR, color="k", ls=":", lw=1.4, label="C floor")
        a.legend(fontsize=7, frameon=False)
    a.set_xlabel("worst-step form factor"); a.set_ylabel("count")
    a.set_title("(b) form factor over the sweep", fontsize=9)

    a = ax[1][0]
    a.plot(asym, fom, "o", ms=4, color="#b03a2e", alpha=0.6)
    a.axhline(nominal["fom"], color="#1f4e79", lw=1.6, label="nominal")
    a.set_yscale("log"); a.set_xlabel("gap asymmetry (mm)"); a.set_ylabel("FOM")
    a.set_title("(c) FOM vs left-right asymmetry", fontsize=9)
    a.legend(fontsize=7, frameon=False); a.grid(alpha=0.25, ls=":")

    a = ax[1][1]
    lo = np.array([s["f"].min() for s in ok if s["f"].size]) / 1e9
    hi = np.array([s["f"].max() for s in ok if s["f"].size]) / 1e9
    a.hist(lo, bins=20, alpha=0.7, color="#1f4e79", label="band low")
    a.hist(hi, bins=20, alpha=0.7, color="#b03a2e", label="band high")
    a.set_xlabel("frequency (GHz)"); a.set_ylabel("count")
    a.set_title("(d) tuning band edges", fontsize=9)
    a.legend(fontsize=7, frameon=False)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    return fig