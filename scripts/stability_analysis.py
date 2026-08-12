"""
Machining-tolerance study for the toaster cavity.

WHY THIS MATTERS HERE
    The nominal geometry is mirror-symmetric, so its six gaps are equal and the
    cells are degenerate -- that degeneracy is exactly what produces the in-phase,
    high-C operating mode. Independent machining errors detune the cells relative
    to one another. Whether that destroys the mode depends on one ratio:

        cell detuning  ~  (dgap / gap) * f        [50 um on 10 mm  ->  ~75 MHz]
        cell coupling  ~  the multiplet spread    [measured ~76 MHz]

    Measured on a representative geometry these are the SAME SIZE, so 50 um sits
    right at the crossover between "tolerable" and "the mode localises into the
    widest cell and C collapses". There is no arguing this one from scale alone.

WHAT IS VARIED
    Every rectangle independently -- both dividers and both side toasts separately,
    which BREAKS the mirror symmetry and is the whole point. Per rectangle:
    width, height, x-centre, y-centre. Plus cavity width, cavity height, and the
    tuning angle. 5*4 + 2 + 1 = 23 independent perturbations.

NO CHANGES TO fem_solve ARE NEEDED
    CavitySpec already takes an arbitrary list of Rects, so an asymmetric geometry
    is expressible as-is. All this module adds is a builder that produces one.
"""

from __future__ import annotations

import os
import csv
import time
from dataclasses import dataclass, asdict, field

import numpy as np

from . import fem_solve as fem


MM = 1e-3

# ── defaults matching the MCMC ──────────────────────────────────────────────
GAP0 = 10.0          # mm
GAP1 = 10.0
CAVITY_HEIGHT = 160.0
X_MAX_FREQ = 8.75
N_MODES = 8          # a touch more than the MCMC: asymmetry splits the multiplet
MESH_SIZE = 0.001   # m
PENALTY = 1e33
C_FLOOR = 0.05

ALUMINIUM = fem.Material("aluminium", sigma=fem.SIGMA_AL_COMSOL)

# the five metal rectangles, in the order used everywhere below
PARTS = ("ctr", "divL", "divR", "sideL", "sideR")


# ═════════════════════════════════════════════════════════════════════════════
# tolerances
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Tolerances:
    """
    1-sigma machining/assembly errors (standard deviations). LENGTHS IN MILLIMETERS, angle in degrees.

    Suggested starting points, and why they differ:

      width, height  : the machined dimension of each bar. Good CNC holds
                       +/-25 um (0.001"), routine work +/-50 um. 0.025 is a
                       realistic default; 0.05 is conservative.
      position       : where the bar actually ends up once mounted. This is
                       usually WORSE than the dimension tolerance and it is the
                       one that matters most, because it changes the GAPS
                       directly and the gap sets the frequency. 0.05 mm is
                       optimistic for a bolted assembly; try 0.1 too.
      cavity_w/h     : a large machined part, +/-50 um typical.
      angle_deg      : NOT a machined dimension -- it is the alignment of the
                       tuning trajectory, set by the stage. 1 deg is very
                       conservative for a linear/rotary stage; 0.1-0.3 deg is
                       achievable. Keep 1 deg as a worst case, but note that at
                       theta = 0 a 1 deg error injects |x|*tan(1 deg) = 0.15 mm
                       of unwanted vertical motion, which is 3x your length
                       tolerance -- so at low theta the stage alignment, not the
                       machining, may dominate.

    Set `correlated_parts=True` to model parts cut from one setup (all five bars
    share a width error) rather than five independent errors.
    """
    width: float = 0.05
    height: float = 0.05
    position: float = 0.05
    cavity_w: float = 0.05
    cavity_h: float = 0.05
    angle_deg: float = 1
    correlated_parts: bool = False

    @classmethod
    def uniform_length(cls, length_err=0.05, angle_err=1.0, **kw):
        """One number for every length, as in the original sketch."""
        return cls(width=length_err, height=length_err, position=length_err,
                   cavity_w=length_err, cavity_h=length_err,
                   angle_deg=angle_err, **kw)


# ═════════════════════════════════════════════════════════════════════════════
# geometry: nominal -> perturbed -> CavitySpec
# ═════════════════════════════════════════════════════════════════════════════

def nominal_geometry(x0_mm, gap0=GAP0, gap1=GAP1, cavity_h=CAVITY_HEIGHT):
    """
    7-vector [angle, div_h, div_w, ctr_w, side_w, ctr_h, side_h] (mm)
    -> the full asymmetric description, with left/right split apart.

    Layout outward from the middle:
        [ctr_w] [gap0] [div_w] [gap1] [side_w] [gap1] | wall
    """
    x = np.asarray(x0_mm, dtype=np.float64).ravel()
    if x.size == 8:                       # legacy vector with gap1 inside
        x = np.delete(x, 3)
    angle, div_h, div_w, ctr_w, side_w, ctr_h, side_h = [float(v) for v in x]

    x_div = ctr_w / 2 + gap0 + div_w / 2
    x_side = ctr_w / 2 + gap0 + div_w + gap1 + side_w / 2
    cav_w = ctr_w + 2 * gap0 + 2 * div_w + 4 * gap1 + 2 * side_w

    g = {"angle": angle, "cav_w": cav_w, "cav_h": cavity_h}
    for name, w, h, xc in (("ctr",   ctr_w,  ctr_h,  0.0),
                           ("divL",  div_w,  div_h, -x_div),
                           ("divR",  div_w,  div_h, +x_div),
                           ("sideL", side_w, side_h, -x_side),
                           ("sideR", side_w, side_h, +x_side)):
        g[f"{name}_w"] = w; g[f"{name}_h"] = h
        g[f"{name}_x"] = xc; g[f"{name}_y"] = 0.0
    return g


def perturb_geometry(g, tol: Tolerances, rng=None):
    """Draw one Gaussian-perturbed geometry. All 23 dimensions independently."""
    rng = rng or np.random.default_rng()
    p = dict(g)
    p["angle"] += rng.normal(0.0, tol.angle_deg)
    p["cav_w"] += rng.normal(0.0, tol.cavity_w)
    p["cav_h"] += rng.normal(0.0, tol.cavity_h)

    # one shared draw per dimension if the parts came off a single setup
    shared_w = rng.normal(0.0, tol.width)
    shared_h = rng.normal(0.0, tol.height)
    for name in PARTS:
        dw = shared_w if tol.correlated_parts else rng.normal(0.0, tol.width)
        dh = shared_h if tol.correlated_parts else rng.normal(0.0, tol.height)
        p[f"{name}_w"] = max(p[f"{name}_w"] + dw, 1e-3)
        p[f"{name}_h"] = max(p[f"{name}_h"] + dh, 1e-3)
        p[f"{name}_x"] += rng.normal(0.0, tol.position)
        p[f"{name}_y"] += rng.normal(0.0, tol.position)
    return p


def geometry_gaps(g):
    """The six gaps (mm), left to right. The frequency lives in these."""
    edges = []
    for name in PARTS:
        edges.append((g[f"{name}_x"] - g[f"{name}_w"] / 2,
                      g[f"{name}_x"] + g[f"{name}_w"] / 2, name))
    edges.sort()
    gaps, prev = [], -g["cav_w"] / 2
    for lo, hi, _n in edges:
        gaps.append(lo - prev); prev = hi
    gaps.append(g["cav_w"] / 2 - prev)
    return np.array(gaps)


def geom_to_spec(g, toast_dx=0.0, toast_dy=0.0, mesh_size=MESH_SIZE,
                 tag="", wall_material=ALUMINIUM, metal_material=ALUMINIUM,
                 mesh_uniform=False):
    """
    Build the CavitySpec. toast_dx/dy are in METRES and move the three TOASTS
    (ctr, sideL, sideR) together; the dividers stay fixed, as in the real tuner.
    """
    metal = []
    for name in PARTS:
        moves = name in ("ctr", "sideL", "sideR")
        cx = g[f"{name}_x"] * MM + (toast_dx if moves else 0.0)
        cy = g[f"{name}_y"] * MM + (toast_dy if moves else 0.0)
        metal.append(fem.Rect.from_center(cx, cy, g[f"{name}_w"] * MM,
                                          g[f"{name}_h"] * MM, name))
    return fem.CavitySpec(
        outer=fem.Rect.from_center(0.0, 0.0, g["cav_w"] * MM, g["cav_h"] * MM,
                                   "cavity"),
        metal=metal, mesh_size=mesh_size, mesh_uniform=mesh_uniform,
        wall_material=wall_material, metal_material=metal_material, tag=tag)


def geom_valid(g, min_gap=0.5):
    """Cheap sanity screen: no overlaps, nothing outside the cavity."""
    if min(geometry_gaps(g)) < min_gap:
        return False
    for name in PARTS:
        if abs(g[f"{name}_y"]) + g[f"{name}_h"] / 2 >= g["cav_h"] / 2:
            return False
    return True


# ═════════════════════════════════════════════════════════════════════════════
# evaluate one geometry
# ═════════════════════════════════════════════════════════════════════════════

def tuning_positions(g, n=16, gap0=GAP0, x_max=X_MAX_FREQ):
    """(dx, dy, f_guess) in METRES / Hz, using this geometry's own angle."""
    t = np.tan(np.radians(float(g["angle"])))
    for x in -np.linspace(0.0, x_max * MM, n):
        yield float(x), float(abs(x) * t), 3e8 / (2.0 * (gap0 * MM + abs(x)))


def evaluate_geometry(g, tuning_steps=16, mesh_size=MESH_SIZE, n_workers=None,
                      timeout=600, c_cutoff=True, min_localisation=0.0):
    """
    Full tuning sweep of one (possibly asymmetric) geometry.

    Returns dict with per-step arrays and the scan-time FOM, matching the MCMC's
    definition:  FOM = integral f^2 / (V^2 C^2 Q) df.
    """
    positions = list(tuning_positions(g, n=tuning_steps))
    specs, results = fem.run_sweep(
        lambda dx, dy, i: geom_to_spec(g, toast_dx=dx, toast_dy=dy,
                                       mesh_size=mesh_size,
                                       tag=f"x={dx*1e3:.2f}mm"),
        positions, n_modes=N_MODES, n_workers=n_workers, timeout=timeout,
        verbose=False)

    C, Q, f, V, loc = [], [], [], [], []
    n_failed = 0
    for r in results:
        if not r["ok"] or not r.get("modes"):
            n_failed += 1; continue
        m = fem.best_mode(r, min_localisation=min_localisation)
        if m is None:
            n_failed += 1; continue
        C.append(m["C"]); Q.append(m["Q"]); f.append(m["f"])
        V.append(m["area"]); loc.append(m["localisation"])

    C, Q, f, V, loc = map(np.asarray, (C, Q, f, V, loc))
    out = {"C": C, "Q": Q, "f": f, "V": V, "loc": loc, "n_failed": n_failed,
           "gaps": geometry_gaps(g)}

    if f.size < 2 or n_failed or (c_cutoff and C.size and C.min() < C_FLOOR):
        out["fom"] = PENALTY
        return out
    fm, Cm = 0.5*(f[:-1]+f[1:]), 0.5*(C[:-1]+C[1:])
    Qm, Vm = 0.5*(Q[:-1]+Q[1:]), 0.5*(V[:-1]+V[1:])
    val = float(np.sum(fm**2 / (Vm**2 * Cm**2 * Qm) * np.abs(np.diff(f))))
    out["fom"] = val if (np.isfinite(val) and val > 0) else PENALTY
    return out


# ═════════════════════════════════════════════════════════════════════════════
# the study
# ═════════════════════════════════════════════════════════════════════════════

def check_stability(x0_mm, num_samples=100, tol: Tolerances | None = None,
                    theta_err=None, length_err=None,
                    tuning_steps=16, mesh_size=MESH_SIZE, n_workers=None,
                    gap0=GAP0, gap1=GAP1, cavity_h=CAVITY_HEIGHT,
                    seed=None, save_path=None, verbose=True):
    """
    Monte-Carlo the machining tolerances around a nominal design.

    x0_mm : the nominal 7-vector [angle, div_h, div_w, ctr_w, side_w, ctr_h, side_h].
    tol   : a Tolerances instance. For the simple "one sigma for all lengths"
            behaviour pass theta_err / length_err instead (in degrees / mm).

    Every sample perturbs all 23 dimensions independently -- the two dividers and
    the two side toasts separately -- so the mirror symmetry is genuinely broken.

    Returns (nominal_result, samples, summary).
    """
    if tol is None:
        tol = (Tolerances.uniform_length(length_err if length_err is not None else 0.05,
                                         theta_err if theta_err is not None else 1.0)
               if (theta_err is not None or length_err is not None)
               else Tolerances())
    rng = np.random.default_rng(seed)
    g0 = nominal_geometry(x0_mm, gap0, gap1, cavity_h)

    if verbose:
        print(f"[stability] nominal gaps (mm): "
              f"{np.round(geometry_gaps(g0), 4)}")
        print(f"[stability] tolerances: {asdict(tol)}")
        print(f"[stability] {num_samples} samples x {tuning_steps} tuning steps",
              flush=True)

    t0 = time.perf_counter()
    nominal = evaluate_geometry(g0, tuning_steps, mesh_size, n_workers)
    if verbose:
        print(f"[stability] nominal: FOM={nominal['fom']:.6g}  "
              f"meanC={nominal['C'].mean():.4f}  meanQ={nominal['Q'].mean():.4g}  "
              f"band {nominal['f'].min()/1e9:.3f}-{nominal['f'].max()/1e9:.3f} GHz "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)

    samples, n_rejected = [], 0
    for i in range(num_samples):
        g = perturb_geometry(g0, tol, rng)
        if not geom_valid(g):
            n_rejected += 1
            continue
        r = evaluate_geometry(g, tuning_steps, mesh_size, n_workers)
        r["geom"] = g
        samples.append(r)
        if verbose:
            print(f"  [{i+1}/{num_samples}] FOM={r['fom']:.4g}  "
                  f"minC={r['C'].min() if r['C'].size else float('nan'):.4f}  "
                  f"gapspread={r['gaps'].max()-r['gaps'].min():.4f} mm", flush=True)

    summary = summarise(nominal, samples, n_rejected, verbose=verbose)
    if save_path:
        write_csv(save_path, nominal, samples)
    return nominal, samples, summary


def summarise(nominal, samples, n_rejected=0, verbose=True):
    """Spread of every quantity, and the failure fraction."""
    ok = [s for s in samples if s["fom"] < PENALTY]
    def col(key, red):
        return np.array([red(s[key]) for s in ok if np.size(s[key])])
    out = {"n_samples": len(samples), "n_rejected_geometry": n_rejected,
           "n_penalty": len(samples) - len(ok),
           "frac_penalty": (len(samples) - len(ok)) / max(1, len(samples))}
    for name, arr, nom in (
            ("fom",   np.array([s["fom"] for s in ok]),      nominal["fom"]),
            ("meanC", col("C", np.mean),  nominal["C"].mean()  if nominal["C"].size else np.nan),
            ("minC",  col("C", np.min),   nominal["C"].min()   if nominal["C"].size else np.nan),
            ("meanQ", col("Q", np.mean),  nominal["Q"].mean()  if nominal["Q"].size else np.nan),
            ("fmin",  col("f", np.min),   nominal["f"].min()   if nominal["f"].size else np.nan),
            ("fmax",  col("f", np.max),   nominal["f"].max()   if nominal["f"].size else np.nan),
            ("minloc", col("loc", np.min), nominal["loc"].min() if nominal["loc"].size else np.nan)):
        if arr.size == 0:
            continue
        out[name] = {"nominal": float(nom), "mean": float(arr.mean()),
                     "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
                     "min": float(arr.min()), "max": float(arr.max()),
                     "p5": float(np.percentile(arr, 5)),
                     "p95": float(np.percentile(arr, 95))}
        out[name]["rel_std"] = (out[name]["std"] / abs(out[name]["mean"])
                                if out[name]["mean"] else np.nan)
    if verbose:
        print(f"\n[stability] {out['n_samples']} evaluated, "
              f"{out['n_rejected_geometry']} rejected before solving, "
              f"{out['n_penalty']} hit the penalty "
              f"({100*out['frac_penalty']:.1f}%)")
        print(f"  {'quantity':>8} {'nominal':>12} {'mean':>12} {'std':>11} "
              f"{'rel std':>9} {'p5':>12} {'p95':>12}")
        for k in ("fom", "meanC", "minC", "meanQ", "fmin", "fmax", "minloc"):
            if k not in out:
                continue
            d = out[k]
            print(f"  {k:>8} {d['nominal']:>12.5g} {d['mean']:>12.5g} "
                  f"{d['std']:>11.4g} {d['rel_std']:>9.2%} {d['p5']:>12.5g} "
                  f"{d['p95']:>12.5g}")
    return out


def write_csv(path, nominal, samples):
    """One row per sample: the FOM, the summary observables, and every dimension."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    geom_keys = sorted(samples[0]["geom"].keys()) if samples else []
    hdr = (["sample", "fom", "meanC", "minC", "meanQ", "fmin", "fmax", "minloc",
            "n_failed", "gap_spread"] + geom_keys)
    with open(path, "w", newline="") as fh:
        wtr = csv.writer(fh); wtr.writerow(hdr)
        for i, s in enumerate(samples):
            C, Q, f, loc = s["C"], s["Q"], s["f"], s["loc"]
            wtr.writerow([i, s["fom"],
                          C.mean() if C.size else "", C.min() if C.size else "",
                          Q.mean() if Q.size else "",
                          f.min() if f.size else "", f.max() if f.size else "",
                          loc.min() if loc.size else "",
                          s["n_failed"], s["gaps"].max() - s["gaps"].min()]
                         + [s["geom"][k] for k in geom_keys])
    return path