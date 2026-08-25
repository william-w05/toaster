r"""
Screening diagnostics for the MCMC surrogate.

    topk    Retrospective precision@k per holdout window, rebuilt from the
            checkpoints. Self-validates against the stored rmse_log.
            This is a FLOOR, not an estimate -- see the note it prints.

    batch   Replicate real screening batches with the exact _batch_proposals
            kernel (Student-t df=3, clipped, linear mode) around the current
            walker positions, filter with proposed_params_within_limits, rank
            with a checkpoint, and emit candidates for FEM evaluation.
            Prints the zero-cost spread diagnostic first.

    score   Ingest evaluated batches and report WITHIN-BATCH precision@1,
            within-batch rank correlation, and the expected gain in nats from
            screening versus taking a random feasible proposal.

WHY WITHIN-BATCH.  Screening is argmin over one batch from one walker. Every
row of fem_evals.csv is the winner of its own batch, so the historical data
contains zero within-batch comparisons -- pooled Spearman there mixes walker
drift (easy, irrelevant) with within-batch discrimination (hard, the only thing
that matters). Only exhaustively evaluating a few real batches measures it.

Typical use:
    python surrogate_screening_eval.py topk
    python surrogate_screening_eval.py batch --n-full-batches 5   # 320 FEM calls
    python surrogate_screening_eval.py score --proposals batches_evaluated.csv
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, kendalltau, hypergeom

try:
    import pandas as pd
except ImportError:
    pd = None

# --------------------------------------------------------------------------
# constants mirrored from the sampler -- keep in sync
# --------------------------------------------------------------------------

PARAM_NAMES = ["angle", "div_h", "div_w", "ctr_w", "side_w", "ctr_h", "side_h"]
N_PARAMS = 7
I_ANGLE, I_DIVH, I_DIVW, I_CTRW, I_SIDEW, I_CTRH, I_SIDEH = range(N_PARAMS)

ANGLE_MIN, ANGLE_MAX = 0.0, 50.0
H_MIN, H_MAX = 90.0, 145.0
H_TOL = 0.2
CTR_W_MIN, CTR_W_MAX = 3.0, 20.0
SIDE_W_MIN, SIDE_W_MAX = 3.0, 20.0
# NOTE: SIDE_W_TOL was 0.2 when the 08_13 run was produced and is 0.4 in the
# current sampler. Replaying an archived run needs the value that was in force
# then, or the feasible pool will not match. Override with --side-w-tol.
SIDE_W_TOL = 0.4
DIV_W_MIN = 3.0
TOTAL_W_MAX = 400.0 / np.sqrt(2.0)
GAP0 = GAP1 = 10
CAVITY_HEIGHT = 160

PENALTY = 1e33
LOG_PENALTY_THR = np.log(PENALTY) - 1e-6      # natural log, matches _split_penalty
TEMP0 = 1.0
COOLING = 0.999
TEMP_MIN = 1e-3
C_FLOOR = 0.05                                # the c_cutoff form-factor threshold
N_WALKERS = 10
PROPOSAL_STD_FRAC = 0.1        # fraction of each parameter's own range
PROPOSAL_DF = 3
PROPOSAL_CLIP = 2.0
N_SCREEN = 64

# Per-parameter range used to turn the fraction into an absolute step.
# CHECK THIS: div_w's upper limit is GAP0, not a named DIV_W_MAX, so if your
# code uses a different span for it these numbers shift.
PARAM_RANGE = np.array([
    ANGLE_MAX - ANGLE_MIN,          # angle   50
    H_MAX - H_MIN,                  # div_h   55
    GAP0 - DIV_W_MIN,               # div_w    7
    CTR_W_MAX - CTR_W_MIN,          # ctr_w   17
    SIDE_W_MAX - SIDE_W_MIN,        # side_w  17
    H_MAX - H_MIN,                  # ctr_h   55
    H_MAX - H_MIN,                  # side_h  55
], dtype=np.float64)

DEFAULT_ROOT = os.path.join("results", "08_13_2026_mcmc_results")
CKPT_RE = re.compile(r"surrogate_step_?(\d+)\.pt$")
K_VALUES = [1, 5, 10, 25, 50, 0.05, 0.10, 0.25]


# --------------------------------------------------------------------------
# feasibility
# --------------------------------------------------------------------------

def within_limits_scalar(p):
    """Verbatim transcription of proposed_params_within_limits, for cross-check."""
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (N_PARAMS,):
        return False
    theta = p[I_ANGLE]
    div_h, div_w = p[I_DIVH], p[I_DIVW]
    ctr_w, side_w = p[I_CTRW], p[I_SIDEW]
    ctr_h, side_h = p[I_CTRH], p[I_SIDEH]

    if np.any(p[1:] <= 0) or np.any(p >= 160):
        return False
    if theta < ANGLE_MIN or theta > ANGLE_MAX:
        return False
    for h in (div_h, ctr_h, side_h):
        if h < H_MIN or h > H_MAX:
            return False
    if ctr_h > CAVITY_HEIGHT - 2 * GAP0 * np.abs(np.tan(np.radians(theta))):
        return False
    if div_h <= (1 - H_TOL) * ctr_h or div_h >= (1 + H_TOL) * ctr_h:
        return False
    if side_h <= (1 - H_TOL) * ctr_h or side_h >= (1 + H_TOL) * ctr_h:
        return False
    if ctr_w < CTR_W_MIN or ctr_w > CTR_W_MAX:
        return False
    if side_w < SIDE_W_MIN or side_w > SIDE_W_MAX:
        return False
    if side_w >= (1 + SIDE_W_TOL) * ctr_w or side_w < (1 - SIDE_W_TOL) * ctr_w:
        return False
    if div_w < DIV_W_MIN or div_w >= GAP0:
        return False
    if (ctr_w + 2 * GAP0 + 2 * div_w + 4 * GAP1 + 2 * side_w) >= TOTAL_W_MAX:
        return False
    return True


def within_limits(P):
    """Vectorised over rows of P (N, 7)."""
    P = np.atleast_2d(np.asarray(P, dtype=np.float64))
    theta = P[:, I_ANGLE]
    div_h, div_w = P[:, I_DIVH], P[:, I_DIVW]
    ctr_w, side_w = P[:, I_CTRW], P[:, I_SIDEW]
    ctr_h, side_h = P[:, I_CTRH], P[:, I_SIDEH]

    ok = np.all(P[:, 1:] > 0, axis=1) & np.all(P < 160, axis=1)
    ok &= (theta >= ANGLE_MIN) & (theta <= ANGLE_MAX)
    for h in (div_h, ctr_h, side_h):
        ok &= (h >= H_MIN) & (h <= H_MAX)
    ok &= ctr_h <= CAVITY_HEIGHT - 2 * GAP0 * np.abs(np.tan(np.radians(theta)))
    ok &= (div_h > (1 - H_TOL) * ctr_h) & (div_h < (1 + H_TOL) * ctr_h)
    ok &= (side_h > (1 - H_TOL) * ctr_h) & (side_h < (1 + H_TOL) * ctr_h)
    ok &= (ctr_w >= CTR_W_MIN) & (ctr_w <= CTR_W_MAX)
    ok &= (side_w >= SIDE_W_MIN) & (side_w <= SIDE_W_MAX)
    ok &= (side_w < (1 + SIDE_W_TOL) * ctr_w) & (side_w >= (1 - SIDE_W_TOL) * ctr_w)
    ok &= (div_w >= DIV_W_MIN) & (div_w < GAP0)
    ok &= (ctr_w + 2 * GAP0 + 2 * div_w + 4 * GAP1 + 2 * side_w) < TOTAL_W_MAX
    return ok


def constraint_breakdown(P):
    """First-failure attribution, in the order the original function tests."""
    P = np.atleast_2d(np.asarray(P, dtype=np.float64))
    theta = P[:, I_ANGLE]
    div_h, div_w = P[:, I_DIVH], P[:, I_DIVW]
    ctr_w, side_w = P[:, I_CTRW], P[:, I_SIDEW]
    ctr_h, side_h = P[:, I_CTRH], P[:, I_SIDEH]
    tests = [
        ("positivity / <160", np.all(P[:, 1:] > 0, axis=1) & np.all(P < 160, axis=1)),
        ("angle range", (theta >= ANGLE_MIN) & (theta <= ANGLE_MAX)),
        ("div_h in [H_MIN,H_MAX]", (div_h >= H_MIN) & (div_h <= H_MAX)),
        ("ctr_h in [H_MIN,H_MAX]", (ctr_h >= H_MIN) & (ctr_h <= H_MAX)),
        ("side_h in [H_MIN,H_MAX]", (side_h >= H_MIN) & (side_h <= H_MAX)),
        ("ctr_h wall clearance",
         ctr_h <= CAVITY_HEIGHT - 2 * GAP0 * np.abs(np.tan(np.radians(theta)))),
        ("div_h within +/-20% ctr_h",
         (div_h > (1 - H_TOL) * ctr_h) & (div_h < (1 + H_TOL) * ctr_h)),
        ("side_h within +/-20% ctr_h",
         (side_h > (1 - H_TOL) * ctr_h) & (side_h < (1 + H_TOL) * ctr_h)),
        ("ctr_w box", (ctr_w >= CTR_W_MIN) & (ctr_w <= CTR_W_MAX)),
        ("side_w box", (side_w >= SIDE_W_MIN) & (side_w <= SIDE_W_MAX)),
        ("side_w within +/-20% ctr_w",
         (side_w < (1 + SIDE_W_TOL) * ctr_w) & (side_w >= (1 - SIDE_W_TOL) * ctr_w)),
        ("div_w in [3, GAP0)", (div_w >= DIV_W_MIN) & (div_w < GAP0)),
        ("total width", (ctr_w + 2 * GAP0 + 2 * div_w + 4 * GAP1
                         + 2 * side_w) < TOTAL_W_MAX),
    ]
    n = len(P)
    alive = np.ones(n, dtype=bool)
    first, marginal = {}, {}
    for name, ok in tests:
        first[name] = int((alive & ~ok).sum())
        marginal[name] = int((~ok).sum())
        alive &= ok
    return first, marginal, int(alive.sum()), n


def selftest_limits(rng, n=4000):
    """Guard against the vectorised form drifting from the scalar one."""
    lo = np.array([0, 85, 2, 2, 2, 85, 85], dtype=float)
    hi = np.array([55, 150, 12, 22, 22, 150, 150], dtype=float)
    P = rng.uniform(lo, hi, size=(n, N_PARAMS))
    v = within_limits(P)
    s = np.array([within_limits_scalar(r) for r in P])
    if not np.array_equal(v, s):
        bad = np.where(v != s)[0][:3]
        raise SystemExit(f"feasibility mismatch at rows {bad}: {P[bad]}")
    return v.mean()


# --------------------------------------------------------------------------
# proposal kernel (mirrors _batch_proposals, PROPOSAL_MODE == 'linear')
# --------------------------------------------------------------------------

def step_std(frac=PROPOSAL_STD_FRAC, absolute=None):
    """Absolute per-dim proposal sd. frac is a fraction of each param's range."""
    if absolute is not None:
        return np.broadcast_to(np.asarray(absolute, dtype=np.float64),
                               (N_PARAMS,)).copy()
    return float(frac) * PARAM_RANGE


def batch_proposals(x, proposal_std=None, n=N_SCREEN,
                    df=PROPOSAL_DF, clip=PROPOSAL_CLIP, rng=None):
    """n heavy-tailed proposals around the physical point x (N_PARAMS,)."""
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=np.float64)
    d = x.shape[0]
    if proposal_std is None:
        proposal_std = step_std()
    z = rng.standard_normal((n, d))
    chi2 = rng.chisquare(df, size=(n, d))
    t = np.clip(z / np.sqrt(chi2 / df), -clip, clip)
    return x[None, :] + t * np.asarray(proposal_std, dtype=np.float64)


def clipped_t_sd(df=PROPOSAL_DF, clip=PROPOSAL_CLIP, rng=None, n=400000):
    """sd of the standardised t step after clipping (~1.118 for df=3, clip=2)."""
    rng = rng or np.random.default_rng(0)
    z = rng.standard_normal(n)
    c = rng.chisquare(df, size=n)
    return float(np.clip(z / np.sqrt(c / df), -clip, clip).std(ddof=1))


def effective_step_sd(std, df=PROPOSAL_DF, clip=PROPOSAL_CLIP):
    return np.asarray(std, dtype=np.float64) * clipped_t_sd(df, clip)


def infer_step_scale(hist_X, n_walkers=N_WALKERS, n_recent=3000):
    """
    Robust per-dim scale of lag-n_walkers differences in arrival order, i.e.
    how far a walker's evaluated point moves between steps. Compare against the
    assumed proposal sd to confirm the fraction-vs-absolute interpretation.
    """
    H = hist_X[-n_recent:]
    if len(H) < 4 * n_walkers:
        return None
    d = H[n_walkers:] - H[:-n_walkers]
    d = d[np.any(d != 0, axis=1)]
    if len(d) < 10:
        return None
    return 1.4826 * np.median(np.abs(d - np.median(d, 0)), axis=0)


# --------------------------------------------------------------------------
# checkpoint
# --------------------------------------------------------------------------

class SurrogateNet(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Ckpt:
    def __init__(self, path, step, device="cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        self.path, self.step, self.device = path, step, device
        sd = ck["net"]
        hidden, d = sd["net.0.weight"].shape
        self.d, self.hidden = int(d), int(hidden)
        self.net = SurrogateNet(self.d, self.hidden).to(device=device,
                                                        dtype=torch.float64)
        self.net.load_state_dict(sd)
        self.net.eval()
        self.Xmu = np.asarray(ck["Xmu"], dtype=np.float64)
        self.Xsi = np.asarray(ck["Xsi"], dtype=np.float64)
        self.ymu, self.ysi = float(ck["ymu"]), float(ck["ysi"])
        self.trained = bool(ck["trained"])
        self.n_trained_on = int(ck["n_trained_on"])
        self.hist_X = np.asarray(ck["hist_X"], dtype=np.float64)
        self.hist_y = np.asarray(ck["hist_y"], dtype=np.float64)
        self.rmse_log = list(ck.get("rmse_log", []))

    @property
    def n_obs(self):
        return len(self.hist_y)

    @property
    def saved_post_fit(self):
        return self.n_trained_on >= self.n_obs

    @torch.no_grad()
    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        Xn = (X - self.Xmu) / self.Xsi
        out = self.net(torch.tensor(Xn, dtype=torch.float64,
                                    device=self.device)).cpu().numpy()
        return np.atleast_1d(out) * self.ysi + self.ymu

    def __repr__(self):
        return (f"<Ckpt step={self.step} d={self.d} hidden={self.hidden} "
                f"n_obs={self.n_obs} n_trained_on={self.n_trained_on}>")


def find_checkpoints(d):
    out = [(int(m.group(1)), os.path.join(d, fn))
           for fn in os.listdir(d) if (m := CKPT_RE.search(fn))]
    if not out:
        raise SystemExit(f"no surrogate_step#####.pt in {d}")
    return sorted(out)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def resolve_k(k, n):
    kk = max(1, int(round(k * n))) if isinstance(k, float) else int(k)
    return kk if kk <= n else None


def ktag(k):
    return f"{int(k * 100)}pct" if isinstance(k, float) else str(k)


def ranking_metrics(y_true, y_pred, k_values=K_VALUES):
    """Natural-log objective, MINIMISED. Penalties already removed."""
    n = len(y_true)
    out = {"N": n}
    out["Spearman"] = float(spearmanr(y_pred, y_true).statistic)
    out["Kendall"] = float(kendalltau(y_pred, y_true).statistic)
    out["RMSE"] = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    out["Pearson"] = float(np.corrcoef(y_pred, y_true)[0, 1])
    out["sigma_true"] = float(np.std(y_true, ddof=1))
    out["sigma_pred"] = float(np.std(y_pred, ddof=1))
    out["R2"] = 1.0 - out["RMSE"] ** 2 / max(np.var(y_true), 1e-300)
    out["bias"] = float(np.mean(y_pred - y_true))
    out["calib_ratio"] = out["sigma_pred"] / max(out["sigma_true"], 1e-300)
    out["calib_target"] = out["Pearson"]
    out["slope_true_on_pred"] = float(np.polyfit(y_pred, y_true, 1)[0])
    out["RMSE_floor"] = out["sigma_true"] * np.sqrt(max(0.0, 1 - out["Pearson"] ** 2))

    best = float(np.min(y_true))
    ranks = np.empty(n)
    ranks[np.argsort(y_true, kind="stable")] = np.arange(n)
    for k in k_values:
        kk, t = resolve_k(k, n), ktag(k)
        if kk is None:
            out[f"P@{t}"] = np.nan
            continue
        true_top = set(np.argsort(y_true, kind="stable")[:kk].tolist())
        pred_top = np.argsort(y_pred, kind="stable")[:kk]
        hits = len(true_top & set(pred_top.tolist()))
        out[f"P@{t}"] = hits / kk
        out[f"P@{t}_base"] = kk / n
        out[f"P@{t}_lift"] = (hits / kk) / (kk / n)
        out[f"P@{t}_p"] = float(hypergeom.sf(hits - 1, n, kk, kk))
        out[f"regret@{t}"] = float(np.min(y_true[pred_top]) - best)
        out[f"meanpct@{t}"] = float(np.mean(ranks[pred_top]) / max(n - 1, 1))
    return out


def drop_penalty(X, y):
    ok = (y < LOG_PENALTY_THR) & np.isfinite(y)
    return X[ok], y[ok], ok


# --------------------------------------------------------------------------
# mode: topk
# --------------------------------------------------------------------------

def build_windows(ckpts, n_test):
    pre = [c for c in ckpts if not c.saved_post_fit]
    if len(pre) >= max(2, len(ckpts) // 2):
        print(f"[align] checkpoints look PRE-fit ({len(pre)}/{len(ckpts)}); "
              f"windows are self-contained.")
        for c in ckpts:
            if not c.trained or c.saved_post_fit:
                continue
            n = min(n_test, c.n_obs - c.n_trained_on)
            if n >= 3:
                yield c, c.hist_X[-n:], c.hist_y[-n:]
        return
    print("[align] checkpoints look POST-fit; pairing ckpt i's net with "
          "ckpt i+1's new observations.")
    for a, b in zip(ckpts[:-1], ckpts[1:]):
        if not a.trained:
            continue
        lo, hi = a.n_trained_on, b.n_obs
        if hi - lo >= 3:
            yield a, b.hist_X[max(lo, hi - n_test):hi], b.hist_y[max(lo, hi - n_test):hi]


def mode_topk(args):
    ckpts = [Ckpt(p, s, args.device) for s, p in find_checkpoints(args.ckpt_dir)]
    print(f"loaded {len(ckpts)} checkpoints: {ckpts[0]} ... {ckpts[-1]}")
    recorded = {int(e[0]): tuple(e[1:5]) for e in ckpts[-1].rmse_log if len(e) >= 5}

    rows = []
    for c, X, y in build_windows(ckpts, args.n_test):
        n_raw = len(y)
        X, y, _ = drop_penalty(X, y)
        if len(y) < 10:
            continue
        m = ranking_metrics(y, c.predict(X))
        m.update({"CkptStep": c.step, "NRaw": n_raw, "NPenalty": n_raw - len(y),
                  "PenaltyFrac": (n_raw - len(y)) / n_raw})
        for key in (c.n_trained_on + n_raw, c.n_trained_on, c.n_obs):
            if key in recorded:
                nt, rmse, rho, tau = recorded[key]
                m.update({"rec_N": nt, "rec_RMSE": rmse,
                          "rec_Spearman": rho, "rec_Kendall": tau})
                break
        rows.append(m)

    if not rows or pd is None:
        raise SystemExit("no windows produced metrics (or pandas missing)")
    res = pd.DataFrame(rows)
    front = ["CkptStep", "N", "NPenalty", "RMSE", "RMSE_floor", "sigma_true",
             "R2", "Spearman", "Kendall", "Pearson", "calib_ratio", "calib_target"]
    res = res[[c for c in front if c in res] +
              [c for c in res.columns if c not in front]]
    res.to_csv(args.out, index=False)

    if "rec_RMSE" in res:
        dn = (res["N"] - res["rec_N"]).abs().max()
        dr = (res["RMSE"] - res["rec_RMSE"]).abs().max()
        ds = (res["Spearman"] - res["rec_Spearman"]).abs().max()
        print(f"\n=== VALIDATION vs stored rmse_log ===")
        print(f"  max|dN| {dn:.0f}   max|dRMSE| {dr:.3e}   max|dSpearman| {ds:.3e}")
        print("  MATCH -- reconstruction exact." if dr < 1e-6 and dn == 0
              else "  MISMATCH -- alignment is wrong; everything below is suspect.")
    else:
        print("\n!! no rmse_log entries matched; cannot self-validate.")

    tags = [c for c in res.columns
            if c.startswith("P@") and not c.endswith(("_base", "_lift", "_p"))]
    print("\n=== TOP-K PRECISION PER WINDOW ===")
    print(res[["CkptStep", "N"] + tags].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== SUMMARY ===")
    for t in tags:
        tg = t[2:]
        print(f"  {t:>12s} mean={res[t].mean():.3f}  "
              f"chance={res[f'P@{tg}_base'].mean():.3f}  "
              f"lift={res[f'P@{tg}_lift'].mean():5.2f}x  "
              f"p<0.05 in {int((res[f'P@{tg}_p'] < 0.05).sum())}/{len(res)}")
    print(f"\n  RMSE {res['RMSE'].mean():.4f}  floor {res['RMSE_floor'].mean():.4f}  "
          f"sigma_true {res['sigma_true'].mean():.4f}")
    print(f"  calib_ratio {res['calib_ratio'].mean():.3f} "
          f"(MSE-optimal = r = {res['calib_target'].mean():.3f})")
    print(f"\nwrote {args.out}")
    print("\nFLOOR, NOT ESTIMATE: every point here is the argmin winner of its own\n"
          "batch, and no two points here ever competed. Run `batch` for the number\n"
          "that describes screening.")


# --------------------------------------------------------------------------
# mode: batch
# --------------------------------------------------------------------------

def walker_positions(ck, args):
    if args.walkers and os.path.isfile(args.walkers):
        w = pd.read_csv(args.walkers)
        xs = sorted([c for c in w.columns if re.fullmatch(r"x\d+", c)],
                    key=lambda c: int(c[1:]))
        if not xs:
            xs = [c for c in w.columns if c in PARAM_NAMES] or list(w.columns[-7:])
        P = w[xs].to_numpy(float)[-args.n_walkers:]
        print(f"[batch] walker positions from {args.walkers}")
        return P
    print("!! no --walkers file; using the last evaluated points as walker state.\n"
          "   Those are screened proposals, not accepted positions, so each batch\n"
          "   centre is off by roughly one full proposal step (~6mm in the heights,\n"
          "   ~2mm in the widths). The batches are still valid clouds, but they are\n"
          "   not centred where the sampler actually sits. Pass --walkers if you can.")
    return ck.hist_X[-args.n_walkers:]


def load_fom(args):
    """Import the sampler module and return its fom callable."""
    if args.mcmc_path:
        sys.path.insert(0, os.path.abspath(args.mcmc_path))
    sys.path.insert(0, os.getcwd())
    tried, mod, last = [], None, ""
    for name in [args.mcmc_module, "scripts.mcmc", "mcmc"]:
        if name in tried:
            continue
        tried.append(name)
        try:
            mod = importlib.import_module(name)
            break
        except Exception as e:                                # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
    if mod is None:
        raise SystemExit(f"could not import any of {tried} ({last})\n"
                         f"Run from the project root, or pass --mcmc-path DIR.")
    fn = getattr(mod, args.fom_attr, None)
    if not callable(fn):
        cands = [a for a in dir(mod) if "fom" in a.lower()]
        raise SystemExit(f"'{mod.__name__}' has no callable '{args.fom_attr}'"
                         + (f"; did you mean one of {cands}?" if cands else ""))
    print(f"[eval] using {mod.__name__}.{args.fom_attr}")
    return fn


def evaluate_rows(df, call, path, args, col="fom", rows=None,
                  key_cols=("batch_id", "cand_index")):
    """
    Fill `col` by calling the real FEM. `call` takes a (7,) array.

    Rows are evaluated in order, so an interrupted run leaves whole usable
    groups. Results flush to disk every --flush-every calls and on Ctrl-C, and
    --resume matches previous values on `key_cols`.
    """
    xs = [f"x{j}" for j in range(N_PARAMS)]
    key_cols = [c for c in key_cols if c in df.columns]
    ecol, scol = f"{col}_error", f"{col}_seconds"
    for c, v in ((col, np.nan), (ecol, ""), (scol, np.nan)):
        if c not in df.columns:
            df[c] = v
    df[col] = pd.to_numeric(df[col], errors="coerce")

    if args.resume and os.path.isfile(path):
        old = pd.read_csv(path)
        if set(key_cols + [col]) <= set(old.columns):
            prev = (old.assign(**{col: pd.to_numeric(old[col], errors="coerce")})
                       .dropna(subset=[col])
                       .drop_duplicates(subset=key_cols)
                       .set_index(key_cols)[col])
            keys = pd.MultiIndex.from_frame(df[key_cols])
            df[col] = df[col].fillna(
                pd.Series(keys.map(prev), index=df.index, dtype="float64"))
            n = int(df[col].notna().sum())
            if n:
                print(f"[eval] resumed {n} '{col}' values from {path}")

    pool = df.index if rows is None else pd.Index(rows)
    todo = [i for i in pool if pd.isna(df.loc[i, col])]
    if args.limit:
        todo = todo[:args.limit]
    print(f"[eval] {len(todo)} FEM calls queued for '{col}' "
          f"({int(df[col].notna().sum())}/{len(df)} filled)")
    if not todo:
        return df

    t0, done = time.perf_counter(), 0
    try:
        for i in todo:
            x = df.loc[i, xs].to_numpy(dtype=np.float64)
            t1 = time.perf_counter()
            try:
                v = float(call(x))
            except Exception as e:                            # noqa: BLE001
                v = np.nan
                df.loc[i, ecol] = f"{type(e).__name__}: {e}"
                print(f"  !! row {i} failed: {type(e).__name__}: {e}", flush=True)
            dt = time.perf_counter() - t1
            df.loc[i, col] = v
            df.loc[i, scol] = dt
            done += 1
            rate = (time.perf_counter() - t0) / done
            lg = np.log(v) if np.isfinite(v) and v > 0 else float("nan")
            flag = "  PENALTY" if np.isfinite(v) and v >= PENALTY * (1 - 1e-9) else ""
            tag = "/".join(str(df.loc[i, c]) for c in key_cols)
            print(f"  [{done}/{len(todo)}] {tag}  log={lg:.4f}  {dt:.1f}s  "
                  f"ETA {(len(todo)-done)*rate/60:.1f} min{flag}", flush=True)
            if done % args.flush_every == 0:
                df.to_csv(path, index=False)
    except KeyboardInterrupt:
        print("\n[eval] interrupted -- saving partial results "
              "(rerun with --resume to continue)")

    df.to_csv(path, index=False)
    print(f"[eval] wrote {path}: {int(df[col].notna().sum())}/{len(df)} '{col}' "
          f"filled, {int((df[ecol].astype(str) != '').sum())} errors")
    return df


def mode_batch(args):
    rng = np.random.default_rng(args.seed)
    frac = selftest_limits(rng)
    print(f"[selftest] vectorised feasibility matches scalar "
          f"(box acceptance {frac:.3f})")

    ckpts = find_checkpoints(args.ckpt_dir)
    step, path = ckpts[-1] if args.step is None else \
        min(ckpts, key=lambda t: abs(t[0] - args.step))
    ck = Ckpt(path, step, args.device)
    print(f"[batch] using {ck}")

    std = step_std(args.proposal_std, args.absolute_std)
    eff = effective_step_sd(std)
    print("\n=== PROPOSAL SCALE ===")
    print(f"  {'param':<8s} {'range':>8s} {'sd':>9s} {'cap':>9s} {'obs lag-10':>11s}")
    obs = infer_step_scale(ck.hist_X, args.n_walkers)
    for j, nm in enumerate(PARAM_NAMES):
        o = f"{obs[j]:11.4f}" if obs is not None else f"{'n/a':>11s}"
        print(f"  {nm:<8s} {PARAM_RANGE[j]:8.1f} {eff[j]:9.4f} "
              f"{std[j]*PROPOSAL_CLIP:9.4f} {o}")
    if obs is not None:
        ratio = np.median(obs / np.maximum(eff, 1e-12))
        print(f"  median observed/assumed = {ratio:.3f}")
        print("  Near 1 confirms the fraction-of-range reading. Orders of magnitude"
              "\n  below 1 would mean the sd is absolute after all; rerun with"
              "\n  --absolute-std. (Selection biases this slightly low, since the"
              "\n  evaluated points are argmin winners, not raw proposals.)")

    W = walker_positions(ck, args)

    # ---- feasibility bottleneck -------------------------------------------
    pool = np.vstack([batch_proposals(x, std, n=args.n_screen, rng=rng) for x in W])
    first, marginal, alive, ntot = constraint_breakdown(pool)
    print(f"\n=== FEASIBILITY ({alive}/{ntot} = {alive/ntot:.3f} survive) ===")
    print(f"  effective screening pool = {args.n_screen * alive / ntot:.1f} "
          f"of {args.n_screen} candidates")
    print(f"  {'constraint':<28s} {'first-fail':>11s} {'marginal':>10s}")
    for name in first:
        if marginal[name]:
            print(f"  {name:<28s} {100*first[name]/ntot:10.1f}% "
                  f"{100*marginal[name]/ntot:9.1f}%")
    print("\n  'first-fail' attributes each rejection to the earliest rule it trips;"
          "\n  'marginal' is how often each rule fails on its own. A dominant row"
          "\n  means your screening pool is set by geometry, not by the surrogate.")

    # ---- zero-cost spread diagnostic --------------------------------------
    within_sd, feas_rates = [], []
    for x in W:
        C = batch_proposals(x, std, n=args.n_screen, rng=rng)
        f = within_limits(C)
        feas_rates.append(f.mean())
        if f.sum() >= 3:
            within_sd.append(ck.predict(C[f]).std(ddof=1))
    _, hy, ok = drop_penalty(ck.hist_X, ck.hist_y)
    hp = ck.predict(ck.hist_X[ok])
    print("\n=== SPREAD DIAGNOSTIC (no FEM calls) ===")
    print(f"  feasible fraction per batch        : {np.mean(feas_rates):.3f}")
    print(f"  predicted sd WITHIN a batch        : {np.mean(within_sd):.5f}")
    print(f"  predicted sd across history        : {hp.std(ddof=1):.5f}")
    print(f"  true log-FoM sd across history     : {hy.std(ddof=1):.5f}")
    print(f"  ratio within/across (predicted)    : "
          f"{np.mean(within_sd) / max(hp.std(ddof=1), 1e-12):.4f}")
    print("\n  Within-batch spread caps what screening can win: no ranker beats it.")

    # ---- covariate shift: proposals vs the training buffer ----------------
    Xtr = ck.hist_X[ok]
    ztr = (Xtr - ck.Xmu) / ck.Xsi
    feas = pool[within_limits(pool)]
    zpr = (feas - ck.Xmu) / ck.Xsi
    # Disjoint reference and query sets. Striding both would leave ~1/k of the
    # query points inside the reference set, matching themselves at distance 0
    # and deflating the training-to-training baseline.
    perm = rng.permutation(len(ztr))
    ref = ztr[perm[:min(2000, len(ztr) // 2)]]
    qry = ztr[perm[len(ref):len(ref) + 500]]
    nn = np.sqrt(((zpr[:, None, :] - ref[None]) ** 2).sum(-1).min(1))
    nnt = np.sqrt(((qry[:, None, :] - ref[None]) ** 2).sum(-1).min(1))
    ratio = np.median(nn) / max(np.median(nnt), 1e-12)
    print("\n=== COVARIATE SHIFT (standardised space) ===")
    print(f"  NN distance, proposal -> training set : median {np.median(nn):.3f}")
    print(f"  NN distance, training -> training     : median {np.median(nnt):.3f}")
    print(f"  ratio                                 : {ratio:.2f}")
    print(f"  implied local density ratio (^{ck.d})     : {ratio ** ck.d:.1f}x sparser")
    print("  The buffer holds only argmin winners, but screening ranks the whole\n"
          "  proposal cloud. In 7-D a modest distance ratio is a large density\n"
          "  ratio, so much of the within-batch predicted spread may be\n"
          "  extrapolation rather than signal. Only the FEM run separates them.")

    # ---- emit batches for FEM --------------------------------------------
    if pd is None:
        raise SystemExit("pandas required")
    recs = []
    for b in range(min(args.n_full_batches, len(W))):
        x = W[b]
        C = batch_proposals(x, std, n=args.n_screen, rng=rng)
        f = within_limits(C)
        C = C[f]
        if len(C) < 4:
            continue
        pred = ck.predict(C)
        order = np.argsort(pred, kind="stable")
        rank = np.empty(len(C), dtype=int)
        rank[order] = np.arange(len(C))
        keep = np.arange(len(C)) if args.n_per_batch >= len(C) else \
            np.sort(rng.choice(len(C), args.n_per_batch, replace=False))
        # the walker's own position: gives gain-vs-incumbent, not just vs random
        if args.eval_centres and within_limits_scalar(x):
            recs.append({**{f"x{j}": x[j] for j in range(ck.d)},
                         "batch_id": b, "cand_index": -1, "role": "centre",
                         "pred_log_fom": float(ck.predict(x[None])[0]),
                         "pred_rank_in_batch": -1, "ckpt_step": step, "fom": ""})
        for i in keep:
            row = {f"x{j}": C[i, j] for j in range(ck.d)}
            row.update({"batch_id": b, "cand_index": int(i), "role": "candidate",
                        "pred_log_fom": float(pred[i]),
                        "pred_rank_in_batch": int(rank[i]),
                        "ckpt_step": step, "fom": ""})
            recs.append(row)
    df = pd.DataFrame(recs)
    ncand = int((df["role"] == "candidate").sum())
    print(f"\nprepared {df['batch_id'].nunique()} batches, {ncand} candidates"
          + (f" + {len(df) - ncand} centres" if len(df) > ncand else "")
          + f" = {len(df)} FEM calls")

    if not args.evaluate:
        df.to_csv(args.out_proposals, index=False)
        print(f"wrote {args.out_proposals} (not evaluated)")
        print("Rerun with --evaluate to call the FEM directly, or fill 'fom' by "
              "hand (linear space, 1e33 for penalties) and then:")
        print(f"    python {os.path.basename(__file__)} score "
              f"--proposals {args.out_proposals}")
        return

    fom = load_fom(args)
    kw = {} if args.c_cutoff is None else {"c_cutoff": bool(args.c_cutoff)}
    print(f"[eval] calling {args.fom_attr}(x{', c_cutoff=' + str(args.c_cutoff) if kw else ''})")
    df = evaluate_rows(df, lambda x: fom(x, **kw), args.out_proposals, args)

    # Penalised points carry no gradient information as-is. Re-evaluating them
    # with c_cutoff=False recovers the underlying smooth value, which tells us
    # whether the cliff sits on terrain the surrogate finds attractive.
    if args.recheck_uncut and args.c_cutoff:
        hit = df.index[df["fom"] >= PENALTY * (1 - 1e-9)]
        if len(hit):
            print(f"\n[eval] {len(hit)} penalised points; re-running with "
                  f"c_cutoff=False to recover the uncut surface")
            df = evaluate_rows(df, lambda x: fom(x, c_cutoff=False),
                               args.out_proposals, args, col="fom_uncut", rows=hit)

    if int(df["fom"].notna().sum()) >= 8:
        print("\n" + "=" * 62)
        args.proposals = args.out_proposals
        mode_score(args)


# --------------------------------------------------------------------------
# mode: sweep -- ranking quality as a function of iteration
# --------------------------------------------------------------------------

_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_param_string(s):
    """'[ 19.454,119.458,  7.312, ...]' -> ndarray(7,)."""
    v = [float(m) for m in _NUM.findall(str(s))]
    if len(v) != N_PARAMS:
        raise ValueError(f"parsed {len(v)} numbers, expected {N_PARAMS}: {s!r}")
    return np.asarray(v, dtype=np.float64)


def _find_col(cols, patterns):
    for exact in (True, False):
        for pat in patterns:
            for c in cols:
                if (re.fullmatch(pat, str(c), re.I) if exact
                        else re.search(pat, str(c), re.I)):
                    return c
    return None


def load_walker_states(path, args):
    """
    Return (states, values, n_walkers):
      states[iteration] -> (n_walkers, 7) accepted positions
      values[iteration] -> (n_walkers,) natural-log objective at those positions

    The file has no iteration column; it is one row per (step, walker) in
    arrival order, so the iteration is each walker's own running count. Row
    block 0 is the pre-MCMC initial evaluation, giving 2501 blocks for a
    2500-step run.
    """
    df = pd.read_csv(path)
    if _find_col(list(df.columns), [r"walker", r"chain"]) is None:
        # no usable header -- re-read positionally
        df = pd.read_csv(path, header=None,
                         names=["Walker", "Parameters", "Value"][:len(df.columns)])
        print(f"[walkers] no header detected; assuming Walker,Parameters,Value")
    cols = list(df.columns)

    wk = args.walker_col or _find_col(cols, [r"walker", r"chain", r"replica"])
    val = _find_col(cols, [r"value", r"fom", r"objective"])
    packed = next((c for c in cols
                   if c != wk and not pd.api.types.is_numeric_dtype(df[c])
                   and df[c].astype(str).str.count(r"[\d.]+").median() >= N_PARAMS),
                  None)
    if wk is None or packed is None:
        raise SystemExit(
            f"{path}: need a walker column and a packed parameter column.\n"
            f"columns: {cols}\nfirst row: {df.iloc[0].to_dict()}")

    P = np.stack([parse_param_string(s) for s in df[packed]])
    w = df[wk].to_numpy()
    it = df.groupby(wk).cumcount().to_numpy() + args.iter_offset
    nw = int(pd.Series(w).nunique())
    print(f"[walkers] {len(df)} rows, {nw} walkers, iterations "
          f"{it.min()}..{it.max()} (derived from per-walker row order)")
    if len(df) % nw:
        print(f"  !! {len(df)} rows is not a multiple of {nw}; the last block is "
              f"incomplete and those iterations will be short")

    y = (np.log(pd.to_numeric(df[val], errors="coerce")
                .where(lambda s: s > 0).to_numpy())
         if val else np.full(len(df), np.nan))
    if val:
        npen = int(np.sum(y >= LOG_PENALTY_THR))
        print(f"[walkers] '{val}' parsed as the incumbent objective; "
              f"{npen} penalty rows ({100*npen/len(y):.2f}%)")

    states, values = {}, {}
    order = np.argsort(w, kind="stable")
    for i in range(it.max() + 1):
        sel = order[it[order] == i]
        if len(sel) == 0:
            continue
        states[i] = P[sel]
        values[i] = y[sel]
    return states, values, nw


def sweep_row(g, ck_step, it, inc=None, T=None):
    """Within-batch ranking stats for one checkpoint, averaged over walkers."""
    pen = g["fom"] >= PENALTY * (1 - 1e-9)
    # three populations: clean (drop penalties), uncut (substitute the smooth
    # value under the cliff), operational (penalties rank worst)
    y_clean = g["y_true"].where(~pen)
    y_uncut = g["y_true"].where(~pen, g.get("y_uncut", np.nan))
    keys = ("rho_clean", "rho_uncut", "rho_oper", "tau", "pearson", "sigma",
            "hit1", "pct1", "gain", "oracle", "n", "gain_inc", "acc_pick",
            "acc_rand")
    per = {k: [] for k in keys}
    for w, b in g.groupby("batch_id"):
        m = b.index
        yp = b["pred_log_fom"].to_numpy()
        for tag, ys in (("rho_clean", y_clean.loc[m]), ("rho_uncut", y_uncut.loc[m])):
            ok = ys.notna().to_numpy()
            if ok.sum() >= 5 and not np.allclose(ys[ok], ys[ok].iloc[0]):
                per[tag].append(float(spearmanr(yp[ok], ys[ok]).statistic))
        yo = b["y_true"].to_numpy().copy()
        finite = np.isfinite(yo)
        if finite.any():
            yo[pen.loc[m].to_numpy()] = np.nanmax(yo[finite]) + 1
        if np.isfinite(yo).all() and not np.allclose(yo, yo[0]):
            per["rho_oper"].append(float(spearmanr(yp, yo).statistic))

        ys = y_clean.loc[m]
        ok = ys.notna().to_numpy()
        if ok.sum() < 5:
            continue
        yt, yq = ys[ok].to_numpy(), yp[ok]
        pick = int(np.argmin(yq))
        per["tau"].append(float(kendalltau(yq, yt).statistic))
        per["pearson"].append(float(np.corrcoef(yq, yt)[0, 1]))
        per["sigma"].append(float(np.std(yt, ddof=1)))
        per["hit1"].append(float(pick == int(np.argmin(yt))))
        per["pct1"].append(float((yt < yt[pick]).sum() / (len(yt) - 1)))
        per["gain"].append(float(yt.mean() - yt[pick]))
        per["oracle"].append(float(yt.mean() - yt.min()))
        per["n"].append(len(yt))
        # incumbent comes free from the walker-state file; no FEM call needed
        y0 = None if inc is None else inc.get(int(w))
        if y0 is not None and np.isfinite(y0):
            per["gain_inc"].append(float(y0 - yt[pick]))
            if T and T > 0:
                per["acc_pick"].append(float(min(1.0, np.exp(-(yt[pick] - y0) / T))))
                per["acc_rand"].append(
                    float(np.mean(np.minimum(1.0, np.exp(-(yt - y0) / T)))))

    if not per["rho_uncut"] and not per["rho_clean"]:
        return None
    out = {"CkptStep": ck_step, "Iteration": it, "n_batches": len(per["sigma"]),
           "n_cand": len(g), "pen_rate": float(pen.mean()), "T": T}
    for k, v in per.items():
        if v:
            out[k] = float(np.mean(v))
            if k.startswith("rho") or k in ("hit1", "pearson"):
                out[k + "_se"] = (float(np.std(v, ddof=1) / np.sqrt(len(v)))
                                  if len(v) > 1 else np.nan)
    if per["oracle"] and np.mean(per["oracle"]) > 0:
        out["captured"] = float(np.mean(per["gain"]) / np.mean(per["oracle"]))
    if per["acc_pick"] and np.mean(per["acc_rand"]) > 0:
        out["acc_ratio"] = float(np.mean(per["acc_pick"]) / np.mean(per["acc_rand"]))
    return out


def mode_sweep(args):
    if pd is None:
        raise SystemExit("pandas required")
    rng = np.random.default_rng(args.seed)
    states, values, nw = load_walker_states(args.walker_states, args)
    iters = np.array(sorted(states))
    print(f"[walkers] {len(iters)} iterations available, range "
          f"{iters[0]}..{iters[-1]}")

    ckpts = find_checkpoints(args.ckpt_dir)[::max(1, args.every)]
    std = step_std(args.proposal_std, args.absolute_std)
    per_ckpt = args.n_walkers_sweep * args.n_per_walker
    total = len(ckpts) * per_ckpt

    # alignment self-check: block i of the walker file should be iteration i,
    # so the checkpoint trained through step S has seen (S+1)*nw evaluations
    try:
        c0 = Ckpt(ckpts[-1][1], ckpts[-1][0], args.device)
        expect = (ckpts[-1][0] + 1) * nw
        print(f"[align] ckpt step {ckpts[-1][0]}: n_obs={c0.n_obs}, "
              f"(step+1)*walkers={expect}"
              + ("  OK" if c0.n_obs == expect else
                 f"  MISMATCH by {c0.n_obs - expect} -- try --iter-offset "
                 f"{(c0.n_obs - expect)//nw}"))
    except Exception as e:                                    # noqa: BLE001
        print(f"[align] could not check: {e}")

    sec = None
    for f in (args.out_proposals, args.sweep_evals):
        if os.path.isfile(f):
            d = pd.read_csv(f)
            if "fom_seconds" in d and d["fom_seconds"].notna().any():
                sec = float(d["fom_seconds"].median())
                break
    print(f"\n=== SWEEP PLAN ===")
    print(f"  checkpoints            : {len(ckpts)} (--every {args.every})")
    print(f"  per checkpoint         : {args.n_walkers_sweep} walkers x "
          f"{args.n_per_walker} = {per_ckpt}")
    print(f"  total FEM calls        : {total}")
    print(f"  incumbent values       : free, from the walker-state file")
    if sec:
        print(f"  median call time       : {sec:.1f}s (from a previous run)")
        print(f"  estimated wall time    : {total*sec/3600:.1f} hours")
    else:
        print("  no timing data found; run `batch` first for an estimate")
    print("  Raise --every to subsample checkpoints, or lower --n-per-walker.")
    if args.dry_run:
        print("\n--dry-run: stopping before any FEM calls.")
        return

    recs = []
    for step, path in ckpts:
        ck = Ckpt(path, step, args.device)
        cand_it = iters[iters <= step]
        if not len(cand_it):
            print(f"  [step {step}] no walker state at or before it; skipping")
            continue
        it = int(cand_it[-1])
        W = states[it][:args.n_walkers_sweep]
        for w, x in enumerate(W):
            if not within_limits_scalar(x):
                continue
            C, tries = [], 0
            while len(C) < args.n_per_walker and tries < 40:
                P = batch_proposals(x, std, n=args.n_screen, rng=rng)
                C.extend(P[within_limits(P)])
                tries += 1
            C = np.asarray(C[:args.n_per_walker])
            if len(C) < 5:
                continue
            pred = ck.predict(C)
            for i, c in enumerate(C):
                recs.append({**{f"x{j}": c[j] for j in range(ck.d)},
                             "ckpt_step": step, "iteration": it,
                             "batch_id": w, "cand_index": i,
                             "pred_log_fom": float(pred[i]), "fom": ""})
    df = pd.DataFrame(recs)
    print(f"\nbuilt {len(df)} proposals across "
          f"{df['ckpt_step'].nunique()} checkpoints")

    keys = ("ckpt_step", "batch_id", "cand_index")
    fom = load_fom(args)
    kw = {} if args.c_cutoff is None else {"c_cutoff": bool(args.c_cutoff)}
    df = evaluate_rows(df, lambda x: fom(x, **kw), args.sweep_evals, args,
                       key_cols=keys)
    hit = df.index[df["fom"] >= PENALTY * (1 - 1e-9)]
    if len(hit) and args.recheck_uncut:
        print(f"\n[sweep] {len(hit)} penalised; recovering the uncut surface")
        df = evaluate_rows(df, lambda x: fom(x, c_cutoff=False), args.sweep_evals,
                           args, col="fom_uncut", rows=hit, key_cols=keys)

    df["y_true"] = np.log(df["fom"].where(df["fom"] > 0, np.nan))
    if "fom_uncut" in df.columns:
        # fom() returns PENALTY for solver failures (f.size < 2 or n_failed > 0)
        # regardless of c_cutoff, so only the C < C_FLOOR route is recoverable.
        # Leaving an unrecovered 1e33 in y_uncut would rank it worst and quietly
        # turn rho_uncut into rho_oper for those points.
        ok = ((df["fom_uncut"] > 0) & (df["fom_uncut"] < PENALTY * (1 - 1e-9)))
        df["y_uncut"] = np.log(df["fom_uncut"].where(ok, np.nan))
        npen = int((df["fom"] >= PENALTY * (1 - 1e-9)).sum())
        if npen:
            print(f"\n[sweep] of {npen} penalised candidates, {int(ok.sum())} were "
                  f"recovered by c_cutoff=False (form-factor cliff) and "
                  f"{npen - int(ok.sum())} were not (solver failure -- these stay "
                  f"penalised either way and are dropped from rho_uncut)")

    rows = []
    for s, g in df.groupby("ckpt_step"):
        it = int(g["iteration"].iloc[0])
        inc = ({w: float(v) for w, v in enumerate(values[it])
                if np.isfinite(v) and v < LOG_PENALTY_THR} if it in values else None)
        T = (args.temperature if args.temperature else
             max(TEMP_MIN, TEMP0 * float(args.cooling ** it)))
        rr = sweep_row(g, int(s), it, inc=inc, T=T)
        if rr is not None:
            rows.append(rr)
    res = pd.DataFrame(rows).sort_values("Iteration")
    res.to_csv(args.out, index=False)

    show = [c for c in ("Iteration", "T", "n_batches", "pen_rate",
                        "rho_clean", "rho_uncut", "rho_oper", "pearson",
                        "sigma", "hit1", "pct1", "gain", "oracle", "captured",
                        "gain_inc", "acc_pick", "acc_rand", "acc_ratio")
            if c in res]
    print("\n=== RANKING QUALITY vs ITERATION ===")
    print(res[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwrote {args.out}")
    plot_sweep(res, args.out)


def plot_sweep(res, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping the plot")
        return
    x = res["Iteration"]
    fig, ax = plt.subplots(2, 1, figsize=(6, 6),sharex=True)

    for key, lab in (("rho_clean", "penalties excluded"),
                     ("rho_uncut", "uncut substituted"),
                     ("rho_oper", "penalties included")):
        if key in res and key != "rho_uncut":
            ln, = ax[0].plot(x, res[key], "o-", ms=4, label=lab)
            #if key + "_se" in res:
            #    ax[0].fill_between(x, res[key] - res[key + "_se"],
            #                       res[key] + res[key + "_se"],
            #                       color=ln.get_color(), alpha=.18)
    ax[0].axhline(0, color="k", lw=.8)
    ax[0].set_ylabel(r"Within-batch Spearman $\rho$", fontsize="large")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)

    if "acc_pick" in res:
        ax[1].plot(x, res["acc_pick"], "o-", ms=4, label="screened pick")
    if "acc_rand" in res:
        ax[1].plot(x, res["acc_rand"], "s-", ms=4, label="random proposal")
    ax[1].set_ylabel("Mean acceptance probability", fontsize="large"); ax[1].set_yscale("log")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
    ax[1].set_xlabel("MCMC iteration", fontsize="large")
    plt.suptitle("Surrogate metrics vs. MCMC iteration", fontsize=20)

    r'''if "hit1" in res:
        ax[1].plot(x, res["hit1"], "o-", ms=4, label="precision@1")
        if "n_cand" in res and "n_batches" in res:
            ax[1].plot(x, res["n_batches"] / res["n_cand"], "k--", lw=1,
                       label="chance")
    if "pct1" in res:
        ax[1].plot(x, res["pct1"], "s-", ms=4, label="percentile of pick")
    ax[1].set_ylabel("selection quality"); ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)

    if "gain" in res:
        ax[2].plot(x, res["gain"], "o-", ms=4, label="captured gain")
    if "oracle" in res:
        ax[2].plot(x, res["oracle"], "-", lw=1.5, label="oracle ceiling")
    if "sigma" in res:
        ax[2].plot(x, res["sigma"], "k:", lw=1.2, label=r"$\sigma$ within batch")
    ax[2].set_ylabel("nats"); ax[2].legend(fontsize=9); ax[2].grid(alpha=.3)

    if "acc_pick" in res:
        ax[3].plot(x, res["acc_pick"], "o-", ms=4, label="screened pick")
    if "acc_rand" in res:
        ax[3].plot(x, res["acc_rand"], "s-", ms=4, label="random proposal")
    ax[3].set_ylabel("Metropolis acceptance"); ax[3].set_yscale("log")
    ax[3].legend(fontsize=9); ax[3].grid(alpha=.3)

    if "pen_rate" in res:
        ax[4].plot(x, res["pen_rate"], "o-", ms=4, color="#b0563a")
    ax[4].set_ylabel("penalty rate"); ax[4].set_xlabel("iteration")
    ax[4].grid(alpha=.3)'''

    fig.tight_layout()
    png = out.replace(".csv", ".png")
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")


# --------------------------------------------------------------------------
# mode: score
# --------------------------------------------------------------------------

def mode_score(args):
    if pd is None:
        raise SystemExit("pandas required")
    raw = pd.read_csv(args.proposals)
    raw["fom"] = pd.to_numeric(raw["fom"], errors="coerce")
    if "role" not in raw.columns:
        raw["role"] = "candidate"
    rc = raw[raw["role"] == "candidate"]

    n_all = len(rc)
    n_fail = int(rc["fom"].isna().sum())
    n_nonpos = int((rc["fom"].notna() & (rc["fom"] <= 0)).sum())
    n_pen_all = int((rc["fom"] >= PENALTY * (1 - 1e-9)).sum())
    n_use = n_all - n_fail - n_nonpos - n_pen_all
    print("=== ROW ACCOUNTING (candidates) ===")
    print(f"  rows written              : {n_all}")
    print(f"  solver failed (NaN)       : {n_fail}")
    print(f"  non-positive FoM          : {n_nonpos}")
    print(f"  penalty (>= 1e33)         : {n_pen_all}")
    print(f"  usable for ranking        : {n_use}")
    if n_fail or n_nonpos:
        bad = rc[rc["fom"].isna() | (rc["fom"] <= 0)]
        errs = [e for e in bad.get("fom_error", pd.Series(dtype=str)).astype(str)
                if e and e != "nan"]
        print(f"  !! {n_fail + n_nonpos} rows excluded from every metric below."
              + (f" First error: {errs[0][:80]}" if errs else
                 " No exception recorded -- the FEM returned these values."))

    df = raw[raw["fom"].notna()].copy()
    if df.empty:
        raise SystemExit("no evaluated rows yet")
    df["y_true"] = np.log(df["fom"].where(df["fom"] > 0, np.nan))
    pen = df["fom"] >= PENALTY * (1 - 1e-9)
    cand = df[(df["role"] == "candidate")]
    print(f"\nloaded {len(cand)} evaluated candidates in "
          f"{cand['batch_id'].nunique()} batches ({100*pen.mean():.1f}% penalty)")

    # walker's own objective, where it was evaluated
    inc = {int(r.batch_id): float(r.y_true)
           for r in df[(df["role"] == "centre") & ~pen].itertuples()
           if np.isfinite(r.y_true)}

    rows = []
    for b, g in cand.groupby("batch_id"):
        gp = g["fom"] >= PENALTY * (1 - 1e-9)
        yp_all = g["pred_log_fom"].to_numpy()
        pick_all = int(np.argmin(yp_all))          # what screening actually picks
        gc = g[~gp & g["y_true"].notna()]
        yt, yp = gc["y_true"].to_numpy(), gc["pred_log_fom"].to_numpy()
        n = len(yt)
        if n < 4 or np.allclose(yt, yt[0]):
            continue
        pick = int(np.argmin(yp))
        # rank of the penalised points among the surrogate's preferences
        pen_pct = (float(np.mean(np.argsort(np.argsort(yp_all))[gp.to_numpy()]
                                 / max(len(g) - 1, 1))) if gp.any() else float("nan"))
        rows.append({
            "batch_id": b, "n": n, "n_pen": int(gp.sum()),
            "rho": float(spearmanr(yp, yt).statistic),
            "tau": float(kendalltau(yp, yt).statistic),
            "pearson": float(np.corrcoef(yp, yt)[0, 1]),
            "sigma_batch": float(np.std(yt, ddof=1)),
            "range_batch": float(yt.max() - yt.min()),
            "hit@1": float(pick == int(np.argmin(yt))),
            "regret@1": float(yt[pick] - yt.min()),
            "gain_vs_random": float(yt.mean() - yt[pick]),
            "gain_vs_random_best": float(yt.mean() - yt.min()),
            "gain_vs_incumbent": float(inc[int(b)] - yt[pick])
                                 if int(b) in inc else float("nan"),
            "pct@1": float((yt < yt[pick]).sum() / (n - 1)),
            "pen_rate": float(gp.mean()),
            "pen_picked": float(bool(gp.to_numpy()[pick_all])),
            "pen_mean_pct": pen_pct,
        })
    if not rows:
        raise SystemExit("no scoreable batches")
    r = pd.DataFrame(rows)

    print("\n=== WITHIN-BATCH (this is the screening task) ===")
    print(r.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    nb = len(r)
    print(f"\n  mean within-batch rho     : {r['rho'].mean():+.3f} "
          f"(sd {r['rho'].std(ddof=1):.3f} over {nb} batches)")
    print(f"  mean within-batch tau     : {r['tau'].mean():+.3f}")
    print(f"  precision@1               : {r['hit@1'].mean():.3f}  "
          f"chance = {np.mean(1.0 / r['n']):.3f}")
    print(f"  mean percentile of pick   : {r['pct@1'].mean():.3f} (0 = best)")
    print(f"  mean regret@1             : {r['regret@1'].mean():.5f} nats")
    print(f"  sigma within batch        : {r['sigma_batch'].mean():.5f} nats")
    print(f"  full range within batch   : {r['range_batch'].mean():.5f} nats")
    print("  (computed on non-penalised candidates only -- the population the\n"
          "   regressor was actually trained on)")

    # ---- the penalty cliff ------------------------------------------------
    base, picked = r["pen_rate"].mean(), r["pen_picked"].mean()
    n_cand = int(r["n"].sum() + r["n_pen"].sum())
    print("\n=== PENALTY CLIFF ===")
    print(f"  penalty base rate in the pool        : {base:.3f} "
          f"({n_pen_all}/{n_cand} candidates)")

    hist_rate = args.hist_penalty_rate
    if hist_rate is None:
        try:
            ck = Ckpt(*find_checkpoints(args.ckpt_dir)[-1][::-1], args.device)
            hist_rate = float((ck.hist_y >= LOG_PENALTY_THR).mean())
        except Exception:                                     # noqa: BLE001
            hist_rate = None

    if base == 0:
        print("  The cliff was not reached at all: no candidate in any batch\n"
              "  triggered the cutoff.")
        if hist_rate:
            from scipy.stats import binomtest
            p = binomtest(0, n_cand, hist_rate, alternative="less").pvalue
            print(f"  historical penalty rate over the run : {hist_rate:.3f}")
            print(f"  P(0 of {n_cand} at the historical rate)  : {p:.2e}")
            if p < 0.01:
                print("  Significantly below the run average -- the walkers have\n"
                      "  migrated away from the forbidden boundary. The cliff is not\n"
                      "  a live concern at this point in the anneal, so a feasibility\n"
                      "  classifier would buy nothing now. Recheck if the chain is\n"
                      "  reheated or the proposal scale is raised.")
    else:
        print(f"  penalty rate among top-1 picks       : {picked:.3f} "
              f"({int(r['pen_picked'].sum())}/{len(r)} batches)")
        print(f"  relative risk (picked / base)        : {picked/base:.2f}")
        print(f"  mean percentile of penalised points  : "
              f"{r['pen_mean_pct'].mean():.3f} (0.5 = cliff invisible to the model)")
        print("\n  The regressor never saw a penalty point, so it cannot represent\n"
              "  the cliff. Relative risk near 1 means screening is blind to it;\n"
              "  above 1 means the model is drawn to it and every such pick is a\n"
              "  wasted FEM call -- the exact cost screening exists to avoid.")

    if "fom_uncut" in df.columns and df["fom_uncut"].notna().any():
        u = df[(df["role"] == "candidate") & df["fom_uncut"].notna()].copy()
        u["y_uncut"] = np.log(u["fom_uncut"].where(u["fom_uncut"] > 0, np.nan))
        clean_mean = cand.loc[~pen.reindex(cand.index, fill_value=False),
                              "y_true"].mean()
        print("\n=== UNDER THE CLIFF (c_cutoff=False on penalised points) ===")
        print(f"  mean uncut log-FoM of penalised pts  : {u['y_uncut'].mean():.4f}")
        print(f"  mean log-FoM of feasible candidates  : {clean_mean:.4f}")
        d = u["y_uncut"].mean() - clean_mean
        print(f"  difference                           : {d:+.4f} nats")
        if d < 0:
            print("  NEGATIVE: the forbidden region looks BETTER on the smooth\n"
                  "  surface. The surrogate is being lured toward the cliff by a\n"
                  "  trend that is real but truncated. A feasibility classifier is\n"
                  "  the fix -- ranking alone cannot solve this.")
        else:
            print("  POSITIVE: the cliff sits on unattractive terrain, so ordinary\n"
                  "  ranking already steers away from it incidentally.")

    g, gmax = r["gain_vs_random"].mean(), r["gain_vs_random_best"].mean()
    print("\n=== WHAT SCREENING BUYS PER STEP ===")
    print(f"  gain over a random feasible proposal : {g:+.5f} nats "
          f"({np.exp(g):.4f}x linear)")
    print(f"  gain of an ORACLE ranker             : {gmax:+.5f} nats")
    gi = r["gain_vs_incumbent"].mean()
    if np.isfinite(gi):
        print(f"  gain over the walker's own position  : {gi:+.5f} nats "
              f"in {int(r['gain_vs_incumbent'].notna().sum())} batches")
        print("    (near zero is expected at equilibrium -- the value of screening\n"
              "     shows up as acceptance rate and staying put, not as steady\n"
              "     descent. The random-proposal row is the real counterfactual.)")
    frac = g / gmax if gmax > 0 else float("nan")
    rbar = r["pearson"].mean()
    print(f"  fraction of achievable gain captured : {frac:.3f}")
    print(f"  predicted by mean within-batch r     : {rbar:.3f}")
    print("\n  Under approximate bivariate normality the expected gain from argmin\n"
          "  selection is r x (oracle gain), so the captured fraction should track\n"
          "  the Pearson r above -- NOT R^2. This is why a model with RMSE ~ sigma\n"
          "  can still be worth running: value is linear in r.")
    if np.isfinite(frac) and frac < 0.6 * rbar:
        print("  Captured well below r: check whether feasibility filtering is\n"
              "  shrinking the pool, or whether tail candidates are extrapolations.")
    print("\n  The oracle row is the ceiling: no surrogate wins more than that per\n"
          "  step. If it is small next to the ~0.22 sd across your run, the lever\n"
          "  is the proposal scale or the constraint geometry, not the model.")

    # ---- what this buys in Metropolis acceptance --------------------------
    T = args.temperature
    if T is None and "ckpt_step" in df.columns:
        T = float(args.cooling ** float(df["ckpt_step"].iloc[0]))
    if T and T > 0 and inc:
        acc_pick, acc_rand, better = [], [], []
        for b, gb in cand[~pen & cand["y_true"].notna()].groupby("batch_id"):
            if int(b) not in inc:
                continue
            yt = gb["y_true"].to_numpy()
            yp = gb["pred_log_fom"].to_numpy()
            d_pick = yt[int(np.argmin(yp))] - inc[int(b)]
            d_all = yt - inc[int(b)]                 # every candidate as a draw
            acc_pick.append(min(1.0, np.exp(-d_pick / T)))
            acc_rand.append(float(np.mean(np.minimum(1.0, np.exp(-d_all / T)))))
            better.append(float(np.mean(d_all < 0)))
        if acc_pick:
            ap, ar = float(np.mean(acc_pick)), float(np.mean(acc_rand))
            print(f"\n=== METROPOLIS ACCEPTANCE (T = {T:.4f}) ===")
            print(f"  accept rate, screened pick           : {ap:.3f}")
            print(f"  accept rate, one random proposal     : {ar:.3f}")
            print(f"  ratio                                : "
                  f"{ap/ar if ar > 0 else float('inf'):.1f}x")
            print(f"  P(random proposal beats incumbent)   : {np.mean(better):.3f}")
            print("\n  Computed exactly from the evaluated candidates as\n"
                  "  min(1, exp(-dlogFoM / T)), treating each candidate as a draw\n"
                  "  from the proposal. This is the operational payoff: screening\n"
                  "  converts FEM calls into accepted moves. Verify the form\n"
                  "  matches your sampler's acceptance rule before quoting it.")

    pooled = df[~pen & df["y_true"].notna()]
    if len(pooled) >= 20:
        m = ranking_metrics(pooled["y_true"].to_numpy(),
                            pooled["pred_log_fom"].to_numpy())
        print(f"\n=== POOLED ACROSS BATCHES (for contrast, n={m['N']}) ===")
        print(f"  Spearman {m['Spearman']:.3f}   RMSE {m['RMSE']:.4f}   "
              f"sigma_true {m['sigma_true']:.4f}   R2 {m['R2']:.3f}   "
              f"bias {m['bias']:+.4f}")
        gap = m["Spearman"] - r["rho"].mean()
        if gap > 0.15:
            print(f"  Pooled exceeds within-batch by {gap:+.3f}: the model is partly\n"
                  "  tracking walker drift rather than local structure.")
        else:
            print(f"  Pooled and within-batch agree ({gap:+.3f}): the model is\n"
                  "  resolving local structure, not just drift. Good.")
        if m["R2"] < 0:
            print(f"  R2 < 0 with rho = {m['Spearman']:.2f} is the original paradox in\n"
                  f"  its purest form. Floor at this correlation is "
                  f"{m['RMSE_floor']:.3f}; you are at {m['RMSE']:.3f}, so roughly\n"
                  f"  {np.sqrt(max(m['RMSE']**2 - m['RMSE_floor']**2, 0)):.3f} nats is "
                  f"pure bias/scale error, not ordering error.")


# --------------------------------------------------------------------------

def main(argv=None):
    global SIDE_W_TOL, H_TOL
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["topk", "batch", "score", "sweep"])
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--side-w-tol", type=float, default=SIDE_W_TOL,
                   help="use the value in force when the run was produced "
                        "(0.2 for the 08_13 run, 0.4 currently)")
    p.add_argument("--h-tol", type=float, default=H_TOL)

    p.add_argument("--n-test", type=int, default=490)
    p.add_argument("--out", default="surrogate_topk.csv")

    p.add_argument("--step", type=int, default=None)
    p.add_argument("--walkers", default=None,
                   help="CSV of current walker positions (x0..x6)")
    p.add_argument("--n-walkers", type=int, default=N_WALKERS)
    p.add_argument("--n-screen", type=int, default=N_SCREEN)
    p.add_argument("--n-full-batches", type=int, default=5)
    p.add_argument("--n-per-batch", type=int, default=10**9,
                   help="evaluate only this many per batch (default: all)")
    p.add_argument("--proposal-std", type=float, default=PROPOSAL_STD_FRAC,
                   help="fraction of each parameter's range (default 0.1)")
    p.add_argument("--absolute-std", type=float, default=None,
                   help="override with an absolute sd applied to every dimension")
    p.add_argument("--out-proposals", default="batches_to_evaluate.csv")

    p.add_argument("--evaluate", action="store_true",
                   help="call the real FEM to fill 'fom', then score")
    p.add_argument("--mcmc-module", default="mcmc")
    p.add_argument("--mcmc-path", default=None,
                   help="directory to add to sys.path before importing")
    p.add_argument("--fom-attr", default="fom")
    p.add_argument("--c-cutoff", type=lambda s: s.lower() in ("1", "true", "yes"),
                   default=True,
                   help="value passed to fom(c_cutoff=...). MUST match what the "
                        "sampler used, or the experiment measures a different "
                        "function. Pass 'none' semantics by omitting via --no-c-arg")
    p.add_argument("--no-c-arg", dest="c_cutoff", action="store_const", const=None,
                   help="do not pass c_cutoff at all; use the function default")
    p.add_argument("--recheck-uncut", dest="recheck_uncut", action="store_true",
                   default=True,
                   help="re-evaluate penalised points with c_cutoff=False")
    p.add_argument("--no-recheck-uncut", dest="recheck_uncut", action="store_false")
    p.add_argument("--eval-centres", dest="eval_centres", action="store_true",
                   default=True, help="also evaluate each walker position")
    p.add_argument("--no-eval-centres", dest="eval_centres", action="store_false")
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--flush-every", type=int, default=1,
                   help="write the CSV every N calls (1 = after every call)")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after this many calls; use --limit 3 as a smoke test")

    p.add_argument("--proposals", default="batches_to_evaluate.csv")
    p.add_argument("--temperature", type=float, default=None,
                   help="annealing T at this step; default cooling**ckpt_step")
    p.add_argument("--cooling", type=float, default=COOLING)
    p.add_argument("--hist-penalty-rate", type=float, default=None,
                   help="penalty fraction over the run; read from the checkpoint "
                        "history if omitted")

    # sweep
    p.add_argument("--walker-states", default=None,
                   help="chain file; defaults to all_params_all_values.csv under "
                        "--root, falling back to all_params_all_vals.csv")
    p.add_argument("--iter-col", default=None)
    p.add_argument("--walker-col", default=None)
    p.add_argument("--param-cols", default=None,
                   help="comma-separated, in the order the MLP was trained on")
    p.add_argument("--wide-order", default="auto",
                   choices=["auto", "walker", "param"],
                   help="column layout if the walker file is in wide format")
    p.add_argument("--iter-offset", type=int, default=0,
                   help="shift derived iterations; see the [align] check")
    p.add_argument("--every", type=int, default=1,
                   help="use every Nth checkpoint (raise this to cut cost)")
    p.add_argument("--n-walkers-sweep", type=int, default=10)
    p.add_argument("--n-per-walker", type=int, default=100,
                   help="proposals per walker; 10 x 100 = 150 per checkpoint")
    p.add_argument("--sweep-evals", default="sweep_evals1.csv")
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan and cost estimate, then stop")

    args = p.parse_args(argv)
    if args.side_w_tol != SIDE_W_TOL or args.h_tol != H_TOL:
        print(f"[limits] SIDE_W_TOL {SIDE_W_TOL} -> {args.side_w_tol}, "
              f"H_TOL {H_TOL} -> {args.h_tol}")
    SIDE_W_TOL, H_TOL = args.side_w_tol, args.h_tol
    if args.ckpt_dir is None:
        args.ckpt_dir = os.path.join(args.root, "surrogate_ckpt")
    if args.walker_states is None:
        for name in ("all_params_all_values.csv", "all_params_all_vals.csv"):
            cand = os.path.join(args.root, name)
            if os.path.isfile(cand):
                args.walker_states = cand
                break
        else:
            args.walker_states = os.path.join(args.root,
                                              "all_params_all_values.csv")
    if args.mode == "sweep" and args.out == "surrogate_topk.csv":
        args.out = "sweep_metrics1.csv"
    {"topk": mode_topk, "batch": mode_batch, "score": mode_score,
     "sweep": mode_sweep}[args.mode](args)


if __name__ == "__main__":
    main()