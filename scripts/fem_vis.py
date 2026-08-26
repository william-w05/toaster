"""
Visual checks for cavity2d geometries.

Three levels, cheapest first:

  plot_spec(spec)          pure matplotlib patches -- NO gmsh, NO meshing, instant.
                           Use this while you are still getting the layout right.
  plot_mesh(spec)          runs the real boolean geometry + mesher and draws the
                           actual elements, coloured by material region. This is
                           what the solver will see, so it catches booleans that
                           silently did the wrong thing.
  plot_modes(spec, res)    E_z field of each mode with f / C / Q / localisation,
                           so you can SEE whether a high-C mode is delocalised or
                           just a field concentrated in one corner.

All functions accept save=<path>; if matplotlib has no display they simply write
the PNG. Dimensions are metres internally but axes are drawn in mm.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")           # safe on headless machines; remove for interactive
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle as MplRect

from . import fem_solve as cv

MM = 1000.0     # metres -> mm for display


# ─────────────────────────────────────────────────────────────────────────────
def _add_rect(ax, r, **kw):
    """
    Draw a Rect as a patch, HONOURING its rotation.

    MplRect takes (lower-left, w, h) and knows nothing about Rect.angle, so a
    tilted bar used to be drawn upright while the mesher rotated it. Rect.corners()
    already returns the rotated corners, so fall through to a Polygon whenever
    there is an angle.
    """
    from matplotlib.patches import Polygon as MplPolygon
    if getattr(r, "angle", 0.0):
        return ax.add_patch(MplPolygon(r.corners() * MM, closed=True, **kw))
    return ax.add_patch(MplRect((r.x0 * MM, r.y0 * MM), r.w * MM, r.h * MM, **kw))

def _label(ax, r, text, i):
    """Place a label legibly: rotate inside tall thin rectangles, stagger heights."""
    rot = 90 if r.h > 2.0 * r.w else 0
    # stagger vertically so neighbouring thin bars do not overprint
    dy = (0.0 if rot else (0.16 * r.h * ((i % 3) - 1)))
    ax.annotate(text, (r.cx * MM, (r.cy + dy) * MM), ha="center", va="center",
                fontsize=7, rotation=rot, zorder=5)


def plot_spec(spec, ax=None, save=None, annotate=True, show_centers=True,
              title=None, zoom=None):
    """
    Draw the geometry from the spec alone -- no gmsh, no meshing. Instant, so use
    it as the fast feedback loop while building a layout.

    Metal (setminus) rectangles are hatched; dielectrics are filled and labelled
    with their eps_r. Centres are marked, since rectangles are specified by centre.

    zoom : None    -> show the whole cavity
           "metal" -> fit to the inclusions (useful when a small assembly sits in
                      a much larger cavity, where the whole view hides the detail)
    """
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(9, 6))

    if hasattr(spec, "radius"):                     # CylSpec
        from matplotlib.patches import Circle as MplCircle
        cx, cy = spec.center
        ax.add_patch(MplCircle((cx * MM, cy * MM), spec.radius * MM,
                               facecolor="#eaf3fb", edgecolor="#1f4e79",
                               lw=2.0, zorder=0))
    else:                                           # CavitySpec
        o = spec.outer
        ax.add_patch(MplRect((o.x0 * MM, o.y0 * MM), o.w * MM, o.h * MM,
                             facecolor="#eaf3fb", edgecolor="#1f4e79",
                             lw=2.0, zorder=0))

    for i, r in enumerate(spec.metal):
        _add_rect(ax, r, facecolor="#9aa5b1", edgecolor="#33404d", lw=1.4,
                  hatch="///", zorder=2)
        if show_centers:
            ax.plot(r.cx * MM, r.cy * MM, "k+", ms=7, zorder=4)
        if annotate:
            _label(ax, r, f"{r.name}  {r.w*MM:.1f}x{r.h*MM:.1f}", i)

    for i, (r, mat) in enumerate(spec.dielectric):
        _add_rect(ax, r, facecolor="#f6d9a0", edgecolor="#a8791f", lw=1.4,
                  alpha=0.9, zorder=1)
        if show_centers:
            ax.plot(r.cx * MM, r.cy * MM, "k+", ms=7, zorder=4)
        if annotate:
            _label(ax, r, f"{mat.name} $\\epsilon_r$={mat.eps_r:g}", i)

    if zoom == "metal" and (spec.metal or spec.dielectric):
        rs = list(spec.metal) + [r for r, _ in spec.dielectric]
        x0 = min(r.x0 for r in rs); x1 = max(r.x0 + r.w for r in rs)
        y0 = min(r.y0 for r in rs); y1 = max(r.y0 + r.h for r in rs)
        pad = 0.25 * max(x1 - x0, y1 - y0) * MM
        ax.set_xlim(x0 * MM - pad, x1 * MM + pad)
        ax.set_ylim(y0 * MM - pad, y1 * MM + pad)
    else:
        ex0, ey0, ex1, ey1 = spec.extent
        pad = 0.04 * max(ex1 - ex0, ey1 - ey0) * MM
        ax.set_xlim(ex0 * MM - pad, ex1 * MM + pad)
        ax.set_ylim(ey0 * MM - pad, ey1 * MM + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.grid(alpha=0.25, ls=":")
    ex0, ey0, ex1, ey1 = spec.extent
    shape = (f"disk r={spec.radius*MM:.1f} mm" if hasattr(spec, "radius")
             else f"outer {(ex1-ex0)*MM:.1f} x {(ey1-ey0)*MM:.1f} mm")
    ax.set_title(title or f"geometry: {spec.tag or '(untagged)'}   {shape}"
                          f"   metal={len(spec.metal)}  diel={len(spec.dielectric)}")
    _overlap_warnings(spec, ax)
    if created and save:
        fig.tight_layout(); fig.savefig(save, dpi=140); plt.close(fig)
    return ax


def _overlap_warnings(spec, ax):
    """Flag rectangles that overlap or stick out -- the usual layout mistakes."""
    msgs = []
    ox0, oy0, ox1, oy1 = spec.extent
    rects = [(r, "metal") for r in spec.metal] + \
            [(r, "diel") for r, _ in spec.dielectric]
    for r, kind in rects:
        x0, y0, x1, y1 = r.bounds
        if x0 < ox0 - 1e-12 or y0 < oy0 - 1e-12 or x1 > ox1 + 1e-12 or y1 > oy1 + 1e-12:
            msgs.append(f"{kind} '{r.name}' extends outside the cavity")
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            a, _ = rects[i]; b, _ = rects[j]
            ax0, ay0, ax1, ay1 = a.bounds        # rotation-aware, like the
            bx0, by0, bx1, by1 = b.bounds        # out-of-bounds test above
            if (ax0 < bx1 - 1e-12 and bx0 < ax1 - 1e-12 and
                    ay0 < by1 - 1e-12 and by0 < ay1 - 1e-12):
                msgs.append(f"'{a.name}' overlaps '{b.name}'")
    if msgs:
        ax.text(0.01, 0.99, "WARNING\n" + "\n".join(msgs[:5]),
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                color="#a11", bbox=dict(fc="#ffecec", ec="#a11", alpha=0.9))
    return msgs


def check_spec(spec, verbose=True):
    """Non-graphical version of the same checks. Returns a list of problems."""
    fig, ax = plt.subplots()
    msgs = _overlap_warnings(spec, ax)
    plt.close(fig)
    if verbose:
        print("[check] " + ("OK, no overlaps or out-of-bounds rectangles"
                            if not msgs else f"{len(msgs)} problem(s):"))
        for m in msgs:
            print("   -", m)
    return msgs


# ─────────────────────────────────────────────────────────────────────────────
def plot_mesh(spec, save=None, show_regions=True, lw=0.25, title=None):
    """
    Mesh the spec for real and draw the elements, coloured by material region.
    This is the ground truth: if a boolean did something unexpected, it shows here.
    """
    import skfem, os
    tmp = cv.tmp_msh_path("viz")          # portable: %TEMP% on Windows, /tmp on Unix
    with cv._quiet():                     # meshio.read prints a stray blank line
        diel_mats = cv.build_mesh(spec, tmp)
        m = skfem.Mesh.load(tmp)
    try:
        os.remove(tmp)
    except OSError:
        pass
    p, t = m.p, m.t

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.triplot(p[0] * MM, p[1] * MM, t.T, lw=lw, color="#7a8b99", alpha=0.8)

    if show_regions:
        colors = ["#eaf3fb", "#f6d9a0", "#cfe8cf", "#f2c8c8", "#ded0ef"]
        names = ["background"] + [f"diel_{i}" for i in range(len(diel_mats))]
        for i, nm in enumerate(names):
            if nm not in m.subdomains:
                continue
            el = m.subdomains[nm]
            ax.tripcolor(p[0] * MM, p[1] * MM, t[:, el].T,
                         facecolors=np.zeros(len(el)),
                         cmap=matplotlib.colors.ListedColormap([colors[i % len(colors)]]),
                         alpha=0.85, zorder=-1)
    for nm, col in (("wall", "#1f4e79"), ("metal", "#b03a2e")):
        if nm in m.boundaries:
            f = m.facets[:, m.boundaries[nm]]
            ax.plot(p[0][f] * MM, p[1][f] * MM, color=col, lw=1.8)

    ax.set_aspect("equal"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    ax.set_title(title or f"mesh: {spec.tag or ''}  {t.shape[1]} elements  "
                          f"(blue = wall BC, red = cut-out/metal BC)")
    ax.set_aspect("equal")
    if save:
        fig.tight_layout(); fig.savefig(save, dpi=140); plt.close(fig)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
def plot_modes(spec, result, n=None, save=None, cmap="RdBu_r"):
    """
    E_z of each mode, annotated with f / C / Q / localisation.

    Requires solve_cavity(..., keep_fields=True). Localised modes are obvious here:
    the field collapses into a corner while a delocalised fundamental fills the
    cavity -- the visual counterpart of the A_part/area number.
    """
    if "fields" not in result:
        raise ValueError("no fields in result: call solve_cavity(..., keep_fields=True)")
    m = result["mesh"]
    nmodes = len(result["fields"]) if n is None else min(n, len(result["fields"]))
    ncol = min(3, nmodes); nrow = int(np.ceil(nmodes / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.4 * nrow),
                             squeeze=False)
    nv = m.p.shape[1]
    for i in range(nmodes):
        ax = axes[i // ncol][i % ncol]
        u = result["fields"][i][:nv]          # P2: vertex dofs come first
        lim = np.max(np.abs(u)) or 1.0
        tp = ax.tripcolor(m.p[0] * MM, m.p[1] * MM, m.t.T, u,
                          cmap=cmap, vmin=-lim, vmax=lim, shading="gouraud")
        for nm, col in (("wall", "#111111"), ("metal", "#111111")):
            if nm in m.boundaries:
                f = m.facets[:, m.boundaries[nm]]
                ax.plot(m.p[0][f] * MM, m.p[1][f] * MM, color=col, lw=1.0)
        md = result["modes"][i]
        ax.set_title(f"f={result['freqs'][i]/1e9:.4f} GHz   C={md['C']:.3f}\n"
                     f"Q={md['Q']:.3g}", fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(tp, ax=ax, fraction=0.035)
    for j in range(nmodes, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"E_z modes: {spec.tag or ''}", y=1.0)
    if save:
        fig.tight_layout(); fig.savefig(save, dpi=140); plt.close(fig)
    return fig

def plot_modes_square_magnitude(spec, result, n=None, save=None, cmap="RdBu_r"):
    """
    |E_z|^2 of each mode, annotated with f / C / Q / localisation.
    """
    if "fields" not in result:
        raise ValueError("no fields in result: call solve_cavity(..., keep_fields=True)")
    m = result["mesh"]
    nmodes = len(result["fields"]) if n is None else min(n, len(result["fields"]))
    ncol = min(3, nmodes); nrow = int(np.ceil(nmodes / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.4 * nrow),
                                squeeze=False)
    nv = m.p.shape[1]
    for i in range(nmodes):
        ax = axes[i // ncol][i % ncol]
        u = result["fields"][i][:nv]          # P2: vertex dofs come first
        val = np.abs(u) ** 2
        lim = np.max(val) or 1.0
        tp = ax.tripcolor(m.p[0] * MM, m.p[1] * MM, m.t.T, val,
                            cmap=cmap, vmin=0.0, vmax=lim, shading="gouraud")
        for nm, col in (("wall", "#111111"), ("metal", "#111111")):
            if nm in m.boundaries:
                f = m.facets[:, m.boundaries[nm]]
                ax.plot(m.p[0][f] * MM, m.p[1][f] * MM, color=col, lw=1.0)
        md = result["modes"][i]
        ax.set_title(f"f={result['freqs'][i]/1e9:.4f} GHz   C={md['C']:.3f}\n"
                        f"Q={md['Q']:.3g}", fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(tp, ax=ax, fraction=0.035)
    for j in range(nmodes, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"E_z modes: {spec.tag or ''}", y=1.0)
    if save:
        fig.tight_layout(); fig.savefig(save, dpi=140); plt.close(fig)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
def _pick_best(result, min_localisation=0.0):
    """Index of the highest-C mode, optionally refusing localised ones."""
    idx = [i for i, m in enumerate(result["modes"])
           if m["localisation"] >= min_localisation]
    if not idx:
        idx = list(range(len(result["modes"])))     # nothing passes -> don't drop the step
    return max(idx, key=lambda i: result["modes"][i]["C"])


def plot_best_modes(entries, save=None, cmap="RdBu_r", ncol=4,
                    min_localisation=0.0, suptitle=None, share_scale=False):
    """
    Field of the HIGHEST-FORM-FACTOR mode at every tuning position, one panel per
    step, so you can watch the operating mode evolve across the scan.

    entries : list of (spec, result) or (spec, result, label). Every result must
              come from solve_cavity(..., keep_fields=True). Each tuning position
              has its own geometry and therefore its own mesh, which is why the
              meshes are drawn per panel rather than reused.
    min_localisation : forwarded to the mode choice -- set it > 0 to pick the best
              DELOCALISED mode instead of letting argmax(C) grab a localised one.
    share_scale : use one colour scale across panels instead of normalising each.
              Per-panel (default) shows mode shape best; shared shows amplitude loss.
    """
    ents = [(e[0], e[1], (e[2] if len(e) > 2 else "")) for e in entries]
    for _, r, _lab in ents:
        if "fields" not in r:
            raise ValueError("a result has no fields: "
                             "call solve_cavity(..., keep_fields=True)")
    n = len(ents)
    ncol = min(ncol, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.9 * nrow),
                             squeeze=False)

    picks = [_pick_best(r, min_localisation) for _, r, _l in ents]
    gmax = 1.0
    if share_scale:
        gmax = max(np.max(np.abs(r["fields"][i][:r["mesh"].p.shape[1]]))
                   for (_, r, _l), i in zip(ents, picks)) or 1.0

    for k, ((spec, r, lab), i) in enumerate(zip(ents, picks)):
        ax = axes[k // ncol][k % ncol]
        m = r["mesh"]
        nv = m.p.shape[1]
        u = r["fields"][i][:nv]
        lim = gmax if share_scale else (np.max(np.abs(u)) or 1.0)
        tp = ax.tripcolor(m.p[0] * MM, m.p[1] * MM, m.t.T, u,
                          cmap=cmap, vmin=-lim, vmax=lim, shading="gouraud")
        for nm in ("wall", "metal"):
            if nm in m.boundaries:
                f = m.facets[:, m.boundaries[nm]]
                ax.plot(m.p[0][f] * MM, m.p[1][f] * MM, color="#111111", lw=0.9)
        md = r["modes"][i]
        head = (lab + "\n") if lab else ""
        ax.set_title(f"{head}f={r['freqs'][i]/1e9:.3f} GHz  C={md['C']:.3f}\n"
                     f"Q={md['Q']:.3g}", fontsize=8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(tp, ax=ax, fraction=0.04)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle(suptitle or "highest form-factor mode across the tuning range",
                 y=1.0, fontsize=11)
    if save:
        fig.tight_layout(); fig.savefig(save, dpi=140); plt.close(fig)
    return fig

def plot_best_modes_magnitude_square(entries, save=None, cmap="RdBu_r", ncol=4,
                    min_localisation=0.0, suptitle=None, share_scale=False):
    """
    Field of the HIGHEST-FORM-FACTOR mode (|E_z|^2) at every tuning position, one panel per
    step, so you can watch the operating mode evolve across the scan.

    entries : list of (spec, result) or (spec, result, label). Every result must
              come from solve_cavity(..., keep_fields=True). Each tuning position
              has its own geometry and therefore its own mesh, which is why the
              meshes are drawn per panel rather than reused.
    min_localisation : forwarded to the mode choice -- set it > 0 to pick the best
              DELOCALISED mode instead of letting argmax(C) grab a localised one.
    share_scale : use one colour scale across panels instead of normalising each.
              Per-panel (default) shows mode shape best; shared shows amplitude loss.
    """
    ents = [(e[0], e[1], (e[2] if len(e) > 2 else "")) for e in entries]
    for _, r, _lab in ents:
        if "fields" not in r:
            raise ValueError("a result has no fields: "
                             "call solve_cavity(..., keep_fields=True)")
    n = len(ents)
    ncol = min(ncol, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.9 * nrow),
                             squeeze=False)

    picks = [_pick_best(r, min_localisation) for _, r, _l in ents]
    gmax = 1.0
    if share_scale:
        gmax = max(np.max(np.abs(r["fields"][i][:r["mesh"].p.shape[1]]))
                   for (_, r, _l), i in zip(ents, picks)) or 1.0

    for k, ((spec, r, lab), i) in enumerate(zip(ents, picks)):
        ax = axes[k // ncol][k % ncol]
        m = r["mesh"]
        nv = m.p.shape[1]
        u = r["fields"][i][:nv]
        val = np.abs(u) ** 2
        lim = gmax if share_scale else (np.max(val) or 1.0)
        tp = ax.tripcolor(m.p[0] * MM, m.p[1] * MM, m.t.T, val,
                          cmap=cmap, vmin=0.0, vmax=lim, shading="gouraud")
        for nm in ("wall", "metal"):
            if nm in m.boundaries:
                f = m.facets[:, m.boundaries[nm]]
                ax.plot(m.p[0][f] * MM, m.p[1][f] * MM, color="#111111", lw=0.9)
        md = r["modes"][i]
        head = (lab + "\n") if lab else ""
        ax.set_title(f"{head}f={r['freqs'][i]/1e9:.3f} GHz  C={md['C']:.3f}\n"
                     f"Q={md['Q']:.3g}", fontsize=8)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(tp, ax=ax, fraction=0.04)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    fig.suptitle(suptitle or "highest form-factor mode across the tuning range",
                 y=1.0, fontsize=20)
    if save:
        fig.tight_layout(); fig.savefig(save, dpi=140); plt.close(fig)
    return fig


def plot_tuning_summary(entries, save=None, min_localisation=0.0, xvals=None,
                        xlabel="tuning step"):
    """
    f, C, Q and localisation of the best mode versus tuning position -- the
    scalar companion to plot_best_modes. A localisation trace that collapses
    after the first step is the signature of the mode fragmenting.
    """
    ents = [(e[0], e[1], (e[2] if len(e) > 2 else "")) for e in entries]
    picks = [_pick_best(r, min_localisation) for _, r, _l in ents]
    x = np.arange(len(ents)) if xvals is None else np.asarray(xvals, dtype=float)
    f = np.array([r["freqs"][i] for (_, r, _l), i in zip(ents, picks)]) / 1e9
    C = np.array([r["modes"][i]["C"] for (_, r, _l), i in zip(ents, picks)])
    Q = np.array([r["modes"][i]["Q"] for (_, r, _l), i in zip(ents, picks)])
    L = np.array([r["modes"][i]["localisation"] for (_, r, _l), i in zip(ents, picks)])

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for ax, y, lab, c in ((axes[0][0], f, "f (GHz)", "#1f4e79"),
                          (axes[0][1], C, "form factor C", "#b03a2e"),
                          (axes[1][0], Q, "Q", "#2e7d32"),
                          (axes[1][1], L, "localisation $A_{part}/A$", "#6a4c93")):
        ax.plot(x, y, "o-", color=c, lw=1.6, ms=5)
        ax.set_ylabel(lab); ax.set_xlabel(xlabel); ax.grid(alpha=0.3, ls=":")
    fig.suptitle("operating mode across the tuning range")
    if save:
        fig.tight_layout(); fig.savefig(save, dpi=140); plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
def toaster_spec(params, cavity_w=None, cavity_h=0.160, gap0=0.010, gap1=None,
                 toast_dx=0.0, toast_dy=0.0, mesh_size=0.001, tag="toaster",
                 wall_material=None, metal_material=None, mesh_uniform=False,
                 center_x=None, center_y=None):
    """
    Build a CavitySpec from the parameter vector, centred on the origin.

    params (METRES for lengths, degrees for the angle). BOTH lengths are accepted:

      7-vector (gap1 fixed, current):
        0 angle | 1 divider_height | 2 divider_width | 3 center_toast_width
        4 side_toast_width | 5 center_toast_height | 6 side_toast_height
        -> gap1 comes from the `gap1` argument (defaulting to gap0)

      8-vector (legacy, gap1 optimised):
        0 angle | 1 divider_height | 2 divider_width | 3 gap1
        4 center_toast_width | 5 side_toast_width | 6 center_toast_height
        7 side_toast_height
        -> gap1 comes from params[3] unless the `gap1` argument overrides it
    toast_dx, toast_dy : tuning displacement applied to ALL THREE TOASTS together
        (centre + both side toasts). The DIVIDERS and the walls stay fixed -- they
        are the shell structure, the toasts are the tuning elements.
        (center_x/center_y are accepted as legacy aliases.)

    LAYOUT (as specified). Going outward from the middle on each side:
        [center_w] [gap0] [div_w] [gap1] [side_w] [gap1] | wall
    so the divider sits BETWEEN the centre toast and the side toast, and gap0 (fixed
    at 10 mm) is the gap flanking the centre toast. Side toasts and dividers are
    placed symmetrically about x = 0 and are NOT moved by the tuning displacement;
    only the centre toast moves.

    cavity_w : defaults to the true assembly width implied by that sequence,
        center_w + 2*gap0 + 2*div_w + 4*gap1 + 2*side_w,
        i.e. the outermost gap1 brings you to the side wall.

        NOTE: this is 2*gap0 = 20 mm WIDER than your constraint expression
        4*gap1 + 2*side_w + 2*div_w + center_w, which contains no gap0 term. With
        the 400/sqrt(2) = 282.8 mm limit and ~85 mm assemblies the difference never
        binds, but the constraint does understate the real width by 20 mm.

        Setting cavity_w to 282.8 mm instead leaves large empty regions beside the
        assembly whose own modes swamp the spectrum near 1.7 GHz -- that bound is a
        "must fit in the bore" limit, not the cavity width.

    Why the toasts move together: with the dividers fixed, translating all three
    toasts by -d turns the six gaps into, left to right,
        gap1-d, gap1+d, gap0-d, gap0+d, gap1-d, gap1+d
    so with gap0 = gap1 = 10 mm THREE gaps widen to (10+d) and stay degenerate with
    one another. That degenerate trio supports an in-phase mode at
    f = c / (2*(10 + d)) = 3e11 / (2*(10 + |x|)) -- your tuning law -- and keeps a
    high form factor across the scan. Moving only the centre toast instead breaks
    the degeneracy and the mode collapses into a single cell after one step.

    PLOT IT AND CHECK IT AGAINST YOUR COMSOL MODEL before trusting any number.
    """
    p = [float(x) for x in np.asarray(params).ravel()]
    if len(p) == 7:
        _, div_h, div_w, ctr_w, side_w, ctr_h, side_h = p
        if gap1 is None:
            gap1 = gap0                       # gap1 is fixed, not a free parameter
    elif len(p) == 8:
        _, div_h, div_w, gap1_p, ctr_w, side_w, ctr_h, side_h = p
        if gap1 is None:
            gap1 = gap1_p                     # legacy: gap1 lives in the vector
    else:
        raise ValueError(f"toaster_spec expects a 7- or 8-vector, got {len(p)}")
    gap1 = float(gap1)
    if cavity_w is None:
        cavity_w = ctr_w + 2.0 * gap0 + 2.0 * div_w + 4.0 * gap1 + 2.0 * side_w

    if center_x is not None:            # legacy alias
        toast_dx = center_x
    if center_y is not None:
        toast_dy = center_y

    # outward:  centre | gap0 | divider | gap1 | side toast | gap1 | wall
    x_div = 0.5 * ctr_w + gap0 + 0.5 * div_w
    x_side = 0.5 * ctr_w + gap0 + div_w + gap1 + 0.5 * side_w

    # TOASTS move with the tuning displacement; DIVIDERS stay put.
    metal = [cv.Rect.from_center(toast_dx, toast_dy, ctr_w, ctr_h, "center_toast")]
    for s in (-1.0, +1.0):
        side = 'L' if s < 0 else 'R'
        metal.append(cv.Rect.from_center(s * x_side + toast_dx, toast_dy,
                                         side_w, side_h, f"side_toast{side}"))
        metal.append(cv.Rect.from_center(s * x_div, 0.0,
                                         div_w, div_h, f"divider{side}"))

    kw = {}
    if wall_material is not None:
        kw["wall_material"] = wall_material
    if metal_material is not None:
        kw["metal_material"] = metal_material
    return cv.CavitySpec(
        outer=cv.Rect.from_center(0.0, 0.0, cavity_w, cavity_h, "cavity"),
        metal=metal, mesh_size=mesh_size, mesh_uniform=mesh_uniform,
        tag=tag, **kw)