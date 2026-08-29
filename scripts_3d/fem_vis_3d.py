"""
Visual checks for 3D cavity geometries -- the counterpart of fem_vis.py.

Same three levels, cheapest first:

  plot_spec_3d(spec)         primitives only -- NO gmsh, NO meshing, instant and
                             INTERACTIVE (rotate/zoom). Use it while you are
                             still getting the layout right.
  plot_mesh_3d(spec)         runs the real boolean geometry + mesher and draws
                             the actual surface, coloured by boundary group, so
                             it catches booleans that silently did the wrong
                             thing. This is what the solver will see.
  plot_field_slices(result)  |E| on horizontal and vertical cut planes, which is
                             the only way to see inside a 3D field.

WHY SLICES AND NOT ISOSURFACES
    A 3D mode fills the volume, so any surface rendering shows you the outside of
    the field and nothing else. Cut planes are the 3D analogue of the 2D
    tripcolor plots you already have: a z-slice through the middle of an
    extruded cavity should look EXACTLY like the 2D solution, and a y- or
    x-slice shows the z-structure (uniform for p = 0, one half-period of a cosine
    for p = 1) that 2D cannot represent at all. Comparing those two views is the
    fastest way to tell which longitudinal mode you have landed on.

INTERACTIVITY
    pyvista (VTK) does the 3D. Three ways out, picked automatically:
      * save=None and a display available -> an interactive window
      * save="....html"                    -> a self-contained HTML file that
        rotates and zooms in any browser, which is what you want on a headless
        machine or for sharing. Needs `pip install "pyvista[jupyter]"` (trame).
      * save="....png"                     -> an off-screen screenshot
    In Jupyter, set pyvista.set_jupyter_backend("trame") once and the same calls
    render inline.

    The SLICE MONTAGES are matplotlib and never need a GL context at all: the
    cutting is done by VTK filters, which are pure computation. Those always
    work, on any machine, and are the ones to use in a script.

UNITS
    Metres internally, millimetres on every axis label, exactly as in 2D.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # safe headless; remove for interactive matplotlib
import matplotlib.pyplot as plt

try:                                    # inside the package
    from . import fem_solve_3d_updated as cv3
    from scripts import fem_solve as cv2
except ImportError:                     # standalone / notebook
    import fem_solve_3d as cv3
    import scripts.fem_solve as cv2

MM = 1000.0        # metres -> mm for display

_AXIS = {"x": 0, "y": 1, "z": 2}
# in-plane axes for each slice normal, and their labels
_INPLANE = {"z": (0, 1, "x", "y"),
            "y": (0, 2, "x", "z"),
            "x": (1, 2, "y", "z")}


def _pv():
    """Import pyvista with a useful message if it is missing."""
    try:
        import pyvista as pv
    except ImportError as e:                                  # pragma: no cover
        raise ImportError(
            "pyvista is required for the 3D views:  pip install 'pyvista[jupyter]'\n"
            "(the [jupyter] extra pulls in trame, which is what makes "
            "save='...html' work)") from e
    return pv


# ─────────────────────────────────────────────────────────────────────────────
# field scaling -- mirrors fem_solve.field_scale, one dimension up
# ─────────────────────────────────────────────────────────────────────────────

def field_scale(result, i, normalize="peak", energy=1.0):
    """
    Multiplier turning the stored eigenvector (peak |E| = 1) into a field with a
    chosen physical normalisation. Returns (scale, unit_label).

    An eigenmode's amplitude is arbitrary, so any absolute |E| needs a convention:

      "peak"   : leave it, max|E| = 1                  -> dimensionless
      "energy" : (eps0/2) int eps_r |E|^2 dV = `energy` JOULES  -> E in V/m.
                 NOTE this differs from the 2D version, where the same option
                 meant joules PER METRE of cavity length because the integral was
                 an area integral. Here it is a real volume integral and the
                 number is a real energy.
      "l2"     : int eps_r |E|^2 dV = 1                -> E in 1/m^(3/2)
    """
    md = result["modes"][i]
    if normalize == "peak":
        return 1.0, r"$|E|$ (peak-normalised)"
    if normalize == "energy":
        U0 = md.get("U_stored")
        if not U0 or U0 <= 0:
            raise ValueError("result lacks U_stored; use normalize='peak'.")
        return float(np.sqrt(energy / U0)), r"$|E|$  (V/m)"
    if normalize == "l2":
        d = md.get("int_eps_E2")
        if not d or d <= 0:
            raise ValueError("result lacks int_eps_E2; use normalize='peak'.")
        return float(1.0 / np.sqrt(d)), r"$|E|$  (m$^{-3/2}$)"
    raise ValueError("normalize must be 'peak', 'energy' or 'l2'")


def pick_mode(result, i=None, min_localisation=0.0):
    """Index of the mode to draw: the given one, or the highest-C delocalised."""
    if i is not None:
        return int(i)
    cand = [j for j, m in enumerate(result["modes"])
            if m["localisation"] >= min_localisation]
    if not cand:
        cand = list(range(len(result["modes"])))
    return max(cand, key=lambda j: result["modes"][j]["C"])


# ─────────────────────────────────────────────────────────────────────────────
# mesh -> pyvista
# ─────────────────────────────────────────────────────────────────────────────

def to_pyvista(result, i=None, normalize="peak", energy=1.0,
               min_localisation=0.0, component=None):
    """
    The solved mesh as a pyvista UnstructuredGrid carrying the field as POINT
    data, in millimetres (so every axis reads in mm).

    Arrays attached: "E" (vector), "|E|", "Ez", "|E|^2".

    Edge-element dofs are edge circulations, not point values, so this goes
    through fem_solve_3d.nodal_field, which L2-projects the field onto P1. That
    projection is the only reason a 3D field can be plotted at all, and it is
    also a mild smoother -- fine for looking, not for extracting peak values.

    Requires solve_cavity_3d(..., keep_fields=True).
    """
    pv = _pv()
    i = pick_mode(result, i, min_localisation)
    mesh = result["mesh"]
    E = cv3.nodal_field(result, i)
    scale, unit = field_scale(result, i, normalize=normalize, energy=energy)
    E = E * scale

    pts = mesh.p.T * MM
    t = mesh.t.T                                    # (n_tet, 4)
    cells = np.column_stack([np.full(len(t), 4), t]).ravel()
    ctypes = np.full(len(t), pv.CellType.TETRA, dtype=np.uint8)
    grid = pv.UnstructuredGrid(cells, ctypes, pts)
    grid.point_data["E"] = E
    grid.point_data["|E|"] = np.linalg.norm(E, axis=1)
    grid.point_data["Ez"] = E[:, 2]
    grid.point_data["|E|^2"] = np.sum(E ** 2, axis=1)
    if component is not None:
        grid.point_data[f"E{component}"] = E[:, _AXIS[component]]
    return grid, i, unit


# ─────────────────────────────────────────────────────────────────────────────
# level 1: the spec, from primitives -- instant, no meshing
# ─────────────────────────────────────────────────────────────────────────────

def _box_polydata(pv, box):
    """A (possibly rotated) Box as pyvista PolyData, in mm."""
    b = pv.Box(bounds=(box.x0 * MM, (box.x0 + box.w) * MM,
                       box.y0 * MM, (box.y0 + box.h) * MM,
                       box.z0 * MM, (box.z0 + box.d) * MM))
    ang = float(getattr(box, "angle", 0.0))
    if ang:
        b = b.rotate_z(ang, point=(box.cx * MM, box.cy * MM, box.cz * MM),
                       inplace=False)
    return b


def _outer_polydata(pv, spec):
    if isinstance(spec, cv3.CylSpec3D):
        cx, cy, cz = spec.center
        return pv.Cylinder(center=(cx * MM, cy * MM, cz * MM),
                           direction=(0, 0, 1), radius=spec.radius * MM,
                           height=spec.length * MM, resolution=90)
    if isinstance(spec, cv3.CavitySpec3D):
        return _box_polydata(pv, spec.outer)
    return None


def plot_spec_3d(spec, save=None, show=True, cavity_opacity=0.12,
                 metal_color="#b03a2e", cavity_color="#1f4e79",
                 show_edges=True, title=None, window_size=(1100, 780),
                 background="white"):
    """
    Draw the geometry from the spec alone -- no gmsh, no meshing. Instant, and
    fully rotatable, so use it as the fast feedback loop while building a layout.

    The cavity is drawn translucent and the metal inclusions solid, which is the
    intuitive reading even though the solver does the opposite (the metal is CUT
    OUT and its walls become boundary conditions).

    save : None -> interactive window (needs a display)
           "x.html" -> self-contained interactive HTML, works headless
           "x.png"  -> off-screen screenshot

    Returns the pyvista Plotter.
    """
    pv = _pv()
    p = pv.Plotter(off_screen=(save is not None), window_size=window_size)
    p.set_background(background)

    outer = _outer_polydata(pv, spec)
    if outer is not None:
        p.add_mesh(outer, color=cavity_color, opacity=cavity_opacity,
                   show_edges=False, label="cavity")
        p.add_mesh(outer.extract_all_edges(), color=cavity_color, line_width=2)
    elif isinstance(spec, cv3.ImportedSpec3D):
        # an imported skin has no primitive form; fall back to the real mesh
        p.close()
        return plot_mesh_3d(spec, save=save, show=show, title=title,
                            window_size=window_size, background=background)

    for b in getattr(spec, "metal", []):
        p.add_mesh(_box_polydata(pv, b), color=metal_color, opacity=1.0,
                   show_edges=show_edges, edge_color="#33404d")
    for b, mat in getattr(spec, "dielectric", []):
        p.add_mesh(_box_polydata(pv, b), color="#f6d9a0", opacity=0.6,
                   show_edges=show_edges,
                   label=f"{mat.name} eps_r={mat.eps_r:g}")

    n_metal = len(getattr(spec, "metal", []))
    p.add_axes(xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)")
    p.show_grid(xtitle="x (mm)", ytitle="y (mm)", ztitle="z (mm)")
    p.add_text(title or f"geometry: {spec.tag or '(untagged)'}   "
                        f"metal={n_metal}", font_size=10,
               position="lower_left")     # keep it clear of the grid labels
    p.camera_position = "iso"
    return _finish(p, save, show)


def _finish(p, save, show):
    """Window, HTML or PNG, depending on `save`."""
    if save is None:
        if show:
            p.show()
        return p
    ext = os.path.splitext(save)[1].lower()
    d = os.path.dirname(os.path.abspath(save))
    if d:
        os.makedirs(d, exist_ok=True)
    if ext in (".html", ".htm"):
        try:
            p.export_html(save)
        except Exception as e:                                # pragma: no cover
            raise RuntimeError(
                f"HTML export needs trame: pip install 'pyvista[jupyter]'  ({e})")
    else:
        p.screenshot(save)
    p.close()
    return p


# ─────────────────────────────────────────────────────────────────────────────
# level 2: the real mesh
# ─────────────────────────────────────────────────────────────────────────────

def plot_mesh_3d(spec, save=None, show=True, show_edges=True,
                 wall_color="#1f4e79", metal_color="#b03a2e",
                 wall_opacity=0.25, title=None, window_size=(1100, 780),
                 background="white", clip=None):
    """
    Mesh the spec for real and draw the boundary, coloured by group: blue = outer
    wall BC, red = cut-out/metal BC. This is the ground truth -- if a boolean did
    something unexpected, or a bar failed to cut, it shows here and nowhere else.

    clip : None, or "x"/"y"/"z" to cut the outer wall away along that axis so you
        can see inside. The metal surfaces are always drawn whole.

    Returns the pyvista Plotter.
    """
    pv = _pv()
    import skfem
    tmp = cv3.tmp_msh_path("viz3d")
    with cv3._quiet():
        cv3.build_mesh_3d(spec, tmp)
        m = skfem.Mesh.load(tmp)
    try:
        os.remove(tmp)
    except OSError:
        pass

    pts = m.p.T * MM
    p = pv.Plotter(off_screen=(save is not None), window_size=window_size)
    p.set_background(background)

    bnd = m.boundaries or {}
    counts = {}
    for name, color, opacity in (("wall", wall_color, wall_opacity),
                                 ("metal", metal_color, 1.0)):
        if name not in bnd:
            continue
        f = m.facets[:, bnd[name]].T                 # (n_facet, 3) triangles
        counts[name] = len(f)
        faces = np.column_stack([np.full(len(f), 3), f]).ravel()
        surf = pv.PolyData(pts, faces)
        if clip and name == "wall":
            surf = surf.clip(normal=clip, invert=True)
        p.add_mesh(surf, color=color, opacity=opacity,
                   show_edges=show_edges, edge_color="#555555", line_width=0.6)

    p.add_axes(xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)")
    p.show_grid(xtitle="x (mm)", ytitle="y (mm)", ztitle="z (mm)")
    p.add_text(title or f"mesh: {spec.tag or ''}   {m.t.shape[1]} tets   "
                        f"(blue = wall BC, red = metal BC)", font_size=10,
               position="lower_left")     # keep it clear of the grid labels
    p.camera_position = "iso"
    return _finish(p, save, show)


# ─────────────────────────────────────────────────────────────────────────────
# level 3: the field -- slices
# ─────────────────────────────────────────────────────────────────────────────

def slice_plane(grid, normal="z", position=None, array="|E|", nx=360):
    """
    One cut plane through the field, resampled onto a REGULAR grid.

    -> (values (ny, nx) with NaN outside the cavity, extent for imshow, labels)

    The regular grid is what makes the metal bars appear as clean holes: VTK's
    probe marks points that fall outside the mesh, and those become NaN rather
    than being interpolated across the gap. A triangulated cut would smear the
    field straight through a bar.

    position : in METRES along `normal`; default the middle of the mesh.
    """
    pv = _pv()
    ax = _AXIS[normal]
    b = np.array(grid.bounds, dtype=float)           # mm, (xmin,xmax,ymin,...)
    lo, hi = b[2 * ax], b[2 * ax + 1]
    pos_mm = 0.5 * (lo + hi) if position is None else float(position) * MM
    pos_mm = float(np.clip(pos_mm, lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo)))

    i0, i1, l0, l1 = _INPLANE[normal]
    u0, u1 = b[2 * i0], b[2 * i0 + 1]
    v0, v1 = b[2 * i1], b[2 * i1 + 1]
    ny = max(8, int(round(nx * (v1 - v0) / (u1 - u0))))
    us = np.linspace(u0, u1, int(nx))
    vs = np.linspace(v0, v1, int(ny))
    U, V = np.meshgrid(us, vs)
    P = np.zeros((U.size, 3))
    P[:, i0] = U.ravel()
    P[:, i1] = V.ravel()
    P[:, ax] = pos_mm

    probe = pv.PolyData(P).sample(grid)
    vals = np.asarray(probe.point_data[array], dtype=float)
    mask = np.asarray(probe.point_data["vtkValidPointMask"]).astype(bool)
    vals = np.where(mask, vals, np.nan).reshape(U.shape)
    return vals, [u0, u1, v0, v1], (l0, l1), pos_mm


def plot_field_slices(result, i=None, normals=("z", "y", "x"), positions=None,
                      array="|E|", normalize="peak", energy=1.0, nx=360,
                      cmap="magma", share_scale=True, save=None, dpi=160,
                      min_localisation=0.0, title=None, spec=None):
    """
    HORIZONTAL AND VERTICAL CUT PLANES through |E| -- the main 3D field view.

    normals   : which cut directions to show, one ROW each. "z" is the transverse
        cross-section (compare it directly against your 2D plots); "y" and "x"
        are longitudinal and show the z-structure that 2D cannot represent.
    positions : None -> three cuts per row at 25%, 50%, 75% of that axis;
        or a dict {"z": [...], "y": [...]} of positions in METRES;
        or a flat list applied to every normal.
    array     : "|E|" (default), "Ez", "|E|^2", or "E" component names.
    share_scale : one colour scale across all panels (default). Turn it off to
        see the shape of a weak slice, but then panels are NOT comparable.

    Returns (fig, info) where info carries the slice positions and the colour
    limits actually used.
    """
    grid, i, unit = to_pyvista(result, i, normalize=normalize, energy=energy,
                               min_localisation=min_localisation)
    b = np.array(grid.bounds, dtype=float)

    if positions is None:
        pos = {}
        for nm in normals:
            ax = _AXIS[nm]
            lo, hi = b[2 * ax], b[2 * ax + 1]
            pos[nm] = [(lo + q * (hi - lo)) / MM for q in (0.25, 0.5, 0.75)]
    elif isinstance(positions, dict):
        pos = positions
    else:
        pos = {nm: list(positions) for nm in normals}

    panels = []
    for nm in normals:
        for q in pos[nm]:
            vals, extent, labels, pos_mm = slice_plane(grid, nm, q, array=array,
                                                       nx=nx)
            panels.append((nm, pos_mm, vals, extent, labels))

    finite = np.concatenate([p[2][np.isfinite(p[2])].ravel() for p in panels]) \
        if panels else np.array([0.0])
    diverging = array in ("Ez", "Ex", "Ey")
    if diverging:
        lim = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
        vmin, vmax = -lim, lim
        cmap = "RdBu_r" if cmap == "magma" else cmap
    else:
        vmin = 0.0
        vmax = float(np.nanmax(finite)) if finite.size else 1.0

    ncol = max(len(pos[nm]) for nm in normals)
    nrow = len(normals)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.7 * nrow),
                             squeeze=False)
    k = 0
    for r_, nm in enumerate(normals):
        for c_ in range(ncol):
            ax = axes[r_][c_]
            if c_ >= len(pos[nm]):
                ax.axis("off")
                continue
            _nm, pos_mm, vals, extent, labels = panels[k]; k += 1
            kw = dict(vmin=vmin, vmax=vmax) if share_scale else {}
            im = ax.imshow(vals, origin="lower", extent=extent, cmap=cmap,
                           interpolation="nearest", aspect="equal", **kw)
            ax.set_title(f"{nm} = {pos_mm:.2f} mm", fontsize=9)
            ax.set_xlabel(f"{labels[0]} (mm)", fontsize=8)
            ax.set_ylabel(f"{labels[1]} (mm)", fontsize=8)
            ax.tick_params(labelsize=7)
            cb = fig.colorbar(im, ax=ax, fraction=0.045)
            cb.ax.tick_params(labelsize=6)
            if not share_scale or (r_ == 0 and c_ == ncol - 1):
                cb.set_label(unit if array == "|E|" else array, fontsize=7)

    md = result["modes"][i]
    fig.suptitle(title or
                 f"{array} of mode {i}:  f = {result['freqs'][i]/1e9:.4f} GHz,  "
                 f"C = {md['C']:.4f},  Q = {md['Q']:.4g}"
                 + ("   (shared colour scale)" if share_scale else ""),
                 fontsize=12)
    fig.tight_layout()
    if save:
        d = os.path.dirname(os.path.abspath(save))
        if d:
            os.makedirs(d, exist_ok=True)
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig, {"mode": i, "positions": pos, "vmin": vmin, "vmax": vmax,
                 "unit": unit}


def plot_modes_3d(result, n=None, normal="z", position=None, array="|E|",
                  normalize="peak", cmap="magma", ncol=3, save=None, dpi=160,
                  title=None):
    """
    One cut plane per MODE, annotated with f / C / Q / localisation -- the 3D
    analogue of fem_vis.plot_modes, and the quickest way to see which mode the
    shift-invert actually landed on.

    Each panel is normalised independently, because the point here is mode SHAPE.
    """
    nmodes = len(result["modes"]) if n is None else min(n, len(result["modes"]))
    ncol = min(ncol, nmodes)
    nrow = int(np.ceil(nmodes / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.9 * nrow),
                             squeeze=False)
    for j in range(nmodes):
        ax = axes[j // ncol][j % ncol]
        grid, _i, unit = to_pyvista(result, j, normalize=normalize)
        vals, extent, labels, pos_mm = slice_plane(grid, normal, position,
                                                   array=array)
        im = ax.imshow(vals, origin="lower", extent=extent, cmap=cmap,
                       interpolation="nearest", aspect="equal")
        md = result["modes"][j]
        ax.set_title(f"f={result['freqs'][j]/1e9:.4f} GHz   C={md['C']:.4f}\n"
                     f"Q={md['Q']:.3g}   loc={md['localisation']:.3f}",
                     fontsize=9)
        ax.set_xlabel(f"{labels[0]} (mm)", fontsize=8)
        ax.set_ylabel(f"{labels[1]} (mm)", fontsize=8)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, fraction=0.045)
    for j in range(nmodes, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(title or f"{array} at {normal} = "
                          f"{'mid-plane' if position is None else position}: "
                          f"{result.get('tag','')}", fontsize=12)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig


def view_field_3d(result, i=None, array="|E|", normalize="peak", energy=1.0,
                  save=None, show=True, cmap="magma", opacity=0.9,
                  slices=("x", "y", "z"), positions=None, outline=True,
                  interactive_widgets=None, min_localisation=0.0,
                  window_size=(1150, 820), background="white"):
    """
    INTERACTIVE 3D view of the field on cut planes.

    interactive_widgets :
        True  -> draggable slice planes you move with the mouse. Needs a live
                 render window, so it does NOT survive an HTML export.
        False -> fixed orthogonal slices, which DO export to HTML and stay
                 rotatable and zoomable in a browser.
        None  -> True for a live window, False when `save` is given. This is
                 almost always what you want.

    positions : dict {"z": metres, ...} or None for the mid-plane of each axis.

    Returns the pyvista Plotter.
    """
    pv = _pv()
    grid, i, unit = to_pyvista(result, i, normalize=normalize, energy=energy,
                               min_localisation=min_localisation)
    grid.set_active_scalars(array)
    if interactive_widgets is None:
        interactive_widgets = save is None

    p = pv.Plotter(off_screen=(save is not None), window_size=window_size)
    p.set_background(background)
    if outline:
        p.add_mesh(grid.outline(), color="#333333", line_width=2)

    if interactive_widgets:
        # one draggable plane per requested normal
        for nm in slices:
            p.add_mesh_slice(grid, normal=nm, cmap=cmap, opacity=opacity,
                             scalars=array)
    else:
        b = np.array(grid.bounds, dtype=float)
        for nm in slices:
            ax = _AXIS[nm]
            if positions and nm in positions:
                q = float(positions[nm]) * MM
            else:
                q = 0.5 * (b[2 * ax] + b[2 * ax + 1])
            origin = [0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]),
                      0.5 * (b[4] + b[5])]
            origin[ax] = q
            sl = grid.slice(normal=nm, origin=origin)
            if sl.n_points:
                p.add_mesh(sl, cmap=cmap, opacity=opacity, scalars=array,
                           scalar_bar_args={"title": array})

    md = result["modes"][i]
    p.add_text(f"mode {i}:  f = {result['freqs'][i]/1e9:.4f} GHz   "
               f"C = {md['C']:.4f}   Q = {md['Q']:.4g}", font_size=10,
               position="lower_left")
    p.add_axes(xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)")
    p.camera_position = "iso"
    return _finish(p, save, show)


# ─────────────────────────────────────────────────────────────────────────────
# scalar summaries
# ─────────────────────────────────────────────────────────────────────────────

def plot_length_scan(result2d, lengths, sigma, i=None, save=None, dpi=160,
                     min_localisation=0.0, title=None):
    """
    Q and f against cavity LENGTH, from a single 2D solve via the exact extrusion
    relations -- no 3D solve at all.

    This is the plot that answers "how long should the cavity be": f is flat
    (the p = 0 mode does not care), C is flat, and Q rises towards its 2D value
    as the endcaps become a smaller fraction of the wall area. The p = 1 curve is
    drawn alongside because it is the neighbour that comes DOWN in frequency as
    the cavity lengthens, and where it crosses the operating mode is where you
    get mode mixing.
    """
    lengths = np.asarray(lengths, dtype=float)
    rows = [cv3.extruded_modes(result2d, L, sigma=sigma, p_max=1, i=i,
                               min_localisation=min_localisation)
            for L in lengths]
    Q0 = np.array([r[0]["Q"] for r in rows])
    f0 = np.array([r[0]["f"] for r in rows])
    f1 = np.array([r[1]["f"] for r in rows])
    Q2d = rows[0][0]["Q_2d"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(lengths * MM, Q0, "o-", color="#1f4e79", lw=1.8, ms=4,
                 label="$p=0$ (operating)")
    axes[0].axhline(Q2d, color="#b03a2e", ls=":", lw=1.4,
                    label=f"2D limit $L\\to\\infty$ = {Q2d:.0f}")
    axes[0].set_xlabel("cavity length $L$ (mm)")
    axes[0].set_ylabel("$Q$")
    axes[0].set_title("endcap loss vs length", fontsize=10)
    axes[0].grid(alpha=0.3, ls=":"); axes[0].legend(fontsize=8)

    axes[1].plot(lengths * MM, f0 / 1e9, "o-", color="#1f4e79", lw=1.8, ms=4,
                 label="$p=0$  (length-independent)")
    axes[1].plot(lengths * MM, f1 / 1e9, "s--", color="#2e7d32", lw=1.6, ms=4,
                 label="$p=1$  ($C=0$)")
    axes[1].set_xlabel("cavity length $L$ (mm)")
    axes[1].set_ylabel("frequency (GHz)")
    axes[1].set_title("longitudinal mode ladder", fontsize=10)
    axes[1].grid(alpha=0.3, ls=":"); axes[1].legend(fontsize=8)

    fig.suptitle(title or "3D from 2D: exact extrusion relations", fontsize=12)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig


def check_spec_3d(spec, verbose=True):
    """
    Non-graphical layout check: inclusions that overlap each other or stick out of
    the cavity, using ROTATED bounding boxes. The 3D counterpart of
    fem_vis.check_spec, and worth running before a long solve because gmsh will
    NOT complain about either problem -- overlapping cut tools just merge into
    one hole.
    """
    msgs = []
    boxes = list(getattr(spec, "metal", [])) + \
            [r for r, _m in getattr(spec, "dielectric", [])]
    if isinstance(spec, cv3.CavitySpec3D):
        ob = spec.outer.bounds
        for r in boxes:
            b = r.bounds
            if (b[0] < ob[0] - 1e-12 or b[1] < ob[1] - 1e-12 or b[2] < ob[2] - 1e-12
                    or b[3] > ob[3] + 1e-12 or b[4] > ob[4] + 1e-12
                    or b[5] > ob[5] + 1e-12):
                msgs.append(f"'{r.name}' extends outside the cavity")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            A, B = boxes[i].bounds, boxes[j].bounds
            if (A[0] < B[3] and B[0] < A[3] and A[1] < B[4] and B[1] < A[4]
                    and A[2] < B[5] and B[2] < A[5]):
                msgs.append(f"'{boxes[i].name}' overlaps '{boxes[j].name}'")
    if verbose:
        print("[check3d] " + ("OK, no overlaps or out-of-bounds boxes"
                              if not msgs else f"{len(msgs)} problem(s):"))
        for m in msgs:
            print("   -", m)
    return msgs