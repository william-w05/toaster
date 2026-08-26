"""
3D vector eigenmode solver for microwave cavities -- the 3D counterpart of
fem_solve.py.

  gmsh (OpenCASCADE)  -> geometry: boxes/cylinders, boolean cut, STEP/IGES import
  scikit-fem          -> FEM assembly with NEDELEC edge elements
  scipy               -> shift-invert eigensolve with EXPLICIT KERNEL DEFLATION

WHY THIS IS A VECTOR PROBLEM (and 2D was not)
    In 2D with no z-dependence the TM modes reduce to a scalar Helmholtz problem
    for E_z. In 3D there is no such reduction: the modes are genuine vector
    fields and the eigenproblem is the curl-curl system

        curl( (1/mu_r) curl E ) = k0^2 eps_r E ,     n x E = 0 on PEC

    Discretised with Whitney edge (Nedelec) elements, which enforce tangential
    continuity and normal discontinuity -- exactly the physics of E across a
    material interface. NODAL (Lagrange) elements applied to this system produce
    a spectrum riddled with spurious non-physical modes; edge elements do not.
    That is the whole reason for ElementTetN0 rather than ElementTetP2.

THE KERNEL, AND WHY NAIVE SHIFT-INVERT CRAWLS
    curl(grad phi) = 0, so the discrete curl-curl matrix K has a null space of
    dimension equal to the number of interior nodes -- hundreds to millions of
    eigenvectors all at lambda = 0. Shift-invert at sigma maps every one of them
    to -1/sigma, a single massively degenerate cluster. ARPACK then has to
    resolve that cluster before it can report anything else, and the cost
    explodes: on a 3157-dof pillbox, asking for 6 modes took 176 s, and 5 of the
    6 returned were null-space junk.

    The fix here is exact rather than heuristic. The kernel is precisely the
    range of the DISCRETE GRADIENT G (for Whitney elements, G is the signed
    edge-node incidence matrix: -1 at the low-numbered vertex, +1 at the high
    one). Two facts make deflation clean:

      * K G = 0 identically -- verified numerically at 1e-16 relative, and
        asserted at runtime by check_gradient_kernel().
      * every PHYSICAL mode is already M-orthogonal to the kernel. If K u =
        lambda M u with lambda != 0 then (G phi)^T K u = 0 because K G = 0, hence
        lambda (G phi)^T M u = 0, hence (G phi)^T M u = 0.

    So the M-orthogonal projector P = I - G (G^T M G)^-1 G^T M annihilates the
    kernel and acts as the identity on every mode we want. Feeding ARPACK
    OPinv = P (K - sigma M)^-1 P therefore returns the physical spectrum
    unchanged while the kernel is mapped to 0 and never selected. The sandwich
    (P on both sides, not one) is what keeps the operator M-self-adjoint, which
    ARPACK's symmetric mode requires.

    Same problem, same mesh: 176 s -> 0.07 s, and no null modes in the output.

LOSS / Q
    Same perturbative treatment as 2D, one dimension up:

        Q = omega U / P_loss
        U      = (eps0/2) integral( eps_r |E|^2 dV )
        P_loss = (R_s/2) surface_integral( |H_t|^2 dS )
        |H|    = |curl E| / (omega mu0 mu_r)
        R_s    = sqrt(omega mu0 / (2 sigma))

    At a PEC wall H has no normal component, so |H_t| = |H| there exactly and no
    projection is needed. Different boundary groups may carry different metals.

FORM FACTOR
    C = |integral E_z dV|^2 / ( V * integral eps_r |E|^2 dV )

    the direct 3D analogue of the 2D expression, with area -> volume. NOTE it
    still singles out E_z: it is the coupling to an axion field along the
    solenoid axis, not a norm.

PARALLELISM
    As in 2D: many small independent solves, one geometry per process. gmsh is
    not thread-safe but is fine in separate processes.
"""

from __future__ import annotations

import os
import io
import re as _re
import uuid
import tempfile
import contextlib
import time
import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed

import gmsh
import skfem
from skfem import (Basis, FacetBasis, ElementTetN0, ElementTetP1,
                   BilinearForm, LinearForm, Functional, asm)
# NOTE skfem's ElementTetN1 is an alias for ElementTetN0 (both are the lowest
# order Whitney edge element, one dof per edge). There is no higher-order
# Nedelec tetrahedron available, so h-refinement is the only accuracy knob.
from skfem.helpers import dot, curl
from scipy.sparse.linalg import eigsh, splu, LinearOperator

# reuse the 2D module's materials and constants so the two solvers cannot drift
try:                                    # inside the package
    from scripts import fem_solve as fem2d
except ImportError:                     # standalone / notebook
    import scripts.fem_solve as fem2d

C0 = fem2d.C0
MU0 = fem2d.MU0
EPS0 = fem2d.EPS0
Material = fem2d.Material
SIGMA_COPPER = fem2d.SIGMA_COPPER
SIGMA_AL = fem2d.SIGMA_AL
SIGMA_COPPER_COMSOL = fem2d.SIGMA_COPPER_COMSOL
SIGMA_AL_COMSOL = fem2d.SIGMA_AL_COMSOL

QUIET = True
_BBOX_TOL = 1e-6


@contextlib.contextmanager
def _quiet(enabled=None):
    """Swallow stdout only; stderr is left alone."""
    if enabled is None:
        enabled = QUIET
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ─────────────────────────────────────────────────────────────────────────────
# CAD units
# ─────────────────────────────────────────────────────────────────────────────

_STEP_UNIT = _re.compile(
    r"LENGTH_UNIT[^;]*?SI_UNIT\s*\(\s*([^,]*),\s*\.METRE\.", _re.I | _re.S)
_SI_PREFIX = {"$": 1.0, "": 1.0, "*": 1.0, ".MILLI.": 1e-3, ".CENTI.": 1e-2,
              ".DECI.": 1e-1, ".DECA.": 1e1, ".HECTO.": 1e2, ".KILO.": 1e3,
              ".MICRO.": 1e-6, ".NANO.": 1e-9}
_IGES_UNIT = {"MM": 1e-3, "M": 1.0, "IN": 0.0254, "FT": 0.3048, "CM": 1e-2}


def detect_cad_units(path, default=1.0, verbose=False):
    """
    METRES PER FILE UNIT, read out of the CAD file itself.

    Every unit mistake in this pipeline has cost an hour, because nothing fails:
    the geometry is simply 1000x the intended size, the mesh estimate explodes or
    the frequency comes out 1000x off, and the numbers all look self-consistent.
    The file knows the answer, so ask it rather than guessing.

    STEP declares it outright, e.g.
        ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT(.MILLI.,.METRE.) )   -> 1e-3
        ( LENGTH_UNIT() NAMED_UNIT(*) SI_UNIT($,.METRE.) )         -> 1.0
    IGES carries it as the units name in the Global section (2HMM, 2HIN, ...).

    Returns `default` if nothing is found, so a truncated or exotic file degrades
    to the old behaviour rather than raising.
    """
    ext = os.path.splitext(path)[1].lower()
    try:
        with open(path, "r", errors="ignore") as fh:
            head = fh.read(400_000)
    except OSError:
        return float(default)
    scale = None
    if ext in (".step", ".stp", ".p21"):
        m = _STEP_UNIT.search(head)
        if m:
            scale = _SI_PREFIX.get(m.group(1).strip().upper())
    elif ext in (".igs", ".iges"):
        m = _re.search(r"\b\d+H(MM|CM|M|IN|FT)\b", head[:20_000], _re.I)
        if m:
            scale = _IGES_UNIT.get(m.group(1).upper())
    if scale is None:
        return float(default)
    if verbose:
        print(f"[units] {os.path.basename(path)} declares "
              f"{scale:g} m per file unit", flush=True)
    return float(scale)


def tmp_msh_path(prefix="cavity3d"):
    return os.path.join(tempfile.gettempdir(),
                        f"{prefix}_{os.getpid()}_{uuid.uuid4().hex[:8]}.msh")


# ─────────────────────────────────────────────────────────────────────────────
# geometry primitives
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Box:
    """
    Axis-aligned-then-rotated box in METRES, given by its lower corner and size.

    angle : degrees CCW about the z-axis through the box's OWN CENTRE, matching
        fem_solve.Rect.angle. gmsh has no rotated-box primitive, so build_mesh_3d
        creates it axis-aligned and rotates it, exactly as the 2D code does.

    The z extent is `d`; a bar that spans the full cavity length is just a box
    with d = cavity length, which is what from_2d() builds.
    """
    x0: float
    y0: float
    z0: float
    w: float
    h: float
    d: float
    name: str = "box"
    angle: float = 0.0

    def as_tuple(self):
        """gmsh occ.addBox signature: (x, y, z, dx, dy, dz) -- UN-rotated."""
        return (self.x0, self.y0, self.z0, self.w, self.h, self.d)

    @classmethod
    def from_center(cls, cx, cy, cz, w, h, d, name="box", angle=0.0):
        return cls(cx - 0.5 * w, cy - 0.5 * h, cz - 0.5 * d, w, h, d, name, angle)

    @classmethod
    def from_rect(cls, rect, z0, d, name=None):
        """Extrude a 2D fem_solve.Rect through z in [z0, z0 + d]."""
        return cls(rect.x0, rect.y0, z0, rect.w, rect.h, d,
                   name or rect.name, getattr(rect, "angle", 0.0))

    @property
    def cx(self):
        return self.x0 + 0.5 * self.w

    @property
    def cy(self):
        return self.y0 + 0.5 * self.h

    @property
    def cz(self):
        return self.z0 + 0.5 * self.d

    @property
    def center(self):
        return (self.cx, self.cy, self.cz)

    def corners(self):
        """(8, 3) array of the actual corners, rotation included."""
        cx, cy, cz = self.center
        dx, dy, dz = 0.5 * self.w, 0.5 * self.h, 0.5 * self.d
        s = np.array([[sx * dx, sy * dy, sz * dz]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        if self.angle:
            t = np.radians(self.angle)
            R = np.array([[np.cos(t), -np.sin(t), 0.0],
                          [np.sin(t), np.cos(t), 0.0],
                          [0.0, 0.0, 1.0]])
            s = s @ R.T
        return s + np.array([cx, cy, cz])

    @property
    def bounds(self):
        """(xmin, ymin, zmin, xmax, ymax, zmax) of the ROTATED box."""
        if not self.angle:
            return (self.x0, self.y0, self.z0,
                    self.x0 + self.w, self.y0 + self.h, self.z0 + self.d)
        c = self.corners()
        return (float(c[:, 0].min()), float(c[:, 1].min()), float(c[:, 2].min()),
                float(c[:, 0].max()), float(c[:, 1].max()), float(c[:, 2].max()))

    def moved_to(self, cx, cy, cz):
        return Box.from_center(cx, cy, cz, self.w, self.h, self.d,
                               self.name, self.angle)

    def shifted(self, dx=0.0, dy=0.0, dz=0.0):
        return Box(self.x0 + dx, self.y0 + dy, self.z0 + dz,
                   self.w, self.h, self.d, self.name, self.angle)


def CBox(cx, cy, cz, w, h, d, name="box", angle=0.0):
    """Shorthand for Box.from_center."""
    return Box.from_center(cx, cy, cz, w, h, d, name, angle)


# ─────────────────────────────────────────────────────────────────────────────
# specs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CavitySpec3D:
    """
    Rectangular-prism cavity with box inclusions -- the direct 3D analogue of
    fem_solve.CavitySpec.

    outer      : the cavity volume.
    metal      : boxes CUT OUT (setminus); their walls become BC surfaces.
    dielectric : (Box, Material) kept in the mesh as distinct material regions.
    mesh_size  : target element size (m). In 3D this bites much harder than in
        2D: halving it multiplies the tetrahedron count by ~8 and the edge-DOF
        count with it. Start coarse.
    """
    outer: Box
    metal: list = field(default_factory=list)
    dielectric: list = field(default_factory=list)
    background: Material = field(default_factory=lambda: Material("vacuum"))
    wall_material: Material = field(default_factory=lambda: Material("cu"))
    metal_material: Material = field(default_factory=lambda: Material("cu"))
    mesh_size: float = 0.004
    mesh_size_min: float | None = None
    mesh_uniform: bool = False
    tag: str = ""

    def add_outer(self, occ):
        return occ.addBox(*self.outer.as_tuple())

    def on_wall(self, pts):
        """True if every sampled point of a surface lies on the OUTER boundary."""
        x0, y0, z0, x1, y1, z1 = self.outer.bounds
        tol = _BBOX_TOL + 1e-6 * max(self.outer.w, self.outer.h, self.outer.d)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        return bool(np.all(np.abs(x - x0) < tol) or np.all(np.abs(x - x1) < tol) or
                    np.all(np.abs(y - y0) < tol) or np.all(np.abs(y - y1) < tol) or
                    np.all(np.abs(z - z0) < tol) or np.all(np.abs(z - z1) < tol))

    @property
    def extent(self):
        return self.outer.bounds

    @property
    def volume(self):
        return self.outer.w * self.outer.h * self.outer.d


@dataclass
class CylSpec3D:
    """
    Circular-cylinder cavity (a real pillbox, endcaps included).

    THE validation case: pillbox_analytic() gives f, C and Q in closed form, and
    unlike the 2D disk it exercises the endcap loss that only exists in 3D.
    """
    radius: float
    length: float
    center: tuple = (0.0, 0.0, 0.0)      # centre of the cylinder
    metal: list = field(default_factory=list)
    dielectric: list = field(default_factory=list)
    background: Material = field(default_factory=lambda: Material("vacuum"))
    wall_material: Material = field(default_factory=lambda: Material("cu"))
    metal_material: Material = field(default_factory=lambda: Material("cu"))
    mesh_size: float = 0.004
    mesh_size_min: float | None = None
    mesh_uniform: bool = False
    tag: str = ""

    def add_outer(self, occ):
        cx, cy, cz = self.center
        return occ.addCylinder(cx, cy, cz - 0.5 * self.length,
                               0.0, 0.0, self.length, self.radius)

    def on_wall(self, pts):
        cx, cy, cz = self.center
        tol = _BBOX_TOL + 1e-6 * max(self.radius, self.length)
        r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        z = pts[:, 2]
        on_barrel = np.all(np.abs(r - self.radius) < tol)
        on_cap = ((np.all(np.abs(z - (cz - 0.5 * self.length)) < tol) or
                   np.all(np.abs(z - (cz + 0.5 * self.length)) < tol))
                  and np.all(r <= self.radius + tol))
        return bool(on_barrel or on_cap)

    @property
    def extent(self):
        cx, cy, cz = self.center
        R, L = self.radius, self.length
        return (cx - R, cy - R, cz - 0.5 * L, cx + R, cy + R, cz + 0.5 * L)

    @property
    def volume(self):
        return np.pi * self.radius ** 2 * self.length


@dataclass
class ImportedSpec3D:
    """
    Cavity whose geometry comes from a CAD file (.igs/.iges, .step/.stp, .brep).

    THE ONE DECISION THAT MATTERS -- `interior`:

      interior=False : the imported solid IS the vacuum cavity. Use this when the
          CAD models the void directly.
      interior=True  : the imported solid is the METAL SHELL, and the cavity is
          the void ENCLOSED BY IT. This is the usual case for a real part: a
          hollow cylinder with walls and endcaps of finite thickness is a model of
          the metal, not of the vacuum. Solving the shell itself would be
          meaningless -- you would get the modes of a solid lump of aluminium
          bounded by its own outer skin.

    HOW THE VOID IS FOUND. The inner surfaces are those whose bounding box does
    not touch the overall bounding box; they are sewn into a surface loop and that
    loop becomes the volume. This is exact for a closed shell (walls plus
    endcaps), which is precisely the case where the void is well defined. It fails
    for an OPEN tube, where the "void" is not enclosed and the notion has no
    meaning -- cap it in CAD first. Set `report=True` to print the classification
    and the recovered volume before committing to a mesh.

    Two things that bite on IGES specifically, both seen in testing:

      * IGES is a SURFACE format. A file that looks like a solid in your CAD
        viewer typically imports as a bag of trimmed surfaces with no volume at
        all, and gmsh's healShapes(makeSolids=True) does NOT reliably fix it.
        This class therefore always falls back to sewing an explicit surface loop,
        which does work.
      * a shell recovered that way is often not a valid solid for BOOLEAN
        operations. Cutting it out of a bounding box silently returns the box
        unchanged -- no error, no warning, just a wrong answer. That is why the
        void is built from its own surfaces rather than by subtraction.

    scale : an EXTRA multiplier, default 1.0, which you should not normally need.
        Units are handled by pinning OCC's target unit to metres at import, so a
        file declaring millimetres and one declaring metres both arrive at the
        same physical size. Only reach for this if the CAD itself is drawn at the
        wrong scale.

    expect_size : the longest OVERALL dimension of the part, in METRES -- the
        number you would read off the CAD model, not the cavity inside it. Optional but
        worth setting: it is checked against what was actually imported and
        raises if they disagree, which turns a silent 1000x error into an
        immediate one. Unit mistakes here do not fail cleanly -- the frequency
        just comes out 1000x off and everything looks self-consistent.

    materials : IGES carries no material information, so wall_material must be
        set here. The whole void boundary is treated as `wall`.
    """
    path: str
    interior: bool = False
    metal: list = field(default_factory=list)
    dielectric: list = field(default_factory=list)
    background: Material = field(default_factory=lambda: Material("vacuum"))
    wall_material: Material = field(default_factory=lambda: Material("cu"))
    metal_material: Material = field(default_factory=lambda: Material("cu"))
    mesh_size: float = 0.004
    mesh_size_min: float | None = None
    mesh_uniform: bool = False
    scale: float | str = 1.0
    expect_size: float | None = None
    report: bool = True
    touch_tol: float = 1e-4
    tag: str = ""
    _extent: tuple | None = None

    def _resolve_scale(self):
        """`scale` is an extra multiplier on top of OCC's unit normalisation.
        "auto" is accepted for backwards compatibility but is now a no-op: with
        OCCTargetUnit pinned to metres there is nothing left to detect."""
        if isinstance(self.scale, str):
            if self.scale.lower() != "auto":
                raise ValueError(f"scale must be a number or 'auto', got "
                                 f"{self.scale!r}")
            object.__setattr__(self, "scale", 1.0)
        return float(self.scale)

    def add_outer(self, occ):
        # THE unit fix. OCC converts an imported shape into a TARGET unit, and
        # what that target is by default DEPENDS ON THE GMSH BUILD: some versions
        # convert to millimetres, others pass the file's raw numbers through. So
        # the same metre-declared STEP arrives as 1.1 on one machine and 1100 on
        # another, and no amount of reading the file's own SI_UNIT can tell you
        # which -- that was the bug in detect_cad_units, which assumed raw
        # passthrough. Pinning the target to metres makes both cases land at 1.1
        # and makes `scale` a plain multiplier you almost never need.
        gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        self._resolve_scale()
        gmsh.option.setNumber("Geometry.OCCAutoFix", 1)
        gmsh.option.setNumber("Geometry.OCCSewFaces", 1)
        gmsh.option.setNumber("Geometry.OCCMakeSolids", 1)
        occ.importShapes(self.path, highestDimOnly=True)
        occ.synchronize()

        surfs = [t for (d, t) in gmsh.model.getEntities(2)]
        if not surfs:
            raise ValueError(f"{self.path} imported no surfaces at all; the file "
                             f"is empty, or the format is not one OCC reads.")
        B = np.array([gmsh.model.getBoundingBox(2, t) for t in surfs])
        g = [B[:, 0].min(), B[:, 1].min(), B[:, 2].min(),
             B[:, 3].max(), B[:, 4].max(), B[:, 5].max()]
        tol = self.touch_tol * max(g[3] - g[0], g[4] - g[1], g[5] - g[2])

        inner, outer_s = [], []
        for t, b in zip(surfs, B):
            touch = (abs(b[0] - g[0]) < tol or abs(b[1] - g[1]) < tol
                     or abs(b[2] - g[2]) < tol or abs(b[3] - g[3]) < tol
                     or abs(b[4] - g[4]) < tol or abs(b[5] - g[5]) < tol)
            (outer_s if touch else inner).append(t)

        if self.interior:
            if not inner:
                raise ValueError(
                    f"{self.path}: interior=True but every surface touches the "
                    f"overall bounding box, so there is no enclosed void. Either "
                    f"the part is solid, or it is an open tube that needs capping "
                    f"in CAD, or the walls are so thin that touch_tol "
                    f"({self.touch_tol}) swallowed them -- try a smaller value.")
            loop = occ.addSurfaceLoop(inner)
            vol = occ.addVolume([loop])
            occ.synchronize()
            chosen, n_used = vol, len(inner)
        else:
            vols = [t for (d, t) in gmsh.model.getEntities(3)]
            if not vols:
                loop = occ.addSurfaceLoop(surfs)
                vols = [occ.addVolume([loop])]
                occ.synchronize()
            if len(vols) > 1:
                out, _ = occ.fuse([(3, vols[0])], [(3, v) for v in vols[1:]])
                occ.synchronize()
                vols = [t for (d, t) in out if d == 3]
            chosen, n_used = vols[0], len(surfs)

        # DO NOT occ.dilate() here. Scaling a shell that was recovered by sewing
        # silently corrupts it: the bounding box comes back wrong (a cone reports
        # its untrimmed extent) and the mesher then tries to fill an unbounded
        # region and is killed by the OOM reaper. build_mesh_3d applies the scale
        # to the finished MESH instead, via affineTransform, which touches only
        # node coordinates and cannot invalidate the geometry.
        bb = tuple(x * self.scale for x in gmsh.model.getBoundingBox(3, chosen))
        object.__setattr__(self, "_extent", bb)
        # check the PART, not the extracted cavity: expect_size is the number you
        # read off the CAD model, and with interior=True the void is smaller
        pbb = gmsh.model.getBoundingBox(-1, -1)
        span = max(pbb[3] - pbb[0], pbb[4] - pbb[1], pbb[5] - pbb[2]) * self.scale
        if self.expect_size:
            ratio = span / float(self.expect_size)
            if not (0.95 <= ratio <= 1.05):
                raise ValueError(
                    f"{os.path.basename(self.path)} imported with a longest "
                    f"dimension of {span:.6g} m, but expect_size says "
                    f"{float(self.expect_size):.6g} m -- a factor of "
                    f"{ratio:.4g}.\n"
                    f"  A factor near 1000 or 1/1000 is a unit problem; check "
                    f"the CAD, or set scale={1.0/ratio:.6g} to force it.")
        if self.report:
            # getMass can come back NEGATIVE when the sewn loop is oriented
            # inward. That is a sign convention, not an error -- the magnitude is
            # the volume and gmsh meshes it either way.
            vol_m3 = abs(occ.getMass(3, chosen)) * self.scale ** 3
            print(f"[import] {os.path.basename(self.path)}: {len(surfs)} surfaces "
                  f"({len(inner)} inner / {len(outer_s)} outer), "
                  f"interior={self.interior}, scale={self.scale:g}", flush=True)
            print(f"[import]   cavity volume = {vol_m3:.6g} m^3, bbox (m) = "
                  f"{tuple(round(float(x), 5) for x in bb)}", flush=True)
            span = max(bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2])
            print(f"[import]   longest dimension = {span:.4g} m", flush=True)
            if not (1e-3 <= span <= 20.0):
                print(f"[import]   WARNING: that is not a plausible microwave "
                      f"cavity size. `scale` is almost", flush=True)
                print(f"[import]   certainly wrong -- a file in millimetres read "
                      f"as metres gives exactly this.", flush=True)
        return chosen

    def on_wall(self, pts):
        """Everything bounding the void is conducting wall. build_mesh_3d asks the
        explicit metal boxes first, so this catch-all is correct."""
        return True

    @property
    def extent(self):
        if self._extent is None:
            raise RuntimeError("extent is only known after the geometry is built; "
                               "call build_mesh_3d or solve_cavity_3d first.")
        return self._extent

    @property
    def volume(self):
        return float("nan")


def from_2d(spec2d, length, mesh_size=None, tag=None, center_z=0.0,
            partial=None):
    """
    Extrude a 2D fem_solve.CavitySpec into a CavitySpec3D: every Rect becomes a
    Box spanning the full cavity length, so the cross-section is exactly the one
    the 2D solver used.

    THIS IS THE BRIDGE from the existing toaster code:

        spec2d = viz.toaster_spec(params_m, gap0=..., gap1=..., cavity_h=...)
        spec3d = fem_solve_3d.from_2d(spec2d, length=0.20)

    partial : optional {name: (z0, dz)} in METRES to make a bar span only part of
        the length -- a tuning rod that does not reach the endcaps, say. Anything
        not named spans the full length. The moment ANY bar is partial the
        geometry stops being a prism and extruded_modes() no longer applies; only
        the full 3D solve is valid then.
    """
    o = spec2d.outer
    outer = Box.from_center(o.cx, o.cy, center_z, o.w, o.h, length, "cavity")
    partial = partial or {}

    def span(name):
        return partial.get(name, (center_z - 0.5 * length, length))

    metal = []
    for r in spec2d.metal:
        z0, dz = span(r.name)
        metal.append(Box(r.x0, r.y0, z0, r.w, r.h, dz, r.name,
                         getattr(r, "angle", 0.0)))
    diel = []
    for r, mat in spec2d.dielectric:
        z0, dz = span(r.name)
        diel.append((Box(r.x0, r.y0, z0, r.w, r.h, dz, r.name,
                         getattr(r, "angle", 0.0)), mat))
    return CavitySpec3D(
        outer=outer, metal=metal, dielectric=diel,
        background=spec2d.background,
        wall_material=spec2d.wall_material,
        metal_material=spec2d.metal_material,
        mesh_size=spec2d.mesh_size if mesh_size is None else mesh_size,
        mesh_size_min=spec2d.mesh_size_min,
        mesh_uniform=spec2d.mesh_uniform,
        tag=tag if tag is not None else (spec2d.tag or "") + "_3d")


# ─────────────────────────────────────────────────────────────────────────────
# the extrusion shortcut: exact 3D answers from a 2D solve
# ─────────────────────────────────────────────────────────────────────────────

def extruded_modes(result2d, length, sigma, p_max=2, i=None,
                   min_localisation=0.0):
    """
    EXACT 3D f, C and Q for a PRISM (2D cross-section extruded by `length` with
    PEC endcaps), computed from a 2D solve. No 3D mesh, no 3D solve.

    For a uniform cross-section the 3D TM modes separate:

        E_z(x,y,z) = psi(x,y) cos(p pi z / L),      k^2 = k_t^2 + (p pi / L)^2

    with psi the 2D eigenfunction. Three consequences, all worth knowing before
    you spend a night on a 3D mesh:

    FREQUENCY  f_p = (c/2pi) sqrt(k_t^2 + (p pi/L)^2). The p = 0 mode is
        INDEPENDENT OF LENGTH -- it is the 2D answer, exactly. Making the cavity
        longer does not detune it.

    FORM FACTOR  C_0 equals the 2D C exactly (the z-integral factors out of both
        numerator and denominator and cancels). For every p >= 1,
        integral_0^L cos(p pi z/L) dz = 0, so C_p = 0 IDENTICALLY. Only the p = 0
        mode couples. This is why the 2D model was a legitimate stand-in for the
        form factor in the first place, and why the p >= 1 modes below matter only
        as neighbours that can hybridise with the operating mode, never as
        candidates for it.

    QUALITY FACTOR  this is where 3D genuinely differs, because the endcaps are
        new lossy surface that a 2D model cannot see. For p = 0,

            1/Q_3D = 1/Q_2D + 2 R_s / (mu0 c k0 L)

        exactly. The correction is pure endcap loss, it scales as 1/L, and it is
        the reason to care about cavity length at all. Sanity check against the
        pillbox: with Q_2D = mu0 c j01 / (2 R_s) and k0 = j01/R this collapses to
        the textbook Q = [mu0 c j01/(2 R_s)] / (1 + R/L).

    result2d : a dict from fem_solve.solve_cavity (uses freqs + modes), or a
        single mode dict with keys f, C, Q.
    sigma    : conductivity of the ENDCAPS, S/m. Required, and not derivable
        from the 2D result: Q_2D is proportional to 1/R_s, so it pins the
        product but never R_s itself, and the endcap term needs R_s on its own.
        Use the wall_material sigma -- the endcaps are outer wall.
    i        : which mode of result2d; default is the highest-C one.

    Returns a list of dicts, one per p in 0..p_max.
    """
    if "modes" in result2d:
        if i is None:
            cand = [j for j, m in enumerate(result2d["modes"])
                    if m["localisation"] >= min_localisation] or \
                   list(range(len(result2d["modes"])))
            i = max(cand, key=lambda j: result2d["modes"][j]["C"])
        f2 = float(result2d["freqs"][i])
        m2 = result2d["modes"][i]
    else:
        f2, m2 = float(result2d["f"]), result2d
    C2, Q2 = float(m2["C"]), float(m2["Q"])

    k_t = 2.0 * np.pi * f2 / C0
    out = []
    for p in range(int(p_max) + 1):
        k = np.sqrt(k_t ** 2 + (p * np.pi / length) ** 2)
        f = C0 * k / (2.0 * np.pi)
        omega = 2.0 * np.pi * f
        if p == 0:
            R_s = float(np.sqrt(omega * MU0 / (2.0 * sigma)))
            Q = 1.0 / (1.0 / Q2 + 2.0 * R_s / (MU0 * C0 * k * length))
            C = C2
        else:
            # p >= 1: E_z has a cos(p pi z/L) factor whose integral vanishes
            C = 0.0
            Q = np.nan          # needs the transverse fields; not the operating mode
        out.append({"p": p, "f": f, "C": C, "Q": Q, "k": k, "length": length,
                    "f_2d": f2, "C_2d": C2, "Q_2d": Q2,
                    "endcap_only": (p == 0)})
    return out


def extruded_Q(Q_2d, f_2d, length, sigma):
    """
    Just the p = 0 quality factor: 1/Q_3D = 1/Q_2D + 2 R_s/(mu0 c k0 L).

    Vectorised over Q_2d/f_2d, so a whole 2D tuning sweep converts in one call:

        Q3 = extruded_Q(d["Q"], d["f"], length=0.2, sigma=fem.SIGMA_AL_COMSOL)
    """
    f = np.asarray(f_2d, dtype=np.float64)
    Q2 = np.asarray(Q_2d, dtype=np.float64)
    omega = 2.0 * np.pi * f
    k0 = omega / C0
    R_s = np.sqrt(omega * MU0 / (2.0 * sigma))
    return 1.0 / (1.0 / Q2 + 2.0 * R_s / (MU0 * C0 * k0 * length))


# ─────────────────────────────────────────────────────────────────────────────
# analytic references
# ─────────────────────────────────────────────────────────────────────────────

def pillbox_analytic(radius, length, sigma=SIGMA_COPPER, n=1, eps_r=1.0,
                     mu_r=1.0):
    """
    Closed-form TM_0n0 of an empty circular pillbox -- f, C and Q together, which
    makes it the end-to-end check on the whole 3D pipeline.

        f = c j_0n / (2 pi R sqrt(eps_r mu_r))          (independent of length)
        C = 4 / j_0n^2                                   (0.6917 for TM_010)
        Q = [mu0 c j_0n / (2 R_s)] / (1 + R/L)

    The Q is the 2D cylinder_analytic value divided by (1 + R/L): the extra term
    is the two endcaps, which is exactly the loss channel 2D cannot represent.
    C is unchanged from 2D because E_z is uniform along z for p = 0.
    """
    from scipy.special import jn_zeros
    j0n = float(jn_zeros(0, n)[-1])
    k = j0n / radius
    f = C0 * k / (2.0 * np.pi * np.sqrt(eps_r * mu_r))
    omega = 2.0 * np.pi * f
    R_s = np.sqrt(omega * MU0 / (2.0 * sigma))
    Q = (MU0 * C0 * j0n / (2.0 * R_s)) / (1.0 + radius / length)
    return {"f": f, "C": 4.0 / j0n ** 2, "Q": Q, "j0n": j0n, "k": k, "R_s": R_s,
            "Q_2d": MU0 * C0 * j0n / (2.0 * R_s)}


def rect_cavity_analytic(a, b, length, sigma=SIGMA_COPPER, m=1, n=1):
    """
    Closed-form TM_mn0 of an empty rectangular box a x b x L.

        E_z = sin(m pi x/a) sin(n pi y/b),   k^2 = (m pi/a)^2 + (n pi/b)^2
        C   = 64/pi^4 = 0.6571 for m = n = 1, and 0 whenever m or n is even
              (the integral of a full sine period vanishes)

    A second, independent check on the solver: unlike the pillbox it has corners,
    which is where edge elements earn their keep.
    """
    kx, ky = m * np.pi / a, n * np.pi / b
    k = np.hypot(kx, ky)
    f = C0 * k / (2.0 * np.pi)
    omega = 2.0 * np.pi * f
    R_s = np.sqrt(omega * MU0 / (2.0 * sigma))
    # C: (int E dV)^2 / (V int E^2 dV)
    ix = (2.0 * a / (m * np.pi)) if m % 2 else 0.0
    iy = (2.0 * b / (n * np.pi)) if n % 2 else 0.0
    num = (ix * iy * length) ** 2
    den = (a * b * length) * (a * b / 4.0 * length)
    # Q: U = (eps0/2)(ab/4)L ; side walls + endcaps
    U = 0.5 * EPS0 * (a * b / 4.0) * length
    # |H|^2 = |grad E|^2/(omega mu0)^2 ; on x=0,a : (kx cos)^2 sin^2 -> integrate
    g = 1.0 / (omega * MU0) ** 2
    P_x = 2 * 0.5 * R_s * g * kx ** 2 * (b / 2.0) * length     # walls x = 0, a
    P_y = 2 * 0.5 * R_s * g * ky ** 2 * (a / 2.0) * length     # walls y = 0, b
    P_z = 2 * 0.5 * R_s * g * (k ** 2) * (a * b / 4.0)         # endcaps
    Q = omega * U / (P_x + P_y + P_z)
    return {"f": f, "C": float(num / den), "Q": float(Q), "k": k, "R_s": R_s}


# ─────────────────────────────────────────────────────────────────────────────
# geometry + mesh
# ─────────────────────────────────────────────────────────────────────────────

def _surface_samples(tag, n=4):
    """(n*n, 3) points spread over a surface, for shape-agnostic BC
    classification -- the 3D analogue of the 2D curve sampling."""
    lo, hi = gmsh.model.getParametrizationBounds(2, tag)
    us = np.linspace(float(lo[0]), float(hi[0]), n)
    vs = np.linspace(float(lo[1]), float(hi[1]), n)
    U, V = np.meshgrid(us, vs)
    par = np.column_stack([U.ravel(), V.ravel()]).ravel()
    return np.asarray(gmsh.model.getValue(2, tag, par),
                      dtype=np.float64).reshape(-1, 3)


# A uniform Delaunay tetrahedral mesh puts roughly this many tets in a cube of
# side mesh_size. Calibrated on the pillbox: 0.785 m^3 at h = 83 mm gave 7891
# tets, i.e. 5.8 per h^3. Used only to catch a runaway BEFORE gmsh starts.
_TETS_PER_H3 = 6.0
MAX_ELEMENTS = 1_500_000


def estimate_elements(volume, mesh_size):
    """Rough tetrahedron count for a volume at a given element size."""
    h = float(mesh_size)
    return float(_TETS_PER_H3 * float(volume) / (h ** 3)) if h > 0 else np.inf


class _Progress:
    """
    Stage timing for a 3D solve.

    3D runs are long and silent, which makes it impossible to tell a slow solve
    from a hung one -- and the two have completely different fixes (wait, versus
    coarsen the mesh). Printing the stage boundaries with elapsed time turns that
    into a glance.
    """

    def __init__(self, on=True, stream=None):
        self.on = bool(on)
        self.t0 = time.perf_counter()
        self.tlast = self.t0
        self.stream = stream

    def __call__(self, msg):
        if not self.on:
            return
        now = time.perf_counter()
        print(f"[3d {now - self.t0:7.1f}s +{now - self.tlast:6.1f}s] {msg}",
              flush=True, file=self.stream)
        self.tlast = now

    def done(self, msg="finished"):
        if self.on:
            print(f"[3d {time.perf_counter() - self.t0:7.1f}s] {msg}",
                  flush=True, file=self.stream)


def build_mesh_3d(spec, msh_path: str, verbose: bool = False,
                  max_elements: int = MAX_ELEMENTS):
    """
    Build the 3D geometry with OCC booleans and write a .msh.

    Works for any spec providing add_outer(occ) and on_wall(pts): CavitySpec3D,
    CylSpec3D and ImportedSpec3D all do.

    Physical groups:
      volumes  : "background", plus "diel_0", "diel_1", ... per dielectric
      surfaces : "wall"  -> outer boundary
                 "metal" -> boundaries of the cut-out boxes

    Returns the list of dielectric materials, ordered to match "diel_i".
    """
    d = os.path.dirname(os.path.abspath(msh_path))
    if d:
        os.makedirs(d, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        if verbose:
            # gmsh reports "Meshing 3D... (n%)" only at this verbosity; without it
            # a long mesh is indistinguishable from a hang
            gmsh.option.setNumber("General.Verbosity", 5)
        gmsh.model.add("cavity3d")
        occ = gmsh.model.occ

        outer = spec.add_outer(occ)
        occ.synchronize()
        dom = [(3, outer)]

        # ---- setminus: cut the metal boxes out of the cavity -----------------
        metal_boxes = list(getattr(spec, "metal", []))
        if metal_boxes:
            tools = []
            for r in metal_boxes:
                t = occ.addBox(*r.as_tuple())
                ang = float(getattr(r, "angle", 0.0))
                if ang:
                    occ.rotate([(3, t)], r.cx, r.cy, r.cz, 0.0, 0.0, 1.0,
                               np.radians(ang))
                tools.append((3, t))
            dom, _ = occ.cut(dom, tools, removeObject=True, removeTool=True)
            if not dom:
                raise ValueError("cut() removed the entire domain: the metal "
                                 "boxes fill the whole cavity.")

        # ---- dielectrics as conformal sub-regions ---------------------------
        diel_mats, diel_tags = [], []
        if getattr(spec, "dielectric", None):
            tools = []
            for r, mat in spec.dielectric:
                t = occ.addBox(*r.as_tuple())
                ang = float(getattr(r, "angle", 0.0))
                if ang:
                    occ.rotate([(3, t)], r.cx, r.cy, r.cz, 0.0, 0.0, 1.0,
                               np.radians(ang))
                tools.append((3, t))
                diel_mats.append(mat)
            frag, _ = occ.fragment(dom, tools)
            occ.synchronize()
            # classify by BOUNDING-BOX CONTAINMENT, not centre of mass: after
            # fragmenting, the background is a non-convex shell whose centroid can
            # land inside a dielectric and be misclassified.
            dom, diel_tags = [], [[] for _ in spec.dielectric]
            for (dd, t) in frag:
                if dd != 3:
                    continue
                bb = gmsh.model.getBoundingBox(3, t)
                placed = False
                for i, (r, _m) in enumerate(spec.dielectric):
                    rb = r.bounds
                    tol = _BBOX_TOL + 1e-6 * max(r.w, r.h, r.d)
                    if (bb[0] >= rb[0] - tol and bb[1] >= rb[1] - tol
                            and bb[2] >= rb[2] - tol and bb[3] <= rb[3] + tol
                            and bb[4] <= rb[4] + tol and bb[5] <= rb[5] + tol):
                        diel_tags[i].append(t); placed = True; break
                if not placed:
                    dom.append((3, t))
        occ.synchronize()

        # ---- physical groups -------------------------------------------------
        bg_tags = [t for (dd, t) in dom if dd == 3]
        if not bg_tags:
            raise ValueError("no background volume left after the boolean ops.")
        g = gmsh.model.addPhysicalGroup(3, bg_tags)
        gmsh.model.setPhysicalName(3, g, "background")
        for i, tags in enumerate(diel_tags):
            if tags:
                g = gmsh.model.addPhysicalGroup(3, tags)
                gmsh.model.setPhysicalName(3, g, f"diel_{i}")

        all_vol = bg_tags + [t for tags in diel_tags for t in tags]
        bnd = gmsh.model.getBoundary([(3, t) for t in all_vol],
                                     combined=True, oriented=False)
        wall, metal = [], []
        for (dd, t) in bnd:
            if dd != 2:
                continue
            pts = _surface_samples(t)
            (wall if spec.on_wall(pts) else metal).append(t)
        # For an imported outer skin on_wall is a catch-all, so anything lying on
        # a cut-out box has to be pulled back out of `wall` explicitly.
        if metal_boxes and getattr(spec, "on_wall", None) is not None:
            keep_wall, moved = [], []
            for t in wall:
                pts = _surface_samples(t)
                if any(_on_box(pts, r) for r in metal_boxes):
                    moved.append(t)
                else:
                    keep_wall.append(t)
            wall, metal = keep_wall, metal + moved
        if wall:
            g = gmsh.model.addPhysicalGroup(2, wall)
            gmsh.model.setPhysicalName(2, g, "wall")
        if metal:
            g = gmsh.model.addPhysicalGroup(2, metal)
            gmsh.model.setPhysicalName(2, g, "metal")

        # ---- mesh ------------------------------------------------------------
        # `scale` means the CAD is in other units (mm, usually) while mesh_size is
        # in METRES, so the size targets have to be expressed in file units.
        fs = float(getattr(spec, "scale", 1.0)) or 1.0
        if spec.mesh_uniform:
            gmsh.option.setNumber("Mesh.MeshSizeMax", spec.mesh_size / fs)
            gmsh.option.setNumber("Mesh.MeshSizeMin", spec.mesh_size / fs)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        else:
            hmin = spec.mesh_size_min if spec.mesh_size_min else spec.mesh_size / 3.0
            gmsh.option.setNumber("Mesh.MeshSizeMax", spec.mesh_size / fs)
            gmsh.option.setNumber("Mesh.MeshSizeMin", hmin / fs)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

        # ---- runaway guard -------------------------------------------------
        # THE most common way to hang this solver is a mesh_size copied from an
        # example built for a different-sized part. Element count goes as
        # mesh_size^-3, so a part 10x larger in each direction needs 1000x the
        # elements at the same setting, and gmsh will sit there for hours (or the
        # OOM reaper takes it) with no indication of why. Estimate first, refuse
        # early, and say what size would actually work.
        bb = gmsh.model.getBoundingBox(-1, -1)
        vol_bbox = abs((bb[3] - bb[0]) * (bb[4] - bb[1]) * (bb[5] - bb[2]))
        h_file = spec.mesh_size / fs
        n_est = estimate_elements(vol_bbox, h_file)
        if verbose:
            print(f"[mesh3d] bbox {vol_bbox:.4g} (file units)^3, h = {h_file:.4g}"
                  f" -> ~{n_est:,.0f} tets (estimate)", flush=True)
        if max_elements and n_est > max_elements:
            h_ok = (_TETS_PER_H3 * vol_bbox / float(max_elements)) ** (1.0 / 3.0)
            raise ValueError(
                f"mesh_size = {spec.mesh_size:g} would produce roughly "
                f"{n_est:,.0f} tetrahedra on this geometry (limit "
                f"{max_elements:,}). That is the classic silent hang: element "
                f"count scales as mesh_size^-3, so a value carried over from a "
                f"smaller part explodes here.\n"
                f"  bounding box (file units): "
                f"{tuple(round(float(x), 4) for x in bb)}\n"
                f"  try mesh_size >= {h_ok * fs:.4g} (in metres), or raise "
                f"max_elements if you really mean it.\n"
                f"  BUT FIRST CHECK THE UNITS: if that bounding box reads in the "
                f"hundreds or thousands\n"
                f"  it is a MILLIMETRE file being read as metres. Use "
                f"scale='auto' (the default) or scale=1e-3;\n"
                f"  a suggested mesh_size of {h_ok * fs:.4g} m is itself the "
                f"giveaway that the scale is wrong.\n"
                f"  rule of thumb: start at about 1/6 of the smallest cavity "
                f"dimension and refine with converge().")

        # HXT (10) is much the fastest, but it FAILS OUTRIGHT on volumes recovered
        # by sewing an imported surface bag ("HXT 3D mesh failed"). Frontal (4)
        # copes with those, so fall back rather than making the caller guess.
        # ORDER MATTERS. HXT is fastest and is right for box/cylinder specs, but
        # on a volume sewn from imported surfaces it fails AND leaves the model
        # dirty, so a retry with another algorithm then dies with a PLC error
        # instead of succeeding. For imported geometry, go straight to Frontal.
        algos = ((4, 1, 10) if isinstance(spec, ImportedSpec3D) else (10, 4, 1))
        last = None
        for algo in algos:
            try:
                gmsh.option.setNumber("Mesh.Algorithm3D", algo)
                gmsh.model.mesh.generate(3)
                last = None
                break
            except Exception as e:                       # noqa: BLE001
                last = e
                gmsh.model.mesh.clear()
        if last is not None:
            raise RuntimeError(
                f"every 3D meshing algorithm failed on this geometry; the last "
                f"error was: {last}. For an imported shell this usually means the "
                f"surfaces do not close -- check the void volume printed by "
                f"ImportedSpec3D(report=True) against what you expect.")
        if fs != 1.0:
            # scale the MESH, not the geometry (see ImportedSpec3D.add_outer)
            gmsh.model.mesh.affineTransform([fs, 0, 0, 0,
                                             0, fs, 0, 0,
                                             0, 0, fs, 0])
        gmsh.write(msh_path)
    finally:
        gmsh.finalize()
    return diel_mats


def diagnose_import(spec, verbose=True):
    """
    What actually arrived from a CAD file, WITHOUT meshing it. Run this before a
    long solve, or after one fails.

    Reports the surface count, the inner/outer classification, the recovered
    volume and its bounding box, and -- the number that matters -- the SIGN of
    the volume. A negative volume means the recovered shell is oriented inward,
    which is the state gmsh cannot fill with tetrahedra. It is the reliable
    early-warning signal for the IGES failure described in solve_cavity_3d.

    Returns a dict; nothing is cached, so it is safe to call repeatedly.
    """
    tmp = tmp_msh_path("diag")
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("diag")
        occ = gmsh.model.occ
        vol_tag = spec.add_outer(occ)
        occ.synchronize()
        mass = occ.getMass(3, vol_tag)
        bb = gmsh.model.getBoundingBox(3, vol_tag)
        n_surf = len(gmsh.model.getEntities(2))
        scale = float(getattr(spec, "scale", 1.0))
        out = {"volume_raw": float(mass),
               "volume_m3": float(abs(mass) * scale ** 3),
               "oriented_ok": bool(mass > 0),
               "bbox_m": tuple(float(x) * scale for x in bb),
               "n_surfaces": int(n_surf)}
    finally:
        gmsh.finalize()
        try:
            os.remove(tmp)
        except OSError:
            pass
    if verbose:
        print(f"[diagnose] surfaces         : {out['n_surfaces']}")
        print(f"[diagnose] cavity volume    : {out['volume_m3']:.6g} m^3")
        print(f"[diagnose] bounding box (m) : "
              f"{tuple(round(x, 5) for x in out['bbox_m'])}")
        if out["oriented_ok"]:
            print("[diagnose] orientation      : OK (positive volume) -- this "
                  "should mesh.")
        else:
            print("[diagnose] orientation      : INVERTED (negative volume).")
            print("[diagnose]   gmsh cannot fill an inward-oriented shell with "
                  "tetrahedra. It will")
            print("[diagnose]   emit PLC errors, write a surface-only mesh, and "
                  "not raise. Re-export")
            print("[diagnose]   the part as STEP rather than IGES; that fixes it "
                  "at the source.")
    return out


def _on_box(pts, box):
    """True if every sampled point lies on the surface of `box`."""
    b = box.bounds
    tol = _BBOX_TOL + 1e-6 * max(box.w, box.h, box.d)
    inside = np.all((pts[:, 0] >= b[0] - tol) & (pts[:, 0] <= b[3] + tol) &
                    (pts[:, 1] >= b[1] - tol) & (pts[:, 1] <= b[4] + tol) &
                    (pts[:, 2] >= b[2] - tol) & (pts[:, 2] <= b[5] + tol))
    if not inside:
        return False
    on_face = ((np.abs(pts[:, 0] - b[0]) < tol) | (np.abs(pts[:, 0] - b[3]) < tol) |
               (np.abs(pts[:, 1] - b[1]) < tol) | (np.abs(pts[:, 1] - b[4]) < tol) |
               (np.abs(pts[:, 2] - b[2]) < tol) | (np.abs(pts[:, 2] - b[5]) < tol))
    return bool(np.all(on_face))


# ─────────────────────────────────────────────────────────────────────────────
# FEM forms
# ─────────────────────────────────────────────────────────────────────────────

@BilinearForm
def _curlcurl(u, v, w):
    return dot(curl(u), curl(v))


@BilinearForm
def _vmass(u, v, w):
    return dot(u, v)


def _assemble(mesh, element, spec, diel_mats):
    """K u = k0^2 M u, region by region so each carries its own eps_r / mu_r."""
    basis = Basis(mesh, element)
    sub = mesh.subdomains or {}
    if "background" not in sub:
        raise ValueError("the mesh has no 'background' volume group; it was not "
                         "written by build_mesh_3d.")
    K = asm(_curlcurl, Basis(mesh, element, elements=sub["background"])) \
        * (1.0 / spec.background.mu_r)
    M = asm(_vmass, Basis(mesh, element, elements=sub["background"])) \
        * spec.background.eps_r
    for i, mat in enumerate(diel_mats):
        key = f"diel_{i}"
        if key not in sub:
            continue
        b = Basis(mesh, element, elements=sub[key])
        K = K + asm(_curlcurl, b) * (1.0 / mat.mu_r)
        M = M + asm(_vmass, b) * mat.eps_r
    return basis, K, M


def discrete_gradient(mesh):
    """
    The discrete gradient G for lowest-order Whitney edge elements: the signed
    edge-node incidence matrix, -1 at the lower-numbered vertex of each edge and
    +1 at the higher.

    grad of a P1 function is exactly representable in N0 -- for phi = sum phi_k
    lambda_k, grad phi = sum_edges (phi_j - phi_i) w_ij -- so range(G) IS the
    kernel of the curl-curl matrix, exactly and not approximately. skfem numbers
    N0 dofs one per mesh edge in the order of mesh.edges, with edges stored as
    ascending vertex pairs, which is precisely this convention.
    """
    ne = mesh.edges.shape[1]
    nn = mesh.p.shape[1]
    rows = np.repeat(np.arange(ne), 2)
    cols = mesh.edges.T.ravel()
    data = np.tile([-1.0, 1.0], ne)
    return sp.csr_matrix((data, (rows, cols)), shape=(ne, nn))


def check_gradient_kernel(mesh, K, tol=1e-10):
    """||K G|| / ||K||, which must be ~machine epsilon. Cheap insurance against a
    future skfem release changing the edge ordering or sign convention under us:
    if this ever grows, the deflation is silently wrong and the eigenvalues come
    out polluted rather than obviously broken."""
    G = discrete_gradient(mesh)
    rel = sp.linalg.norm(K @ G) / max(sp.linalg.norm(K), 1e-300)
    if rel > tol:
        raise RuntimeError(
            f"the discrete gradient no longer spans the curl-curl kernel "
            f"(||KG||/||K|| = {rel:.3e}). skfem's N0 edge ordering or sign "
            f"convention has changed; fix discrete_gradient() before trusting "
            f"any eigenvalue.")
    return float(rel)


# ─────────────────────────────────────────────────────────────────────────────
# observables
# ─────────────────────────────────────────────────────────────────────────────

# Which Cartesian component the form factor projects onto. C measures coupling to
# an axion field along the SOLENOID axis, so this must be the CAVITY AXIS -- not
# "z" by habit. CAD parts are frequently modelled with the axis along x or y (the
# eMachineShop STEP export in testing used y), and getting it wrong does not fail
# loudly: C simply comes out ~0 for the operating mode, which looks exactly like a
# badly-aimed f_target.
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}


@Functional
def _int_Eaxis(w):
    return w["uh"][w["iax"]]


@Functional
def _int_eps_E2(w):
    return w["eps"] * dot(w["uh"], w["uh"])


@Functional
def _int_E2(w):
    return dot(w["uh"], w["uh"])


@Functional
def _int_E4(w):
    return dot(w["uh"], w["uh"]) ** 2


@Functional
def _volume(w):
    return 1.0 + 0.0 * w["uh"][0]


@Functional
def _int_curlE2(w):
    return dot(w["cu"], w["cu"])


def _observables(mesh, element, basis, u, k0, spec, diel_mats, axis="z"):
    """Form factor C, quality factor Q, volume, and a localisation diagnostic.

    axis : the CAVITY AXIS, "x" / "y" / "z". C projects onto this component."""
    iax = AXIS_INDEX[str(axis).lower()]
    num = den = vol = l2 = l4 = 0.0
    sub = mesh.subdomains or {}
    regions = [("background", spec.background)] + \
              [(f"diel_{i}", m) for i, m in enumerate(diel_mats)]
    for key, mat in regions:
        if key not in sub:
            continue
        b = Basis(mesh, element, elements=sub[key])
        uh = b.interpolate(u)
        num += _int_Eaxis.assemble(b, uh=uh, iax=iax)
        den += _int_eps_E2.assemble(b, uh=uh, eps=mat.eps_r)
        vol += _volume.assemble(b, uh=uh)
        l2 += _int_E2.assemble(b, uh=uh)
        l4 += _int_E4.assemble(b, uh=uh)

    C = (num ** 2) / (vol * den) if vol > 0 and den > 0 else 0.0

    # participation volume: (int|E|^2)^2 / int|E|^4 -- equals the volume for a
    # uniform field, collapses for a localised one
    V_part = (l2 ** 2) / l4 if l4 > 0 else 0.0

    # Q from wall currents. At a PEC surface H has no normal component, so
    # |H_t| = |H| = |curl E|/(omega mu0 mu_r) there and no projection is needed.
    omega = C0 * k0
    P = 0.0
    bnd = mesh.boundaries or {}
    for key, mat in (("wall", spec.wall_material), ("metal", spec.metal_material)):
        if key not in bnd:
            continue
        fb = FacetBasis(mesh, element, facets=bnd[key])
        uh = fb.interpolate(u)
        g2 = _int_curlE2.assemble(fb, cu=curl(uh))
        R_s = np.sqrt(omega * MU0 / (2.0 * mat.sigma))
        P += 0.5 * R_s * g2 / (omega * MU0) ** 2
    U = 0.5 * EPS0 * den
    Q = (omega * U / P) if P > 0 else np.inf

    return dict(C=float(C), Q=float(Q), volume=float(vol), V_part=float(V_part),
                localisation=float(V_part / vol) if vol else 0.0,
                int_eps_E2=float(den), U_stored=float(U),
                int_Eaxis=float(num), axis=str(axis).lower())


# ─────────────────────────────────────────────────────────────────────────────
# single solve
# ─────────────────────────────────────────────────────────────────────────────

def solve_cavity_3d(spec, n_modes: int = 6, f_target: float | None = None,
                    msh_path: str | None = None, verbose: bool = False,
                    keep_fields: bool = False, deflate: bool = True,
                    element=None, tol: float = 0.0, ncv=None,
                    drop_below: float = 1e-3, check_kernel: bool = True,
                    axis: str = "z", max_elements: int = MAX_ELEMENTS,
                    progress: bool = False, progress_every: int = 25,
                    permc_spec: str = "COLAMD"):
    """
    Build -> mesh -> solve one 3D configuration.

    f_target : Hz. STRONGLY RECOMMENDED. With no target the solve targets the
        lowest modes, which for a multi-cell cavity are not the operating mode.

        A BAD TARGET IS EASY TO SPOT: every returned mode has C ~ 0. In a prism
        only the p = 0 family has nonzero form factor, so a spectrum of zeros
        means the shift landed among the channel or p >= 1 modes rather than the
        operating one. Plot a y- or x-slice: if the field vanishes at mid-length
        and peaks either side, that is p = 1 and the target is too low. Solving
        the cross-section in 2D first and using its frequency is the reliable way
        to aim.
        It also sets the deflation shift, and the deflated operator is much
        better conditioned near a sensible target.

    deflate : project the gradient kernel out of the shift-invert operator (see
        the module docstring). Leave it on. With it off, ARPACK has to chew
        through a null cluster of dimension (number of interior nodes) and the
        solve goes from tenths of a second to minutes, most of the returned modes
        being numerical noise at f ~ 0.

    axis : the CAVITY AXIS for the form factor, "x" / "y" / "z" (default "z").
        MUST match the geometry. The built-in CylSpec3D and from_2d() put the axis
        along z, so the default is right for those. An IMPORTED part is whatever
        the CAD used -- check the bounding box printed by ImportedSpec3D: the long
        direction is usually the axis. Getting this wrong gives C ~ 0 for the
        operating mode with no other symptom.

    permc_spec : fill-reducing ordering for the shift-invert factorisation, which
        is where a 3D solve actually spends its time (measured: 35 s of a 44 s run
        at 34k tets, producing 68M nonzeros of fill from a 37k-dof matrix).
        "COLAMD" (default) is the fastest to compute. "MMD_AT_PLUS_A" produces
        ~2.4x LESS FILL but takes ~6x longer to order, so it is a MEMORY switch,
        not a speed one -- reach for it when a factorisation will not fit, not
        when it is merely slow.

    progress : print stage timings (mesh / assemble / factorise / eigensolve) and
        an ARPACK operator-application counter, plus gmsh's own meshing percentage.
        Costs nothing and is the only way to tell a slow solve from a hung one.

    drop_below : discard returned modes with f < drop_below * f_target as kernel
        residue. Only ever fires when deflate=False.

    element : defaults to ElementTetN0, the lowest-order Whitney edge element.
        NOTE skfem exposes only this one for tetrahedra -- ElementTetN1 is an
        ALIAS for it, not a higher-order element (both give one dof per mesh
        edge), so passing it changes nothing. Accuracy is improved by refining
        the mesh, or by extrapolating with converge().

    Returns a dict with 'freqs' (Hz) and per-mode C / Q / V_part, sorted by
    frequency, plus n_dofs / n_elements / kernel_dim.
    """
    pg = _Progress(progress)
    pg(f"meshing {spec.tag or '(untagged)'} at mesh_size={spec.mesh_size:g} m")
    tmp = msh_path or tmp_msh_path("cavity3d")
    with _quiet(QUIET and not (verbose or progress)):
        diel_mats = build_mesh_3d(spec, tmp, verbose=(verbose or progress),
                                  max_elements=max_elements)
        mesh = skfem.Mesh.load(tmp)
    pg(f"mesh: {mesh.t.shape[1]:,} tets, {mesh.p.shape[1]:,} nodes")
    if msh_path is None:
        try:
            os.remove(tmp)
        except OSError:
            pass
    if not isinstance(mesh, skfem.MeshTet):
        raise RuntimeError(
            f"the mesher produced no tetrahedra -- skfem loaded a "
            f"{type(mesh).__name__} (surface elements only), so there is no "
            f"volume to solve.\n"
            f"\n"
            f"For an IMPORTED geometry this almost always means the CAD file is "
            f"IGES. IGES is a SURFACE format: the part arrives as a bag of "
            f"trimmed surfaces with no solid, the void has to be recovered by "
            f"sewing them, and the sewn shell comes out with inverted "
            f"orientation. gmsh then emits 'PLC Error: a segment and a facet "
            f"intersect', writes a surface-only .msh, and does NOT raise -- which "
            f"is why the failure surfaces here rather than at meshing time.\n"
            f"\n"
            f"THE FIX IS UPSTREAM: re-export the same part as STEP (.step/.stp) "
            f"from your CAD and point `path` at that instead. STEP carries a real "
            f"B-rep solid, so orientation is well defined and the identical "
            f"interior=True path meshes and solves normally. Verified on the same "
            f"part: IGES fails here, STEP gives 23k tets in ~13 s.\n"
            f"\n"
            f"If STEP is genuinely unavailable, run "
            f"fem_solve_3d.diagnose_import(spec) to see what did come in.")

    element = ElementTetN0() if element is None else element
    basis, K, M = _assemble(mesh, element, spec, diel_mats)
    pg(f"assembled: {basis.N:,} edge dofs")
    if check_kernel:
        check_gradient_kernel(mesh, K)

    # PEC: tangential E = 0 on every conducting surface
    bnd = mesh.boundaries or {}
    keys = [k for k in ("wall", "metal") if k in bnd]
    if not keys:
        raise ValueError("the mesh has no 'wall' or 'metal' boundary group.")
    facets = np.concatenate([bnd[k] for k in keys])
    D = basis.get_dofs(facets=facets)
    I = basis.complement_dofs(D)
    Kc = K[I][:, I].tocsc()
    Mc = M[I][:, I].tocsc()

    sigma = (2 * np.pi * f_target / C0) ** 2 if f_target else 0.0
    if sigma <= 0:
        # no target: a small positive shift, still above the kernel
        sigma = 1e-6 * float(sp.linalg.norm(Kc)) / max(float(sp.linalg.norm(Mc)), 1e-300)

    kernel_dim = 0
    OPinv = None
    if deflate:
        bnodes = np.unique(mesh.facets[:, facets])
        inodes = np.setdiff1d(np.arange(mesh.p.shape[1]), bnodes)
        Gc = discrete_gradient(mesh)[I][:, inodes].tocsc()
        kernel_dim = Gc.shape[1]
        S = (Gc.T @ Mc @ Gc).tocsc()
        Slu = splu(S)
        pg(f"gradient projector factorised ({kernel_dim:,} kernel modes deflated)")
        A = (Kc - sigma * Mc).tocsc()
        Alu = splu(A, permc_spec=permc_spec)
        pg(f"shift-invert factorised: fill-in {Alu.L.nnz + Alu.U.nnz:,} nnz")

        def _P(x):
            return x - Gc @ Slu.solve(Gc.T @ (Mc @ x))

        # P on BOTH sides: that is what keeps the operator M-self-adjoint, which
        # ARPACK's symmetric shift-invert mode assumes. One-sided is not.
        # ARPACK gives no progress of its own; counting operator applications is
        # the only honest signal of whether it is converging or stalling
        _count = {"n": 0, "t": time.perf_counter()}

        def _op(x):
            _count["n"] += 1
            if progress and progress_every and _count["n"] % progress_every == 0:
                dt = time.perf_counter() - _count["t"]
                pg(f"eigensolve: {_count['n']} operator applications "
                   f"({_count['n'] / max(dt, 1e-9):.1f}/s)")
            return _P(Alu.solve(_P(x)))

        OPinv = LinearOperator(Kc.shape, dtype=np.float64, matvec=_op)

    pg(f"eigensolve: {n_modes} modes near {(f_target or 0)/1e6:.3f} MHz "
       f"({len(I):,} free dofs)")
    vals, vecs = eigsh(Kc, k=n_modes, M=Mc, sigma=sigma, which="LM",
                       OPinv=OPinv, tol=tol, ncv=ncv)
    pg("eigensolve done; computing C and Q")

    order = np.argsort(np.real(vals))
    out = {"freqs": [], "modes": [], "n_dofs": int(basis.N),
           "n_elements": int(mesh.t.shape[1]), "n_free": int(len(I)),
           "kernel_dim": int(kernel_dim), "tag": spec.tag,
           "deflated": bool(deflate)}
    f_floor = drop_below * (f_target or 0.0)
    for idx in order:
        lam = float(np.real(vals[idx]))
        if lam <= 0:
            continue
        k0 = np.sqrt(lam)
        f = C0 * k0 / (2 * np.pi)
        if f < f_floor:
            continue                      # kernel residue, only when not deflated
        u = np.zeros(basis.N)
        u[I] = np.real(vecs[:, idx])
        # normalise to peak |E| = 1 over the quadrature points, mirroring the 2D
        # convention (the eigenproblem is homogeneous, so scale is a choice)
        uh = basis.interpolate(u)
        peak = float(np.sqrt(np.max(sum(np.asarray(uh[c]) ** 2 for c in range(3)))))
        if peak > 0:
            u = u / peak
        obs = _observables(mesh, element, basis, u, k0, spec, diel_mats, axis=axis)
        out["freqs"].append(f)
        out["modes"].append(obs)
        if keep_fields:
            out.setdefault("fields", []).append(u.copy())
    if keep_fields:
        out["mesh"] = mesh
        out["element"] = element
    pg.done(f"{len(out['freqs'])} modes")
    return out


def converge(spec, f_target, mesh_sizes, n_modes=4, min_localisation=0.0,
             verbose=True, **kw):
    """
    Solve at several mesh sizes and Richardson-extrapolate f, C and Q to h -> 0.

    Only h-refinement is available (skfem has no higher-order Nedelec
    tetrahedron), so this is the accuracy knob. Lowest-order Whitney elements
    converge from BELOW and roughly as h^2 in the eigenvalue, so two or three
    meshes give a usable estimate of where the answer is actually going, plus an
    error bar that is honest rather than assumed.

    Measured behaviour on the analytic cases: frequency converges fastest, then
    the form factor, then Q. Q is the slowest because it is a SURFACE integral of
    a field DERIVATIVE, so it inherits both the geometric faceting error and one
    extra derivative of the discretisation error. On a flat-walled rectangular
    cavity Q lands within ~0.6%; on a curved pillbox it is still 5% out at
    h = R/7, almost entirely from faceting the barrel. Sharp metal edges are the
    other slow case: a thin bar with a field singularity at its corner gave ~5%
    in Q where the same mesh gave 0.01% in frequency.

    Returns {"h": [...], "f": [...], "C": [...], "Q": [...],
             "f_extrap": ..., "C_extrap": ..., "Q_extrap": ..., "order": ...}
    where *_extrap comes from the two finest meshes.
    """
    hs, fs, Cs, Qs = [], [], [], []
    for h in sorted(mesh_sizes, reverse=True):
        import copy
        sp = copy.copy(spec)
        sp.mesh_size = float(h)
        r = solve_cavity_3d(sp, n_modes=n_modes, f_target=f_target, **kw)
        m = best_mode(r, min_localisation=min_localisation)
        if m is None:
            continue
        hs.append(float(h)); fs.append(m["f"]); Cs.append(m["C"]); Qs.append(m["Q"])
        if verbose:
            print(f"[converge] h={h:.4g} tets={r['n_elements']:>7d} "
                  f"dof={r['n_free']:>7d} | f={m['f']/1e9:.5f} GHz "
                  f"C={m['C']:.5f} Q={m['Q']:.4g}", flush=True)
    out = {"h": hs, "f": fs, "C": Cs, "Q": Qs, "order": np.nan,
           "f_extrap": np.nan, "C_extrap": np.nan, "Q_extrap": np.nan}
    if len(hs) >= 3:
        # observed order from the three finest points, if they are monotone
        f0, f1, f2 = fs[-3], fs[-2], fs[-1]
        r01, r12 = f1 - f0, f2 - f1
        if r01 != 0 and r12 / r01 > 0 and r12 / r01 < 1:
            out["order"] = float(np.log(r12 / r01) /
                                 np.log(hs[-2] / hs[-3] * hs[-1] / hs[-2]))
    if len(hs) >= 2:
        pwr = 2.0
        ratio = (hs[-2] / hs[-1]) ** pwr
        for key, arr in (("f", fs), ("C", Cs), ("Q", Qs)):
            out[f"{key}_extrap"] = float(arr[-1] + (arr[-1] - arr[-2]) /
                                         (ratio - 1.0))
        if verbose:
            print(f"[converge] Richardson (assuming h^2): "
                  f"f={out['f_extrap']/1e9:.5f} GHz  C={out['C_extrap']:.5f}  "
                  f"Q={out['Q_extrap']:.4g}", flush=True)
    return out


def best_mode(result, min_localisation: float = 0.0):
    """
    Pick the operating mode: highest C among modes that are not localised.

    In 3D the localisation number is V_part/volume rather than A_part/area, and a
    delocalised fundamental sits lower than its 2D counterpart simply because the
    field now also varies along z for p >= 1.
    """
    cand = [(f, m) for f, m in zip(result["freqs"], result["modes"])
            if m["localisation"] >= min_localisation]
    if not cand:
        return None
    f, m = max(cand, key=lambda fm: fm[1]["C"])
    return {"f": f, **m}


# ─────────────────────────────────────────────────────────────────────────────
# field sampling (for the visualiser, and for anything that needs nodal values)
# ─────────────────────────────────────────────────────────────────────────────

def nodal_field(result, i=0, min_localisation=None):
    """
    (n_nodes, 3) Cartesian E at the mesh VERTICES, by L2 projection of the edge
    element field onto P1.

    Edge-element dofs are circulations along edges, not point values, so there is
    nothing to plot directly: every visualisation needs this projection first.
    Projecting the three components separately (rather than |E|) keeps the vector,
    so glyphs and slices of any component both work afterwards.

    Requires solve_cavity_3d(..., keep_fields=True).
    """
    if "fields" not in result:
        raise ValueError("no fields in result: call solve_cavity_3d(..., "
                         "keep_fields=True)")
    if min_localisation is not None:
        cand = [j for j, m in enumerate(result["modes"])
                if m["localisation"] >= min_localisation] or \
               list(range(len(result["modes"])))
        i = max(cand, key=lambda j: result["modes"][j]["C"])
    mesh = result["mesh"]
    element = result.get("element") or ElementTetN0()
    nb = Basis(mesh, element, intorder=2)
    pb = Basis(mesh, ElementTetP1(), intorder=2)
    uh = nb.interpolate(result["fields"][i])
    Mp = asm(BilinearForm(lambda u, v, w: u * v), pb)
    Mlu = splu(Mp.tocsc())
    comps = []
    for c in range(3):
        @LinearForm
        def rhs(v, w, _c=c):
            return w["Ec"] * v
        b = asm(rhs, pb, Ec=uh[c])
        comps.append(Mlu.solve(b))
    return np.column_stack(comps)


# ─────────────────────────────────────────────────────────────────────────────
# batch (process-parallel)
# ─────────────────────────────────────────────────────────────────────────────

def _worker(args):
    spec, n_modes, f_target, keep_fields, deflate = args
    try:
        out = solve_cavity_3d(spec, n_modes=n_modes, f_target=f_target,
                              keep_fields=keep_fields, deflate=deflate)
        out.pop("element", None)          # not needed downstream, keeps pickles small
        return {"ok": True, **out}
    except Exception as e:
        return {"ok": False, "tag": spec.tag, "error": f"{type(e).__name__}: {e}"}


def run_batch(specs, n_modes: int = 6, f_target=None, n_workers: int | None = None,
              timeout: float | None = None, verbose: bool = True,
              keep_fields: bool = False, deflate: bool = True):
    """
    Solve many 3D configurations in parallel, one process per configuration.

    f_target : scalar (same shift for every spec) or one per spec.

    MEMORY, not cores, is usually the binding constraint in 3D: each worker holds
    its own mesh, sparse factorisation and fill-in. A 3D factorisation can be
    tens of times the size of the matrix, so n_workers = cpu_count will thrash on
    anything but a coarse mesh. Start at 2-4 and watch the resident size.

    timeout : seconds. NOTE this is a budget for the WHOLE batch, not per solve
    (it goes to as_completed), and overrunning it raises rather than marking the
    stragglers as failures -- same semantics as the 2D run_batch.
    """
    n_workers = n_workers or max(1, (os.cpu_count() or 2) // 2)
    if f_target is None or np.isscalar(f_target):
        targets = [f_target] * len(specs)
    else:
        targets = list(f_target)
        if len(targets) != len(specs):
            raise ValueError(f"f_target has {len(targets)} entries but there are "
                             f"{len(specs)} specs.")
    if all(t is None for t in targets) and verbose:
        print("[batch3d] WARNING: no f_target -- returning the LOWEST modes of "
              "each geometry, which for a multi-cell cavity are not the "
              "operating mode.", flush=True)
    payload = [(s, n_modes, t, keep_fields, deflate) for s, t in zip(specs, targets)]
    results = [None] * len(specs)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_worker, p): i for i, p in enumerate(payload)}
        done = 0
        for fut in as_completed(futs, timeout=timeout):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                results[i] = {"ok": False, "tag": specs[i].tag,
                              "error": f"{type(e).__name__}: {e}"}
            done += 1
            if verbose:
                r = results[i]
                msg = (f"f0={r['freqs'][0]/1e9:.4f} GHz" if r.get("ok") and r["freqs"]
                       else r.get("error", "no modes"))
                print(f"[batch3d] {done}/{len(specs)} tag={r.get('tag','')!r} {msg}",
                      flush=True)
    return results


def run_sweep(spec_fn, positions, n_modes: int = 6, n_workers: int | None = None,
              timeout: float | None = None, keep_fields: bool = False,
              verbose: bool = True):
    """
    Parallel tuning sweep, 3D.

    spec_fn(dx, dy, i) -> a 3D spec for tuning position i.
    positions          -> iterable of (dx, dy, f_guess).

    Returns (specs, results) in position order.
    """
    pos = list(positions)
    specs = [spec_fn(dx, dy, i) for i, (dx, dy, _f) in enumerate(pos)]
    guesses = [f for _dx, _dy, f in pos]
    results = run_batch(specs, n_modes=n_modes, f_target=guesses,
                        n_workers=n_workers, timeout=timeout,
                        keep_fields=keep_fields, verbose=verbose)
    return specs, results