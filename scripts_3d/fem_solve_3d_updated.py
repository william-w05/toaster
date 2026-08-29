"""
3D vector eigenmode solver for microwave cavities -- the 3D counterpart of
fem_solve.py.  NGSOLVE EDITION.

  gmsh (OpenCASCADE)  -> geometry: boxes/cylinders, boolean cut, STEP/IGES import
  NGSolve             -> FEM assembly with HCurl (Nedelec) elements, ORDER 2
  scipy               -> shift-invert eigensolve with EXPLICIT KERNEL DEFLATION
                         (or NGSolve's own ArnoldiSolver, eigensolver="arnoldi")

WHAT CHANGED FROM THE scikit-fem VERSION, AND WHAT DID NOT
    UNCHANGED, deliberately: every geometry class (Box, CavitySpec3D, CylSpec3D,
    ImportedSpec3D), from_2d, the extrusion shortcuts, the analytic references,
    build_mesh_3d and the entire gmsh pipeline including the CAD unit handling
    and the runaway guard. build_mesh_3d still writes a .msh carrying the same
    "background"/"diel_i"/"wall"/"metal" physical groups, so
    fem_vis_3d.plot_mesh_3d and check_spec_3d keep working untouched, and
    solve_cavity_3d still returns the same dict keys, so plot_field_slices,
    plot_modes_3d and view_field_3d do too.

    REPLACED: the FEM layer. skfem offered exactly ONE Nedelec tetrahedron --
    ElementTetN0, with ElementTetN1 an alias for it -- so h-refinement was the
    only accuracy knob. NGSolve's HCurl is hierarchical to arbitrary order and
    the default here is order=2. Measured on the analytic pillbox (R = 50 mm,
    L = 200 mm, TM010, curved mesh, h = 14 mm):

        f  +0.008%      C  +0.037%      Q  -3.3%

    at 33,840 dofs. Note the SIGN of the frequency error: order 2 converges from
    ABOVE here, opposite to what the N0 version did, so do not read old
    convergence notes as still applying.

    In NGSolve's numbering HCurl(order=0) IS the lowest-order Whitney/Nedelec
    element -- the exact skfem ElementTetN0 equivalent, one dof per edge -- so
    order=0 reproduces the old discretisation if you ever need to compare.
    Cost grows steeply: measured ~11 dofs per tetrahedron at order 2 against
    ~1.2 at order 0, i.e. the SAME MESH is about 9x the dofs and far more than
    9x the factorisation. Coarsen the mesh when you raise the order; that is the
    whole point of raising it.

WHY THIS IS A VECTOR PROBLEM (and 2D was not)
    In 2D with no z-dependence the TM modes reduce to a scalar Helmholtz problem
    for E_z. In 3D there is no such reduction: the modes are genuine vector
    fields and the eigenproblem is the curl-curl system

        curl( (1/mu_r) curl E ) = k0^2 eps_r E ,     n x E = 0 on PEC

    Discretised with HCurl (edge/Nedelec) elements, which enforce tangential
    continuity and normal discontinuity -- exactly the physics of E across a
    material interface. NODAL (Lagrange) elements applied to this system produce
    a spectrum riddled with spurious non-physical modes; HCurl does not.

CURVED ELEMENTS -- THE BIGGEST SINGLE ACCURACY WIN HERE
    The old module's own convergence notes said Q was "still 5% out at h = R/7,
    almost entirely from faceting the barrel" of a pillbox. That error is
    GEOMETRIC, not FEM: a straight-sided tetrahedral mesh replaces the cylinder
    by an inscribed prism. Raising the element order does nothing for it.

    So solve_cavity_3d asks gmsh for a SECOND-ORDER mesh (mesh_order=2, the
    default) and NGSolve treats it as curved. Measured on the same pillbox mesh,
    1098 tets, identical dof count, only the geometry order changed:

        volume error   -1.98%  ->  -0.003%
        surface area   -0.90%  ->  -0.002%
        frequency      +0.70%  ->  +0.031%
        form factor    -1.31%  ->  +0.187%

    Set mesh_order=1 to go back to straight tetrahedra (do that if gmsh reports
    invalid curved elements, which can happen when a coarse mesh meets strong
    curvature). For all-planar geometry the two are identical, so the default
    costs nothing there.

THE KERNEL, AND WHY NAIVE SHIFT-INVERT CRAWLS
    curl(grad phi) = 0, so the discrete curl-curl matrix K has a null space of
    dimension equal to the number of interior H1 dofs -- thousands to millions of
    eigenvectors all at lambda = 0. Shift-invert at sigma maps every one of them
    to -1/sigma, a single massively degenerate cluster, and the eigensolver has to
    deal with that cluster before it can report anything else. With the previous
    scikit-fem/N0 implementation the cost was extreme: on a 3157-dof pillbox,
    asking for 6 modes took 176 s and 5 of the 6 returned were null-space junk.

    The fix here is exact rather than heuristic, and NGSolve makes it cheaper
    than it was. The kernel is precisely the range of the DISCRETE GRADIENT, and
    fes.CreateGradient() hands it over directly: NGSolve's HCurl basis is built
    so that the gradients of the H1 hierarchical basis ARE basis functions of the
    HCurl space, so the gradient matrix is a sparse +-1 selection matrix, not a
    dense projection. Two facts make deflation clean:

      * K G = 0 identically -- verified numerically at 1.4e-16 relative for
        HCurl order 2, and asserted at runtime by check_gradient_kernel().
      * every PHYSICAL mode is already M-orthogonal to the kernel. If K u =
        lambda M u with lambda != 0 then (G phi)^T K u = 0 because K G = 0, hence
        lambda (G phi)^T M u = 0, hence (G phi)^T M u = 0.

    So the M-orthogonal projector P = I - G (G^T M G)^-1 G^T M annihilates the
    kernel and acts as the identity on every mode we want. Feeding ARPACK
    OPinv = P (K - sigma M)^-1 P therefore returns the physical spectrum
    unchanged while the kernel is mapped to 0 and never selected. The sandwich
    (P on both sides, not one) is what keeps the operator M-self-adjoint, which
    ARPACK's symmetric mode requires.

    Which H1 dofs to deflate is decided WITHOUT asking the space about boundary
    conditions: a gradient column is a boundary column exactly when it has a
    nonzero entry in a Dirichlet HCurl row (an interior vertex function has zero
    tangential trace on every boundary edge). That is a pure linear-algebra test,
    so it cannot drift with an NGSolve release.

LOSS / Q
    Same perturbative treatment as 2D, one dimension up:

        Q = omega U / P_loss
        U      = (eps0/2) integral( eps_r |E|^2 dV )
        P_loss = (R_s/2) surface_integral( |H_t|^2 dS )
        |H|    = |curl E| / (omega mu0 mu_r)
        R_s    = sqrt(omega mu0 / (2 sigma))

    At a PEC wall H has no normal component, so |H_t| = |H| there and no
    projection is needed. That is not just the continuous argument: because the
    tangential trace of E is zero on the whole conductor, the normal component of
    curl E vanishes in the DISCRETE space too -- measured at exactly 0.0 relative
    on the pillbox, so subtracting (n.curl E)^2 changes Q in the 16th digit.
    Different boundary groups may carry different metals.

    THE TRAP, and it is a silent one: NGSolve cannot evaluate curl(gfu) on a
    boundary element, and rather than raising it INTEGRATES TO ZERO -- which
    comes out as Q = inf, not as an error. Every surface integral here therefore
    goes through BoundaryFromVolumeCF(). Do not "simplify" that away.

    Q CONVERGES MUCH MORE SLOWLY THAN f. Measured on the pillbox with order 2 as
    h went 20 mm -> 14 mm: the frequency error fell as h^4 (+0.031% -> +0.008%)
    but Q only as h^2 (-6.9% -> -3.3%). Q is a SURFACE integral of a field
    DERIVATIVE, so it carries one fewer derivative of accuracy than the
    eigenvalue does. converge() extrapolates the two with their own powers
    because of this; do not judge a mesh by its frequency error alone.

FORM FACTOR
    C = |integral E_z dV|^2 / ( V * integral eps_r |E|^2 dV )

    the direct 3D analogue of the 2D expression, with area -> volume. NOTE it
    still singles out E_z: it is the coupling to an axion field along the
    solenoid axis, not a norm. Use axis= to point it elsewhere.

PARALLELISM
    As in 2D: many small independent solves, one geometry per process. gmsh is
    not thread-safe but is fine in separate processes. In 3D MEMORY binds long
    before cores do -- see run_batch.
"""

from __future__ import annotations

import os
import io
import sys
import re as _re
import uuid
import gc
import tempfile
import contextlib
import time
import numpy as np
import scipy.sparse as sp
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed

import gmsh
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

# NGSolve is imported LAZILY, by _ng(). Geometry, meshing, the extrusion
# shortcuts and the analytic references all work without it, and fem_vis_3d
# imports this module purely to reach them -- so a missing NGSolve must not stop
# `import fem_solve_3d`.
NG_ORDER = 2                    # default HCurl order
NG_MESH_ORDER = 2               # default geometric order (2 = curved elements)
NG_THREADS = 0                  # 0 -> NGSolve's own default (all cores)


@contextlib.contextmanager
def task_manager(threads=None):
    """
    NGSolve's shared-memory parallelism, around the phases that are element
    loops: ASSEMBLY and the OBSERVABLES.

    Without this NGSolve runs those single-threaded, and on a 32-core box that is
    the difference between a coffee and a glance. It matters most for the
    observables: each mode costs a handful of Integrate() passes over every
    element, so 50 modes on a 155k-tet mesh is minutes of pure element looping
    that parallelises almost perfectly.

    It is deliberately NOT wrapped around the sparse factorisation. PARDISO
    spawns its own MKL threads, and having an idle NGSolve pool alongside is
    harmless but nesting the two thread pools around the same work is not.

    threads : None/0 -> NGSolve's default (all cores), or an explicit count.
        Set it to 1 in run_batch workers, where the parallelism should come from
        running several geometries at once instead.
    """
    ngs = _ng()
    n = NG_THREADS if threads is None else threads
    if n:
        try:
            ngs.SetNumThreads(int(n))
        except Exception:                                     # pragma: no cover
            pass
    with ngs.TaskManager():
        yield

# MSH 2.2, NOT the gmsh default of 4.1, and this is not a preference:
# netgen's ReadGmsh cannot parse 4.1 at all (it dies on the $Entities block),
# while meshio/skfem read 2.2 happily -- and 2.2 avoids the spurious
# "gmsh:bounding_entities" subdomain that 4.1 hands to skfem. One format keeps
# the solver and the visualiser reading identical files.
MSH_FILE_VERSION = 2.2


def _ng():
    """The ngsolve module, with an actionable message if it is missing."""
    try:
        import ngsolve
    except ImportError as e:                                  # pragma: no cover
        raise ImportError(
            "NGSolve is required for the 3D FEM solve:\n"
            "    pip install ngsolve\n"
            "  (or: conda install -c ngsolve ngsolve)\n"
            "Everything in this module that does NOT solve -- the geometry "
            "classes, build_mesh_3d, extruded_modes/extruded_Q, "
            "pillbox_analytic, rect_cavity_analytic -- works without it.") from e
    return ngsolve


@contextlib.contextmanager
def _quiet(enabled=None):
    """Swallow stdout only; stderr is left alone.

    NOTE netgen's gmsh reader prints its "Physical groups detected" notice from
    C++, which a Python-level stdout redirect does not always catch. Harmless.
    """
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
        2D: halving it multiplies the tetrahedron count by ~8 and the dof count
        with it. With HCurl order 2 each tetrahedron carries ~11 dofs rather than
        the ~1.2 of the old lowest-order element, so start COARSER than the 2D
        habit suggests and refine with converge().

    extrude_layers : if set, mesh the CROSS-SECTION in 2D and extrude it through
        `n` structured layers instead of filling the volume with an isotropic
        Delaunay mesh. Only valid when every metal and dielectric box spans the
        full z extent -- i.e. when the cavity is a PRISM -- and it is checked.

        THIS IS THE RIGHT MESH FOR A MULTI-CELL PRISM, for two reasons:

          * an isotropic mesh ties the transverse resolution to the longitudinal
            one, so most of its elements go into resolving z, where the operating
            mode is CONSTANT. At the same transverse h and an element aspect
            ratio A = (L/layers)/h, the extruded mesh costs about 1.16/A times
            the isotropic count -- so roughly 3x fewer elements at A = 4, or
            equivalently a noticeably finer cross-section for the same budget.
            Since the transverse resolution is what keeps neighbouring cells
            tuned to each other, that is the axis worth spending on.
          * the extruded mesh is EXACTLY invariant under z -> -z and under
            translation in z (mesh points land on exact z-planes -- verified).
            Longitudinal parity is therefore an exact symmetry of the DISCRETE
            problem, so the p = 0 operating mode cannot hybridise with the p = 1
            neighbour sitting (pi/L / k_t)^2 / 2 above it. On an unstructured
            mesh that symmetry is broken at the level of the mesh's own
            asymmetry.

        HOW MANY LAYERS. Few. The p = 0 mode is constant along z and order-2
        elements represent that exactly: measured on a 3-cell prism, layers=1 and
        layers=2 agreed on the form factor to 1% (0.09716 vs 0.09824) even at an
        8.6:1 element aspect ratio, and the extruded answer agreed with an
        isotropic mesh of the same element count to 3%. So 1-2 layers is enough
        for f, C and Q of the operating mode.

        BUT: with very few layers the p >= 1 ladder is badly placed, and since
        shift-invert selects on frequency alone, a target that lands among those
        misplaced neighbours returns them instead of the mode you want. That cost
        me a wrong conclusion during testing. Use 4+ layers when you are still
        hunting for the target, then drop back once it is known.
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
    refine_edges: bool = True
    refine_dist: float | None = None
    extrude_layers: int | None = None
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
    Leave mesh_order=2 on for it -- the barrel is curved and that is where the
    faceting error lives.
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

    AXIS. An imported part is oriented however the CAD drew it, and the form
    factor is a projection onto ONE axis -- pass axis="x"/"y"/"z" to
    solve_cavity_3d to match. The long direction of the bounding box printed
    below is usually it.
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
            partial=None, extrude_layers=None):
    """
    Extrude a 2D fem_solve.CavitySpec into a CavitySpec3D: every Rect becomes a
    Box spanning the full cavity length, so the cross-section is exactly the one
    the 2D solver used.

    THIS IS THE BRIDGE from the existing toaster code:

        spec2d = viz.toaster_spec(params_m, gap0=..., gap1=..., cavity_h=...)
        spec3d = fem_solve_3d.from_2d(spec2d, length=0.20)

    extrude_layers : mesh the cross-section and extrude it through n structured
        layers, instead of filling the volume isotropically. For a multi-cell
        cavity this is what makes the operating mode findable at an affordable
        cost -- see CavitySpec3D.extrude_layers. Incompatible with `partial`,
        which is checked.

    partial : optional {name: (z0, dz)} in METRES to make a bar span only part of
        the length -- a tuning rod that does not reach the endcaps, say. Anything
        not named spans the full length. The moment ANY bar is partial the
        geometry stops being a prism and extruded_modes() no longer applies; only
        the full 3D solve is valid then.
    """
    if partial and extrude_layers:
        raise ValueError(
            "extrude_layers and partial are mutually exclusive: a partial-length "
            "bar is exactly the case where the cross-section depends on z, so "
            "there is nothing to extrude. Use the isotropic mesh there.")
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
        extrude_layers=extrude_layers,
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
    which is where edge elements earn their keep -- and because every wall is
    planar there is no geometric error at all, so this one isolates the FEM.
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
MAX_ELEMENTS = 10_000_000

# HCurl dofs per tetrahedron, asymptotically. Measured on two pillbox meshes at
# order 2: 12,387 dofs on 1,098 tets and 33,840 on 3,143 -- ~11 either way. The
# asymptotic counts follow from n_edge ~ 1.17 n_tet and n_face ~ 2 n_tet with
# (order+1) dofs per edge and, at order 2, 3 per face.
_DOFS_PER_TET = {0: 1.2, 1: 2.4, 2: 9.5, 3: 22.0}

# Refuse outright above this many dofs, and merely warn above WARN_DOFS. These
# are about the DIRECT FACTORISATION, which is what a shift-invert solve costs:
# fill-in grows far faster than the matrix, and 3 GB of RAM died on a 78k-dof
# curl-curl LU in testing while a 34k-dof one took 23 s on one core. Raise them
# if you have the memory and PARDISO; they exist to convert an overnight hang
# into a message.
WARN_DOFS = 300_000
MAX_DOFS = 2_000_000


def estimate_elements(volume, mesh_size):
    """Rough tetrahedron count for a volume at a given element size."""
    h = float(mesh_size)
    return float(_TETS_PER_H3 * float(volume) / (h ** 3)) if h > 0 else np.inf


def estimate_dofs(n_elements, order=NG_ORDER):
    """Rough HCurl dof count for a tetrahedron count. See _DOFS_PER_TET."""
    per = _DOFS_PER_TET.get(int(order), 9.5 * (int(order) / 2.0) ** 3)
    return float(per * float(n_elements))


# ─────────────────────────────────────────────────────────────────────────────
# linear solver: SuperLU (always there) or MKL PARDISO (much faster)
# ─────────────────────────────────────────────────────────────────────────────

LINEAR_SOLVER = "auto"          # "auto" | "pardiso" | "superlu"
MESH_THREADS = 0                # 0 -> all cores; set to 1 inside run_batch workers
_PARDISO_WARNED = [False]


def openmp_conflict_hint():
    """
    Explain an OpenMP/MKL clash if one is likely, else "".

    On Windows, PyTorch ships its own libiomp5md.dll in torch/lib and the `mkl`
    wheel behind pypardiso ships another in Library/bin. Whichever loads first
    wins; the second prints "OMP: Error #15 ... already initialized" and PARDISO
    then dies with an access violation inside MKL. It is not a missing package,
    and reinstalling pypardiso will not help. NGSolve links its own threaded BLAS
    as well, so the same clash can appear even without pypardiso.
    """
    if "torch" not in sys.modules:
        return ""
    return (
        "\n  LIKELY CAUSE: torch is imported in this process. PyTorch and MKL each "
        "bundle their own\n"
        "  OpenMP runtime (libiomp5md.dll) and they cannot coexist -- that is the "
        "'OMP: Error #15'\n"
        "  above, and the access violation after it.\n"
        "  FIX: keep torch out of the process that runs the 3D solver. Nothing in "
        "fem_solve_3d or\n"
        "  fem_vis_3d needs it; it arrives via mcmc.py, which imports torch at "
        "module level for the\n"
        "  surrogate. Either drop the `import mcmc` from your 3D script, or make "
        "that import lazy\n"
        "  (move `import torch` inside mcmc.Surrogate).\n"
        "  KMP_DUPLICATE_LIB_OK=TRUE silences the message but Intel warns it can "
        "silently produce\n"
        "  WRONG ANSWERS -- do not use it for a solver you intend to trust.")


def pardiso_available(explain=False):
    """
    True if MKL PARDISO can actually be used -- importable AND able to factorise.

    A bare `import pypardiso` succeeding proves nothing: the OpenMP clash below
    only shows up when MKL is first asked to do work. So this runs a 3x3 solve.
    """
    try:
        import pypardiso
        A = sp.eye(3, format="csr") * 2.0
        x = pypardiso.spsolve(A, np.ones(3))
        ok = bool(np.allclose(x, 0.5))
        if explain and not ok:
            print("[3d] pypardiso imported but returned a wrong answer",
                  flush=True)
        return ok
    except Exception as e:
        if explain:
            print(f"[3d] pypardiso unusable: {type(e).__name__}: {e}"
                  f"{openmp_conflict_hint()}", flush=True)
        return False


def _is_symmetric(A, tol=1e-10):
    """Cheap structural+numeric symmetry test, O(nnz)."""
    n = sp.linalg.norm(A)
    if n == 0:
        return True
    return bool(sp.linalg.norm((A - A.T).tocsr()) <= tol * n)


def _factorized(A, kind=None, permc_spec="COLAMD", spd=False, progress=None):
    """
    Factorise A once and return a callable solve(b) -> x, plus a label.

    WHY THIS EXISTS. A 3D shift-invert solve is dominated by ONE sparse LU.
    Measured on a 36,909-dof curl-curl matrix, SuperLU took 30.6 s and produced
    68 million nonzeros of fill. MKL PARDISO factorises the same matrix in 1.4 s,
    and in 0.75 s if told the matrix is symmetric -- a 27x speedup on a SINGLE
    core, before any threading. (Those figures are from the scikit-fem/N0 era but
    they are properties of the linear algebra, not of the assembler: a 33,840-dof
    NGSolve order-2 matrix took SuperLU 23 s here.) That is the difference
    between iterating on a design and not.

    spd : the matrix is symmetric POSITIVE DEFINITE (the gradient projector
        G^T M G is; the shift-invert matrix K - sigma*M is not, it is symmetric
        INDEFINITE). PARDISO wants different matrix types for the two.

    THE TRAP with PARDISO's symmetric modes: it expects ONLY THE UPPER TRIANGLE.
    Hand it the full matrix and you get a silently wrong answer, not an error. So
    symmetry is verified numerically first and the triangle is extracted here;
    if the matrix turns out not to be symmetric we fall back to the unsymmetric
    mode rather than guessing.

    Falls back to SuperLU on any PARDISO failure, warning once, so a machine
    without MKL still runs.
    """
    kind = (LINEAR_SOLVER if kind is None else kind).lower()
    if kind in ("auto", "pardiso"):
        try:
            import pypardiso
            sym = _is_symmetric(A)
            mtype = (2 if spd else -2) if sym else 11
            ps = pypardiso.PyPardisoSolver(mtype=mtype)
            Af = (sp.triu(A, format="csr") if mtype in (2, -2) else A.tocsr())
            Af.indptr = Af.indptr.astype(np.int32, copy=False)
            Af.indices = Af.indices.astype(np.int32, copy=False)
            Af.data = np.ascontiguousarray(Af.data, dtype=np.float64)
            ps.factorize(Af)

            def solve(b, _ps=ps, _A=Af):
                return _ps.solve(_A, np.ascontiguousarray(b, dtype=np.float64))

            solve.free = ps.free_memory
            return solve, f"pardiso(mtype={mtype})"
        except Exception as e:                                # pragma: no cover
            if kind == "pardiso":
                raise
            if not _PARDISO_WARNED[0]:
                _PARDISO_WARNED[0] = True
                hint = openmp_conflict_hint()
                if not hint and "pypardiso" not in sys.modules:
                    hint = "\n  pip install pypardiso"
                print(f"[3d] MKL PARDISO unavailable ({type(e).__name__}: {e}); "
                      f"falling back to SuperLU, which is ~25x slower on the "
                      f"shift-invert factorisation.{hint}", flush=True)
    lu = splu(A, permc_spec=permc_spec)
    solve = lambda b: lu.solve(b)                              # noqa: E731
    solve.free = lambda: None
    solve._lu = lu
    return solve, f"superlu({permc_spec})"


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
                  max_elements: int = MAX_ELEMENTS, mesh_order: int = 1,
                  msh_version: float = MSH_FILE_VERSION):
    """
    Build the 3D geometry with OCC booleans and write a .msh.

    Works for any spec providing add_outer(occ) and on_wall(pts): CavitySpec3D,
    CylSpec3D and ImportedSpec3D all do.

    Physical groups:
      volumes  : "background", plus "diel_0", "diel_1", ... per dielectric
      surfaces : "wall"  -> outer boundary
                 "metal" -> boundaries of the cut-out boxes

    mesh_order : 1 for straight tetrahedra, 2 for a second-order (curved) mesh.
        Only the solver benefits -- NGSolve reads the curvature and integrates on
        the true surface -- so this defaults to 1 here and solve_cavity_3d asks
        for 2 explicitly. fem_vis_3d.plot_mesh_3d draws surfaces only, so it is
        unaffected either way.

    msh_version : the .msh format. 2.2, not gmsh's 4.1 default, because netgen's
        reader cannot parse 4.1; meshio/skfem read both, so 2.2 is the one format
        that serves the solver and the visualiser at once.

    Returns the list of dielectric materials, ordered to match "diel_i".
    """
    d = os.path.dirname(os.path.abspath(msh_path))
    if d:
        os.makedirs(d, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        # gmsh is single-threaded unless told otherwise, which is a large part of
        # why CPU sits near one core during meshing. HXT in particular scales.
        gmsh.option.setNumber("General.NumThreads", int(MESH_THREADS or
                                                        (os.cpu_count() or 1)))
        gmsh.option.setNumber("Mesh.MaxNumThreads3D", int(MESH_THREADS or
                                                          (os.cpu_count() or 1)))
        if verbose:
            # gmsh reports "Meshing 3D... (n%)" only at this verbosity; without it
            # a long mesh is indistinguishable from a hang
            gmsh.option.setNumber("General.Verbosity", 5)
        gmsh.model.add("cavity3d")
        occ = gmsh.model.occ

        layers = int(getattr(spec, "extrude_layers", 0) or 0)
        metal_boxes = list(getattr(spec, "metal", []))
        diel_mats = [mat for _r, mat in getattr(spec, "dielectric", [])]
        diel_tags = [[] for _ in getattr(spec, "dielectric", [])]

        if layers:
            # ---- prism path: mesh the cross-section, extrude it -------------
            frag = _build_extruded_domain(spec, occ, layers)
            dom = frag
        else:
            outer = spec.add_outer(occ)
            occ.synchronize()
            dom = [(3, outer)]

            # ---- setminus: cut the metal boxes out of the cavity ------------
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

            # ---- dielectrics as conformal sub-regions ----------------------
            frag = dom
            if getattr(spec, "dielectric", None):
                tools = []
                for r, mat in spec.dielectric:
                    t = occ.addBox(*r.as_tuple())
                    ang = float(getattr(r, "angle", 0.0))
                    if ang:
                        occ.rotate([(3, t)], r.cx, r.cy, r.cz, 0.0, 0.0, 1.0,
                                   np.radians(ang))
                    tools.append((3, t))
                frag, _ = occ.fragment(dom, tools)
                occ.synchronize()

        # ---- sort the volumes into background and dielectric regions -------
        # classify by BOUNDING-BOX CONTAINMENT, not centre of mass: after
        # fragmenting, the background is a non-convex shell whose centroid can
        # land inside a dielectric and be misclassified. Shared by both paths.
        if getattr(spec, "dielectric", None):
            dom = []
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
        mesh_info = None
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

        # ---- graded refinement at the metal edges ---------------------------
        # WHY EDGES AND NOT VOLUME. The operating mode is broad and smooth in the
        # open gaps between bars -- that is where the stored energy is, and it
        # needs almost no resolution. What needs resolution is the RE-ENTRANT
        # EDGES of the bars, where the field has an r^(-1/3) singularity and
        # where a uniform mesh spends its error budget badly.
        #
        # That error is not just inaccuracy here, it is symmetry breaking: a
        # 5-cell toaster holds its transverse modes within ~0.2% of one another,
        # so a mesh whose per-cell error exceeds 0.2% detunes the cells against
        # each other and the in-phase operating mode decouples into localised
        # ones (form factor collapses to ~0). Grading buys accuracy exactly where
        # the asymmetry is generated, at a fraction of the element count.
        edge_field = None
        if (getattr(spec, "refine_edges", False) and metal and not
                spec.mesh_uniform):
            mcurves = set()
            for t in metal:
                for (dd, c) in gmsh.model.getBoundary([(2, t)], combined=False,
                                                      oriented=False):
                    if dd == 1:
                        mcurves.add(abs(c))
            if mcurves:
                h_max = spec.mesh_size / fs
                h_min = ((spec.mesh_size_min or spec.mesh_size / 4.0) / fs)
                d_far = ((spec.refine_dist or 3.0 * spec.mesh_size) / fs)
                fd = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(fd, "CurvesList",
                                                 sorted(mcurves))
                gmsh.model.mesh.field.setNumber(fd, "Sampling", 200)
                ft = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(ft, "InField", fd)
                gmsh.model.mesh.field.setNumber(ft, "SizeMin", h_min)
                gmsh.model.mesh.field.setNumber(ft, "SizeMax", h_max)
                gmsh.model.mesh.field.setNumber(ft, "DistMin", h_min)
                gmsh.model.mesh.field.setNumber(ft, "DistMax", d_far)
                gmsh.model.mesh.field.setAsBackgroundMesh(ft)
                # the field must be the ONLY source of element size, or gmsh's
                # boundary-driven sizing overrides it and the grading vanishes
                gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
                gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
                edge_field = ft
                mesh_info = {"hmin": h_min * fs, "hmax": h_max * fs}
                if verbose:
                    print(f"[mesh3d] edge grading on {len(mcurves)} metal curves:"
                          f" {h_min*fs*1e3:.2f} mm at the edges -> "
                          f"{h_max*fs*1e3:.2f} mm at {d_far*fs*1e3:.1f} mm away",
                          flush=True)

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
        if mesh_info is not None:
            # A GRADED mesh is nowhere near mesh_size on average. Calibrated
            # against a real run (hmin 0.1 mm, hmax 5 mm -> 4.9M tets where the
            # coarse-size estimate said 150k, a 33x miss that let the guard wave
            # through a mesh which then took 214 s and 7.5M dofs):
            #     h_eff = hmin^(1/3) * hmax^(2/3)
            # predicts 7.5M there, which is the right order and errs high.
            # (This branch was dead in the scikit-fem version -- mesh_info was
            # never assigned -- and a flat x4 fudge stood in for it. The
            # calibrated estimator is used now, and the fudge is gone.)
            h_file = ((mesh_info["hmin"] / fs) ** (1.0 / 3.0) *
                      (mesh_info["hmax"] / fs) ** (2.0 / 3.0))
            ratio = mesh_info["hmax"] / max(mesh_info["hmin"], 1e-30)
            if ratio > 20:
                print(f"[mesh3d] WARNING: mesh_size/mesh_size_min = {ratio:.0f}. "
                      f"A refinement ratio above ~20 rarely buys accuracy and "
                      f"explodes the element count; {mesh_info['hmin']*1e3:.3f} mm "
                      f"features need a reason to exist.", flush=True)
        n_est = estimate_elements(vol_bbox, h_file)
        if layers:
            # an extruded mesh scales as h^-2, not h^-3: the z direction is fixed
            # by `layers`, so the isotropic estimate is meaningless here.
            area = cross_section_area(spec) / (fs ** 2)
            n_est = estimate_elements_extruded(area, h_file, layers)
        if verbose:
            print(f"[mesh3d] bbox {vol_bbox:.4g} (file units)^3, h = {h_file:.4g}"
                  f" -> ~{n_est:,.0f} tets (estimate), "
                  f"~{estimate_dofs(n_est, NG_ORDER):,.0f} dofs at order "
                  f"{NG_ORDER}", flush=True)
        if max_elements and n_est > max_elements:
            if layers:
                h_ok = (3.0 * layers * (cross_section_area(spec) / fs ** 2) /
                        (0.36 * float(max_elements))) ** 0.5
            else:
                h_ok = (_TETS_PER_H3 * vol_bbox /
                        float(max_elements)) ** (1.0 / 3.0)
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
                f"dimension and refine with converge(). At HCurl order 2 that "
                f"rule is if anything too fine -- each tetrahedron carries ~11 "
                f"dofs, not ~1.2.")

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

        # ---- curved elements -------------------------------------------------
        # Second order puts the mid-edge nodes ON THE CAD SURFACE, so a cylinder
        # stops being an inscribed prism. Measured on a pillbox: volume error
        # -1.98% -> -0.003%, frequency +0.70% -> +0.031%, at the SAME dof count.
        # For all-planar geometry it is a no-op (the extra nodes are just
        # midpoints), so it is safe to leave on.
        if int(mesh_order) > 1:
            try:
                gmsh.model.mesh.setOrder(int(mesh_order))
            except Exception as e:                       # noqa: BLE001
                print(f"[mesh3d] WARNING: could not build an order-"
                      f"{int(mesh_order)} mesh ({type(e).__name__}: {e}); "
                      f"falling back to straight tetrahedra. On a curved wall "
                      f"expect ~1% errors in f and several % in Q from faceting.",
                      flush=True)
        if fs != 1.0:
            # scale the MESH, not the geometry (see ImportedSpec3D.add_outer)
            gmsh.model.mesh.affineTransform([fs, 0, 0, 0,
                                             0, fs, 0, 0,
                                             0, 0, fs, 0])
        if msh_version:
            gmsh.option.setNumber("Mesh.MshFileVersion", float(msh_version))
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


def _rect_in_plane(occ, box, z0):
    """The z-footprint of a Box as a 2D rectangle in the plane z = z0, rotation
    included."""
    t = occ.addRectangle(box.x0, box.y0, z0, box.w, box.h)
    ang = float(getattr(box, "angle", 0.0))
    if ang:
        occ.rotate([(2, t)], box.cx, box.cy, z0, 0.0, 0.0, 1.0, np.radians(ang))
    return t


def _build_extruded_domain(spec, occ, layers):
    """
    Mesh the cross-section and extrude it: the prism path (see
    CavitySpec3D.extrude_layers).

    Returns the list of (3, tag) volumes, which the caller then classifies into
    background and dielectric regions exactly as it does for the boolean path.

    The PRISM CONDITION is enforced rather than assumed. A bar that stops short
    of the endcaps makes the cross-section z-dependent, so extruding it would
    silently model a DIFFERENT CAVITY -- one whose bars run the full length --
    and the answer would look perfectly reasonable. That is the one failure mode
    this function must not have.
    """
    o = spec.outer
    z0, L = o.z0, o.d
    tol = 1e-9 + 1e-6 * L
    for label, boxes in (("metal", list(getattr(spec, "metal", []))),
                         ("dielectric", [r for r, _m in
                                         getattr(spec, "dielectric", [])])):
        for r in boxes:
            if abs(r.z0 - z0) > tol or abs(r.d - L) > tol:
                raise ValueError(
                    f"extrude_layers requires a PRISM, but the {label} box "
                    f"'{r.name}' spans z = [{r.z0:.6g}, {r.z0 + r.d:.6g}] while "
                    f"the cavity spans [{z0:.6g}, {z0 + L:.6g}].\n"
                    f"  A partial-length bar makes the cross-section depend on "
                    f"z, so there is nothing to extrude -- and extruding anyway "
                    f"would quietly solve a cavity whose bars run the full "
                    f"length.\n"
                    f"  Drop extrude_layers for this geometry: the isotropic "
                    f"mesh is the correct tool once the prism assumption is "
                    f"gone (and that is exactly when the 3D solve earns its "
                    f"keep).")

    face = _rect_in_plane(occ, o, z0)
    faces = [(2, face)]
    metal_boxes = list(getattr(spec, "metal", []))
    if metal_boxes:
        tools = [(2, _rect_in_plane(occ, r, z0)) for r in metal_boxes]
        faces, _ = occ.cut(faces, tools, removeObject=True, removeTool=True)
        if not faces:
            raise ValueError("the metal footprints cover the whole "
                             "cross-section; nothing left to extrude.")
    if getattr(spec, "dielectric", None):
        tools = [(2, _rect_in_plane(occ, r, z0)) for r, _m in spec.dielectric]
        faces, _ = occ.fragment(faces, tools)
    occ.synchronize()
    faces = [(d, t) for (d, t) in faces if d == 2]
    ext = occ.extrude(faces, 0.0, 0.0, L, numElements=[int(layers)],
                      recombine=False)
    occ.synchronize()
    vols = [(d, t) for (d, t) in ext if d == 3]
    if not vols:
        raise RuntimeError("the extrusion produced no volume; check that the "
                           "cross-section is a valid 2D face.")
    return vols


def cross_section_area(spec):
    """Cross-sectional area of a prism spec, metal footprints removed. Used by
    the extruded element estimate; the footprints are assumed not to overlap,
    which fem_vis_3d.check_spec_3d verifies."""
    o = spec.outer
    a = float(o.w * o.h)
    for r in getattr(spec, "metal", []):
        a -= float(r.w * r.h)
    return max(a, 0.0)


def estimate_elements_extruded(area, mesh_size, layers):
    """
    Tetrahedron count for an extruded prism mesh: (triangles) x layers x 3.

    Calibrated, not guessed: a 0.01416 m^2 cross-section at h = 7 mm meshed to
    801 triangles, i.e. 0.36 h^2 of area per triangle.
    """
    h = float(mesh_size)
    if h <= 0:
        return np.inf
    return float(3.0 * int(layers) * float(area) / (0.36 * h ** 2))


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
# NGSolve: mesh, space, assembly
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MeshView:
    """
    The solved mesh reduced to the two arrays anything downstream needs:

        p : (3, n_points) vertex coordinates, metres
        t : (4, n_tets)   corner-vertex indices into p

    Deliberately the SAME layout skfem used, because fem_vis_3d.to_pyvista reads
    result["mesh"].p and .t directly. It is plain numpy, so it survives pickling
    out of a run_batch worker, which an NGSolve mesh handle would not.

    With a curved (mesh_order=2) mesh, p also carries the mid-edge nodes. They
    belong to no tetrahedron in `t`, so they are invisible to VTK's cell
    interpolation; the field values stored at them are exactly zero.
    """
    p: np.ndarray
    t: np.ndarray

    @property
    def n_points(self):
        return int(self.p.shape[1])

    @property
    def n_tets(self):
        return int(self.t.shape[1])

    def __repr__(self):
        return f"MeshView({self.n_points} points, {self.n_tets} tets)"


def _bit_mask(bits, n=None):
    """NGSolve BitArray -> numpy bool array. There is no NumPy() on BitArray in
    the versions tested, hence the explicit loop."""
    if n is None:
        n = len(bits)
    try:
        return np.asarray(bits.NumPy(), dtype=bool)            # pragma: no cover
    except AttributeError:
        return np.fromiter((bool(bits[i]) for i in range(n)), dtype=bool, count=n)


def _to_scipy(mat):
    """NGSolve sparse matrix -> scipy CSR (real).

    Two traps, both silent:
      * assemble WITHOUT symmetric=True, or COO() hands back a triangle only and
        every norm and product downstream is quietly halved.
      * a matrix from a complex FESpace would be cast to float64 with nothing
        worse than a ComplexWarning, so it is refused outright here. The scipy
        eigensolve path must run on a REAL space; only eigensolver="arnoldi"
        builds a complex one, and it never converts.
    """
    rows, cols, vals = mat.COO()
    vals = np.asarray(vals)
    if np.iscomplexobj(vals):
        raise TypeError(
            "refusing to convert a COMPLEX NGSolve matrix to a real scipy "
            "matrix -- the imaginary part would be discarded silently. The "
            "scipy/ARPACK path needs a real HCurl space; complex spaces are "
            "only for eigensolver='arnoldi'.")
    A = sp.coo_matrix((vals.astype(np.float64),
                       (np.asarray(rows), np.asarray(cols))),
                      shape=(mat.height, mat.width))
    return A.tocsr()


def load_ng_mesh(msh_path):
    """Read a build_mesh_3d .msh into an NGSolve mesh, materials and boundary
    names included."""
    ngs = _ng()
    try:
        from netgen.read_gmsh import ReadGmsh
    except ImportError as e:                                  # pragma: no cover
        raise ImportError("netgen.read_gmsh is missing; reinstall netgen/ngsolve."
                          ) from e
    try:
        ngmesh = ReadGmsh(msh_path)
    except Exception as e:
        raise RuntimeError(
            f"netgen could not read {msh_path} ({type(e).__name__}: {e}).\n"
            f"  If you wrote that mesh yourself, it is almost certainly MSH 4.1: "
            f"netgen's reader handles 2.2 only and dies on 4.1's $Entities "
            f"block. build_mesh_3d writes 2.2 (MSH_FILE_VERSION) for exactly "
            f"this reason; re-export with gmsh -format msh22, or set "
            f"Mesh.MshFileVersion = 2.2.") from e
    return ngs.Mesh(ngmesh)


def mesh_arrays(mesh):
    """(p, t) as MeshView wants them. The element loop is Python-level, so this
    is called only when keep_fields=True asks for it."""
    ngs = _ng()
    p = np.asarray([v.point for v in mesh.vertices], dtype=np.float64).T
    t = np.asarray([[vv.nr for vv in el.vertices]
                    for el in mesh.Elements(ngs.VOL)], dtype=np.int64).T
    return MeshView(p=p, t=t)


def _material_cfs(mesh, spec, diel_mats, verbose=False):
    """
    (eps_r, 1/mu_r) as domain-wise CoefficientFunctions, plus a report string.

    The mapping is BY PHYSICAL NAME -- "background" and "diel_i", exactly the
    names build_mesh_3d writes -- not by region index, because index order is the
    reader's business and not ours. A region whose name we do not recognise falls
    back to the background material and says so; silently treating it as vacuum
    is how a dielectric ends up doing nothing.
    """
    ngs = _ng()
    lookup = {"background": spec.background}
    for i, mat in enumerate(diel_mats):
        lookup[f"diel_{i}"] = mat
    eps, muinv, unknown = [], [], []
    for name in mesh.GetMaterials():
        mat = lookup.get(name)
        if mat is None:
            unknown.append(name or "(unnamed)")
            mat = spec.background
        eps.append(float(mat.eps_r))
        muinv.append(1.0 / float(mat.mu_r))
    if unknown:
        print(f"[3d] WARNING: mesh regions {sorted(set(unknown))} carry no "
              f"physical name this module recognises; treating them as "
              f"'{spec.background.name}' (eps_r={spec.background.eps_r:g}). "
              f"Expected 'background' and 'diel_i'.", flush=True)
    report = ", ".join(f"{n}: eps_r={e:g}" for n, e in
                       zip(mesh.GetMaterials(), eps))
    return (ngs.CoefficientFunction(eps), ngs.CoefficientFunction(muinv),
            report)


def hcurl_space(mesh, order=NG_ORDER, nograds=False, complex=False):
    """
    The HCurl space with PEC (tangential-E = 0) on every conducting group.

    Returns (fes, dirichlet_pattern, boundary_names_present).

    complex : complex-valued dofs. Needed ONLY by eigensolver="arnoldi" -- see
        the note there; a real space makes NGSolve's ArnoldiSolver return
        eigenvectors that do not satisfy the boundary condition.

    nograds : drop the HIGH-ORDER gradient shape functions, which shrinks both
        the dof count and the curl-curl kernel. NGSolve offers it and it is a
        recognised trick for eigenproblems, but it changes the discrete space and
        I have not validated its effect on the non-zero spectrum for these
        cavities -- so it is off by default. Deflation (below) removes the
        kernel's effect on the SOLVER without touching the space at all, which is
        the safer half of the same idea.
    """
    ngs = _ng()
    present = [k for k in ("wall", "metal") if k in set(mesh.GetBoundaries())]
    if not present:
        raise ValueError(
            f"the mesh has no 'wall' or 'metal' boundary group -- it carries "
            f"{sorted(set(mesh.GetBoundaries()))}. Either it was not written by "
            f"build_mesh_3d, or the physical names did not survive the .msh "
            f"round trip.")
    pattern = "|".join(present)
    fes = ngs.HCurl(mesh, order=int(order), dirichlet=pattern,
                    nograds=bool(nograds), complex=bool(complex))
    return fes, pattern, present


def assemble_curlcurl(mesh, fes, spec, diel_mats):
    """
    The pair (K, M) of the generalised problem K u = k0^2 M u, as NGSolve
    BilinearForms:

        K = integral (1/mu_r) curl u . curl v dV
        M = integral eps_r u . v dV

    One form each with a domain-wise material coefficient, rather than the 2D
    module's region-by-region assembly: NGSolve evaluates the CoefficientFunction
    per element, so a dielectric insert costs nothing extra and cannot be left
    out of one of the two matrices by accident.
    """
    ngs = _ng()
    eps_cf, muinv_cf, report = _material_cfs(mesh, spec, diel_mats)
    u, v = fes.TnT()
    K = ngs.BilinearForm(muinv_cf * ngs.curl(u) * ngs.curl(v) * ngs.dx)
    M = ngs.BilinearForm(eps_cf * u * v * ngs.dx)
    K.Assemble()
    M.Assemble()
    return K, M, eps_cf, muinv_cf, report


# ─────────────────────────────────────────────────────────────────────────────
# the curl-curl kernel
# ─────────────────────────────────────────────────────────────────────────────

def discrete_gradient(fes):
    """
    The discrete gradient of the HCurl space, as scipy CSR of shape
    (fes.ndof, n_h1_dof), or None if this NGSolve build cannot supply it.

    NGSolve hands this over exactly, via fes.CreateGradient(): its HCurl basis is
    constructed so that gradients of the H1 hierarchical basis functions ARE
    HCurl basis functions, so the matrix comes out as a sparse +-1 selection
    matrix (1,159 nonzeros for a 1,515 x 776 example) rather than anything dense.
    range(G) IS the kernel of the curl-curl matrix, exactly and not
    approximately, at every order -- for order 2 the H1 space behind it is order
    3, which is why hand-rolling an edge-node incidence matrix (correct for the
    lowest-order element only) would deflate just part of the kernel.
    """
    try:
        gradmat, _fesh1 = fes.CreateGradient()
    except (AttributeError, RuntimeError) as e:               # pragma: no cover
        print(f"[3d] WARNING: fes.CreateGradient() unavailable "
              f"({type(e).__name__}: {e}); kernel deflation is off. Expect the "
              f"eigensolve to be far slower and to return modes near f = 0 that "
              f"drop_below then discards.", flush=True)
        return None
    return _to_scipy(gradmat)


def kernel_basis(G, free):
    """
    The columns of the discrete gradient that live entirely inside the FREE dofs,
    restricted to those rows -- i.e. a basis for the kernel of the CONSTRAINED
    curl-curl matrix.

    Which H1 dofs to keep is decided by linear algebra, not by asking the space
    about its boundary conditions: a gradient column is a boundary column exactly
    when it has a nonzero entry in a Dirichlet row, because an interior vertex
    function has zero tangential trace on every boundary edge. Dropping those
    also removes the constant function, whose gradient is zero, which is what
    makes G^T M G nonsingular.
    """
    Gr = G.tocsr()
    dirichlet = ~free
    if dirichlet.any():
        touched = np.asarray(abs(Gr[dirichlet]).sum(axis=0)).ravel() > 0
    else:
        touched = np.zeros(Gr.shape[1], dtype=bool)
    Gc = Gr[free].tocsc()[:, ~touched]
    keep = np.asarray(abs(Gc).sum(axis=0)).ravel() > 0
    return Gc[:, keep].tocsc()


def check_gradient_kernel(K, G, tol=1e-10, mode="probe", n_probe=4, seed=0):
    """
    ||K G|| / ||K||, which must be ~machine epsilon (measured 1.2e-16 for HCurl
    order 2 across three mesh sizes). Cheap insurance against an NGSolve release
    changing the HCurl basis or the meaning of CreateGradient under us: if this
    ever grows, the deflation is silently wrong and the eigenvalues come out
    polluted rather than obviously broken.

    mode :
      "probe" (default) apply K G to a few random vectors and take the worst
          ||K G r|| / (||K|| ||G r||). A handful of matvecs, so the cost is
          O(nnz) and independent of the kernel dimension.
      "full"  form the sparse-sparse product K G. Measured at 0.05 s on a
          15,625-tet problem and scaling roughly linearly, so it is affordable
          too -- but it allocates a whole extra matrix, which at a million dofs
          lands on top of the peak memory of the run.
      False / None  skip.

    A probe cannot prove the product is zero the way the full norm can, but it
    fails just as loudly on the failure that matters: a changed sign or edge
    ordering breaks the kernel property for a GENERIC vector, not for a rare one.
    """
    if not mode:
        return float("nan")
    kn = max(sp.linalg.norm(K), 1e-300)
    if str(mode).lower() == "full":
        rel = sp.linalg.norm(K @ G) / kn
    else:
        rng = np.random.default_rng(seed)
        R = rng.standard_normal((G.shape[1], int(n_probe)))
        GR = G @ R
        KGR = K @ GR
        rel = float(np.max(np.linalg.norm(KGR, axis=0) /
                           np.maximum(kn * np.linalg.norm(GR, axis=0), 1e-300)))
    if rel > tol:
        raise RuntimeError(
            f"the discrete gradient no longer spans the curl-curl kernel "
            f"(||KG||/||K|| = {rel:.3e}). NGSolve's HCurl basis or "
            f"CreateGradient() has changed; fix discrete_gradient()/"
            f"kernel_basis() before trusting any eigenvalue, or pass "
            f"deflate=False to bypass both.")
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


def observable_context(mesh, fes, spec, diel_mats, eps_cf=None, order=NG_ORDER,
                       localisation=True, quartic_order=None):
    """
    Everything the per-mode observables need, built ONCE.

    The scikit-fem version rebuilt a volume Basis and two FacetBasis objects
    inside the per-mode function, which at N quadrature points per element is
    hundreds of megabytes of temporaries -- so asking for 50 modes rebuilt them
    150 times and drove RAM into swap. Nothing here depends on the mode, so it is
    hoisted out. What is left per mode is a handful of Integrate() calls and one
    interpolation.

    QUADRATURE ORDER IS THE WHOLE COST STORY. Measured per mode on a 6,551-tet
    order-2 problem:

        int |E|^4  at order 10   0.75 s      <- 67% of the per-mode cost
        int eps|E|^2 at order 6  0.12 s
        int |E|^2    at order 6  0.12 s
        int E_axis   at order 6  0.07 s
        3x H1 Set (nodal)        0.06 s
        int |curl E|^2 over BND  0.01 s      (surface, so far fewer elements)

    |E|^4 is degree 8 in the field and only feeds V_part/localisation, which is a
    SHAPE DIAGNOSTIC and not part of f, C or Q. Order 8 reproduces the order-10
    value to 10 significant digits at 0.46 s, so that is the default here; order
    6 is 0.24 s but wrong in the 4th digit. localisation=False drops the term
    entirely and roughly triples the per-mode throughput.

    Returns a dict; treat it as opaque.
    """
    ngs = _ng()
    if eps_cf is None:
        eps_cf, _mu, _rep = _material_cfs(mesh, spec, diel_mats)
    surfaces = [(name, mesh.Boundaries(name), mat) for name, mat in
                (("wall", spec.wall_material), ("metal", spec.metal_material))
                if name in set(mesh.GetBoundaries())]
    # the projection space must match the SCALAR TYPE of the solve space, or
    # Set() dies with "BaseVector<Complex>::GetIndirect<double> called" on the
    # arnoldi path. The real part is taken when the values are read back.
    h1 = ngs.H1(mesh, order=1, complex=bool(getattr(fes, "is_complex", False)))
    return {
        "ngs": ngs, "mesh": mesh, "fes": fes, "eps": eps_cf,
        "surfaces": surfaces,
        "gfu": ngs.GridFunction(fes),
        "h1": h1,
        "gfp": [ngs.GridFunction(h1) for _ in range(3)],
        # quadrature: the integrands are |E|^2 (degree 2*order) and |E|^4
        "iord": 2 * int(order) + 2,
        "iord4": int(quartic_order) if quartic_order else 2 * int(order) + 4,
        "localisation": bool(localisation),
        "one": ngs.CoefficientFunction(1.0),
    }


def _real(x, what="integral", tol=1e-8):
    """
    float(x) for a real integral; the real part, checked, for a complex one.

    On the "arnoldi" path the space is complex, so every Integrate() comes back
    complex even though the mode stored in it has been rotated to be real. A
    residual imaginary part means the phase rotation did not take -- worth
    knowing rather than silently truncating.
    """
    z = complex(x)
    if z.imag and abs(z.imag) > tol * max(abs(z.real), 1e-300):
        print(f"[3d] WARNING: {what} came back complex ({z.real:.6g} + "
              f"{z.imag:.6g}j); using the real part. The eigenvector's global "
              f"phase was not fully removed, which usually means that mode has "
              f"not converged.", flush=True)
    return float(z.real)


def nodal_from_gridfunction(ctx, gfu):
    """
    (n_points, 3) Cartesian E at the mesh POINTS, by interpolation onto H1 order 1.

    HCurl dofs are edge/face moments, not point values, so there is nothing to
    plot directly: every visualisation needs this projection first. NGSolve's
    GridFunction.Set does it element-wise and averages, with no global solve --
    which is both far cheaper than the L2 projection the skfem version had to
    assemble and factorise, and the reason the old plot_modes_3d(n=50) hang is
    gone (that hang was in the PLOTTING, after the solve had finished).

    It is a mild smoother: fine for looking at, and for a peak-|E| convention,
    not for extracting a true field maximum. On a curved mesh the mid-edge points
    belong to no element and come back as exact zeros.
    """
    cols = []
    for c in range(3):
        ctx["gfp"][c].Set(gfu[c])
        vals = np.asarray(ctx["gfp"][c].vec.FV().NumPy())
        # the H1 projection space is real, but on the arnoldi path the source
        # GridFunction is complex; the mode has already been rotated to be real,
        # so the real part is the field
        cols.append(np.real(vals).astype(np.float64).copy())
    return np.column_stack(cols)


def _observables(ctx, u, k0, axis="z"):
    """
    Form factor C, quality factor Q, volume, localisation -- for ONE mode.

    Returns (obs_dict, scale, nodal) where `scale` renormalises the mode to
    peak |E| = 1 and `nodal` is the vertex field (already scaled).

    NOTE ON NORMALISATION. Everything is integrated on the RAW eigenvector and
    rescaled afterwards, rather than normalising first and integrating again.
    C, Q and the localisation are all ratios of equal powers of |E| and so are
    scale-INVARIANT; only the absolute energies are not. The peak itself is read
    off the H1-projected field rather than off quadrature points, so it is a
    slightly smoothed maximum -- which moves U_stored and int_eps_E2 a little,
    and moves nothing that is a ratio.
    """
    ngs = ctx["ngs"]
    mesh, gfu = ctx["mesh"], ctx["gfu"]
    iax = AXIS_INDEX[str(axis).lower()]
    gfu.vec.FV().NumPy()[:] = u

    io_, io4 = ctx["iord"], ctx["iord4"]
    num = _real(ngs.Integrate(gfu[iax], mesh, order=io_), "int E_axis dV")
    den = _real(ngs.Integrate(ctx["eps"] * (gfu * gfu), mesh, order=io_),
                "int eps|E|^2 dV")
    vol = _real(ngs.Integrate(ctx["one"], mesh, order=2), "volume")
    if ctx["localisation"]:
        l2 = _real(ngs.Integrate(gfu * gfu, mesh, order=io_), "int |E|^2 dV")
        l4 = _real(ngs.Integrate((gfu * gfu) * (gfu * gfu), mesh, order=io4),
                   "int |E|^4 dV")
        V_part = (l2 ** 2) / l4 if l4 > 0 else 0.0
    else:
        # the dominant integral, skipped on request. best_mode() and
        # fem_vis_3d.pick_mode() treat a NaN localisation as "unknown" and stop
        # filtering on it rather than silently ranking every mode as delocalised.
        V_part = float("nan")

    C = (num ** 2) / (vol * den) if vol > 0 and den > 0 else 0.0

    omega = C0 * k0
    # |H_t| = |H| at a PEC wall, so the full |curl E|^2 is what we want -- and it
    # holds DISCRETELY too: with tangential E constrained to zero over the whole
    # conductor, n.curl E came out at exactly 0.0 relative on the pillbox.
    #
    # BoundaryFromVolumeCF IS LOAD-BEARING. curl(gfu) is a volume-space
    # derivative; NGSolve will not evaluate it on a boundary element and returns
    # ZERO for the integral instead of raising, which surfaces as Q = inf.
    H = ngs.BoundaryFromVolumeCF(ngs.curl(gfu))
    P = 0.0
    for _name, region, mat in ctx["surfaces"]:
        g2 = _real(ngs.Integrate(H * H, mesh, definedon=region, order=io_),
                   f"int |curl E|^2 dS on '{_name}'")
        R_s = np.sqrt(omega * MU0 / (2.0 * mat.sigma))
        P += 0.5 * R_s * g2 / (omega * MU0) ** 2
    Q = (omega * (0.5 * EPS0 * den) / P) if P > 0 else np.inf

    nodal = nodal_from_gridfunction(ctx, gfu)
    peak = float(np.max(np.linalg.norm(nodal, axis=1))) if nodal.size else 0.0
    scale = (1.0 / peak) if peak > 0 else 1.0
    den_n = den * scale ** 2
    return (dict(C=float(C), Q=float(Q), volume=float(vol),
                 V_part=float(V_part),
                 localisation=(float(V_part / vol) if (vol and
                               np.isfinite(V_part)) else float("nan")),
                 int_eps_E2=float(den_n), U_stored=float(0.5 * EPS0 * den_n),
                 int_Eaxis=float(num * scale), axis=str(axis).lower()),
            float(scale), nodal * scale)


# ─────────────────────────────────────────────────────────────────────────────
# single solve
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_order(element, order):
    """`element` used to take an skfem element object. Accept an int (or None)
    for backwards compatibility and say something useful otherwise."""
    if element is None:
        return int(order)
    if isinstance(element, (int, np.integer)):
        return int(element)
    raise TypeError(
        f"element= took a scikit-fem element object in the old solver; this one "
        f"uses NGSolve, so pass order=<int> instead (got "
        f"{type(element).__name__}). order=0 is the lowest-order Whitney element, "
        f"the exact ElementTetN0 equivalent; order=2 is the default here.")


def solve_cavity_3d(spec, n_modes: int = 6, f_target: float | None = None,
                    msh_path: str | None = None, verbose: bool = False,
                    keep_fields: bool = False, deflate: bool = True,
                    element=None, tol: float = 0.0, ncv=None,
                    drop_below: float = 1e-3, check_kernel: bool = True,
                    axis: str = "z", max_elements: int = MAX_ELEMENTS,
                    progress: bool = False, progress_every: int = 25,
                    permc_spec: str = "COLAMD", linear_solver: str = None,
                    order: int = NG_ORDER, mesh_order: int = NG_MESH_ORDER,
                    nograds: bool = False, eigensolver: str = "scipy",
                    max_dofs: int | None = MAX_DOFS,
                    warn_dofs: int | None = WARN_DOFS,
                    threads: int | None = None, localisation: bool = True,
                    quartic_order: int | None = None):
    """
    Build -> mesh -> solve one 3D configuration, with NGSolve HCurl elements.

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

    order : HCurl order, default 2. In NGSolve's numbering 0 is the lowest-order
        Whitney/Nedelec element (one dof per edge) -- i.e. order=0 reproduces the
        old skfem ElementTetN0 discretisation. Cost: ~1.2 dofs per tetrahedron at
        order 0 against ~11 at order 2, so RAISE THE ORDER AND COARSEN THE MESH
        together, never one alone.

    mesh_order : geometric order of the mesh, default 2 (curved). Straight
        tetrahedra (1) inscribe a curved wall and that error does not care how
        good the elements are: on the analytic pillbox, going 1 -> 2 at fixed dof
        count took the frequency error from +0.70% to +0.031% and the volume
        error from -2.0% to -0.003%. Harmless for all-planar geometry.

    eigensolver :
        "scipy"   (default) ARPACK shift-invert on the scipy copies of K and M,
            with the gradient kernel projected out -- the deflation the module
            docstring describes. Deterministic, and it is the path the drop_below
            / permc_spec / linear_solver knobs act on.
        "arnoldi" NGSolve's own ArnoldiSolver, which does its own shift-invert
            with its own internal direct solver and never builds a scipy copy of
            K and M (so it needs less memory). Three things to know:
              * it forces a COMPLEX space. Handed real matrices it returns
                correct eigenvalues with eigenvectors that ignore the boundary
                condition, so f looks right while C collapses to ~0 -- a trap
                worth naming because nothing raises. The mode is rotated back to
                real here by removing its global phase.
              * it does NOT deflate the kernel: modes at f ~ 0 can appear among
                the returned ones (measured: one in eight at k=8 on a pillbox)
                and are filtered afterwards by drop_below.
              * it requires an f_target.
            The "scipy" path is the validated one; this is the lower-memory
            alternative.

    deflate : project the gradient kernel out of the shift-invert operator (see
        the module docstring). Leave it on for the scipy path. With it off,
        ARPACK has to chew through a null cluster of dimension (number of
        interior H1 dofs) and most of the returned modes are numerical noise at
        f ~ 0.

    axis : the CAVITY AXIS for the form factor, "x" / "y" / "z" (default "z").
        MUST match the geometry. The built-in CylSpec3D and from_2d() put the axis
        along z, so the default is right for those. An IMPORTED part is whatever
        the CAD used -- check the bounding box printed by ImportedSpec3D: the long
        direction is usually the axis. Getting this wrong gives C ~ 0 for the
        operating mode with no other symptom.

    linear_solver : "auto" (default, module LINEAR_SOLVER), "pardiso" or
        "superlu"; applies to the scipy path only. "auto" uses MKL PARDISO when
        pypardiso imports and silently falls back to SuperLU otherwise, warning
        once. On a 36,909-dof curl-curl matrix, single core: SuperLU 30.6 s to
        factorise, PARDISO 1.4 s unsymmetric and 0.8 s symmetric, answers
        agreeing to 1e-13. THREADS: PARDISO uses all cores via MKL, so set
        MKL_NUM_THREADS=1 in run_batch workers or they will oversubscribe and run
        slower than one process would.

    permc_spec : fill-reducing ordering for the shift-invert factorisation, which
        is where a 3D solve actually spends its time. "COLAMD" (default) is the
        fastest to compute. "MMD_AT_PLUS_A" produces ~2.4x LESS FILL but takes
        ~6x longer to order, so it is a MEMORY switch, not a speed one -- reach
        for it when a factorisation will not fit, not when it is merely slow.

    max_dofs / warn_dofs : refuse, and warn, above these dof counts. A direct
        factorisation is the binding constraint in 3D and its cost is nowhere
        near linear: a 34k-dof order-2 matrix factorised in 23 s on one core here
        while a 78k-dof one exhausted 3 GB and was OOM-killed. Pass max_dofs=None
        to disable.

    threads : NGSolve threads for assembly and the observables, which are the
        two element-loop phases. None uses module NG_THREADS (0 = all cores).
        NGSolve runs single-threaded outside a TaskManager, so on a many-core box
        this is the largest easy win available: the observables are a handful of
        full-mesh quadrature passes PER MODE, so 50 modes on a 155k-tet mesh is
        minutes of element looping that parallelises almost perfectly. Set 1 in
        run_batch workers and get the parallelism from the batch instead.

    localisation / quartic_order : the localisation diagnostic needs
        int |E|^4 dV, which is degree 8 in the field and, at quadrature order 10,
        was 67% of the per-mode observable cost (0.75 s of 1.15 s on a 6,551-tet
        problem). Order 8 is the default now and reproduces the order-10 value to
        10 significant digits at 0.46 s. localisation=False skips the term
        outright, roughly tripling per-mode throughput; f, C and Q do not depend
        on it, only best_mode's optional filter does.

    WHAT ACTUALLY COSTS TIME, and what does not overlap. There is no pipelining
    to be had between the eigensolve and the observables: ARPACK hands over all
    eigenvectors at once, at the end, so there is nothing to start early on. The
    levers are (a) TaskManager, above, (b) fewer modes -- n_modes drives BOTH the
    ARPACK work (ncv ~ 2*n_modes Lanczos vectors, each costing a full
    back-substitution through the projector sandwich) and the observable count
    linearly, so asking for 50 modes when you want the one nearest a known 2D
    frequency costs about 5x what asking for 10 does -- and (c) the mesh.

    drop_below : discard returned modes with f < drop_below * f_target as kernel
        residue. Fires when deflate=False, and on the "arnoldi" path.

    progress : print stage timings (mesh / convert / factorise / eigensolve /
        observables) and an ARPACK operator-application counter, plus gmsh's own
        meshing percentage. Costs nothing and is the only way to tell a slow
        solve from a hung one.

    element : deprecated alias for `order`, accepted as an int.

    Returns a dict with 'freqs' (Hz) and per-mode C / Q / V_part, sorted by
    frequency, plus n_dofs / n_elements / n_free / kernel_dim. With
    keep_fields=True it also carries 'mesh' (a MeshView), 'fields' (raw HCurl dof
    vectors) and 'nodal_fields' ((n_points, 3) vertex fields), which is what
    fem_vis_3d needs -- all plain numpy, so it survives a run_batch pickle.
    """
    ngs = _ng()
    pg = _Progress(progress)
    order = _resolve_order(element, order)
    eigensolver = str(eigensolver).lower()
    if eigensolver not in ("scipy", "arnoldi"):
        raise ValueError(f"eigensolver must be 'scipy' or 'arnoldi', got "
                         f"{eigensolver!r}")

    # ---- geometry + mesh ---------------------------------------------------
    pg(f"meshing {spec.tag or '(untagged)'} at mesh_size={spec.mesh_size:g} m "
       f"(geometric order {mesh_order})")
    tmp = msh_path or tmp_msh_path("cavity3d")
    with _quiet(QUIET and not (verbose or progress)):
        diel_mats = build_mesh_3d(spec, tmp, verbose=(verbose or progress),
                                  max_elements=max_elements,
                                  mesh_order=mesh_order)
        mesh = load_ng_mesh(tmp)
    if msh_path is None:
        try:
            os.remove(tmp)
        except OSError:
            pass
    if mesh.ne == 0:
        raise RuntimeError(
            f"the mesher produced no tetrahedra -- netgen loaded a mesh with "
            f"volume elements: 0 (surface elements only), so there is no volume "
            f"to solve.\n"
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
    pg(f"mesh: {mesh.ne:,} tets, {mesh.nv:,} points, "
       f"materials {list(mesh.GetMaterials())}")

    # ---- space -------------------------------------------------------------
    # ArnoldiSolver needs COMPLEX dofs. With a real space it returns the right
    # eigenvalues and useless eigenvectors: measured on a rectangular cavity,
    # the returned vector's largest entries sat ON the Dirichlet dofs and its
    # Rayleigh quotient was 15x the reported eigenvalue, so f looked perfect
    # while C came out ~0. With complex=True the same call gives Rayleigh
    # quotients matching to 5 digits and exactly zero on the Dirichlet dofs.
    fes, dirichlet, bnames = hcurl_space(mesh, order=order, nograds=nograds,
                                         complex=(eigensolver == "arnoldi"))
    free = _bit_mask(fes.FreeDofs(), fes.ndof)
    n_free = int(free.sum())
    if warn_dofs and fes.ndof > warn_dofs:
        print(f"[3d] WARNING: {fes.ndof:,} dofs at HCurl order {order}. The "
              f"shift-invert factorisation, not the assembly, is what this "
              f"costs, and its memory grows much faster than the dof count. If "
              f"this run dies or swaps, coarsen mesh_size (dofs scale as "
              f"mesh_size^-3) before reaching for a lower order.", flush=True)
    if max_dofs and fes.ndof > max_dofs:
        h_ok = spec.mesh_size * (fes.ndof / float(max_dofs)) ** (1.0 / 3.0)
        raise ValueError(
            f"this mesh gives {fes.ndof:,} HCurl dofs at order {order} (limit "
            f"max_dofs={max_dofs:,}), which no direct factorisation on a normal "
            f"machine will survive.\n"
            f"  {mesh.ne:,} tets x ~{fes.ndof/max(mesh.ne,1):.1f} dofs each.\n"
            f"  try mesh_size >= {h_ok:.4g} m, or order={max(0, order-1)} at the "
            f"same mesh, or max_dofs=None if you really mean it.\n"
            f"  NOTE a mesh_size that was comfortable with the old lowest-order "
            f"element is ~9x the dofs here; that is the trade you took for the "
            f"accuracy, so coarsen to match.")
    pg(f"HCurl order {order}: {fes.ndof:,} dofs, {n_free:,} free, "
       f"PEC on '{dirichlet}'")

    # ---- assemble ----------------------------------------------------------
    with task_manager(threads):
        K, M, eps_cf, muinv_cf, matreport = assemble_curlcurl(mesh, fes, spec,
                                                              diel_mats)
    pg(f"assembled ({matreport})")

    sigma = (2 * np.pi * f_target / C0) ** 2 if f_target else 0.0

    raw = []            # (lambda, full dof vector)
    kernel_dim = 0
    if eigensolver == "arnoldi":
        # ---- NGSolve's own Arnoldi -----------------------------------------
        if sigma <= 0:
            raise ValueError(
                "eigensolver='arnoldi' needs an f_target: ArnoldiSolver takes a "
                "shift, and a shift of zero sits exactly on the curl-curl "
                "kernel, so the solve returns null-space modes and nothing else. "
                "Pass f_target, or use eigensolver='scipy'.")
        gf = ngs.GridFunction(fes, multidim=n_modes)
        pg(f"eigensolve (NGSolve Arnoldi): {n_modes} modes near "
           f"{f_target/1e6:.3f} MHz")
        lams = ngs.ArnoldiSolver(K.mat, M.mat, fes.FreeDofs(),
                                 list(gf.vecs), float(sigma))
        worst_imag = 0.0
        for j, lam in enumerate(lams):
            w = np.asarray(gf.vecs[j].FV().NumPy()).copy()
            # A closed lossless cavity has REAL eigenvectors up to one global
            # phase, which Arnoldi picks arbitrarily. Rotate it out using the
            # largest entry; what is left in the imaginary part is a convergence
            # diagnostic, not physics.
            k = int(np.argmax(np.abs(w)))
            if np.abs(w[k]) > 0:
                w = w * (np.conj(w[k]) / np.abs(w[k]))
            ur, ui = np.real(w), np.imag(w)
            amp = max(float(np.abs(ur).max()), 1e-300)
            worst_imag = max(worst_imag, float(np.abs(ui).max()) / amp)
            raw.append((float(complex(lam).real),
                        np.ascontiguousarray(ur, dtype=np.float64)))
        if worst_imag > 1e-3:
            print(f"[3d] NOTE: after removing the global phase the worst "
                  f"eigenvector still has an imaginary part at "
                  f"{worst_imag:.1%} of its amplitude. Those modes are not "
                  f"converged -- ask for fewer modes, target them better, or "
                  f"use eigensolver='scipy'.", flush=True)
    else:
        # ---- scipy shift-invert with kernel deflation ----------------------
        # THIS IS AN EXPENSIVE STAGE IN ITS OWN RIGHT, and it used to hide
        # inside the "kernel check" timing. Converting NGSolve's matrices to
        # scipy and then extracting the free-dof submatrix means, briefly,
        # FOUR copies of a curl-curl matrix in memory (NGSolve K and M, the
        # scipy originals, and the constrained ones), which is both the peak
        # RAM of the whole solve and tens of seconds at a million dofs. The
        # prints below break it up so you can see which part costs what.
        Ks, Ms = _to_scipy(K.mat), _to_scipy(M.mat)
        pg(f"converted to scipy ({Ks.nnz:,} + {Ms.nnz:,} nnz)")
        Gfull = discrete_gradient(fes) if deflate else None
        del K, M
        gc.collect()
        Kc = Ks[free][:, free].tocsc()
        Mc = Ms[free][:, free].tocsc()
        del Ks, Ms
        gc.collect()
        pg(f"constrained to {n_free:,} free dofs ({Kc.nnz:,} nnz)")
        if sigma <= 0:
            # no target: a small positive shift, still above the kernel
            sigma = 1e-6 * float(sp.linalg.norm(Kc)) / max(
                float(sp.linalg.norm(Mc)), 1e-300)

        OPinv = None
        Asolve = Ssolve = None
        if Gfull is not None:
            Gc = kernel_basis(Gfull, free)
            del Gfull
            kernel_dim = int(Gc.shape[1])
            if check_kernel:
                kmode = ("probe" if check_kernel is True
                         else str(check_kernel).lower())
                rel = check_gradient_kernel(Kc, Gc, mode=kmode)
                pg(f"kernel check [{kmode}]: ||KG||/||K|| = {rel:.2e} "
                   f"({kernel_dim:,} kernel vectors)")
            S = (Gc.T @ Mc @ Gc).tocsc()
            Ssolve, slabel = _factorized(S, linear_solver, permc_spec, spd=True)
            pg(f"gradient projector factorised [{slabel}] "
               f"({kernel_dim:,} kernel modes deflated)")
            del S
            A = (Kc - sigma * Mc).tocsc()
            Asolve, alabel = _factorized(A, linear_solver, permc_spec, spd=False)
            fill = (f", fill-in {Asolve._lu.L.nnz + Asolve._lu.U.nnz:,} nnz"
                    if hasattr(Asolve, "_lu") else "")
            pg(f"shift-invert factorised [{alabel}]{fill}")
            del A
            gc.collect()

            def _P(x):
                return x - Gc @ Ssolve(Gc.T @ (Mc @ x))

            # P on BOTH sides: that is what keeps the operator M-self-adjoint,
            # which ARPACK's symmetric shift-invert mode assumes. One-sided is
            # not. ARPACK gives no progress of its own; counting operator
            # applications is the only honest signal of whether it is converging
            # or stalling.
            _count = {"n": 0, "t": time.perf_counter()}

            def _op(x):
                _count["n"] += 1
                if progress and progress_every and _count["n"] % progress_every == 0:
                    dt = time.perf_counter() - _count["t"]
                    pg(f"eigensolve: {_count['n']} operator applications "
                       f"({_count['n'] / max(dt, 1e-9):.1f}/s)")
                return _P(Asolve(_P(x)))

            OPinv = LinearOperator(Kc.shape, dtype=np.float64, matvec=_op)

        pg(f"eigensolve (ARPACK): {n_modes} modes near "
           f"{(f_target or 0)/1e6:.3f} MHz ({n_free:,} free dofs)")
        vals, vecs = eigsh(Kc, k=n_modes, M=Mc, sigma=sigma, which="LM",
                           OPinv=OPinv, tol=tol, ncv=ncv)
        for j in range(len(vals)):
            u = np.zeros(fes.ndof)
            u[free] = np.real(vecs[:, j])
            raw.append((float(np.real(vals[j])), u))
        # PARDISO holds its factors in MKL-side memory that Python's GC does not
        # own; without this a sweep of many geometries leaks until the machine
        # swaps. Harmless no-op on the SuperLU path.
        for _s in (Asolve, Ssolve):
            if _s is not None:
                try:
                    _s.free()
                except Exception:
                    pass
        del Kc, Mc, vals, vecs
        gc.collect()

    pg("eigensolve done; computing C and Q")

    # ---- observables -------------------------------------------------------
    # ONE TaskManager around the whole loop, not one per mode: entering and
    # leaving the thread pool 50 times would cost more than it saves.
    with task_manager(threads):
        ctx = observable_context(mesh, fes, spec, diel_mats, eps_cf=eps_cf,
                                 order=order, localisation=localisation,
                                 quartic_order=quartic_order)
        raw.sort(key=lambda lu: lu[0])
        out = {"freqs": [], "modes": [], "n_dofs": int(fes.ndof),
               "n_elements": int(mesh.ne), "n_free": n_free,
               "kernel_dim": int(kernel_dim), "tag": spec.tag,
               "deflated": bool(kernel_dim > 0), "order": int(order),
               "mesh_order": int(mesh_order), "eigensolver": eigensolver,
               "element": f"HCurl(order={order})"}
        # a floor for kernel residue: relative to the target when there is one,
        # else to the largest frequency found, so f ~ 0 junk is still caught
        f_all = [C0 * np.sqrt(lam) / (2 * np.pi) for lam, _ in raw if lam > 0]
        f_ref = f_target or (max(f_all) if f_all else 0.0)
        f_floor = drop_below * f_ref
        t_obs = time.perf_counter()
        for _n, (lam, u) in enumerate(raw):
            if lam <= 0:
                continue
            k0 = np.sqrt(lam)
            f = C0 * k0 / (2 * np.pi)
            if f < f_floor:
                continue                      # kernel residue
            obs, scale, nodal = _observables(ctx, u, k0, axis=axis)
            if not out["freqs"]:
                pg(f"first frequency: {f/1e9:.4f} GHz "
                   f"({time.perf_counter() - t_obs:.2f} s per mode)")
            out["freqs"].append(f)
            out["modes"].append(obs)
            if keep_fields:
                out.setdefault("fields", []).append(u * scale)
                out.setdefault("nodal_fields", []).append(nodal)
            if progress and progress_every and (_n + 1) % max(1, progress_every // 5) == 0:
                pg(f"observables: {_n + 1}/{len(raw)} modes "
                   f"(f={f/1e9:.4f} GHz, C={obs['C']:.4f})")
        if keep_fields:
            out["mesh"] = mesh_arrays(mesh)
            # the live NGSolve handles, for anything that wants to keep working
            # in NGSolve. Underscore-prefixed so run_batch's worker strips them
            # before pickling.
            out["_ng"] = {"mesh": mesh, "fes": fes, "ctx": ctx}
    # ---- degeneracy warning ------------------------------------------------
    # A near-degenerate cluster has no unique eigenbasis, so the form factor of
    # any ONE returned eigenvector is an artefact of whichever basis ARPACK
    # landed on. Silence here is how a correct solve gets read as C ~ 0.
    if len(out["freqs"]) > 1:
        bc = best_cluster(out)
        if bc and bc["n_modes"] > 1 and bc["C"] > 1.5 * max(bc["C_best"], 1e-12):
            print(f"[3d] NOTE: the largest form factor of a single mode is "
                  f"{bc['C_best']:.4f}, but the {bc['n_modes']} near-degenerate "
                  f"modes between {bc['f_min']/1e9:.4f} and "
                  f"{bc['f_max']/1e9:.4f} GHz sum to C = {bc['C']:.4f}. Inside a "
                  f"degenerate cluster the eigenbasis is arbitrary, so no single "
                  f"eigenvector carries the coupling -- the sum does. Use "
                  f"best_cluster()/combine_cluster(), not best_mode().",
                  flush=True)
    pg.done(f"{len(out['freqs'])} modes")
    return out


def converge(spec, f_target, mesh_sizes, n_modes=4, min_localisation=0.0,
             verbose=True, order=NG_ORDER, richardson_power=None, **kw):
    """
    Solve at several mesh sizes and Richardson-extrapolate f, C and Q to h -> 0.

    EACH QUANTITY GETS ITS OWN POWER, because they do not converge at the same
    rate and using one for all three is how an extrapolation ends up worse than
    the data. Measured on the analytic pillbox at order 2, curved mesh, h going
    20 mm -> 14 mm:

        frequency   +0.031% -> +0.0076%     ~ h^4   ( = h^(2*order) )
        form factor +0.187% -> +0.0365%     ~ h^4
        Q           -6.89%  -> -3.27%       ~ h^2

    Q is the slow one and always will be: it is a SURFACE integral of a field
    DERIVATIVE, so it carries one fewer derivative of accuracy than the
    eigenvalue. Do not judge a mesh by its frequency error. Two other slow cases
    worth knowing: a curved wall meshed with STRAIGHT tetrahedra caps everything
    at h^2 no matter the element order (use mesh_order=2, the default), and sharp
    metal edges with a field singularity converge slowly in Q while frequency
    looks fine.

    richardson_power : override, either a number for all three or a dict like
        {"f": 4, "C": 4, "Q": 2}. The default is 2*order for f and C and 2 for Q.

    Returns {"h": [...], "f": [...], "C": [...], "Q": [...],
             "f_extrap": ..., "C_extrap": ..., "Q_extrap": ..., "order": ...,
             "powers": {...}}
    where *_extrap comes from the two finest meshes and "order" is the frequency
    convergence order OBSERVED from the three finest, which is the number to
    check against 2*order before believing any of it.
    """
    import copy
    p_default = {"f": 2.0 * int(order), "C": 2.0 * int(order), "Q": 2.0}
    if richardson_power is None:
        powers = p_default
    elif isinstance(richardson_power, dict):
        powers = {**p_default, **{k: float(v)
                                  for k, v in richardson_power.items()}}
    else:
        powers = {k: float(richardson_power) for k in p_default}

    hs, fs, Cs, Qs = [], [], [], []
    for h in sorted(mesh_sizes, reverse=True):
        spc = copy.copy(spec)
        spc.mesh_size = float(h)
        r = solve_cavity_3d(spc, n_modes=n_modes, f_target=f_target, order=order,
                            **kw)
        m = best_mode(r, min_localisation=min_localisation)
        if m is None:
            continue
        hs.append(float(h)); fs.append(m["f"]); Cs.append(m["C"]); Qs.append(m["Q"])
        if verbose:
            print(f"[converge] h={h:.4g} tets={r['n_elements']:>7d} "
                  f"dof={r['n_free']:>7d} | f={m['f']/1e9:.5f} GHz "
                  f"C={m['C']:.5f} Q={m['Q']:.4g}", flush=True)
    out = {"h": hs, "f": fs, "C": Cs, "Q": Qs, "order": np.nan,
           "f_extrap": np.nan, "C_extrap": np.nan, "Q_extrap": np.nan,
           "powers": powers}
    if len(hs) >= 3:
        # observed order from the three finest points, if they are monotone
        f0, f1, f2 = fs[-3], fs[-2], fs[-1]
        r01, r12 = f1 - f0, f2 - f1
        if r01 != 0 and 0 < r12 / r01 < 1:
            out["order"] = float(np.log(r12 / r01) /
                                 np.log(hs[-1] / hs[-3]))
    if len(hs) >= 2:
        for key, arr in (("f", fs), ("C", Cs), ("Q", Qs)):
            ratio = (hs[-2] / hs[-1]) ** powers[key]
            out[f"{key}_extrap"] = float(arr[-1] + (arr[-1] - arr[-2]) /
                                         (ratio - 1.0))
        if verbose:
            print(f"[converge] Richardson (f,C at h^{powers['f']:g}, "
                  f"Q at h^{powers['Q']:g}): "
                  f"f={out['f_extrap']/1e9:.5f} GHz  C={out['C_extrap']:.5f}  "
                  f"Q={out['Q_extrap']:.4g}", flush=True)
            if np.isfinite(out["order"]):
                print(f"[converge] observed frequency order: "
                      f"{out['order']:.2f} (expected {2*int(order)} at HCurl "
                      f"order {order}; a much lower number means geometry, not "
                      f"the elements, is the limit)", flush=True)
    return out


def mode_clusters(result, rel_tol=2e-3):
    """
    Group returned modes into NEAR-DEGENERATE CLUSTERS: consecutive frequencies
    within rel_tol of each other. Returns a list of index lists.

    rel_tol defaults to 2e-3 because that is the scale that matters for a
    multi-cell cavity: the transverse modes of a 5-cell toaster sit within ~0.2%
    of one another, so anything inside that window is physically one cluster
    whatever the solver labels it.
    """
    order = np.argsort(np.asarray(result["freqs"], dtype=float))
    out, cur = [], []
    for j in order:
        f = float(result["freqs"][j])
        if cur and abs(f / max(float(result["freqs"][cur[-1]]), 1e-300) - 1.0) > rel_tol:
            out.append(cur)
            cur = []
        cur.append(int(j))
    if cur:
        out.append(cur)
    return out


def cluster_form_factor(result, indices):
    """
    The LARGEST form factor obtainable from any combination of the given modes,
    which for a degenerate or near-degenerate cluster is the physically
    meaningful number -- and it is simply the SUM of the individual C.

    WHY THIS IS THE RIGHT QUANTITY, AND WHY A SINGLE EIGENVECTOR'S C IS NOT.
    Inside a degenerate cluster the eigenvectors are not unique: any orthogonal
    rotation of them is equally an eigenbasis, and ARPACK returns whichever one
    the Krylov iteration happened to land on. The form factor is NOT invariant
    under that rotation, so "the C of mode 7" is an artefact of the basis. What
    is invariant is the coupling of the whole subspace.

    The algebra: eigenvectors of the same pencil are M-orthogonal, so with
    n_i = integral eps_r |E_i|^2 and m_i = integral E_i,axis dV, a combination
    u = sum a_i E_i has

        integral u_axis  = sum a_i m_i ,   integral eps_r |u|^2 = sum a_i^2 n_i

    and maximising (sum a_i m_i)^2 / (V sum a_i^2 n_i) over a gives
    a_i proportional to m_i / n_i and the maximum

        C_max = sum_i m_i^2 / (V n_i) = sum_i C_i .

    VERIFIED: a cavity split into two identical halves by a full-height divider
    has an exactly degenerate pair. The solver returned the LOCALISED basis, one
    mode per half, C = 0.32952 and 0.32754 -- each meaningless on its own -- and
    their sum was 0.65707 against the analytic single-box 64/pi^4 = 0.65702, a
    0.01% agreement.

    SO: if your operating mode comes back with C ~ 0, sum the cluster before
    concluding anything. If the sum is right, the solve is right and only the
    labelling was wrong.
    """
    return float(sum(result["modes"][int(j)]["C"] for j in indices))


def best_cluster(result, rel_tol=2e-3):
    """
    The cluster with the largest total form factor -- the operating mode when it
    has been smeared across a near-degenerate group.

    Returns a dict: C (the summed form factor), f (the M-weighted mean
    frequency), f_min / f_max, indices, n_modes, and C_best (the largest single
    member, i.e. what best_mode would have reported).
    """
    clusters = mode_clusters(result, rel_tol=rel_tol)
    if not clusters:
        return None
    best = max(clusters, key=lambda idx: cluster_form_factor(result, idx))
    fs = [float(result["freqs"][j]) for j in best]
    Cs = [float(result["modes"][j]["C"]) for j in best]
    w = np.array(Cs, dtype=float)
    fmean = float(np.average(fs, weights=w) if w.sum() > 0 else np.mean(fs))
    return {"C": float(sum(Cs)), "f": fmean, "f_min": min(fs), "f_max": max(fs),
            "indices": list(best), "n_modes": len(best),
            "C_best": max(Cs) if Cs else 0.0}


def combine_cluster(result, indices=None, rel_tol=2e-3, axis=None):
    """
    Build the ACTUAL maximum-form-factor field from a cluster and measure it, so
    the operating mode can be plotted and its Q read off rather than inferred.

    The optimal weights are a_i = m_i / n_i (see cluster_form_factor). Both are
    already stored per mode as int_Eaxis and int_eps_E2, so the combination costs
    one weighted sum of dof vectors -- no re-solve.

    Requires solve_cavity_3d(..., keep_fields=True) IN THIS PROCESS: C and Q of
    the combination are re-integrated from the combined field, which needs the
    live NGSolve space (result["_ng"], which run_batch strips before pickling).

    Returns a dict shaped like a mode -- f, C, Q, localisation, ... -- plus
    'indices', 'weights', 'C_sum' (the algebraic prediction) and 'field' /
    'nodal_field'. It also APPENDS the combination to result as an extra mode so
    fem_vis_3d can plot it directly:

        cl = fem_solve_3d.combine_cluster(r3)
        fem_vis_3d.plot_field_slices(r3, i=cl["mode_index"])

    A disagreement between the re-integrated C and the algebraic sum means the
    returned eigenvectors are not M-orthogonal, which would invalidate the sum
    rule; that is checked and warned about rather than assumed.
    """
    if "fields" not in result:
        raise ValueError("combine_cluster needs the eigenvectors: re-run "
                         "solve_cavity_3d(..., keep_fields=True).")
    ng = result.get("_ng")
    if ng is None:
        raise ValueError(
            "combine_cluster needs the live NGSolve space to re-integrate C and "
            "Q, and this result no longer carries it (run_batch strips it so the "
            "result can be pickled). Call it in the process that solved, or use "
            "cluster_form_factor(), which needs only the stored numbers.")
    if indices is None:
        bc = best_cluster(result, rel_tol=rel_tol)
        if bc is None:
            return None
        indices = bc["indices"]
    indices = [int(j) for j in indices]

    w, lam_w = [], []
    for j in indices:
        m = result["modes"][j]
        n_i = float(m["int_eps_E2"])
        w.append(float(m["int_Eaxis"]) / n_i if n_i > 0 else 0.0)
        lam_w.append(n_i)
    w = np.asarray(w, dtype=float)
    if not np.any(w):
        return None
    u = np.zeros_like(np.asarray(result["fields"][indices[0]], dtype=float))
    for a, j in zip(w, indices):
        u = u + a * np.asarray(result["fields"][j], dtype=float)

    # Rayleigh quotient of the combination, exact given M-orthogonality:
    # lambda_eff = sum a_i^2 n_i lambda_i / sum a_i^2 n_i
    k0s = np.array([2 * np.pi * float(result["freqs"][j]) / C0 for j in indices])
    wt = (w ** 2) * np.asarray(lam_w, dtype=float)
    lam_eff = float(np.average(k0s ** 2, weights=wt)) if wt.sum() > 0 else \
        float(np.mean(k0s ** 2))
    k0 = float(np.sqrt(lam_eff))

    ctx = ng["ctx"]
    obs, scale, nodal = _observables(ctx, u, k0,
                                    axis=axis or result["modes"][indices[0]].get(
                                        "axis", "z"))
    C_sum = cluster_form_factor(result, indices)
    if C_sum > 0 and abs(obs["C"] / C_sum - 1.0) > 0.02:
        print(f"[3d] WARNING: the combined field's form factor ({obs['C']:.5f}) "
              f"disagrees with the sum over the cluster ({C_sum:.5f}) by "
              f"{abs(obs['C']/C_sum-1)*100:.1f}%. The eigenvectors are not "
              f"M-orthogonal to the accuracy the sum rule assumes -- trust the "
              f"combined field, not the sum.", flush=True)
    f = C0 * k0 / (2 * np.pi)
    result["freqs"].append(f)
    result["modes"].append(obs)
    if "fields" in result:
        result["fields"].append(u * scale)
    if "nodal_fields" in result:
        result["nodal_fields"].append(nodal)
    return {"f": f, **obs, "indices": indices, "weights": w.tolist(),
            "C_sum": C_sum, "field": u * scale, "nodal_field": nodal,
            "mode_index": len(result["freqs"]) - 1}


def mode_diagnostics(result, indices=None, verbose=True):
    """
    WHY IS C ~ 0 FOR THIS MODE? There are three different answers with three
    different fixes, and the form factor alone cannot tell them apart because all
    three produce the same near-zero number. Two extra integrals can.

    For each mode this reports:

      cancel = |integral E_axis dV| / integral |E_axis| dV
          How much of the axial field survives its own sign. 1 means E_axis never
          changes sign, so the mode couples as well as its amplitude allows. Near
          0 means the coupling is being cancelled by phase, not by weakness --
          the field is there, it just sums to nothing.

      parity = (I_hi + I_lo) / (|I_hi| + |I_lo|),  I_hi/I_lo being integral
          E_axis over the two halves either side of the mid-plane.
          +1  E_axis has the same sign in both halves  -> p = 0, LONGITUDINALLY
              UNIFORM, a candidate for the operating mode.
           0  equal and opposite                       -> p = 1 (or any odd p),
              a half-cosine along the axis whose integral vanishes IDENTICALLY.
              No mesh refinement will ever give this mode a form factor; it is a
              neighbour, not a candidate.

      localisation, already computed, says whether the energy is spread through
          the cavity or sitting in one channel or cell.

    The verdict combines them:

      "couples"      cancel is high -- this mode's C is its own, believe it.
      "p>=1 (axial)" parity ~ 0 -- cancelled along the axis. Your target is
                     sitting on the longitudinal ladder; the p = 0 member is
                     LOWER in frequency, by (p pi/L)^2 in k^2.
      "transverse"   parity ~ +1 but cancel ~ 0 -- cancelled ACROSS the
                     cross-section, i.e. cells or lobes of alternating sign.
                     This is the degenerate/detuned cluster case: the coupling is
                     in the subspace, so use best_cluster()/combine_cluster().
      "localised"    the energy is in one region. A channel mode above the bars
                     looks exactly like this, and typically has a HIGH Q because
                     it avoids the lossy bar surfaces -- a high Q with C = 0 is
                     the signature.

    Requires solve_cavity_3d(..., keep_fields=True) in this process.
    """
    if "fields" not in result or result.get("_ng") is None:
        raise ValueError(
            "mode_diagnostics needs the eigenvectors and the live NGSolve space: "
            "re-run solve_cavity_3d(..., keep_fields=True) and call this in the "
            "same process.")
    ctx = result["_ng"]["ctx"]
    ngs, mesh, gfu = ctx["ngs"], ctx["mesh"], ctx["gfu"]
    axis = str(result["modes"][0].get("axis", "z")).lower()
    iax = AXIS_INDEX[axis]
    coord = {"x": ngs.x, "y": ngs.y, "z": ngs.z}[axis]
    io_ = ctx["iord"]
    vol = _real(ngs.Integrate(ctx["one"], mesh, order=2), "volume")
    mid = _real(ngs.Integrate(coord, mesh, order=2), "axis centroid") / max(vol,
                                                                           1e-300)
    if indices is None:
        indices = range(len(result["modes"]))
    rows = []
    for i in indices:
        i = int(i)
        gfu.vec.FV().NumPy()[:] = np.asarray(result["fields"][i], dtype=float)
        Ea = gfu[iax]
        num = _real(ngs.Integrate(Ea, mesh, order=io_), "int E_axis")
        absn = _real(ngs.Integrate(ngs.sqrt(Ea * Ea), mesh, order=io_),
                     "int |E_axis|")
        hi = _real(ngs.Integrate(ngs.IfPos(coord - mid, Ea, 0.0), mesh,
                                 order=io_), "int E_axis (far half)")
        lo = num - hi
        cancel = abs(num) / absn if absn > 0 else 0.0
        denom = abs(hi) + abs(lo)
        parity = ((hi + lo) / denom) if denom > 0 else 0.0
        # PARITY IS ONLY MEANINGFUL IF THE HALVES INTEGRATE TO SOMETHING. When
        # the cancellation happens WITHIN each half (alternating lobes across the
        # cross-section) both halves are ~0 and their ratio is pure noise, which
        # would otherwise be read as odd-p. For a genuine odd-p mode,
        # E_axis ~ sin(p pi z/L), each half integrates to L/pi against a total
        # |E| of 2L/pi, so this ratio is 0.5; for a transversely cancelled p = 0
        # mode it is cancel/2, i.e. tiny.
        half = (max(abs(hi), abs(lo)) / absn) if absn > 0 else 0.0
        loc = float(result["modes"][i].get("localisation", float("nan")))
        if cancel > 0.5:
            verdict = "couples"
        elif half > 0.25 and abs(parity) < 0.3:
            verdict = "p>=1 (axial)"
        elif np.isfinite(loc) and loc < 0.15:
            verdict = "localised"
        else:
            verdict = "transverse"
        rows.append({"i": i, "f": float(result["freqs"][i]),
                     "C": float(result["modes"][i]["C"]),
                     "Q": float(result["modes"][i]["Q"]),
                     "localisation": loc, "cancel": float(cancel),
                     "parity": float(parity), "half": float(half),
                     "verdict": verdict})
    if verbose:
        print("  i    f (GHz)        C        Q      loc   cancel   half  "
              "parity  verdict")
        for r in rows:
            print("  %-4d %9.5f  %8.5f  %8.3g  %5.3f  %6.3f  %5.3f  %+6.3f  %s"
                  % (r["i"], r["f"] / 1e9, r["C"], r["Q"], r["localisation"],
                     r["cancel"], r["half"], r["parity"], r["verdict"]))
        kinds = {}
        for r in rows:
            kinds[r["verdict"]] = kinds.get(r["verdict"], 0) + 1
        print("  verdicts:", ", ".join(f"{v} {k}" for k, v in
                                       sorted(kinds.items())))
        tot = sum(r["C"] for r in rows)
        print(f"  sum of C over these {len(rows)} modes = {tot:.5f}  "
              f"(compare against the 2D form factor: if it matches, the coupling "
              f"is present but split across the basis)")
    return rows


def best_mode(result, min_localisation: float = 0.0):
    """
    Pick the operating mode: highest C among modes that are not localised.

    BEWARE OF DEGENERACY. This ranks INDIVIDUAL eigenvectors, and inside a
    near-degenerate cluster an individual eigenvector's form factor is an
    artefact of the arbitrary basis the eigensolver returned -- so a multi-cell
    cavity can report C ~ 0 here while the cluster couples perfectly well. See
    best_cluster() and combine_cluster(); solve_cavity_3d prints a NOTE when the
    two disagree.

    In 3D the localisation number is V_part/volume rather than A_part/area, and a
    delocalised fundamental sits lower than its 2D counterpart simply because the
    field now also varies along z for p >= 1.

    A NaN localisation means it was not computed (localisation=False in the
    solve, which skips the expensive int |E|^4). Those modes are still eligible
    when min_localisation <= 0; asking for a positive threshold without the
    diagnostic raises rather than quietly ranking on nothing.
    """
    loc = [m.get("localisation", float("nan")) for m in result["modes"]]
    unknown = [not np.isfinite(x) for x in loc]
    if min_localisation > 0 and any(unknown):
        raise ValueError(
            "min_localisation > 0 but this result has no localisation numbers -- "
            "it was solved with localisation=False, which skips int |E|^4. "
            "Re-solve with localisation=True (the default) or drop the filter.")
    cand = [(f, m) for f, m, x, unk in zip(result["freqs"], result["modes"],
                                           loc, unknown)
            if unk or x >= min_localisation]
    if not cand:
        return None
    f, m = max(cand, key=lambda fm: fm[1]["C"])
    return {"f": f, **m}


# ─────────────────────────────────────────────────────────────────────────────
# field sampling (for the visualiser, and for anything that needs nodal values)
# ─────────────────────────────────────────────────────────────────────────────

def nodal_field(result, i=0, min_localisation=None):
    """
    (n_points, 3) Cartesian E at the mesh points, matching result["mesh"].p
    column for column -- which is exactly what fem_vis_3d.to_pyvista wants.

    HCurl dofs are edge and face moments, so there is nothing to plot directly;
    solve_cavity_3d(keep_fields=True) does the H1 interpolation once per mode
    while the space is still alive and stores the result, so this is now a lookup
    rather than a computation. That is deliberate: the arrays are plain numpy and
    survive being pickled out of a run_batch worker, and plot_modes_3d(n=50) no
    longer re-projects (and, in the skfem version, re-factorised a mass matrix)
    fifty times.

    Requires solve_cavity_3d(..., keep_fields=True).
    """
    if "nodal_fields" not in result:
        if "fields" in result:
            raise ValueError(
                "this result carries raw HCurl dof vectors but no nodal fields; "
                "it was probably produced by an older call. Re-run "
                "solve_cavity_3d(..., keep_fields=True).")
        raise ValueError("no fields in result: call solve_cavity_3d(..., "
                         "keep_fields=True)")
    if min_localisation is not None:
        cand = [j for j, m in enumerate(result["modes"])
                if m["localisation"] >= min_localisation] or \
               list(range(len(result["modes"])))
        i = max(cand, key=lambda j: result["modes"][j]["C"])
    return result["nodal_fields"][int(i)]


# ─────────────────────────────────────────────────────────────────────────────
# batch (process-parallel)
# ─────────────────────────────────────────────────────────────────────────────

def _worker(args):
    spec, n_modes, f_target, keep_fields, deflate, kw = args
    try:
        out = solve_cavity_3d(spec, n_modes=n_modes, f_target=f_target,
                              keep_fields=keep_fields, deflate=deflate, **kw)
        for k in [k for k in out if k.startswith("_")]:
            out.pop(k)                    # live NGSolve handles are not picklable
        return {"ok": True, **out}
    except Exception as e:
        return {"ok": False, "tag": spec.tag, "error": f"{type(e).__name__}: {e}"}


def run_batch(specs, n_modes: int = 6, f_target=None, n_workers: int | None = None,
              timeout: float | None = None, verbose: bool = True,
              keep_fields: bool = False, deflate: bool = True, **kw):
    """
    Solve many 3D configurations in parallel, one process per configuration.
    Extra keyword arguments (order, mesh_order, axis, linear_solver, ...) are
    passed through to solve_cavity_3d.

    f_target : scalar (same shift for every spec) or one per spec.

    MEMORY, not cores, is usually the binding constraint in 3D: each worker holds
    its own mesh, sparse factorisation and fill-in, and at HCurl order 2 there
    are ~9x the dofs of the old lowest-order element for the same mesh. A 3D
    factorisation can be tens of times the size of the matrix, so
    n_workers = cpu_count will thrash on anything but a coarse mesh. Start at 2-4
    and watch the resident size. Set MKL_NUM_THREADS=1 as well, or PARDISO in
    each worker will try to use every core.

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
    payload = [(s, n_modes, t, keep_fields, deflate, kw)
               for s, t in zip(specs, targets)]
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
              verbose: bool = True, **kw):
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
                        keep_fields=keep_fields, verbose=verbose, **kw)
    return specs, results