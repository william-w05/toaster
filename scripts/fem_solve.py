"""
2D TM eigenmode solver for rectangle-built microwave cavities.

Open-source replacement for the COMSOL eigenfrequency step, built on
  gmsh  (OpenCASCADE kernel)  -> geometry with boolean subtraction + meshing
  scikit-fem                  -> FEM assembly
  scipy                       -> shift-invert eigensolve

WHY THIS IS A SCALAR PROBLEM
    For a 2D cross-section with no z-dependence, the TM modes have E = E_z(x,y) z_hat.
    Maxwell then collapses to a scalar Helmholtz eigenproblem

        -div( (1/mu_r) grad E_z ) = k0^2 eps_r E_z ,      E_z = 0 on PEC

    so no Nedelec/edge elements are needed and there are no spurious modes.
    Discretised:  K u = k0^2 M u  with
        K = sum_regions (1/mu_r) * integral(grad u . grad v)
        M = sum_regions   eps_r  * integral(u v)

BOOLEAN GEOMETRY ("setminus")
    Metal inclusions are CUT out of the cavity (gmsh.model.occ.cut) -> they leave
    holes whose boundaries carry PEC/impedance conditions. Dielectric inclusions are
    FRAGMENTED into the domain (gmsh.model.occ.fragment) -> they stay part of the
    mesh as separate material regions with a conformal interface.

LOSS / Q
    A surface-impedance boundary makes the exact eigenproblem quadratic (and, since
    Z_s depends on omega, nonlinear). We do what is standard for high-Q cavities:
    solve the lossless problem, then get Q perturbatively from the wall currents,

        Q = omega * U / P_loss,
        U      = (eps0/2) * integral(eps_r |E_z|^2 dA)          (total stored energy)
        P_loss = (R_s/2)  * contour_integral(|H_t|^2 ds),
        |H_t|  = |dE_z/dn| / (omega mu0 mu_r),
        R_s    = Re(Z_s) = sqrt(omega mu0 / (2 sigma))          (good conductor)

    Different boundary groups may carry different metals, so each named boundary
    gets its own sigma.

PARALLELISM
    The workload is many small INDEPENDENT solves, so it parallelises across
    processes, not across GPU threads: run_batch() uses multiprocessing, one
    geometry per worker. gmsh is not thread-safe but is fine in separate processes
    (each worker calls initialize/finalize itself).
"""

from __future__ import annotations

import os
import io
import uuid
import tempfile
import contextlib
import numpy as np
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed

import gmsh
import skfem
from skfem import (Basis, FacetBasis, ElementTriP1, ElementTriP2,
                   BilinearForm, Functional, asm)
from skfem.helpers import dot, grad
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

# physical constants (SI)
C0   = 299792458.0
MU0  = 4.0e-7 * np.pi
EPS0 = 1.0 / (MU0 * C0**2)

# meshio.read() emits a stray blank line on every call. With one mesh read per
# tuning step per objective evaluation that floods the console, so mesh I/O is
# silenced by default. Set cavity2d.QUIET = False to see it again.
QUIET = True


@contextlib.contextmanager
def _quiet(enabled=None):
    """Swallow stdout (only) for the duration. stderr is left alone so genuine
    warnings and tracebacks still reach you."""
    if enabled is None:
        enabled = QUIET
    if not enabled:
        yield
        return
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def tmp_msh_path(prefix="cavity"):
    """Portable scratch .msh path. Uses the platform temp dir (%TEMP% on Windows,
    /tmp on Unix) instead of a hardcoded POSIX path, and includes the pid plus a
    uuid so concurrent workers never collide."""
    return os.path.join(tempfile.gettempdir(),
                        f"{prefix}_{os.getpid()}_{uuid.uuid4().hex[:8]}.msh")


# gmsh getBoundingBox() inflates the returned box by this much (absolute, metres).
# Any geometric classification tolerance must be comfortably larger than it.
_BBOX_TOL = 1e-6

# Conductivities (S/m). Q ~ sqrt(sigma), so a small sigma mismatch shows up as a
# small-but-visible Q offset when cross-checking against another code. The
# _COMSOL values are what COMSOL's built-in material library uses; match them if
# you are comparing against a COMSOL model that picked materials from that library.
SIGMA_COPPER = 5.8e7          # generic handbook copper
SIGMA_AL     = 3.5e7          # generic handbook aluminium
SIGMA_COPPER_COMSOL = 5.998e7 # COMSOL material library "Copper"
SIGMA_AL_COMSOL     = 3.774e7 # COMSOL material library "Aluminum"
#   3.774e7 / 3.5e7 = 1.0783 -> sqrt = 1.0384, i.e. COMSOL's aluminium gives Q
#   about 3.8% higher (a few hundred at Q ~ 1e4) purely from the constant.


# ─────────────────────────────────────────────────────────────────────────────
# specification
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Material:
    """eps_r/mu_r are used in the volume forms; sigma only for boundary loss."""
    name: str
    eps_r: float = 1.0
    mu_r: float = 1.0
    sigma: float = SIGMA_COPPER


@dataclass
class Rect:
    """Axis-aligned rectangle, lower-left corner + size, in METRES."""
    x0: float
    y0: float
    w: float
    h: float
    name: str = "rect"

    def as_tuple(self):
        """gmsh occ.addRectangle signature: (x, y, z, dx, dy)."""
        return (self.x0, self.y0, 0.0, self.w, self.h)

    # ---- centre-based interface (usually easier to verify by eye) -------------
    @classmethod
    def from_center(cls, cx, cy, w, h, name="rect"):
        """Rectangle from its CENTRE and size. Rect.from_center(0, 0, w, h) is
        centred on the origin, which matches how a symmetric cavity is described."""
        return cls(cx - 0.5 * w, cy - 0.5 * h, w, h, name)

    @property
    def cx(self):
        return self.x0 + 0.5 * self.w

    @property
    def cy(self):
        return self.y0 + 0.5 * self.h

    @property
    def center(self):
        return (self.cx, self.cy)

    @property
    def bounds(self):
        """(xmin, ymin, xmax, ymax)"""
        return (self.x0, self.y0, self.x0 + self.w, self.y0 + self.h)

    def moved_to(self, cx, cy):
        """Copy translated so its centre is at (cx, cy) -- handy for tuning sweeps."""
        return Rect.from_center(cx, cy, self.w, self.h, self.name)

    def shifted(self, dx=0.0, dy=0.0):
        return Rect(self.x0 + dx, self.y0 + dy, self.w, self.h, self.name)


def CRect(cx, cy, w, h, name="rect"):
    """Shorthand alias for Rect.from_center(...)."""
    return Rect.from_center(cx, cy, w, h, name)


@dataclass
class CavitySpec:
    """
    outer       : the cavity cross-section.
    metal       : rectangles CUT OUT of the cavity (setminus). Their walls become
                  boundary condition surfaces. This is where your toasts/dividers go.
    dielectric  : (Rect, Material) kept in the mesh as distinct material regions.
    wall_material / metal_material : conductors used for the Q surface integral on
                  the outer wall and on the cut-out boundaries respectively.
    background  : material filling the rest of the cavity (vacuum by default).
    mesh_size   : target element size (m). mesh_size_min defaults to mesh_size/3.
    """
    outer: Rect
    metal: list = field(default_factory=list)
    dielectric: list = field(default_factory=list)
    background: Material = field(default_factory=lambda: Material("vacuum"))
    wall_material: Material = field(default_factory=lambda: Material("cu"))
    metal_material: Material = field(default_factory=lambda: Material("cu"))
    mesh_size: float = 0.004
    mesh_size_min: float | None = None
    mesh_uniform: bool = False   # see build_mesh: pin every element to mesh_size
    tag: str = ""            # free-form label carried through to the results

    # ---- outer-shape hooks: everything else in build_mesh is shape-agnostic --
    def add_outer(self, occ):
        """Create the outer surface and return its tag."""
        return occ.addRectangle(*self.outer.as_tuple())

    def on_wall(self, pts):
        """True if every sampled point of a curve lies on the OUTER boundary."""
        x0, y0, x1, y1 = self.outer.bounds
        tol = _BBOX_TOL + 1e-6 * max(self.outer.w, self.outer.h)
        x, y = pts[:, 0], pts[:, 1]
        return bool(np.all(np.abs(x - x0) < tol) or np.all(np.abs(x - x1) < tol) or
                    np.all(np.abs(y - y0) < tol) or np.all(np.abs(y - y1) < tol))

    @property
    def extent(self):
        """(xmin, ymin, xmax, ymax) of the outer shape, for plotting."""
        return self.outer.bounds


@dataclass
class CylSpec:
    """
    Circular cross-section (an infinite cylinder in 2D). Interface-compatible with
    CavitySpec, so solve_cavity / run_batch / plot_mesh take it unchanged.

    Useful because the TM modes have closed forms -- see cylinder_analytic() --
    which makes it the natural end-to-end check on f, C and Q together.

    radius : metres.
    metal / dielectric : optional Rect inclusions, exactly as in CavitySpec.
    """
    radius: float
    center: tuple = (0.0, 0.0)
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
        cx, cy = self.center
        return occ.addDisk(cx, cy, 0.0, self.radius, self.radius)

    def on_wall(self, pts):
        cx, cy = self.center
        r = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        tol = _BBOX_TOL + 1e-6 * self.radius
        return bool(np.all(np.abs(r - self.radius) < tol))

    @property
    def extent(self):
        cx, cy = self.center
        R = self.radius
        return (cx - R, cy - R, cx + R, cy + R)

@dataclass
class HalfPipeSpec:
    """
    Half-disk cross-section (an infinite half-cylinder in 2D): the disk of `radius`
    about `center`, cut along the diameter y = cy. Interface-compatible with
    CavitySpec, so solve_cavity / run_batch / plot_mesh take it unchanged.
 
    upper : True keeps y >= cy, False keeps y <= cy (a trough). The flat face is a
        conducting wall either way, so it joins the "wall" boundary group.
 
    Also a clean validation case: with PEC on both the arc and the diameter the
    modes are exactly the sin(m theta) branch of the full disk,
 
        f_mn = c * j_{m,n} / (2 pi R),   m >= 1, and NON-degenerate
 
    -- the cos branch is killed by E_z = 0 on the diameter, which is why CylSpec's
    doublets collapse to single modes here. See halfpipe_analytic().
 
    radius : metres.
    metal / dielectric : optional Rect inclusions, exactly as in CavitySpec.
    """
    radius: float
    center: tuple = (0.0, 0.0)
    upper: bool = True
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
        """
        Must return a SURFACE tag -- build_mesh does dom = [(2, outer)].
 
        Build the disk, then cut away the unwanted half with an oversized
        rectangle. The cutter extends 1.5R past the disk in x so the shapes
        properly overlap rather than meeting tangentially at a corner, and tags are
        left for gmsh to assign rather than hardcoded (hardcoded tags collide with
        the metal rectangles added afterwards).
        """
        cx, cy = self.center
        R = self.radius
        disk = occ.addDisk(cx, cy, 0.0, R, R)
        y0 = (cy - 1.5 * R) if self.upper else cy      # the half to REMOVE
        cutter = occ.addRectangle(cx - 1.5 * R, y0, 0.0, 3.0 * R, 1.5 * R)
        out, _ = occ.cut([(2, disk)], [(2, cutter)],
                         removeObject=True, removeTool=True)
        surfs = [t for (d, t) in out if d == 2]
        if len(surfs) != 1:
            raise ValueError(f"half-pipe cut produced {len(surfs)} surfaces, "
                             f"expected 1 (radius={R}, center={self.center})")
        return surfs[0]
 
    def on_wall(self, pts):
        """The boundary is the arc PLUS the flat diameter; both are conducting."""
        cx, cy = self.center
        x, y = pts[:, 0], pts[:, 1]
        r = np.hypot(x - cx, y - cy)
        tol = _BBOX_TOL + 1e-6 * self.radius
        on_arc = np.all(np.abs(r - self.radius) < tol)
        # flat face: on the line y = cy AND inside the disk, so an interior metal
        # edge that happens to sit at y = cy is not swept up as outer wall
        on_flat = (np.all(np.abs(y - cy) < tol) and
                   np.all(r <= self.radius + tol))
        return bool(on_arc or on_flat)
 
    @property
    def extent(self):
        cx, cy = self.center
        R = self.radius
        return ((cx - R, cy, cx + R, cy + R) if self.upper
                else (cx - R, cy - R, cx + R, cy))

# ─────────────────────────────────────────────────────────────────────────────
# geometry + mesh
# ─────────────────────────────────────────────────────────────────────────────

def _curve_samples(tag, n=7):
    """n points spread along a curve, as an (n,3) array, for shape-agnostic
    boundary classification."""
    lo, hi = gmsh.model.getParametrizationBounds(1, tag)
    ts = np.linspace(float(lo[0]), float(hi[0]), n)
    return np.asarray(gmsh.model.getValue(1, tag, ts),
                      dtype=np.float64).reshape(-1, 3)


def cylinder_analytic(radius, sigma=SIGMA_COPPER, n=1, eps_r=1.0, mu_r=1.0):
    """
    Closed-form TM_0n of an empty circular cross-section -- the reference for
    validating the solver on all three outputs at once.

        k     = j_{0,n} / R,           f = c k / (2 pi sqrt(eps_r mu_r))
        C     = 4 / j_{0,n}^2          (0.6917 for TM_01: the classic value)
        Q     = mu0 c j_{0,n} / (2 R_s),   R_s = sqrt(omega mu0 / (2 sigma))

    C follows from  int J0(kr) dA = 2 pi R^2 J1(j)/j  and  int J0^2 dA = pi R^2 J1(j)^2.
    """
    from scipy.special import jn_zeros
    j0n = float(jn_zeros(0, n)[-1])
    k = j0n / radius
    f = C0 * k / (2.0 * np.pi * np.sqrt(eps_r * mu_r))
    omega = 2.0 * np.pi * f
    R_s = np.sqrt(omega * MU0 / (2.0 * sigma))
    return {"f": f, "C": 4.0 / j0n**2, "Q": MU0 * C0 * j0n / (2.0 * R_s),
            "j0n": j0n, "k": k, "R_s": R_s}


def build_mesh(spec, msh_path: str, verbose: bool = False):
    """
    Build the geometry with OCC booleans and write a .msh.

    Works for any spec that provides add_outer(occ) and on_wall(pts) --
    CavitySpec (rectangle) and CylSpec (disk) both do.

    Physical groups created:
      surfaces : "background", and one per dielectric ("diel_0", "diel_1", ...)
      curves   : "wall"  -> outer boundary
                 "metal" -> boundaries of the cut-out (setminus) rectangles
    Returns the list of dielectric material objects, ordered to match "diel_i".
    """
    d = os.path.dirname(os.path.abspath(msh_path))
    if d:
        os.makedirs(d, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.model.add("cavity")
        occ = gmsh.model.occ

        outer = spec.add_outer(occ)        # rectangle, disk, ...
        dom = [(2, outer)]

        # ---- setminus: cut the metal rectangles out of the cavity -----------
        if spec.metal:
            tools = [(2, occ.addRectangle(*r.as_tuple())) for r in spec.metal]
            dom, _ = occ.cut(dom, tools, removeObject=True, removeTool=True)
            if not dom:
                raise ValueError("cut() removed the entire domain: the metal "
                                 "rectangles cover the whole cavity.")

        # ---- embed dielectrics as conformal sub-regions ---------------------
        diel_mats, diel_tags = [], []
        if spec.dielectric:
            tools = []
            for r, mat in spec.dielectric:
                tools.append((2, occ.addRectangle(*r.as_tuple())))
                diel_mats.append(mat)
            frag, _ = occ.fragment(dom, tools)
            occ.synchronize()
            # Identify which fragments are the dielectric rectangles by BOUNDING-BOX
            # CONTAINMENT, not centre of mass: after fragmenting, the background is a
            # frame-shaped (non-convex) region whose centroid can land inside a
            # dielectric rectangle and be misclassified. A fragment's bbox is
            # contained in rect i's bbox only if it really is that sub-region.
            dom, diel_tags = [], [[] for _ in spec.dielectric]
            for (d, t) in frag:
                if d != 2:
                    continue
                bb = gmsh.model.getBoundingBox(2, t)
                placed = False
                for i, (r, _m) in enumerate(spec.dielectric):
                    # gmsh's getBoundingBox pads the box by ~1e-7 (absolute), so the
                    # tolerance MUST exceed that or every fragment is rejected.
                    tol = _BBOX_TOL + 1e-6 * max(r.w, r.h)
                    if (bb[0] >= r.x0 - tol and bb[1] >= r.y0 - tol and
                            bb[3] <= r.x0 + r.w + tol and bb[4] <= r.y0 + r.h + tol):
                        diel_tags[i].append(t); placed = True; break
                if not placed:
                    dom.append((2, t))
        occ.synchronize()

        # ---- physical groups -------------------------------------------------
        bg_tags = [t for (d, t) in dom if d == 2]
        if not bg_tags:
            raise ValueError("no background region left after boolean ops.")
        g = gmsh.model.addPhysicalGroup(2, bg_tags)
        gmsh.model.setPhysicalName(2, g, "background")
        for i, tags in enumerate(diel_tags):
            if tags:
                g = gmsh.model.addPhysicalGroup(2, tags)
                gmsh.model.setPhysicalName(2, g, f"diel_{i}")

        # Boundaries: outer wall vs cut-out (metal) walls. Classified by SAMPLING
        # POINTS ALONG EACH CURVE and asking the spec whether they lie on its outer
        # boundary. A bounding-box test cannot do this for a disk (an arc's bbox is
        # not the disk's), whereas point sampling is shape-agnostic.
        all_surf = bg_tags + [t for tags in diel_tags for t in tags]
        bnd = gmsh.model.getBoundary([(2, t) for t in all_surf],
                                     combined=True, oriented=False)
        wall, metal = [], []
        for (d, t) in bnd:
            if d != 1:
                continue
            (wall if spec.on_wall(_curve_samples(t)) else metal).append(t)
        if wall:
            g = gmsh.model.addPhysicalGroup(1, wall); gmsh.model.setPhysicalName(1, g, "wall")
        if metal:
            g = gmsh.model.addPhysicalGroup(1, metal); gmsh.model.setPhysicalName(1, g, "metal")

        # ---- mesh ------------------------------------------------------------
        # Two regimes:
        #
        #  mesh_uniform=False (default): elements may shrink to mesh_size_min
        #    (default mesh_size/3) near small geometric features. gmsh does this
        #    through MeshSizeExtendFromBoundary and MeshSizeFromPoints, both on by
        #    default. Better accuracy per element, but the element size is not a
        #    single number, so it is NOT directly comparable with another code.
        #
        #  mesh_uniform=True: pin min = max = mesh_size and switch off the
        #    feature-driven refinement, giving a near-uniform mesh. This is the
        #    setting to use when matching COMSOL with min element size = max
        #    element size, where COMSOL's growth rate / curvature factor /
        #    narrow-region resolution are all inoperative because the size is
        #    clamped from both sides.
        if spec.mesh_uniform:
            gmsh.option.setNumber("Mesh.MeshSizeMax", spec.mesh_size)
            gmsh.option.setNumber("Mesh.MeshSizeMin", spec.mesh_size)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        else:
            hmin = spec.mesh_size_min if spec.mesh_size_min else spec.mesh_size / 3.0
            gmsh.option.setNumber("Mesh.MeshSizeMax", spec.mesh_size)
            gmsh.option.setNumber("Mesh.MeshSizeMin", hmin)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)          # frontal-Delaunay
        gmsh.model.mesh.generate(2)
        gmsh.write(msh_path)
    finally:
        gmsh.finalize()
    return diel_mats


# ─────────────────────────────────────────────────────────────────────────────
# FEM forms
# ─────────────────────────────────────────────────────────────────────────────

@BilinearForm
def _stiff(u, v, w):
    return dot(grad(u), grad(v))

@BilinearForm
def _mass(u, v, w):
    return u * v


def _assemble(mesh, element, spec, diel_mats):
    """K u = k0^2 M u, assembled region by region so each carries its own eps/mu."""
    basis = Basis(mesh, element)
    K = asm(_stiff, Basis(mesh, element, elements=mesh.subdomains["background"])) \
        * (1.0 / spec.background.mu_r)
    M = asm(_mass, Basis(mesh, element, elements=mesh.subdomains["background"])) \
        * spec.background.eps_r
    for i, mat in enumerate(diel_mats):
        key = f"diel_{i}"
        if key not in mesh.subdomains:
            continue
        b = Basis(mesh, element, elements=mesh.subdomains[key])
        K = K + asm(_stiff, b) * (1.0 / mat.mu_r)
        M = M + asm(_mass, b) * mat.eps_r
    return basis, K, M


# ─────────────────────────────────────────────────────────────────────────────
# observables
# ─────────────────────────────────────────────────────────────────────────────

@Functional
def _int_u(w):        return w["uh"]
@Functional
def _int_eps_u2(w):   return w["eps"] * w["uh"] ** 2
@Functional
def _int_u2(w):       return w["uh"] ** 2
@Functional
def _int_u4(w):       return w["uh"] ** 4
@Functional
def _area(w):         return 1.0 + 0.0 * w["uh"]
@Functional
def _dudn2(w):        return dot(w["uh"].grad, w.n) ** 2


def _observables(mesh, element, basis, u, k0, spec, diel_mats):
    """Form factor C, quality factor Q, areas, and a localisation diagnostic."""
    # region-weighted integrals (eps_r differs per region)
    num = den = area = l2 = l4 = 0.0
    regions = [("background", spec.background)] + \
              [(f"diel_{i}", m) for i, m in enumerate(diel_mats)]
    for key, mat in regions:
        if key not in mesh.subdomains:
            continue
        b = Basis(mesh, element, elements=mesh.subdomains[key])
        uh = b.interpolate(u)
        num  += _int_u.assemble(b, uh=uh)
        den  += _int_eps_u2.assemble(b, uh=uh, eps=mat.eps_r)
        area += _area.assemble(b, uh=uh)
        l2   += _int_u2.assemble(b, uh=uh)
        l4   += _int_u4.assemble(b, uh=uh)

    C = (num ** 2) / (area * den) if area > 0 and den > 0 else 0.0

    # participation area: (int|E|^2)^2 / int|E|^4. Equals the area for a uniform
    # field and collapses for a localised one -> use it to reject localised modes.
    A_part = (l2 ** 2) / l4 if l4 > 0 else 0.0

    # Q from wall currents, each boundary group with its own conductor
    omega = C0 * k0
    P = 0.0
    for key, mat in (("wall", spec.wall_material), ("metal", spec.metal_material)):
        if key not in mesh.boundaries:
            continue
        fb = FacetBasis(mesh, element, facets=mesh.boundaries[key])
        g2 = _dudn2.assemble(fb, uh=fb.interpolate(u))
        R_s = np.sqrt(omega * MU0 / (2.0 * mat.sigma))
        P += 0.5 * R_s * g2 / (omega * MU0) ** 2
    U = 0.5 * EPS0 * den
    Q = (omega * U / P) if P > 0 else np.inf

    return dict(C=float(C), Q=float(Q), area=float(area),
                A_part=float(A_part), localisation=float(A_part / area) if area else 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# single solve
# ─────────────────────────────────────────────────────────────────────────────

def solve_cavity(spec, n_modes: int = 6, f_target: float | None = None,
                 order: int = 2, msh_path: str | None = None, verbose: bool = False,
                 keep_fields: bool = False):
    """
    Build -> mesh -> solve one configuration.

    f_target : Hz. If given, the shift-invert eigensolve targets modes near it
               (the analogue of your COMSOL FreqGuess). Otherwise the lowest modes.
    keep_fields : also return the mesh, basis and nodal E_z of every mode, so the
               field can be plotted (see cavity_viz.plot_modes). Off by default --
               it makes the result far larger, which matters when batching.

    Returns dict with 'freqs' (Hz) and per-mode C / Q / A_part, sorted by frequency.
    """
    tmp = msh_path or tmp_msh_path("cavity")
    with _quiet(QUIET and not verbose):   # QUIET=False or verbose=True -> show it
        diel_mats = build_mesh(spec, tmp, verbose=verbose)
        mesh = skfem.Mesh.load(tmp)
    if msh_path is None:                     # only clean up files we created
        try:
            os.remove(tmp)
        except OSError:
            pass
    element = ElementTriP2() if order == 2 else ElementTriP1()

    basis, K, M = _assemble(mesh, element, spec, diel_mats)

    # PEC (E_z = 0) on every conducting boundary
    facets = np.concatenate([mesh.boundaries[k] for k in ("wall", "metal")
                             if k in mesh.boundaries])
    D = basis.get_dofs(facets=facets)
    I = basis.complement_dofs(D)
    Kc, Mc = K[I][:, I], M[I][:, I]

    sigma_shift = (2 * np.pi * f_target / C0) ** 2 if f_target else 0.0
    vals, vecs = eigsh(Kc, k=n_modes, M=Mc, sigma=sigma_shift, which="LM")

    order_idx = np.argsort(np.abs(vals))
    out = {"freqs": [], "modes": [], "n_dofs": int(basis.N),
           "n_elements": int(mesh.t.shape[1]), "tag": spec.tag}
    for idx in order_idx:
        lam = float(np.real(vals[idx]))
        if lam <= 0:
            continue
        k0 = np.sqrt(lam)
        u = np.zeros(basis.N)
        u[I] = np.real(vecs[:, idx])
        nrm = np.max(np.abs(u))
        if nrm > 0:
            u = u / nrm
        obs = _observables(mesh, element, basis, u, k0, spec, diel_mats)
        out["freqs"].append(C0 * k0 / (2 * np.pi))
        out["modes"].append(obs)
        if keep_fields:
            out.setdefault("fields", []).append(u.copy())
    if keep_fields:
        out["mesh"] = mesh
        out["basis"] = basis
    return out


def best_mode(result, min_localisation: float = 0.0):
    """
    Pick the operating mode: highest C among modes that are not localised.

    min_localisation is A_part/area; a delocalised fundamental sits near ~0.5-0.7,
    a corner-localised mode much lower. Set it > 0 to refuse localised modes
    instead of letting argmax(C) select them.
    """
    cand = [(f, m) for f, m in zip(result["freqs"], result["modes"])
            if m["localisation"] >= min_localisation]
    if not cand:
        return None
    f, m = max(cand, key=lambda fm: fm[1]["C"])
    return {"f": f, **m}


# ─────────────────────────────────────────────────────────────────────────────
# batch (process-parallel)
# ─────────────────────────────────────────────────────────────────────────────

def _worker(args):
    spec, n_modes, f_target, order, keep_fields = args
    try:
        out = solve_cavity(spec, n_modes, f_target, order, keep_fields=keep_fields)
        out.pop("basis", None)      # not picklable / not needed for plotting
        return {"ok": True, **out}
    except Exception as e:
        return {"ok": False, "tag": spec.tag, "error": f"{type(e).__name__}: {e}"}


def run_batch(specs, n_modes: int = 6, f_target=None,
              order: int = 2, n_workers: int | None = None,
              timeout: float | None = None, verbose: bool = True,
              keep_fields: bool = False):
    """
    Solve many configurations in parallel, one process per configuration.

    f_target : Hz. EITHER a scalar (same shift for every spec) OR a sequence with
        one target per spec -- which is what a tuning sweep needs, since each step
        resonates at a different frequency.

        WITH NO TARGET the shift-invert returns the LOWEST modes of each geometry.
        For a multi-cell cavity those are not the operating mode: they are the
        low-frequency modes of the channels above/below the bars, and they come out
        at a similar frequency for every tuning step (which looks like the sweep
        doing nothing). Always pass the analytic guess, exactly as you set
        FreqGuess in COMSOL.

    n_workers : defaults to os.cpu_count(). Each worker holds its own gmsh session.
    timeout   : seconds per future; a hung/pathological geometry is reported as a
                failure instead of stalling the batch (the analogue of the COMSOL
                hang problem -- here the worker is a separate, killable process).

    Returns a list of result dicts in the same order as `specs`.
    """
    n_workers = n_workers or os.cpu_count() or 1
    if f_target is None or np.isscalar(f_target):
        targets = [f_target] * len(specs)
    else:
        targets = list(f_target)
        if len(targets) != len(specs):
            raise ValueError(f"f_target has {len(targets)} entries but there are "
                             f"{len(specs)} specs; pass one target per spec or a scalar.")
    if all(t is None for t in targets) and verbose:
        print("[batch] WARNING: no f_target -- returning the LOWEST modes of each "
              "geometry, which for a multi-cell cavity are channel modes, not the "
              "operating mode.", flush=True)
    payload = [(s, n_modes, t, order, keep_fields) for s, t in zip(specs, targets)]
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
                print(f"[batch] {done}/{len(specs)} tag={r.get('tag','')!r} {msg}",
                      flush=True)
    return results

def plot_modes_square_magnitude(spec, result, n=None, save=None, cmap="RdBu_r"):
    """
    Duplicate of plot_modes_square magnitude in fem_vis to avoid circular import.
    """
    MM = 1000.0
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
        #lim = np.max(val) or 1.0
        tp = ax.tripcolor(m.p[0] * MM, m.p[1] * MM, m.t.T, val,
                            cmap=cmap, shading="gouraud")
        for nm, col in (("wall", "#111111"), ("metal", "#111111")):
            if nm in m.boundaries:
                f = m.facets[:, m.boundaries[nm]]
                ax.plot(m.p[0][f] * MM, m.p[1][f] * MM, color=col, lw=1.0)
        md = result["modes"][i]
        ax.set_title(f"f={result['freqs'][i]/1e9:.4f} GHz   C={md['C']:.3f}\n"
                        f"Q={md['Q']:.3g}   loc={md['localisation']:.3f}", fontsize=9)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(tp, ax=ax, fraction=0.035)
    for j in range(nmodes, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"E_z modes: {spec.tag or ''}", y=1.0)
    if save:
        fig.tight_layout(); fig.savefig(save, dpi=140); plt.close(fig)
    return fig


def run_sweep(spec_fn, positions, n_modes: int = 6, order: int = 2,
              n_workers: int | None = None, timeout: float | None = None,
              verbose: bool = True, plot_all: bool = False):
    """
    Parallel tuning sweep -- the usual reason a run feels slow is that the steps
    are done one at a time on one core while the rest of the machine sits idle.

    spec_fn(dx, dy, i) -> CavitySpec for tuning position i.
    positions          -> iterable of (dx, dy, f_guess); f_guess becomes that
                          step's shift-invert target (REQUIRED to land on the
                          operating mode rather than the lowest channel modes).

    Returns (specs, results), results in position order.
    """
    pos = list(positions)
    specs = [spec_fn(dx, dy, i) for i, (dx, dy, _f) in enumerate(pos)]
    guesses = [f for _dx, _dy, f in pos]

    results = run_batch(specs, n_modes=n_modes, f_target=guesses, order=order,
                        n_workers=n_workers, timeout=timeout, verbose=verbose,
                        keep_fields=plot_all)

    if plot_all:
        for i, spec in enumerate(specs):
            plot_modes_square_magnitude(spec, results[i], save=f"results/08_11_2026_stability_nominal_tuning_{i+1}_pos_mode.png")
    
    return specs, results