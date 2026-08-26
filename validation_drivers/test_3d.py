"""
Two worked examples for the 3D solver.

  A) EXTRUDE, VISUALISE AND SOLVE a 7-parameter toaster geometry.
  B) VISUALISE AND SOLVE build_files/Hollow-cylinder.step.

Run either from VS Code with:   python examples_3d.py A     (or B, or both)

READ THIS FIRST, it will save you a night:

  * For a PRISM -- which the extruded toaster is -- you almost certainly do NOT
    want the 3D solve. The p = 0 mode has f and C identical to the 2D answer, and
    Q follows exactly from  1/Q_3D = 1/Q_2D + 2 R_s/(mu0 c k0 L).  Measured, the
    extrusion formula lands within 0.2-0.4% of analytic while a 3D mesh you can
    actually afford was 4-10% out in Q. Example A therefore does the cheap exact
    calculation FIRST and treats the 3D solve as a cross-check.
    The 3D solve earns its keep only when the geometry is NOT a prism: a rod that
    does not span the full length, endcap features, or an imported part.

  * A spectrum where every mode has C ~ 0 means the frequency target is wrong,
    not that the code is broken. In a prism only p = 0 has nonzero form factor.
"""

import sys
import numpy as np

from scripts import fem_solve as f2          # or: import fem_solve as f2
from scripts import fem_vis as v2
from scripts_3d import fem_solve_3d as f3
from scripts_3d import fem_vis_3d as v3
from scripts import mcmc


# ═════════════════════════════════════════════════════════════════════════════
# A) the 7-parameter toaster: extrude -> visualise -> solve
# ═════════════════════════════════════════════════════════════════════════════

def example_A(length_m=0.20, mesh_size=0.004, outdir="TEMP/ex3d",
              run_3d=True, n_modes=6):
    """
    params_mm is the usual DESIGN VECTOR:
        angle, div_h, div_w, ctr_w, side_w, ctr_h, side_h
    gap0 and gap1 are fixed at 10 mm and are not in the vector.
    """
    params_mm = [9.71851258, 8.0, 127.23060036, 14.4991172,
                 15.65917039, 124.90989748, 123.52196735]
    # NOTE the order: angle, div_h, div_w, ctr_w, side_w, ctr_h, side_h.
    # Written out explicitly so it cannot be got backwards:
    params_mm = [
        9.71851258,      # angle   (deg)
        127.23060036,    # div_h   (mm)
        8.0,             # div_w   (mm)  -- must be < gap0 = 10
        14.4991172,      # ctr_w   (mm)
        15.65917039,     # side_w  (mm)
        124.90989748,    # ctr_h   (mm)
        123.52196735,    # side_h  (mm)
    ]

    # ---- 1. the 2D cross-section, exactly as you already build it ------------
    params_m = mcmc._params_to_m(params_mm)          # THE mm -> m boundary
    spec2d = v2.toaster_spec(params_m, gap0=mcmc.GAP0_M, gap1=mcmc.GAP1_M,
                             cavity_h=mcmc.CAV_H_M, mesh_size=mcmc.MESH_SIZE,
                             wall_material=mcmc.ALUMINIUM,
                             metal_material=mcmc.ALUMINIUM, tag="toaster")

    # ---- 2. solve it in 2D (cheap, accurate, and the frequency you need) -----
    f_guess = 3e8 / (2.0 * mcmc.GAP0_M)              # ~15 GHz at x = 0
    r2 = f2.solve_cavity(spec2d, n_modes=6, f_target=f_guess, keep_fields=True)
    m2 = f2.best_mode(r2)
    print(f"[2D]  f = {m2['f']/1e9:.4f} GHz   C = {m2['C']:.4f}   "
          f"Q = {m2['Q']:.0f}")

    # ---- 3. THE 3D ANSWER, exactly, with no 3D mesh --------------------------
    sigma = mcmc.ALUMINIUM.sigma
    modes = f3.extruded_modes(r2, length_m, sigma=sigma, p_max=2)
    print(f"[ext] prism of length {length_m*1e3:.0f} mm, exact relations:")
    for e in modes:
        print(f"        p={e['p']}  f = {e['f']/1e9:8.4f} GHz   "
              f"C = {e['C']:.4f}   Q = {e['Q'] if np.isfinite(e['Q']) else float('nan'):.0f}")
    print(f"        (p=0 f and C are the 2D values; Q dropped from "
          f"{m2['Q']:.0f} to {modes[0]['Q']:.0f} purely from endcap loss)")

    # how Q depends on the length you have not chosen yet
    v3.plot_length_scan(r2, np.linspace(0.05, 0.60, 24), sigma=sigma,
                        save=f"{outdir}/length_scan.png")

    # ---- 4. extrude to a 3D spec --------------------------------------------
    spec3d = f3.from_2d(spec2d, length=length_m, mesh_size=mesh_size)
    v3.check_spec_3d(spec3d)

    # ---- 5. visualise, interactively ----------------------------------------
    v3.plot_spec_3d(spec3d, save=f"{outdir}/toaster_spec.html", show=False)
    v3.plot_mesh_3d(spec3d, save=f"{outdir}/toaster_mesh.html", show=False,
                    clip="y")     # clip the wall away to see the bars inside
    print(f"[vis] wrote {outdir}/toaster_spec.html and toaster_mesh.html "
          f"-- open them in a browser, they rotate and zoom")

    if not run_3d:
        return {"r2": r2, "extruded": modes, "spec3d": spec3d}

    # ---- 6. the 3D solve, as a CROSS-CHECK ----------------------------------
    # Aim at the 2D frequency. Anything else lands among the channel modes and
    # every C comes back 0.
    r3 = f3.solve_cavity_3d(spec3d, n_modes=n_modes, f_target=m2["f"],
                            keep_fields=True)
    m3 = f3.best_mode(r3)
    print(f"[3D]  {r3['n_elements']} tets, {r3['n_free']} dofs")
    print(f"      f = {m3['f']/1e9:.4f} GHz ({(m3['f']/modes[0]['f']-1)*100:+.2f}% "
          f"vs exact)   C = {m3['C']:.4f} "
          f"({(m3['C']/modes[0]['C']-1)*100:+.2f}%)   "
          f"Q = {m3['Q']:.0f} ({(m3['Q']/modes[0]['Q']-1)*100:+.2f}%)")

    # ---- 7. look at the field -----------------------------------------------
    # z-slices are the transverse cross-section: compare directly against 2D.
    # y- and x-slices show the longitudinal structure 2D cannot represent.
    v3.plot_field_slices(r3, save=f"{outdir}/toaster_slices.png")
    v3.plot_modes_3d(r3, n=n_modes, save=f"{outdir}/toaster_modes.png")
    v3.view_field_3d(r3, save=f"{outdir}/toaster_field.html", show=False)
    print(f"[vis] wrote {outdir}/toaster_slices.png, _modes.png, _field.html")
    return {"r2": r2, "extruded": modes, "r3": r3, "spec3d": spec3d}


# ═════════════════════════════════════════════════════════════════════════════
# B) the imported hollow cylinder
# ═════════════════════════════════════════════════════════════════════════════

def example_B(path="build_files/Hollow-cylinder.STEP", f_target=229.5e6, outdir="TEMP/ex3d", n_modes=4):
    """
    The part is the METAL SHELL -- walls and endcaps of finite, varying
    thickness -- so the cavity is the void it encloses, NOT the imported solid.
    That is what interior=True does.

    IGES carries no materials, so the conductor is supplied here.
    """
    AL = f2.Material("aluminium", sigma=f2.SIGMA_AL_COMSOL)

    spec = f3.ImportedSpec3D(path=path,
                         interior=True,
                         # scale="auto" — detected from the file
                         mesh_size=0.3/6,
                         wall_material=AL, metal_material=AL)

    f3.diagnose_import(spec)
    
    # Look at it BEFORE solving: this both meshes the void and shows you which
    # surfaces became the wall. If the "cavity" you see is the shell rather than
    # the hole in it, interior= is the wrong way round.
    v3.plot_mesh_3d(spec, save=f"{outdir}/sample_3d_mesh.html", show=False)
    print(f"[vis] wrote {outdir}/sample_3d_mesh.html")

    r = f3.solve_cavity_3d(spec, n_modes=n_modes, f_target=f_target,
                           keep_fields=True, axis="y", progress=True)
    print(f"[3D]  {r['n_elements']} tets, {r['n_free']} dofs")
    for j, (f, m) in enumerate(zip(r["freqs"], r["modes"])):
        print(f"      mode {j}: f = {f/1e9:8.4f} GHz   C = {m['C']:.4f}   "
              f"Q = {m['Q']:9.0f}   loc = {m['localisation']:.3f}")
    m = f3.best_mode(r)
    print(f"[3D]  operating mode: f = {m['f']/1e9:.4f} GHz   "
          f"C = {m['C']:.4f}   Q = {m['Q']:.0f}")

    # sanity: a uniform cylinder of the mean inner radius, TM010
    #R_eff = 0.0425
    #ref = f2.cylinder_analytic(R_eff, sigma=AL.sigma)
    #print(f"[ref] uniform R = {R_eff*1e3:.1f} mm: f = {ref['f']/1e9:.4f} GHz, "
    #      f"C = {ref['C']:.4f} (the taper and the endcaps move both)")

    v3.plot_field_slices(r, save=f"{outdir}/sample_3d_slices.png")
    v3.view_field_3d(r, save=f"{outdir}/sample_3d_field.html", show=False)
    print(f"[vis] wrote {outdir}/sample_3d_slices.png and sample_3d_field.html")
    return r


if __name__ == "__main__":
    which = (sys.argv[1] if len(sys.argv) > 1 else "both").upper()
    if which in ("A", "BOTH"):
        example_A()
    if which in ("B", "BOTH"):
        example_B()