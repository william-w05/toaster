"""
Two worked examples for the 3D solver.

  A) EXTRUDE, CALIBRATE AND SOLVE a 7-parameter toaster geometry.
  B) VISUALISE AND SOLVE build_files/Hollow-cylinder.step.

Run either from VS Code with:   python examples_3d.py A     (or B, or both)

READ THIS FIRST, it will save you a night:

  * For a PRISM -- which the extruded toaster is -- the EXACT answer needs no 3D
    mesh at all. The p = 0 mode has f and C identical to the 2D solve, and Q
    follows from  1/Q_3D = 1/Q_2D + 2 R_s/(mu0 c k0 L).  Example A computes that
    FIRST and uses it as the reference everything else is judged against.

  * So why solve in 3D? Because the parameters you eventually need to optimise
    are 3D-only (bar length vs cavity length, endcap features, a rod that does
    not reach the endcaps), and for those there is no exact relation. The prism
    case is where you CALIBRATE the 3D mesh against an answer you already know.
    Do that before trusting a single 3D-only number. Step 5 below is that
    calibration and it is the most important part of this file.

  * MESH THE CROSS-SECTION, NOT THE VOLUME. The operating mode is constant along
    z, so an isotropic mesh spends most of its elements resolving a direction in
    which nothing happens. extrude_layers meshes the cross-section in 2D and
    extrudes it, which is what makes the 2D solver's own 1 mm resolution
    affordable in 3D:

        transverse h    isotropic            extruded (layers=2)
        1 mm            11.4M tets (never)   168,900 tets / 1.85M dofs
        2 mm            1.4M tets            43,152 tets / 480k dofs
        3 mm            420k tets / 4.0M     19k tets / 215k dofs

    Those are measured on this cross-section. The extruded mesh is also exactly
    symmetric under z -> -z, so the p = 0 mode cannot hybridise with the p = 1
    neighbour sitting only ~0.2% above it.

  * A spectrum where every mode has C ~ 0 has THREE possible causes and they
    need different fixes. Do not guess -- f3.mode_diagnostics() separates them:
    an axial p >= 1 mode (target too high), a localised channel mode (high Q,
    C = 0), or a detuned multi-cell cluster (transverse cancellation, meaning
    the transverse mesh is too coarse to hold the cells in tune).
"""

import os
#os.environ["MKL_NUM_THREADS"] = "1"

import sys
import copy
import numpy as np

from scripts import fem_solve as f2          # or: import fem_solve as f2
from scripts import fem_vis as v2
from scripts_3d import fem_solve_3d_updated as f3
from scripts_3d import fem_vis_3d as v3
from scripts import mcmc


# ═════════════════════════════════════════════════════════════════════════════
# A) the 7-parameter toaster: extrude -> calibrate -> solve
# ═════════════════════════════════════════════════════════════════════════════

def example_A(length_m=0.16, outdir="TEMP/ex3d", run_3d=True, calibrate=True,
              n_modes=20, layers=4, h_bulk=0.003, h_edge=0.001):
    """
    params_mm is the usual DESIGN VECTOR:
        angle, div_h, div_w, ctr_w, side_w, ctr_h, side_h
    gap0 and gap1 are fixed at 10 mm and are not in the vector.

    layers / h_bulk / h_edge are the 3D mesh knobs that matter:
        layers  structured z layers. 4 while you are still confirming the
                frequency target (few layers misplace the p >= 1 ladder, and
                shift-invert selects on frequency alone, so a misplaced ladder
                hands you the wrong modes); 1-2 once the target is known, since
                the p = 0 mode is constant along z.
        h_bulk  transverse element size away from the bars.
        h_edge  transverse element size AT the bar edges, where the field
                singularity lives and where cell-to-cell detuning is generated.
                Measured on this geometry: h_bulk=3 mm / h_edge=1 mm gives
                ~59k tets and ~665k dofs -- about what the old isotropic
                12 -> 3 mm run cost, but 3x finer everywhere.
    """
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
    r2 = f2.solve_cavity(spec2d, n_modes=n_modes, f_target=f_guess,
                         keep_fields=True)
    m2 = f2.best_mode(r2)
    print(f"[2D]  f = {m2['f']/1e9:.4f} GHz   C = {m2['C']:.4f}   "
          f"Q = {m2['Q']:.0f}   (mesh_size = {mcmc.MESH_SIZE*1e3:.2f} mm)")

    # ---- 3. THE EXACT 3D ANSWER, with no 3D mesh ----------------------------
    # This is the REFERENCE. For a prism it is not an approximation: f and C of
    # the p = 0 mode are the 2D values identically, and Q differs only by the
    # endcap term. Everything below is judged against it.
    sigma = mcmc.ALUMINIUM.sigma
    modes = f3.extruded_modes(r2, length_m, sigma=sigma, p_max=2)
    exact = modes[0]
    print(f"[ext] prism of length {length_m*1e3:.0f} mm, exact relations:")
    for e in modes:
        Q = e["Q"] if np.isfinite(e["Q"]) else float("nan")
        print(f"        p={e['p']}  f = {e['f']/1e9:8.4f} GHz   "
              f"C = {e['C']:.4f}   Q = {Q:.0f}")
    print(f"        p=1 sits {(modes[1]['f']/exact['f']-1)*100:+.3f}% above p=0 "
          f"-- that is how close the nearest wrong answer is")

    # ---- 4. extrude to a 3D spec -------------------------------------------
    # from_2d inherits mesh_size from spec2d (1 mm), which is the resolution the
    # 2D answer you trust was computed at. Grade it rather than coarsen it: fine
    # where the bars are, coarser in the open gaps where the mode is smooth.
    spec3d = f3.from_2d(spec2d, length=length_m, extrude_layers=layers)
    spec3d.mesh_size = h_bulk
    spec3d.mesh_size_min = h_edge
    spec3d.refine_edges = True
    spec3d.refine_dist = 0.004
    v3.check_spec_3d(spec3d)

    # ---- 5. CALIBRATE: does C converge to the exact value? ------------------
    # The whole point. Sweep the transverse element size on a UNIFORM mesh (a
    # convergence study must refine everything, not just the bulk) and watch C
    # approach the exact prism value. The h where it arrives is the h you need
    # for the 3D-only runs, where no exact answer exists to check against.
    #
    # Per-quantity Richardson powers: f and C converge as h^4 at HCurl order 2,
    # Q only as h^2 because it is a surface integral of a field derivative.
    if calibrate:
        spec_conv = copy.copy(spec3d)
        spec_conv.mesh_uniform = True        # clean h-refinement, no grading
        spec_conv.refine_edges = False
        spec_conv.extrude_layers = 2         # p = 0 needs no more than this
        spec_conv.tag = "toaster_conv"
        print(f"[cal] sweeping transverse h; C must approach "
              f"{exact['C']:.4f} and f approach {exact['f']/1e9:.4f} GHz")
        c = f3.converge(spec_conv, f_target=m2["f"],
                        mesh_sizes=[0.004, 0.003, 0.002],
                        n_modes=n_modes, threads=0)
        for h, f_, C_, Q_ in zip(c["h"], c["f"], c["C"], c["Q"]):
            print(f"[cal]   h={h*1e3:4.1f} mm : f {(f_/exact['f']-1)*100:+7.3f}%  "
                  f"C {(C_/exact['C']-1)*100:+8.2f}%  "
                  f"Q {(Q_/exact['Q']-1)*100:+7.2f}%")
        print(f"[cal]   h -> 0  : f {(c['f_extrap']/exact['f']-1)*100:+7.3f}%  "
              f"C {(c['C_extrap']/exact['C']-1)*100:+8.2f}%  "
              f"Q {(c['Q_extrap']/exact['Q']-1)*100:+7.2f}%")
        print("[cal]   if C is still far off at the finest h, the transverse "
              "mesh -- not the solver -- is the limit.")

    # ---- 6. visualise ------------------------------------------------------
    # Draw a COARSE copy: a 1 mm mesh makes an unreadable (and enormous) HTML,
    # and what you are checking here is the geometry, not the mesh.
    spec_vis = copy.copy(spec3d)
    spec_vis.mesh_size, spec_vis.mesh_size_min = 0.006, 0.003
    v3.plot_spec_3d(spec3d, save=f"{outdir}/toaster_spec.html", show=False)
    v3.plot_mesh_3d(spec_vis, save=f"{outdir}/toaster_mesh.html", show=False,
                    clip="y")     # clip the wall away to see the bars inside
    print(f"[vis] wrote {outdir}/toaster_spec.html and toaster_mesh.html "
          f"-- open them in a browser, they rotate and zoom")

    if not run_3d:
        return {"r2": r2, "extruded": modes, "spec3d": spec3d}

    # ---- 7. the production 3D solve ----------------------------------------
    # threads=0 puts NGSolve's assembly and observables on all cores; without it
    # they run single-threaded and the per-mode C/Q cost dominates the run.
    r3 = f3.solve_cavity_3d(spec3d, n_modes=n_modes, f_target=m2["f"],
                            keep_fields=True, linear_solver="pardiso",
                            threads=0, progress=True)
    m3 = f3.best_mode(r3)
    print(f"[3D]  {r3['n_elements']} tets, {r3['n_free']} free dofs "
          f"({r3['n_dofs']} total), {r3['kernel_dim']} kernel modes deflated")
    print(f"      f = {m3['f']/1e9:.4f} GHz "
          f"({(m3['f']/exact['f']-1)*100:+.2f}% vs exact)   "
          f"C = {m3['C']:.4f} ({(m3['C']/exact['C']-1)*100:+.2f}%)   "
          f"Q = {m3['Q']:.0f} ({(m3['Q']/exact['Q']-1)*100:+.2f}%)")

    # ---- 8. if C is wrong, find out WHY before changing anything -----------
    # cancel  = |int E_z| / int |E_z|  -- how much axial field survives its sign
    # half    = the larger half-integral, relative to int |E_z|
    # parity  = +1 longitudinally uniform (p = 0), 0 odd p
    print("[dia] per-mode diagnosis:")
    f3.mode_diagnostics(r3)
    print(f"[dia] sum of C over all {len(r3['modes'])} modes = "
          f"{sum(m['C'] for m in r3['modes']):.4f}  (exact single-mode value "
          f"{exact['C']:.4f})")
    # Only a diagnostic: if the sum is right while every individual C is small,
    # the coupling is present but split across a degenerate basis, which means
    # the mesh is not holding the cells in tune. The EXPERIMENT sees one mode at
    # a time, so the number to optimise is the single best mode -- never the sum.
    bc = f3.best_cluster(r3)
    if bc and bc["n_modes"] > 1:
        print(f"[dia] largest near-degenerate cluster: {bc['n_modes']} modes, "
              f"{bc['f_min']/1e9:.4f}-{bc['f_max']/1e9:.4f} GHz, "
              f"best single C = {bc['C_best']:.4f}, summed C = {bc['C']:.4f}")

    # ---- 9. look at the field ----------------------------------------------
    # z-slices are the transverse cross-section: compare directly against 2D.
    # y- and x-slices show the longitudinal structure 2D cannot represent -- for
    # the operating mode they must be featureless along z.
    i_best = int(np.argmax([m["C"] for m in r3["modes"]]))
    v3.plot_field_slices(r3, i=i_best, save=f"{outdir}/toaster_slices.png")
    v3.plot_modes_3d(r3, n=n_modes, save=f"{outdir}/toaster_modes.png")
    v3.view_field_3d(r3, i=i_best, save=f"{outdir}/toaster_field.html",
                     show=False)
    print(f"[vis] wrote {outdir}/toaster_slices.png, _modes.png, _field.html")
    return {"r2": r2, "extruded": modes, "r3": r3, "spec3d": spec3d}


# ═════════════════════════════════════════════════════════════════════════════
# A') the 3D-only case: bars that stop short of the endcaps
# ═════════════════════════════════════════════════════════════════════════════

def example_A_partial(length_m=0.16, gap_m=0.010, outdir="TEMP/ex3d",
                      n_modes=20, mesh_size=0.004):
    """
    THE CASE THE 3D SOLVE EXISTS FOR: bars that do not reach the endcaps, so the
    cross-section depends on z and no exact relation applies.

    Note what changes and what does not:
      * extrude_layers is NOT available here -- the cross-section is
        z-dependent, so there is nothing to extrude, and from_2d refuses the
        combination rather than quietly modelling full-length bars.
      * so the mesh is isotropic again, and the transverse resolution you
        calibrated in example_A is the resolution you must now afford in all
        three directions. That is the real cost of leaving the prism.
      * the 2D frequency is now only a starting guess, not the answer. Take the
        target from the calibrated prism run at the same parameters.
    """
    params_mm = [9.71851258, 127.23060036, 8.0, 14.4991172, 15.65917039,
                 124.90989748, 123.52196735]
    params_m = mcmc._params_to_m(params_mm)
    spec2d = v2.toaster_spec(params_m, gap0=mcmc.GAP0_M, gap1=mcmc.GAP1_M,
                             cavity_h=mcmc.CAV_H_M, mesh_size=mcmc.MESH_SIZE,
                             wall_material=mcmc.ALUMINIUM,
                             metal_material=mcmc.ALUMINIUM, tag="toaster")
    r2 = f2.solve_cavity(spec2d, n_modes=n_modes,
                         f_target=3e8 / (2.0 * mcmc.GAP0_M), keep_fields=True)
    m2 = f2.best_mode(r2)

    # every bar retracted by gap_m at BOTH ends
    z0 = -0.5 * length_m + gap_m
    dz = length_m - 2.0 * gap_m
    names = [r.name for r in spec2d.metal]
    spec3d = f3.from_2d(spec2d, length=length_m, mesh_size=mesh_size,
                        partial={n: (z0, dz) for n in names},
                        tag="toaster_partial")
    spec3d.mesh_size_min = 0.002
    spec3d.refine_edges = True
    v3.check_spec_3d(spec3d)
    v3.plot_mesh_3d(spec3d, save=f"{outdir}/partial_mesh.html", show=False,
                    clip="x")   # clip x and look end-on: is the gap really there?

    r3 = f3.solve_cavity_3d(spec3d, n_modes=n_modes, f_target=m2["f"],
                            keep_fields=True, linear_solver="pardiso",
                            threads=0, progress=True)
    print(f"[3D]  {r3['n_elements']} tets, {r3['n_free']} free dofs")
    f3.mode_diagnostics(r3)
    m3 = f3.best_mode(r3)
    print(f"[3D]  best mode: f = {m3['f']/1e9:.4f} GHz   C = {m3['C']:.4f}   "
          f"Q = {m3['Q']:.0f}   "
          f"(2D prism gave f = {m2['f']/1e9:.4f} GHz, C = {m2['C']:.4f})")
    v3.plot_field_slices(r3, save=f"{outdir}/partial_slices.png")
    return r3


# ═════════════════════════════════════════════════════════════════════════════
# B) the imported hollow cylinder
# ═════════════════════════════════════════════════════════════════════════════

def example_B(path="build_files/Hollow-cylinder.STEP", f_target=229.5e6,
              outdir="TEMP/ex3d", n_modes=6):
    """
    The part is the METAL SHELL -- walls and endcaps of finite, varying
    thickness -- so the cavity is the void it encloses, NOT the imported solid.
    That is what interior=True does.

    STEP carries no material, so the conductor is supplied here.

    axis="y" because this part is modelled with its axis along y, and the form
    factor is a projection onto ONE axis. Get it wrong and C comes out ~0 for
    the operating mode with no other symptom -- check the bounding box printed
    by ImportedSpec3D and use its long direction.
    """
    AL = f2.Material("aluminium", sigma=f2.SIGMA_AL_COMSOL)

    spec = f3.ImportedSpec3D(path=path,
                             interior=True,
                             mesh_size=0.3 / 6,
                             wall_material=AL, metal_material=AL)

    f3.diagnose_import(spec)

    # Look at it BEFORE solving: this both meshes the void and shows you which
    # surfaces became the wall. If the "cavity" you see is the shell rather than
    # the hole in it, interior= is the wrong way round.
    v3.plot_mesh_3d(spec, save=f"{outdir}/sample_3d_mesh.html", show=False)
    print(f"[vis] wrote {outdir}/sample_3d_mesh.html")

    # mesh_order=2 (the default) curves the elements onto the true barrel. On a
    # pillbox that took the frequency error from +0.70% to +0.031% at the same
    # dof count, so leave it on for anything with a curved wall.
    r = f3.solve_cavity_3d(spec, n_modes=n_modes, f_target=f_target,
                           keep_fields=True, axis="y", threads=0, progress=True)
    print(f"[3D]  {r['n_elements']} tets, {r['n_free']} free dofs")
    for j, (f, m) in enumerate(zip(r["freqs"], r["modes"])):
        print(f"      mode {j}: f = {f/1e9:8.4f} GHz   C = {m['C']:.4f}   "
              f"Q = {m['Q']:9.0f}   loc = {m['localisation']:.3f}")
    print("[dia] per-mode diagnosis:")
    f3.mode_diagnostics(r)
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
    if which in ("P", "PARTIAL"):
        example_A_partial()
    if which in ("B", "BOTH"):
        example_B()