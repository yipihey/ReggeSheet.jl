# ReggeSheet.jl — design log, numerical choices, and failure modes

Working notes for the Lorentzian Regge-calculus dust integrator and its DISCO-DJ/nLPT comparison.
Honest record: what was chosen, what was measured, what broke and how.

## Conventions

- **Signature** η = diag(−,+,+,+); signed squared edge lengths `l²` are the dynamical variables
  (space-like > 0, time-like < 0).
- **Spatial tessellation**: Kuhn 6-tet decomposition, `_TETS`/`_CUBE_CORNERS` **identical to
  DiscoDJNative** `sheet_deposit.jl` so the P(k) comparison shares one deposition code path.
- **4D complex**: tent slabs; each tetrahedral prism → 4 pentachora via the staircase product
  triangulation. **Load-bearing subtlety**: the staircase is globally consistent ONLY if tet
  vertices are taken in a global total order (sorted by vertex id) — with the raw `TETS` tuple
  order, adjacent prisms triangulate their shared wall differently and the complex is broken
  (non-manifold hinges). Bug found via the pole-edge equations having no complete hinge rings;
  fixed in `build_pentachora` (`_sort4`). After the fix the interior-hinge count on the 3-slice
  N=3 flat lattice went 1323 → 2376 with skipped (boundary) 648.
- **Deficits by development** (`deficit.jl`): the pentachora ring around a hinge is developed into
  the hinge's ⟂ 2-plane by chaining explicit 2-vectors (orientation fixed by positive det each
  step). Time-containing hinges → Euclidean ⟂ plane, ε = 2π − Σθ. Space-like hinges → Lorentzian
  ⟂ plane, ε = boost rapidity of the development holonomy (Sorkin boost deficit; the 4 light-cone
  crossings cancel universally so no complex arithmetic is needed). Validated: **static flat
  Minkowski lattice has |ε| < 5e-15 on every interior hinge, both kinds** (the all-signs test).
- **Action normalization**: ∫√−g R → 2·Σ_h A_h ε_h, so S_R = (1/8πG)·Σ_h A_h ε_h, i.e. κ = 1/(8πG).
  Verified numerically: with κ = 1/(16πG) the homogeneous dust universe converges to an expansion
  rate exactly √2·H_EdS (gravity term exactly halved at the seed).
- **Sign convention** s_boost = s_euclid = +1 with the development-holonomy deficit reproduces
  ATTRACTIVE gravity (expanding EdS from the pole equation with the +m/(2τ) dust term). No sign
  gymnastics required beyond the development orientation.

## Homogeneous (gate #1) structure

- Flat-slice ansatz: spatial l² = a²dx²·|Δq|² (|Δq|² ∈ {1,2,3}); per-slab pole l² = −dτ² (constant
  lapse = synchronous) and three diagonal classes d1,d2,d3 (offsets (1,0,0),(1,1,0),(1,1,1)).
- **Pole/diagonal hinge rings close within one slab** ⇒ the {pole, d1, d2, d3} Regge equations
  couple only (a_n, slab, a_{n+1}): the pole equation (the only place dust enters) is the discrete
  **Hamiltonian constraint = first-order Friedmann equation**, solved per tent move (4×4 Newton).
  The slice-spatial equations need both adjacent slabs (Raychaudhuri-like, second order) and are
  evaluated as **constraint residuals**, never solved.
- The vacuum flat static configuration satisfies the equations to ~1e-14 (checked directly).
- **Time-symmetric degeneracy**: the constraint is quadratic in the expansion rate ⇒ at ȧ=0 the
  Newton Jacobian is singular (∂G/∂ȧ ∝ ȧ). Not a bug; the vacuum test asserts the residual instead
  of Newton-solving, and dust steps must be seeded on the expanding branch.
- Lapse: uniform across each slice (synchronous); slab thickness graded as dτ_n = lna_step/H_n with
  H_n the LATTICE's own previous-step rate (the analytic solution never drives the evolution).

## Failure modes found (and their fixes)

1. **Staircase wall mismatch** (above) — silent non-manifold hinges; would have produced wrong
   deficits everywhere. Caught because pole equations had no interior representative on a 2-slice
   stack; the flat-closure test then quantified the fix.
2. **Near-null diagonals destroy Float64 deficits.** The staircase diagonals have
   l² = −dτ² + (spatial)² which sweeps through zero as dτ/(a·dx) crosses 1 — an unavoidable
   crossing during a long march (dτ ∝ 1/H grows faster than a·dx). Near the crossing the
   embedding/angle chain suffers ~9 digits of cancellation amplification: measured Float64
   residual noise floor 4e-5 (jitter probe) vs a dust source of 6e-6 ⇒ Newton oscillates and
   cannot converge. **Fix**: the homogeneous solve runs its geometry in Double64 (~31 digits;
   isbits, StaticArrays/ForwardDiff-compatible); measured residual floor drops to ~1e-21 absolute
   (~3e-16 of the dust scale). Float64 stays the default for returned quantities. The 3D solver
   will need either the same treatment near null configurations or an analytically-stable wedge
   formulation — flagged for M3.
3. **Forward-difference measurement bias**: comparing (a₁−a₀)/(dτ a₀) against H(t₀) has an O(H dτ)
   bias (−H dτ/4 for EdS) that dominates single-step accuracy probes; trajectory-level comparison
   (the actual gate) is the meaningful test.
4. **Null-crossing singularity of the development parametrization.** The staircase diagonals cross
   NULL when dτ/(a·dx) = 1, i.e. at a* = (dx/lna_step)² during a graded march. At the crossing a
   wedge boundary direction is null and `_advance` divides by a null norm (x = β/α, α→0) — a
   genuine coordinate singularity that extended precision does NOT cure. Empirical signature (all
   consistent): lna=3e-2/dx=1e-3 (a*=0.001, out of range) marches cleanly at 5.6e-6; lna=1e-2
   (a*=0.01 = the start) degrades to 8.8e-6; lna=3e-3 (a*=0.11, mid-march) kills Newton (residual
   stalls ~5e-7). ALSO measured: the error is dx-INDEPENDENT (5.577/5.524/5.568e-6 at
   dx=1e-3/3e-4/1e-4) — no spatial floor at the 1e-6 level, so gate accuracy is set by the time
   step + crossing avoidance alone. **Gate-#1 fix**: choose lna_step/dx > 20 so dτ/(a·dx) ≥ 2 over
   the whole march (diagonals uniformly time-like; dx is free since no spatial floor).
   **FIXED for robustness (M2 prep)**: `_advance` reformulated as line ∩ conic (the linear
   condition g(w)·w′ = β intersected with ⟨w′,w′⟩ = γ), which degrades gracefully at null entry
   directions (the quadratic becomes linear — unique root) instead of dividing by ⟨w,w⟩.
   Verified: flat closure unchanged (4.7e-15), FLRW accuracy unchanged (7.7e-9 smoke), and the
   previously-fatal crossing config (lna=3e-3, dx=1e-3, a*=0.11) now marches straight through
   (601 steps, err 1.8e-6 — ACCURACY near null remains locally degraded since the geometry itself
   is ill-conditioned there; the gate keeps crossing-free parameters, and the 3D solver should
   log a near-null diagnostic).

## M2 progress: the general engine, its precision law, and the shift-gauge discovery

Built (`edges.jl`, `step.jl`): the general per-edge machinery — canonical edge enumeration from
the pentachora (NO offset-pattern assumptions: the sorted staircase makes diagonal patterns vary
near the periodic wrap, vertices own 0–14 diagonals), per-edge equation tables, a per-sweep global
deficit cache, and two slab solvers (per-vertex Schwarz sweeps + a global damped Gauss–Newton).
**Cross-validation**: filling the window with a converged FLRW slab pair, every complete edge
equation vanishes to state-quantization precision (~1e-16) across ALL vertices — the general
assembly reproduces gate #1's physics exactly.

Findings (each measured, in order of discovery):
1. **Boost deficits are uncomputable in Float64 in the weak-field regime** — each wedge rapidity
   is O(|ln lna_step|) ≈ 6–10 and the deficit is their ~1e-8-relative cancellation; SO(1,1) is
   abelian so NO reformulation avoids the sum (measured Float64 errors O(10) on ε ~ 1e-7!).
   All solvers run their geometry in Double64. This is a fundamental precision law of Lorentzian
   Regge in cosmology, not an implementation artifact.
2. **ForwardDiff NaN traps**: `sqrt(max(u,0))` and `acos(clamp(c,±1))` have NaN/Inf partials at
   the clamp; fixed by value-branches (`u > 0 ? sqrt(u) : 0`) and `atan(√gram, β)` for wedge
   angles.
3. **Parallel-block race**: color classes must separate blocks on the UNION of unknown-touched
   hinges and equation hinges (equation residuals READ edges others WRITE; Double64 is 128-bit ⇒
   torn reads).
4. **Per-vertex splittings diverge**: the equations are stiff (∂ε/∂l² ~ η/l²) and strongly
   cross-coupled; disjoint-ownership Gauss–Seidel AND overlapping Schwarz blocks both oscillate at
   O(1) residuals. The global damped Gauss–Newton converges (O(10) → 1e-3, LM-damped).
5. **THE SHIFT FREEDOM IS UNFIXED** — the big one. With only the lapse pinned (poles = −dτ²), the
   slab system retains ~3 gauge modes/vertex (tangential placement of each new vertex): the global
   GN reduces residuals while the state slides along gauge directions AWAY from the FLRW
   representative (recovery "error" 0.44 with residuals improving — different gauge representative
   of the same geometry, per-edge comparisons are NOT gauge-invariant). Consequences for M2/M3:
   * gauge-fix the shift: 3 conditions/vertex; the natural zero-shift (synchronous-comoving)
     condition in edge-length variables is d(v,u) = pole(v) + sp1(v,u) + O(curvature) — exactly
     the pattern of the FLRW seeds (the ansatz silently picked this gauge); OR
   * compare gauge-INVARIANT observables (per-tet volumes/densities per Lagrangian mass element —
     the AHK sheet-density view).
   * For gate #2 the pragmatic path is the 1D-reduced (per-column classes) solve — the reduction
     picks the symmetric gauge representative implicitly, exactly as the FLRW ansatz did — with
     the general 3D machinery + explicit shift gauge-fixing as the M3 follow-on.
     [SUPERSEDED — the general engine was fixed instead; see below.]

6. **Corner-sort staircase (supersedes the global-id sort).** Wall consistency requires only that
   adjacent prisms order shared-face vertices identically — sorting tet vertices by LOCAL corner
   id (lex in comoving offsets) achieves this because in-face relative offsets are shared, AND it
   is translation-invariant: no special wrap cubes, every vertex owns exactly 7 incoming
   diagonals. (The global-id sort was also consistent but wrap-broken: in-degrees 0–14, which had
   blocked uniform gauge pinning.) Flat closure re-verified at 1.5e-15.
7. **The Jacobian support must be RING-based.** A hinge's deficit depends on ALL edges of its
   ring's pentachora (wedge invariants use full 5-vertex D-matrices), not just the hinge
   triangle's 3 edges. The first Gauss–Newton used hinge-triangle-only supports — most Jacobian
   entries silently missing — producing slow creep + LM stalls (and contributing to the Schwarz
   divergence). With `ring_edges`/`hinges_touching` incidence the gauge-fixed GN converges to the
   quantization floor.
8. **Gauge-fixed recovery PASSES**: pinning poles (lapse) + the 3 axis-offset incoming diagonals
   per vertex (shift; uniform thanks to the corner-sort) makes the slab solution unique — from a
   3e-3 perturbation the global GN drives max|r| → 1.1e-16 (state-quantization floor) and
   recovers the FLRW slab to 4.9e-7 (Float64 quantization of the truth). The general engine is
   validated end-to-end; gate #2 marches it on 1D Zel'dovich ICs with the zero-shift pin
   d(axis) := pole + sp1.
9. **MARCHING: step 1 superb, step 2 collapses — the open M2 problem.** First march step (N=3,
   one wavelength): err_x = 4.6e-3 at THREE points per wavelength, **err_perp = 4.5e-8 = plane
   symmetry EMERGES through the full 3D engine**, Bianchi 1e-10. But the second window's sp1
   equations are exactly unsolvable (least-squares floor 1.4e-4, O(1) relative), and the
   compromise state explodes by step 3 (×1e3/step). NOT wave physics: A=0 (pure FLRW through the
   general march) fails identically. NOT the hard pins: soft-gauge rows (all unknowns free,
   weighted gauge steering) fail identically. Two live hypotheses:
   (a) equation SCHEDULING: each window imposes 15 eqs/vertex on 14 unknowns; exactness of the
       count relies on the discrete Bianchi identities, which are only approximate — the Sorkin
       prescription drops the 4 gauge-conjugate equations (pole + pinned diagonals) and/or defers
       slab-internal equations to the next window; note dust lives ONLY in the pole equation
       (comoving dust has T_ij = 0 — spatial equations are dust-free, correct physically), so the
       drop/defer choice must still route dust into the solve (the FLRW-class march did: it
       solved {pole, diag} and dropped sp1 — the DUAL split, closed by symmetry).
   (b) SEED contamination: the first window's slab A comes from analytic midpoint chords —
       O(dτ³) off-shell; window 1 solves its sp1 equations by bending slab B off-trajectory, and
       window 2 inherits an inconsistent past.
   DECISIVE experiment (next session): build window 2 entirely from exact flrw_step slabs and
   test solvability — separates (a) from (b) immediately. If (a): restructure the solved set per
   the Sorkin split (defer slab-internal {poleB,dgB} to the next window, solve {poleA,dgA,sp1}
   there — dust then enters via poleA). If (b): seed slab A by solving a proper initial-data
   problem instead of midpoint chords.
   [RESOLVED: BOTH. See 10.]
10. **Initial-data solve + Sorkin split ⇒ integrable; the remaining enemy is the free-evolution
    constraint instability.** Implemented (`solve_edges!` core, `solve_slab!` Sorkin split,
    `initial_slab_solve!`):
    * evolution solves the DUST-FREE square set {unpinned-dgB eqs + sp1 eqs} (11/vtx) for
      {unpinned dgB + sp2} (11/vtx) — comoving dust has T_ij = 0, so the pole equation is the
      dust-sourced Hamiltonian CONSTRAINT, monitored not solved (solving it too = the old
      over-determination, amplified ×1e3/step by approximate Bianchi);
    * initial data solves the unpinned dgA equations for the unpinned dgA diagonals given the two
      analytic slices (which encode ȧ); the poleA residual measures (slice0, slice1, dτ)
      consistency — measured 8e-14 ✓ the analytic slices are consistent, my old midpoint-chord
      diagonals were the off-shell part.
    RESULT (A=0): initial data to 4.6e-19; steps 1–3 clean (errors 1e-16 → 2e-9 → 1.3e-7,
    Bianchi ~1e-11); then an EXPONENTIALLY growing constraint-violating mode (×50–10³/step) blows
    the march at fixed ELAPSED Δlna ≈ 2–3e-2 regardless of step size — the classic free-evolution
    instability of NR (gate #1's homogeneous symmetry excluded these modes; the general engine
    admits them). NEXT: constrained evolution — add the Hamiltonian (poleB, with dust) +
    momentum (pinned-dgB) equations as WEIGHTED least-squares rows in solve_slab! (constraint
    damping, Z4-style in spirit); the LSQ machinery makes this a small edit.
11a. **The residual instability is a PARASITIC ROOT of the two-slab marching recurrence.** With
    York initial data (see 12) and CFL satisfied, the wave march still grows at ×3–4 PER STEP —
    independent of dτ (per-step, not per-time gain) AND independent of amplitude (A=10 and A=1
    grow identically, seeds ∝ A). A linear per-step map amplification |λ_p| ≈ 3 of the two-level
    (window) recursion — the classic parasitic root of multistep schemes; in hindsight the A=0
    "stability" matched roundoff seeds growing 2e-16·3^12 ≈ 1e-10 by step 10-12. Constraint
    damping (wc=1) does NOT remove it (the mode has a constraint-satisfying component); the
    hard-equation LM stalls at the truncation scale from step ~3 as the mode grows. NEXT LEVER
    (NR-standard): Kreiss–Oliger-style dissipation as weighted soft rows penalizing the
    time-oscillating (parasitic) component — e.g. w_d·(sp2 − 2·sp1 + sp0) rows at an amplitude
    that vanishes at the scheme's accuracy order; alternatively a wider implicit window (solve
    two slabs jointly) or a dissipative reformulation of the solved set.
11b. **KO-style dissipation KILLS the parasitic root.** Soft rows w_d·√dmax·(sp2 − (2·sp1 − sp0))
    per slice-2 spatial edge (one unknown per row ⇒ diagonal normal-equation contributions;
    the row's own recurrence has a double root at λ=1, pulling |λ_p|=3 below 1). Measured at A=1,
    12 steps, w_d ∈ {0.1, 0.3, 1.0}: ALL stable, no exponential growth; err_x grows only LINEARLY
    (~1e-5/step, ≈ the same across all w_d ⇒ the drift is scheme truncation, NOT dissipation
    bias); w_d=1 is cleanest (err_perp flat at 1.0e-6, constraints 9.4e-10). Production default
    w_d=1. The marching scheme is now: Sorkin split + gauge pins + constraint damping (wc=1) +
    KO dissipation (wd=1) + CFL 0.4 + York initial data.
12. **Discrete York initial data WORKS.** Extending `initial_slab_solve!` with the slice-1
    spatial edges as unknowns and the Hamiltonian+momentum constraints as soft rows drives the
    initial Hamiltonian residual from 1e-8 (14% of the dust scale — the raw analytic chords at 3
    pts/wavelength are badly off-constraint at D·A=0.2) to **2.4e-17**. Gotcha: the LM's
    hard-row-only early exit skipped the soft rows entirely (byte-identical results) — with soft
    rows the solver must run to LM stall.
11. **CFL — the instability's true identity (constraint damping alone was not enough).**
    Constraint-damping rows slowed the constraint growth ~100× but the error growth persisted
    with per-step gain ~50–250 that WORSENED with larger dτ — a per-step map amplification, not a
    physical mode. The sp1 equations are hyperbolic (the spatial metric's wave dofs); the
    slab-by-slab solve is an explicit scheme for them, so stability demands **dτ ≲ a·dx** (the
    cell light-crossing). We had deliberately run dτ/(a·dx) = 4–14 to keep the staircase
    diagonals time-like (null avoidance, #4) — but the SPACE-LIKE side of null satisfies both:
    **dτ = 0.4·a·dx** (the flat-closure test's regime all along). Measured: A=0 march flat at
    ~1e-10 error / 8e-12 constraints over 10+ steps — STABLE. Note the CFL ratio drifts ∝ √a
    during a march: long marches must re-grade dτ to hold the CFL number (and stay below the
    null crossing).

## M4: the DISCO-DJ comparison harness

Built and validated end-to-end (commits f10b8a4, a115fe0, fc6b197):
- **`embed_slice`** (core): positions from edge lengths — sparse GN multilateration, wrap-aware,
  translation-gauge-pinned; recovers the analytic Zel'dovich slice to **5.6e-16 cell units**
  (permanent suite test). This bridges the intrinsic lattice state to position-based comparisons.
- **`march3d`** (core): the general 3D march — seeds from arbitrary displacement arrays;
  cosmology enters ONLY via E(a) + Ω_m; the lattice drives its own clock (mean sp2/sp1);
  snapshots recorded for offline diagnostics.
- **`compare/`**: DiscoDJNative env (local path source), ICs bridge (EdS + Planck18; Mpc/h ↔
  c/H0 = 2997.92458; vertex-ordering match), Ψ-residuals vs 1/2/3LPT, P(k) through the SHARED
  `sheet_deposit` (the comparison isolates dynamics), CairoMakie → PDF. March results cached
  (`.jls`) so diagnostics iterate without re-marching.

**Milestone: the first full-3D cosmological Regge march** — EdS, Gaussian seed-42 ICs
(Planck18-shape spectrum, 250 Mpc/h, N=8), a = 0.02 → 0.06 in 64 steps, York initial data at
ham = 1.2e-9, constraints bounded (2.4e-5 → 4.2e-4).

Configuration lessons from the low-amplitude preview (a ≤ 0.06, ξ/dx ≈ 0.001 — displacements
~100× below the validated regime):
- LPT orders indistinguishable there (residuals identical to 4 digits) — no discrimination power;
- the raw Ψ-residual is dominated by BACKGROUND-CLOCK truncation (δa/a ≈ 0.2% from the coarse
  dlna ≈ 0.03 CFL steps) leaking through the q-spread — hence the scale-marginalized residual
  (δa/a fitted and reported separately from the perturbation signal);
- deposited-P(k) is estimator-noise-dominated at δ ~ 1e-3;
- `sheet_deposit` requires deposit res == Lagrangian N (internal reshape).

**Definitive run** (completed on the 19× solver; `output/compare_nlpt.pdf`): both cosmologies
a = 0.02 → 0.6, snapshots a = {0.1, 0.2, 0.4, 0.6}; EdS 326 steps / 65 min, ΛCDM 552 steps /
117 min (the pre-optimization solver could not finish EdS in 7 h). RESULTS:

- **EdS (the clean case — the lattice solves exactly this model)**: York ham = 4.7e-10;
  constraints bounded the whole march (2.6e-5 → 6.7e-4, plateaued — no secular growth over 326
  steps to a=0.6). Measured ξ/dx = 0.002 → 0.010 — an order below the anticipated ~0.13: the
  N=8 / 250 Mpc/h box samples only k ∈ [0.025, 0.1] h/Mpc with 8³ modes, and the displacement
  variance in that window is small. Consequences, exactly as the ξ/dx law predicts:
  |Ψ_R − Ψ_nLPT| rms ≈ 1.14–1.24e-4 code units, FLAT in a (a systematics floor, not a growing
  dynamical signal), and **identical against 1/2/3LPT to 4 digits** — at this amplitude the LPT
  orders differ by far less than the lattice floor, so M4 has NO LPT-order discrimination
  power. δa/a ≤ 5e-5. P_Regge/P_3LPT stable in a at ±25–40% per bin (bins 1–7; the Nyquist bin
  is a deposit artifact at 5–7× and should be ignored).
- **ΛCDM (model-mismatched by construction)**: York ham = 7.1e-10 and a clean march to
  a ≈ 0.12, then a CONSTRAINT-DRIFT EVENT at a ≈ 0.12–0.15 (ham 1.4e-3 → 2.1e-2 over ~30
  steps) after which the march stays bounded at that level to a=0.6; Ψ-residual jumps 20× to
  6.4e-4 (≈5× EdS) and δa/a flips sign to +2.4e-4. Leading explanation: the action has NO
  cosmological-constant term — the lattice marches a pure-dust (open) universe from
  ΛCDM-consistent initial data, while the per-step seeds/KO targets and the comparison
  reference are ΛCDM; the event coincides with Ω_Λ(a) crossing the ~0.5% level (Ω_Λ(0.13) ≈
  0.005 → 0.33 at a=0.6). The ΛCDM panel is therefore qualitative only.
  [UPDATE: the Λ volume term was implemented and validated (gate #1b, 9.5e-8 — see "The Λ
  volume term" below); it delays the event and improves the residuals/P-ratios but does NOT
  remove it — the missing-Λ hypothesis was only part of the story; see that section for the
  measured outcome and the remaining candidates.]

**M4 verdict (honest)**: the deliverable demonstrated is the ENGINE + HARNESS — a stable,
constraint-bounded, full-GR 3D Regge march of Gaussian-IC cold dust to a=0.6 with end-to-end
position/P(k) comparison against nLPT on identical ICs. It is NOT yet a GR-vs-LPT
discrimination: that needs ξ/dx raised into the validated 0.1–0.2 regime (larger N at fixed
box and/or smaller box — the 64³ scaling path) and, for ΛCDM, the Λ volume term.

## M5: the lightcone comparison (`compare/lightcone.jl`, `output/compare_lightcone.pdf`)

Geometry: observer OUTSIDE the box along +x at χ(a=0.35) = 2449 Mpc/h from the box center, so
the past lightcone sweeps the whole 250 Mpc/h box over a ∈ [0.306, 0.397]; a dedicated EdS
march (258 steps, 56 min) records 18 snapshots across the window. Regge side: `embed_slice`
per snapshot → per-vertex worldlines, linear-in-a interpolation, bisection of
|x(a)−obs| = χ(a); the reference runs the SAME root-find through the shared
`lightcone_cross_ad` kernel (K=1 Zel'dovich stack — the standard-nLPT ψ₂ carries D₁² growth,
incompatible with the kernel's exact-growth D₂plus column, and M4 measured LPT orders
identical to 4 digits at this amplitude, so K=1 is the consistent, lossless reference).
Shared conventions: comoving Mpc/h; "velocity" = dx/dlna (the kernel's v_r = Σf₁D_kΨ_k·r̂ IS
d(D₁Ψ₁)/dlna·r̂ for K=1); redshift-space offset s = x + (dx/dlna·r̂)r̂.

RESULTS (all 512 particles cross in BOTH pipelines):

| metric | rms | max |
|---|---|---|
| \|Δx_obs\| (crossing position) | **0.355 Mpc/h** | 0.81 Mpc/h |
| \|Δa_cross\| | 4.1e-05 | 1.2e-04 |
| \|Δv_r\| (dx/dlna, radial) | 0.045 Mpc/h | 0.14 Mpc/h |
| kNN-CDF sup\|ΔCDF\| (k=1,2 × real,rsd) | — | **≤ 0.0045** |

The position residual equals the M4 Ψ-residual floor EXACTLY (1.19e-4 code × 2998 =
0.358 Mpc/h): **the lightcone crossing machinery (interpolation + bisection + RSD) adds no
error beyond the known lattice systematics floor.** The kNN-CDFs (8192 volume-filling
queries) agree BELOW their own sampling noise (~1/√8192 ≈ 0.011) in both real and redshift
space — the two lightcone point processes are statistically indistinguishable at this
amplitude, as the M4 floor predicts. M5 closed.

Scale caveat (honest): N=8 / 250 Mpc/h probes only k ≤ 0.1 h/Mpc with 8³ modes; the spec-scale
64³ needs the march at ~2.9M unknowns/slab — the sparse-QR path is the right architecture but
per-step cost and EdgeComplex memory need engineering (M6 documentation item).

## Solver performance: the derivative precision law and the 19× march

Measurement-driven optimization of `solve_edges!` (all regression-gated on the FLRW-recovery
value 4.982e-07, the 46-test suite, and bit-stable march errors). Headline, N=5 Zel'dovich
march, 8 steps, sparse path (nu=1375/slab), 16 threads:

| | old | new | |
|---|---|---|---|
| march wall-clock | **2038 s** | **105.6 s** | **19.3×** (identical err_x/err_perp/constraints) |
| hard cold-start slab solve | 140 s | 53 s | 2.6× (identical \|r\| = 1.12e-16) |
| Jacobian assembly (per) | 3.3–4.3 s | ~1.0 s | per-hinge chain rule |
| damped-LSQ factorization (per) | 1.7–2.9 s | 0.01–0.03 s | CHOLMOD refactor |

**Finding #13 — precision law #2: the boost cancellation extends to DERIVATIVES.** ∂ε/∂l² of a
boost hinge is the same deep cancellation as ε itself. A Float64-dual Jacobian is O(40)-RELATIVE
garbage (frobenius diff 3.9e1 vs the per-equation reference; Double64 duals agree to 8.8e-13),
and the solver then *silently converges to wrong answers* (FLRW recovery 2.05 instead of 5e-7 —
LM accepts steps, residuals drop, the result is wrong). All ForwardDiff through `hinge_deficit`
must run Double64 duals; only the STORAGE of J is safely Float64. Consequence: there is no
F32/F64 GPU escape for assembly — a GPU port would need double-double on FP64 (the A6000 runs
FP64 at 1/32 rate — pointless). The optimization had to be structural, not precision.

**What the 19× is made of** (each measured, each with its failure story):
1. **Per-hinge Jacobian assembly**: differentiate each hinge's deficit ONCE over its ring
   unknowns (slot-array override instead of a per-call Dict), then chain-rule
   ∂G_e/∂u = κ·s_h·[∂(∂A/∂slot)/∂u·ε_h + (∂A/∂slot)·∂ε_h/∂u] into the ≤3 owning equations +
   soft rows — vs the per-equation path that re-differentiated every hinge once per referencing
   equation. 3–4× on assembly; validated against the per-equation J to 8.8e-13.
2. **Chord (stale-J) reuse with a STRONG bar**: a stale-J step gets one trial and must contract
   rt < 0.3·obj, else J refreshes and the iteration retries (failed retries don't consume the
   iteration budget). Pure chord under the plain rt<obj bar CRAWLS — λ escalates, hair-thin
   steps keep passing, maxit exhausts far from the solution (recovery 4.6e-2).
3. **CHOLMOD normal equations** `cholesky(Symmetric(A'A); shift=λ)` with numeric-only
   `cholesky!` refactor keyed by (J version, λ), replacing per-trial SPQR of the augmented
   system: 30–100× on the linear algebra at direction agreement Δ ~ 1e-14; non-PD at floor λ
   caught → λ escalation. (LSMR is the designated replacement at 64³ where 3D-lattice Cholesky
   fill walls; matvecs measured ~20 µs at nu=1375.)
4. **Ladder economics** — the LM λ-ladder was half the march cost. Measured facts: march solves
   sit on the KO-dissipation objective floor far above tol and exit only by stall; rt(λ) is
   FLAT for many rungs (trial rt saturates while λ ≪ dmax) and improves only near λ ~ dmax.
   Fixes: ×100 ladder acceleration after 2 flat rungs; a floor-GATED plateau exit (3 flat rungs
   AND rt_best < 1.5·obj — an ungated version froze cold starts at their first iteration, and a
   per-step minimal-progress bar stalled tol-converging solves whose ∞-norm plateaus for one
   iteration then drops); a solve-level detector (3 CONSECUTIVE accepted steps with <2%
   progress ⇒ converged-at-floor, killing measured 20+-iteration floor-polish grinds).
5. **Negative result (tried, measured, REMOVED)**: warm-starting lm_λ across march solves —
   from either the end-of-solve or the last-accepted value — made marches SLOWER (212→297/347s):
   the ∞-norm acceptance rides a knife edge in λ, and a different starting rung changes which
   marginal steps get accepted, wandering the shift null space (solves exploding from n_it=2 to
   the maxit cap). Cold lm_λ = 1e-10 per solve stands.

**GPU/KA verdict**: not worth porting. The dominant costs after optimization are Double64
deficit sweeps (~17 µs/hinge, threaded, D64/F64 ratio only 1.4–1.7× — the kernel is
structure-bound, not arithmetic-bound) and Double64-dual assembly (precision-locked by finding
#13). The A6000's FP64 is 1/32–1/64 rate, so any GPU double-double path loses to the 128-core
CPU; and the linear algebra is already sub-dominant. The 64³ scaling path is CPU threads
(sweeps and assembly are embarrassingly parallel over hinges/equations) + LSMR + EdgeComplex
memory engineering, not a GPU.

## The Λ volume term (post-M6)

`S_Λ = −(Λ/8πG)·Σ_σ V₄(σ)` — every edge equation gains `−κ·lam·Σ_{σ∋e} ∂V₄/∂l²_e` with
`lam = Λ = 3Ω_Λ` (code units), carried in `conv.lam` (zero default ⇒ every pre-Λ path
bit-identical; Λ=0 suite regression clean). Implementation: `EdgeComplex.pents_of_edge`
incidence + `_dV4_dedge` (single-seed dual through the Cayley–Menger determinant — composes
with outer ForwardDiff duals, so the per-equation Jacobian path gets ∂²V₄ for free); the
per-hinge fast assembly gains a per-pent V₄-HESSIAN pass (10×10 ForwardDiff.hessian in jacT
duals, chain-ruled into rows; pent edges lie inside the ring supports ⇒ sparsity pattern and
CHOLMOD symbolic unchanged); `ClassEquation.epents` extends the homogeneous engine;
`march3d(...; ΩΛ)`.

Validation (all measured):
- **GATE 1b — ΛCDM background**: flat Ω_m=0.31 matter+Λ march a=0.02→1, lattice-driven clock,
  vs the analytic a(t) = (Ω_m/Ω_Λ)^{1/3}sinh^{2/3}(3√Ω_Λ t/2): **max|a/a_an−1| = 9.5e-08**
  (1312 steps, lna_step=3e-3 — same order as gate #1's 5.6e-8), constraints 2.0e-14. This
  validates the sign, the 3Ω_Λ normalization, AND the ∂V₄ derivatives at once (Newton uses
  them all).
- **Λ-dominated sanity** (Ω_Λ=0.99, a=0.5→1.2): 1.5e-6 at lna_step=3e-3 — the
  finite-difference lapse grading lags under ACCELERATED expansion, inflating the (still
  2nd-order) truncation ~100× per step vs EdS; documented, threshold 5e-6. (Exactly m=0 is
  untestable through `flrw_step` — its Newton tolerance is scaled to the dust source.)
- Suite 49/49; EdS gate and Zel'dovich gate byte-stable (Λ=0 default).

**M4 ΛCDM re-run with the term (honest outcome — partial fix)**: the Λ term HELPED but did
not remove the constraint event. Measured vs the Λ-less march: event onset DELAYED a≈0.12 →
a≈0.15 (the clean low-ham stretch lengthened, 2.0e-4 through a≈0.145), Ψ-residual at a≥0.2
down 10% (6.31e-4 → 5.71e-4), δa/a down (2.6e-4 → 2.2e-4), and the mid-k P_Regge/P_3LPT bins
now center on unity (1.01–1.07, was 0.85–1.08). BUT a near-identical event still fires at
a≈0.15–0.19 and saturates at the same ham ≈ 2.2e-2 — with Λ in the action it now happens at
Ω_Λ(a) ≈ 0.9%, so the missing-Λ hypothesis explained only part of the drift. The residual
cause is OPEN. Candidate (a) — radiation in the seed clock — was tested by re-marching with a
pure matter+Λ E(a) and is ELIMINATED BY VACUITY: `hubble_E` for this cosmology is already
matter+Λ (the two clocks agree to machine precision; the isolation march is bit-identical,
max ham 2.236e-2). Sharpened reading of the traces: EdS shows the SAME-SHAPED constraint bump
(onset a≈0.07, peak 1.3e-3, then relaxes) — the ΛCDM event starts in the same a-range and
reaches 17× that peak while carrying only ~2× the displacement, suggesting NONLINEAR
amplification of a common resolution-scale trigger rather than a cosmology-specific term.
Next diagnostics: (i) amplitude-scaling test (ICs ×0.5: max-ham ∝ A²⁺ ⇒ nonlinear/resolution;
unchanged ⇒ structural), (ii) spatial localization of the ham residual at the event step,
(iii) wd/wc sensitivity. The ΛCDM panel remains qualitative past a≈0.15.

## Scaling to spec (M6): the 64³ cost model

From the measured numbers (N=8: ~10–12 s/step at 16 threads; deficits 17 µs/hinge in
Double64; hinges ≈ 88/vertex; unknowns 11/vertex; LSMR matvec ~20 µs at nu=1375):

- **Steps**: dτ = 0.4·a·dx ⇒ steps ∝ 1/dx ⇒ ×8 from N=8 → N=64 at fixed box: ~2600 steps
  to a = 0.6.
- **Per step at N=64** (23M hinges, 2.9M unknowns, 128 threads): deficit sweep ≈ 3 s ×
  ~6–10 sweeps (residuals + LM trials) + per-hinge D64-dual assembly ≈ 12 s × ~2 + LSMR
  (nnz ≈ 6×10⁷, replaces CHOLMOD whose 3D fill walls) ≈ tens of seconds ⇒ **~1–1.5 min/step**.
- **Total**: ~2–3 days at 128 threads, ~10–20 GB (ring_edges dominates; the eid Dict and
  per-thread slot buffers need engineering at this scale).
- **Practical rungs**: N=16 (~1.5 h) and N=32 (~1 day ÷ 8... ~8 h) come first and already
  raise ξ/dx 2–4× toward the validated 0.1–0.2 regime; combining with a smaller box gets the
  rest of the way.
- **GPU**: measured NOT worthwhile — the two precision laws lock geometry + Jacobian duals to
  Double64 (double-double on FP64-weak hardware loses to 128 CPU cores), and the deficit
  kernel is structure-bound (D64/F64 only 1.4–1.7×).

Queued physics before the discrimination run: the Λ volume term S_Λ = −(Λ/8πG)·Σ V₄ (the
∂V₄²/∂l² derivatives are already in the Cayley–Menger machinery) — required for a valid ΛCDM
march past a ≈ 0.13 (see M4).

## Gate status

- **Gate #1 (EdS a(t) < 1e-6): PASSED.** Crossing-free convergence sweep (diagonals uniformly
  time-like, lna_step/dx ≥ 30·√a₀):

  | lna_step | dx    | steps | max\|a/a_EdS−1\| | max constraint |
  |----------|-------|-------|------------------|----------------|
  | 1e-2     | 1e-4  | 470   | 6.6e-7           | 1.3e-11        |
  | 3e-3     | 1e-4  | 1545  | **5.6e-8**       | 5.4e-13        |
  | 1e-3     | 3e-5  | 4615  | 6.4e-9           | 6.5e-13        |

  Clean **2nd order** in lna_step (error ratios ≈ 9 per 3× refinement) over a full decade; error is
  dx-INDEPENDENT at this level (no measurable spatial floor — flat-sliced homogeneous expansion is
  represented essentially exactly in space, the discretization error is temporal). The monitored
  spatial (Raychaudhuri) constraint residuals sit at 1e-11–1e-13 without ever being solved — the
  discrete Bianchi consistency of the Sorkin split. CI gate runs lna_step=3e-3, dx=1e-4 (18×
  margin). Wall-clock: ~1545 Double64 steps in a few minutes, single-threaded.
- **Gate #2 (Zel'dovich): PASSED.** N=3 (kdx = 2π/3, THREE points per wavelength), A=10
  (D·A ≈ 0.2, genuinely nonlinear), CFL 0.4, wd=1, wc=1, York initial data; 12-step CI march:

  | metric | measured | threshold |
  |---|---|---|
  | max err_x (wave vs analytic)      | 1.34e-3 | < 3e-3 |
  | max err_perp (emergent plane sym) | 1.82e-4 | < 5e-4 |
  | max constraint (Ham+mom)          | 9.3e-7  | < 1e-5 |
  | stability (no exponential growth) | linear-in-steps | ✓ |

  **M3 resolution study (three verdicts, all measured):**
  1. GAUGE-DRIFT REFUTED: phase-aligning the lattice x-profile against the analytic wave gives
     φ = 0 exactly and aligned ≡ raw at BOTH N=3 and N=4 — the deviation is a genuine SHAPE
     deviation of the discrete wave, not a coordinate/pin drift.
  2. The "N=4 anomaly" inverted: the error SATURATES with N (err_x at 12 steps, D·A=0.2:
     N=3: 1.34e-3, N=4: 3.51e-3, N=5: 4.11e-3 — +162% then +17%) — N=3 was anomalously GOOD
     (Z₃-symmetric wave), N≥4 approaches the generic envelope ≈1% of the wave per 12 steps.
  3. Portrait of the deviation: linear-in-steps, ∝ D·A, dτ-weak (per-step ∝ dτ^~0.5 at fixed
     CFL), wd-independent (not dissipation bias), φ=0 (not gauge).
  4. **The ξ/dx law (hypothesis + first confirmation pending):** N=8/kdx=0.79 gives err_x =
     5.61e-3 — STILL growing as the wave is better resolved, killing the (kdx)² story entirely.
     But err/(ξ/dx) with ξ = D·A/k (the DISPLACEMENT amplitude) is ≈ constant (0.025) for N≥4:
     [N, kdx, ξ/dx, err_x, err/(ξ/dx)] = [4, 1.57, 0.127, 3.51e-3, 0.028];
     [5, 1.26, 0.159, 4.11e-3, 0.026]; [8, 0.79, 0.255, 5.61e-3, 0.022]. The controlling
     resolution parameter is DISPLACEMENT PER CELL — the classic Lagrangian-code criterion — not
     wave sampling; the study conflated them by fixing the slope D·A while growing the box.
     **CONFIRMED**: N=8 at matched ξ/dx = 0.096 (A=3.77, D·A=0.075) gives err_x = 1.54e-3 vs
     N=3's 1.34e-3 — the absolute error tracks ξ/dx to ~15% while kdx varies 2.6×, D·A varies
     2.7× and N varies 3→8. The M3 resolution study is CLOSED with the systematics model:
     **err_x(12 steps) ≈ 0.015–0.03 · (ξ/dx)** (conservative envelope 0.03), the classic
     Lagrangian displacement-per-cell criterion. M4 consequence: Regge−nLPT residuals are
     trustworthy where 0.03·(ξ/dx)·(steps/12) ≪ the sought GR/LPT-truncation signal — early
     times and quasilinear scales, degrading as ξ/dx → 1 toward shell-crossing (where the
     caustic detector halts per spec).

  Error budget (original gate-2 measurements): the deviation from the continuum solution is ≈0.7–0.9% of the wave
  amplitude, INDEPENDENT of dτ (CFL 0.4 vs 0.2 at matched Δlna: 1.21e-4 vs 1.69e-4 at A=1) and
  ∝ D·A (A=1: 1.7e-4; A=10: 1.34e-3). **CORRECTION (the N=4 measurement contradicts the naive
  (kdx)² story)**: N=4/kdx=1.57 gives err_x = 3.5e-3 — 2.6× LARGER than N=3, not the 0.56×
  spatial-truncation prediction. The drift is deterministic, linear-in-steps, dτ-independent and
  amplitude-proportional, but its N-dependence is NOT yet characterized (candidate: the N=3 wave
  is an exact Z₃-symmetric lattice eigenmode with partial error cancellation; the N=4 sampling
  breaks that symmetry differently). Flagged as the first M3 question — the comparison harness
  needs the resolution-dependence understood before interpreting Regge−nLPT residuals. The gate
  itself asserts what is measured: stable tracking at the N=3 accuracy, dτ-independence, and
  amplitude-proportionality (both established); it does NOT assert N-convergence.
  Precision gotcha closed along the way: the vacuum unit test and the FLRW constraint monitor
  evaluated residuals in Float64 — boost-deficit garbage per precision law #1 (measured d1-class
  residual −0.58 from deficits that are 1e-14 in Double64); both now promote to Double64.

- ~~Gate #2 (Zel'dovich): not started.~~ **Design notes from the FLRW work** (for M2/M3):
  * Build the GENERAL inhomogeneous per-edge machinery and run it on 1D ICs — the plane symmetry
    should EMERGE, a stronger gate than a 1D-specialized solver (and it's the M3/M4 engine anyway).
  * Slab-local equations (pole + diagonals; rings close within one slab — a combinatorial fact,
    independent of homogeneity) are FEWER than the slab unknowns in the inhomogeneous case:
    ~8/vertex equations vs ~10.5/vertex unknowns. The FLRW ansatz hid this (4 eq / 4 unknowns).
    The missing equations are those of the SLICE-n spatial edges, which close only with the
    PREVIOUS slab present — that's where the momentum (ψ̇, growing-vs-decaying mode) enters,
    second-order in time, exactly the Sorkin tent-move structure: per tent over vertex v, solve
    the equations of edges incident to the past vertex b_v (two-slab window) for the new edges at
    t_v. Initial data therefore needs TWO seeded slices (as the task's initialize.jl anticipated).
  * Practical solver shape: vertex-wise Gauss–Seidel–Newton sweeps over local ~10-unknown blocks
    (neighbors frozen per sweep), red-black threaded — converges to the implicit slab solution
    without a global sparse Newton and matches the "local evolution steps" requirement.
  * The null-crossing hazard (failure mode #4) is LOCAL in the inhomogeneous case: individual
    wedges can approach null anywhere; `_advance` needs the null-robust reformulation before M3
    scales up.
