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
- **Gate #2 (Zel'dovich)**: not started. **Design notes from the FLRW work** (for M2/M3):
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
