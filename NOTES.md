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
