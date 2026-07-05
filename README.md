# ReggeSheet.jl

Lorentzian Regge-calculus evolution of cold dust on a perturbed-FLRW simplicial lattice, built
for head-to-head validation against DISCO-DJ's nLPT displacements (pre-shell-crossing only).

## What it is

- **Intrinsic Lorentzian Regge calculus**: signed squared edge lengths `l²` are the variables
  (space-like > 0, time-like < 0); simplex volumes via signed Cayley–Menger determinants; deficit
  angles by *developing* the pentachora ring around each hinge into its orthogonal 2-plane —
  ordinary angles for time-containing hinges, boost rapidities (Sorkin) for space-like hinges.
- **Lattice**: periodic Kuhn 6-tetrahedra cubes (vertex/tet convention shared verbatim with
  DiscoDJNative's `sheet_deposit`, so the P(k) comparison isolates dynamics), swept into
  tent slabs of pentachora by an order-consistent staircase triangulation.
- **Dust**: conserved vertex masses coupled only through proper time along tent poles,
  `S = (1/8πG) Σ_h A_h ε_h − Σ_i m_i τ_i`; synchronous (constant per-slab) lapse.
  (No cosmological-constant volume term yet — see "Queued" below.)
- **Marching scheme** (every ingredient forced by a measured failure — NOTES.md): Sorkin split
  (solve the dust-free diagonal + spatial equations; the dust-sourced pole equation is the
  *monitored* Hamiltonian constraint) + gauge pins (lapse via poles, shift via 3 axis diagonals
  per vertex) + soft constraint-damping rows + Kreiss–Oliger-style dissipation rows + CFL 0.4
  + discrete York initial data.
- **Two precision laws** (measured, fundamental): boost deficits **and their l²-derivatives**
  are catastrophic cancellations in the cosmological weak-field regime — all residual geometry
  AND all Jacobian dual arithmetic run in Double64 (storage may be Float64). This rules out
  F32/F64 GPU paths for the geometry kernels.

## Status

| Milestone | Result |
|---|---|
| Geometry primitives (CM volumes, Lorentzian dihedrals/boosts, embeddings) | ✅ unit-tested vs analytic configs |
| **Gate #1**: homogeneous lattice reproduces EdS a(t) | ✅ 5.6e-8 (2nd order in the step, 18× margin) |
| **Gate #2**: 1D plane wave vs Zel'dovich (full 3D engine, symmetry emergent) | ✅ err ≈ 0.7–0.9% of wave amplitude at 3 pts/λ, O((kdx)²)-convergent |
| M3: resolution study | ✅ the ξ/dx law: march error ≈ 0.015–0.03·(displacement per cell) |
| M4: Ψ + P(k) vs 1/2/3LPT, EdS + ΛCDM, a → 0.6 | ✅ engine+harness demonstrated (see caveats) |
| M5: lightcone crossings + kNN-CDF vs `lightcone_cross_ad` | ✅ kNN-CDFs agree below query noise |
| Solver | 19× march speedup, all regression-gated (NOTES) |

**M4/M5 headline (honest)**: at N=8 / 250 Mpc/h the box carries ξ/dx ≤ 0.02, so the
Regge−nLPT residual sits at the lattice-systematics floor (≈0.36 Mpc/h ≈ 1% of a cell), LPT
orders are indistinguishable, and the lightcone point processes agree statistically
(sup|ΔCDF| ≤ 0.0045). What is demonstrated is a stable, constraint-bounded full-GR 3D march
with an end-to-end comparison pipeline — not yet a GR-vs-Newtonian discrimination.

## Queued (the discrimination program)

1. **Λ volume term** `S_Λ = −(Λ/8πG)·Σ V₄` — the ΛCDM march currently evolves pure dust from
   ΛCDM initial data and drifts once Ω_Λ(a) ≳ 0.5% (constraint event at a ≈ 0.13; NOTES M4).
2. **Raise ξ/dx into the validated 0.1–0.2 regime** — larger N at fixed box and/or a smaller
   box; this is where LPT orders separate and the GR question becomes answerable.
3. **Scale**: the 64³ slab is ~2.9M unknowns. The measured cost structure says the path is CPU
   threads (deficit sweeps + per-hinge Jacobian assembly are embarrassingly parallel over
   hinges/equations, Double64-bound) + LSMR for the damped least squares (CHOLMOD's 3D-lattice
   fill walls; matvecs are ~20 µs at nu=1375) + EdgeComplex memory engineering. A GPU port is
   measured NOT worthwhile (precision laws + structure-bound kernels + FP64-weak hardware).

## Layout

- `src/` — core (no DISCO-DJ dependency): geometry, lattice/complex, deficits, FLRW,
  edge machinery, the LM Gauss–Newton solver, Zel'dovich references, slice embedding, 3D march.
- `test/` — gates + primitive unit tests: `julia --project=test test/runtests.jl` (46 tests).
- `compare/` — DiscoDJNative-coupled comparison env:
  `julia --project=compare -t 16 compare/compare_nlpt.jl` (M4) and
  `compare/compare_lightcone.jl` (M5) → `output/*.pdf`.
- `NOTES.md` — the lab notebook: numerical choices, calibrations, all measured findings
  (13+ numbered), gate tables, and the honest failure-mode log.
