# ReggeSheet.jl

Lorentzian Regge-calculus evolution of cold dust on a perturbed-FLRW simplicial lattice, built for
head-to-head validation against DISCO-DJ's nLPT displacements (pre-shell-crossing only).

## What it is

- **Intrinsic Lorentzian Regge calculus**: signed squared edge lengths `l²` are the variables
  (space-like > 0, time-like < 0); simplex volumes via signed Cayley–Menger determinants; deficit
  angles by *developing* the pentachora ring around each hinge into its orthogonal 2-plane —
  ordinary angles for time-containing hinges, boost rapidities (Sorkin) for space-like hinges.
- **Lattice**: periodic Kuhn 6-tetrahedra cubes (vertex/tet convention shared verbatim with
  DiscoDJNative's `sheet_deposit`, so the eventual P(k) comparison isolates dynamics), swept into
  tent slabs of pentachora by an order-consistent staircase triangulation.
- **Dust**: conserved vertex masses coupled only through proper time along tent poles,
  `S = (1/8πG) Σ_h A_h ε_h − Σ_i m_i τ_i`; synchronous (constant per-slab) lapse.
- All geometry/residual code is pure and type-generic (ForwardDiff/Enzyme-ready; Float64 default,
  Double64 where near-null edges demand it — see NOTES.md).

## Status

| Milestone | Status |
|---|---|
| Geometry primitives (CM volumes, Lorentzian dihedrals/boosts, embeddings) | ✅ unit-tested vs analytic configs |
| 4D tent-slab complex + flat-closure (ε=0 to 5e-15, all hinge kinds) | ✅ |
| **Gate #1**: homogeneous lattice reproduces EdS a(t) < 1e-6 over a=0.01→1 | ✅ 5.6e-8 (2nd order, 18× margin) |
| **Gate #2**: 1D plane wave = Zel'dovich | ⏳ next |
| 3D dynamics, DISCO-DJ comparison (Ψ, P(k), lightcone kNN) | after gate #2 |

## Layout

- `src/` — core (no DISCO-DJ dependency): geometry, lattice/complex, deficits, homogeneous FLRW.
- `test/` — gates + primitive unit tests: `julia --project=test test/runtests.jl`.
- `compare/` — (post-gate-#2) DiscoDJNative-coupled comparison env.
- `NOTES.md` — numerical choices, calibrations, and the honest failure-mode log.
