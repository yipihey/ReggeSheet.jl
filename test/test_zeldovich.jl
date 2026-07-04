# GATE #2 — the 1D Zel'dovich plane wave. Plane-symmetric dust is EXACTLY Zel'dovich in GR
# (pre-crossing), so the lattice march is validated against the closed-form solution:
#   * err_x    : lattice slice x-edge l² vs the analytic displaced-chord values (the wave),
#   * err_perp : transverse edges vs a(t)²dx² — plane symmetry must EMERGE (the full 3D engine
#                is marched; no symmetry is imposed),
#   * bianchi  : the monitored Hamiltonian (dust-sourced) + momentum constraint residuals.
#
# Scheme (every ingredient forced by a measured failure — NOTES.md #9–12): Sorkin split + gauge
# pins + constraint damping (wc=1) + KO dissipation (wd=1) + CFL 0.4 (dτ = 0.4·a·dx, space-like
# staircase diagonals) + discrete York initial data (ham → 2e-17).
#
# Accuracy (measured, per the accepted convergent-gate criterion): the deviation from the
# continuum solution is SPATIALLY dominated — ≈0.7–0.9% of the wave amplitude at kdx = 2π/3
# (3 points/wavelength; dτ-refinement leaves it unchanged; it scales ∝ D·A and shrinks with kdx —
# the N=4 point is documented in NOTES.md). Thresholds below are the measured N=3 values with
# ~2× margin.
using ReggeSheet, Test, Printf

@testset "GATE 2 — Zel'dovich plane wave (emergent plane symmetry)" begin
    N = 3; dx = 1e-4
    box = N * dx; k = 2π / box
    a0 = 0.02
    lna = 0.4 * dx / sqrt(a0)                 # CFL 0.4
    A = 10.0                                  # D·A ≈ 0.2 — genuinely nonlinear, pre-crossing
    nsteps = GATE2_STEPS
    res = march_zeldovich(N, dx, a0, nsteps; A=A, k=k, lna_step=lna, verbose=false, wd=1.0)
    ex = maximum(r -> r.err_x, res)
    ep = maximum(r -> r.err_perp, res)
    bi = maximum(r -> r.bianchi, res)
    @printf("GATE 2: steps=%d  max err_x=%.3e  max err_perp=%.3e  max constraint=%.3e\n",
            nsteps, ex, ep, bi)
    @test ex < 3e-3                           # wave tracks analytic at the (kdx)² truncation scale
    @test ep < 5e-4                           # plane symmetry emergent through the 3D engine
    @test bi < 1e-5                           # constraints hold along the march
    # stability: no exponential growth across the march (parasitic root dead)
    late = maximum(r -> r.err_x, res[end-2:end])
    early = max(maximum(r -> r.err_x, res[1:3]), 1e-12)
    @test late / early < 50 * nsteps          # ≲ linear-in-steps accumulation, not ×3^n
end
