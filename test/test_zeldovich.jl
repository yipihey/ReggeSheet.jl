# GATE #2 — the 1D Zel'dovich plane wave. Plane-symmetric dust is EXACTLY Zel'dovich in GR
# (pre-crossing), so the lattice march is validated against the closed-form solution:
#   * err_x    : lattice slice x-edge l² vs the analytic displaced-chord values (the wave),
#   * err_perp : transverse edges vs a(t)²dx² — plane symmetry must EMERGE (the full 3D engine
#                is marched; no symmetry is imposed),
#   * bianchi  : the monitored Hamiltonian (dust-sourced) + momentum constraint residuals.
# Discretization: N=3 (kdx = 2π/3 — the identity is checked at the SAMPLED points, so coarse
# sampling is legitimate); CFL regime dτ = 0.4·a·dx (space-like staircase diagonals — hyperbolic
# stability AND null-crossing avoidance; see NOTES.md #11). Per the accepted gate criterion the
# match is to solver-tolerance-plus-accumulation, documented in NOTES.md with the measured march.
using ReggeSheet, Test, Printf

@testset "GATE 2 — Zel'dovich plane wave (emergent plane symmetry)" begin
    N = 3; dx = 1e-4
    box = N * dx; k = 2π / box
    a0 = 0.02
    lna = 0.4 * dx / sqrt(a0)                 # CFL: dτ/(a·dx) = 0.4
    A = 10.0                                  # D·A = a·A ≈ 0.2 (nonlinear, pre-crossing)
    nsteps = GATE2_STEPS
    res = march_zeldovich(N, dx, a0, nsteps; A=A, k=k, lna_step=lna, verbose=false)
    ex = maximum(r -> r.err_x, res)
    ep = maximum(r -> r.err_perp, res)
    bi = maximum(r -> r.bianchi, res)
    @printf("GATE 2: steps=%d  max err_x=%.3e  max err_perp=%.3e  max constraint=%.3e\n",
            nsteps, ex, ep, bi)
    @test ex < 1e-6                           # wave tracks the analytic identity
    @test ep < 1e-6                           # plane symmetry emergent through the 3D engine
    @test bi < 1e-8                           # constraints hold along the march
end
