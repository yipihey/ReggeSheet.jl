# GATE #1 — the unperturbed (homogeneous) lattice must reproduce EdS a(t) to <1e-6 relative
# over Δa = 0.01 → 1. The lattice's own expansion rate drives the (graded, synchronous) lapse;
# the analytic solution is used ONLY for the comparison.
using ReggeSheet, Test, Printf

@testset "vacuum: flat static lattice satisfies the Regge equations" begin
    using DoubleFloats: Double64
    dx = 1e-3
    M0 = FLRWModel(N=3, dx=dx, dτ=1e-5, m=0.0)
    s = dx^2
    # Double64 inputs: boost deficits are Float64-garbage at these scale ratios (the boost
    # precision law, NOTES.md M2 #1) — the equations themselves are fine.
    x = Double64[-M0.dτ^2 + s, -M0.dτ^2 + 2s, -M0.dτ^2 + 3s, s]
    r = ReggeSheet.step_residual(M0, Double64(s), Double64(M0.dτ), x)
    @test maximum(abs, Float64.(r)) < 1e-12
end

@testset "GATE 1 — EdS a(t) to <1e-6 over a=0.01→1" begin
    # Calibrated by the convergence sweep (NOTES.md): lna_step=3e-3, dx=1e-4 gives 5.6e-8 (2nd
    # order, 18× margin) with lna_step/dx = 30 keeping the staircase diagonals uniformly time-like
    # (no null crossing; see failure mode #4). Error is dx-independent at this level.
    dx = 1e-4
    ρ0 = 3 / (8π)                       # H0 = 1, G = 1 ⇒ ρ0 = 3/(8π)
    M = FLRWModel(N=3, dx=dx, dτ=1e-5, m=ρ0 * dx^3)
    t0 = 2 / 3                          # EdS: a(t) = (t/t0)^{2/3}, t0 = 2/(3H0)
    a0 = 0.01
    t, a, cons = flrw_march(M, a0; a_end=1.0, lna_step=GATE1_LNA_STEP)
    ts = t0 * a0^1.5
    aE = @. ((t + ts) / t0)^(2 / 3)
    err = maximum(abs.(a ./ aE .- 1))
    cvals = filter(!isnan, cons)
    cmax = isempty(cvals) ? NaN : maximum(cvals)
    @printf("GATE 1: steps=%d  max|a/a_EdS−1| = %.3e  (target <1e-6)   max|constraint| = %.3e\n",
            length(t) - 1, err, cmax)
    @test err < 1e-6
    @test a[end] >= 1.0
end
