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

@testset "Λ-dominated sanity: near-de Sitter expansion (Ω_Λ = 0.99)" begin
    # Λ-dominated regime (Ω_Λ(a) ≈ 0.93→0.998 over the march) — distinct from gate 1b's
    # matter→Λ handover. (Exactly m = 0 is untestable through flrw_step: its Newton tolerance
    # is scaled to the dust source m/(2dτ), which vanishes.)
    dx = 1e-4
    Ωm = 0.01; ΩΛ = 1 - Ωm
    M = FLRWModel(N=3, dx=dx, dτ=1e-5, m=(3Ωm / (8π)) * dx^3, lam=3ΩΛ)
    a0 = 0.5
    t_of_a(a) = (2 / (3 * sqrt(ΩΛ))) * asinh(sqrt(ΩΛ / Ωm) * a^1.5)
    a_of_t(t) = (sqrt(Ωm / ΩΛ) * sinh(1.5 * sqrt(ΩΛ) * t))^(2 / 3)
    t0 = t_of_a(a0)
    t, a, _ = flrw_march(M, a0; a_end=1.2, lna_step=3e-3, H_seed=sqrt(Ωm / a0^3 + ΩΛ))
    err = maximum(abs(a[i] / a_of_t(t0 + t[i]) - 1) for i in eachindex(t))
    @printf("Λ-dominated: steps=%d  max|a/a_an−1| = %.3e\n", length(t) - 1, err)
    # measured 1.52e-6 at lna_step=3e-3: the finite-difference lapse grading lags under
    # ACCELERATED expansion (ä>0), inflating the (still 2nd-order) truncation ~100× per step
    # vs EdS; a sign/normalization error would miss by orders of magnitude, which is what this
    # regime-coverage test is for (convergence order is established by gates 1/1b).
    @test err < 5e-6
end

@testset "GATE 1b — ΛCDM a(t) to <1e-6 over a=0.02→1 (Λ volume term)" begin
    # Flat matter+Λ: a(t) = (Ωm/ΩΛ)^{1/3}·sinh^{2/3}(3√ΩΛ·t/2). Validates the S_Λ value AND
    # its edge-derivatives (the Newton solve uses both) against the exact Friedmann evolution.
    dx = 1e-4
    Ωm = 0.31; ΩΛ = 1 - Ωm
    M = FLRWModel(N=3, dx=dx, dτ=1e-5, m=(3Ωm / (8π)) * dx^3, lam=3ΩΛ)
    a0 = 0.02
    t_of_a(a) = (2 / (3 * sqrt(ΩΛ))) * asinh(sqrt(ΩΛ / Ωm) * a^1.5)
    a_of_t(t) = (sqrt(Ωm / ΩΛ) * sinh(1.5 * sqrt(ΩΛ) * t))^(2 / 3)
    t0 = t_of_a(a0)
    t, a, cons = flrw_march(M, a0; a_end=1.0, lna_step=GATE1_LNA_STEP,
                            H_seed=sqrt(Ωm / a0^3 + ΩΛ))
    err = maximum(abs(a[i] / a_of_t(t0 + t[i]) - 1) for i in eachindex(t))
    cmax = maximum(x -> isnan(x) ? 0.0 : x, cons)
    @printf("GATE 1b: steps=%d  max|a/a_ΛCDM−1| = %.3e  (target <1e-6)   max|constraint| = %.3e\n",
            length(t) - 1, err, cmax)
    @test err < 1e-6
    @test a[end] >= 1.0
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
