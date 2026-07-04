# Differentiability contract: the geometry → deficit → residual chain is pure and type-generic,
# so ForwardDiff must differentiate it exactly (validated against central finite differences).
using ReggeSheet, Test, ForwardDiff

@testset "ForwardDiff through the deficit chain" begin
    # d(residual)/d(edge l²) on the homogeneous 2-slice window, vs central FD — at O(1),
    # WELL-CONDITIONED scales (dx=1, dτ=0.3, far from the null crossing) where a Float64 FD
    # reference is trustworthy (at gate scales the FD itself drowns in the near-null noise floor;
    # see NOTES.md failure mode #2/#4 — precision, not AD, limits there).
    dx = 1.0; dτ = 0.3
    M = FLRWModel(N=3, dx=dx, dτ=dτ, m=0.0)
    s = dx^2
    base = [-dτ^2 + 1.02s, -dτ^2 + 2.05s, -dτ^2 + 2.95s, 1.04s]   # slightly curved config

    function pole_res(x)
        esq = ReggeSheet._esq_window2((eltype(x)(s), x[4], -eltype(x)(dτ)^2, x[1], x[2], x[3]),
                                      M.ns, M.N)
        ReggeSheet.class_residual(M.eq_pole, esq, M.pents2, M.κ, M.conv)
    end

    g_ad = ForwardDiff.gradient(pole_res, base)
    h = 1e-6
    g_fd = map(1:4) do i
        xp = copy(base); xp[i] += h
        xm = copy(base); xm[i] -= h
        (pole_res(xp) - pole_res(xm)) / (2h)
    end
    @test any(abs.(g_ad) .> 1e-3)                      # non-trivial gradient
    for i in 1:4
        denom = max(abs(g_fd[i]), 1e-6)
        @test abs(g_ad[i] - g_fd[i]) / denom < 1e-6    # central FD is O(h²)=1e-12-limited here
    end
end

@testset "ForwardDiff = analytic on tri-area derivative" begin
    a, b, c = 1.1, 0.9, 1.3
    g = ForwardDiff.gradient(v -> tri_area_sq(v[1], v[2], v[3]), [a, b, c])
    @test g[1] ≈ (b + c - a) / 8
    @test g[2] ≈ (a + c - b) / 8
    @test g[3] ≈ (a + b - c) / 8
end
