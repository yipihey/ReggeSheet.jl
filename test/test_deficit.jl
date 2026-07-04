# THE all-signs validation: a static lattice in flat Minkowski spacetime must have ZERO deficit
# around every interior hinge — Euclidean (time-containing hinges, 2π−Σθ) and Lorentzian
# (space-like hinges, boost-holonomy) alike. Any sign/orientation/branch error shows up here.
using ReggeSheet, Test

# Static flat lattice: slice s at t = s·dt, spatial comoving cube (spacing dx), periodic min-image.
function static_esq(N, dx, dt)
    ns = N^3
    mi(d) = (d = mod(d, N); d > N ÷ 2 ? d - N : d)
    return function (v::Int, w::Int)
        sv, pv = divrem(v - 1, ns); sw, pw = divrem(w - 1, ns)
        iv = pv % N; jv = (pv ÷ N) % N; kv = pv ÷ (N * N)
        iw = pw % N; jw = (pw ÷ N) % N; kw = pw ÷ (N * N)
        di = mi(iw - iv); dj = mi(jw - jv); dk = mi(kw - kv)
        return -((sw - sv) * dt)^2 + (di^2 + dj^2 + dk^2) * dx^2
    end
end

@testset "flat static lattice: every interior hinge has zero deficit" begin
    N, S = 3, 3
    dx, dt = 1.0, 0.3
    pents = build_pentachora_stack(N, S)
    inc = hinge_incidence(pents)
    esq = static_esq(N, dx, dt)

    n_euclid = 0; n_boost = 0; n_skipped = 0
    worst_e = 0.0; worst_b = 0.0
    for (h, pids) in inc
        ring = hinge_ring(h, pids, pents)
        ring === nothing && (n_skipped += 1; continue)
        ε, kind, closure = hinge_deficit(h, ring, esq, pents)
        if kind == :euclid
            n_euclid += 1; worst_e = max(worst_e, abs(ε))
        else
            n_boost += 1; worst_b = max(worst_b, max(abs(ε), closure))
        end
    end
    @info "flat-lattice deficits" n_euclid n_boost n_skipped worst_e worst_b
    @test n_euclid > 0                     # time-containing hinges present
    @test n_boost > 0                      # space-like (boost) hinges present
    @test worst_e < 1e-10                  # 2π − Σθ = 0 exactly (to roundoff)
    @test worst_b < 1e-10                  # boost holonomy closes exactly
end
