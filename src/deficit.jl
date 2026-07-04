# Deficit angles by DEVELOPMENT: the ring of pentachora around a triangular hinge is developed
# (unrolled) into the hinge's orthogonal 2-plane by chaining explicit 2-vectors wedge by wedge; the
# deficit is the failure of the development to close.
#
#   * Time-containing (Lorentzian) hinge triangle → its ⟂ plane is Euclidean: wedge angles are
#     ordinary and additive, ε = 2π − Σθ.
#   * Space-like hinge triangle → its ⟂ plane is Lorentzian (−,+): "angles" are boosts and NOT
#     additive across light-cone quadrants, so we develop explicitly. Flat space ⇒ the development
#     closes exactly (returns the start vector); the deficit is the boost rapidity of the holonomy
#     (Sorkin's boost deficit — the complex-angle convention with the universal 4 light-cone
#     crossings already cancelled).
#
# All functions are pure; per-wedge data comes from `wedge_invariants` (geometry.jl), which is
# intrinsic to the squared edge lengths.

"""
    hinge_ring(h, pids, pents) -> Vector{(pid, apex_in, apex_out)} or nothing

Cyclically order the pentachora `pids` incident on hinge `h` (3 spacetime vertex ids) so that
consecutive entries share the side tetrahedron h∪{apex}. Returns `nothing` for boundary /
non-manifold hinges (some side tet not shared by exactly 2 incident pentachora).
"""
function hinge_ring(h::NTuple{3,Int}, pids::Vector{Int}, pents::Vector{NTuple{5,Int}})
    length(pids) < 3 && return nothing
    apexpair(pid) = begin
        p = pents[pid]
        a = 0; b = 0
        @inbounds for v in p
            if v != h[1] && v != h[2] && v != h[3]
                a == 0 ? (a = v) : (b = v)
            end
        end
        (a, b)
    end
    side = Dict{Int,Vector{Int}}()          # apex vertex -> incident pents having that side tet
    for pid in pids
        a, b = apexpair(pid)
        push!(get!(() -> Int[], side, a), pid)
        push!(get!(() -> Int[], side, b), pid)
    end
    any(length(v) != 2 for v in values(side)) && return nothing
    p0 = pids[1]; a0, _ = apexpair(p0)
    ring = Tuple{Int,Int,Int}[]
    cur, ain = p0, a0
    while true
        aa, bb = apexpair(cur)
        aout = (ain == aa) ? bb : aa
        push!(ring, (cur, ain, aout))
        pr = side[aout]
        cur = (pr[1] == cur) ? pr[2] : pr[1]
        ain = aout
        (cur == p0 && ain == a0) && break
        length(ring) > length(pids) && return nothing        # safety: broken walk
    end
    return length(ring) == length(pids) ? ring : nothing
end

# 5×5 signed squared-edge matrix of a pentachoron via the l² lookup `esq(v,w)`.
function pent_D(p::NTuple{5,Int}, esq)
    T = typeof(esq(p[1], p[2]))
    D = MMatrix{5,5,T}(undef)
    @inbounds for i in 1:5, j in 1:5
        D[i, j] = i == j ? zero(T) : esq(p[i], p[j])
    end
    return SMatrix(D)
end

# One development step in the 2-plane with metric g = diag(pl_sign, 1):
# given current direction w, return w' with ⟨w,w'⟩ = β, ⟨w',w'⟩ = γ and positive orientation
# det[w, w'] > 0.
#
# NULL-ROBUST formulation (line ∩ conic): the linear condition g(w)·w' = β defines a line with
# direction n = Euclidean-perp of the covector g(w) = (pl·w₁, w₂); intersecting the conic
# ⟨w',w'⟩ = γ gives a quadratic in the line parameter that DEGRADES GRACEFULLY when w is null
# (⟨n,n⟩→0 ⇒ the quadratic becomes linear — a unique, well-conditioned root), unlike the previous
# {w, w⊥} expansion which divides by ⟨w,w⟩. det[w,n] = pl·w₁² + w₂²·(pl…) — computed explicitly —
# fixes the orientation branch.
@inline function _advance(w::SVector{2,T}, α::T, β::T, γ::T, pl::T) where {T}
    cv = SVector(pl * w[1], w[2])                          # covector g(w): g(w)·w' = ⟨w,w'⟩
    c2 = cv[1]^2 + cv[2]^2                                 # Euclidean norm² of the covector (>0)
    wp0 = (β / c2) * cv                                    # particular solution on the line
    n = SVector(-cv[2], cv[1])                             # line direction (g(w)·n = 0)
    nn = pl * n[1]^2 + n[2]^2                              # ⟨n,n⟩ (→0 iff w null in Lorentzian)
    b = pl * wp0[1] * n[1] + wp0[2] * n[2]                 # ⟨wp0,n⟩
    c = (pl * wp0[1]^2 + wp0[2]^2) - γ                     # ⟨wp0,wp0⟩ − γ
    dwn = w[1] * n[2] - w[2] * n[1]                        # det[w,n] (orientation slope in t)
    dw0 = w[1] * wp0[2] - w[2] * wp0[1]                    # det[w,wp0]
    local t::T
    if abs(nn) * abs(c) < 100 * eps(one(T)) * max(b^2, one(T)) && abs(b) > 0
        t = -c / (2b)                                      # near-degenerate conic: linear root
    else
        disc = sqrt(max(b^2 - nn * c, zero(T)))
        t1 = nn == 0 ? -c / (2b) : (-b + disc) / nn
        t2 = nn == 0 ? t1 : (-b - disc) / nn
        # pick the root giving positive orientation det[w,w'] = dw0 + t·dwn > 0
        t = (dw0 + t1 * dwn > 0) ? t1 : t2
    end
    return SVector(wp0[1] + t * n[1], wp0[2] + t * n[2])
end

"""
    hinge_deficit(h, ring, esq, pents) -> (ε, kind, closure)

Deficit angle at hinge `h` given its cyclic `ring` (from `hinge_ring`) and the squared-edge lookup
`esq(v,w)`. `kind` is `:euclid` (ε = 2π − Σθ; time-containing hinge) or `:boost` (ε = holonomy
boost rapidity; space-like hinge). `closure` is the normalized development-closure error — ≈0 for
flat space, the sharp all-signs test.
"""
function hinge_deficit(h::NTuple{3,Int}, ring, esq, pents::Vector{NTuple{5,Int}})
    invs = map(ring) do (pid, ain, aout)
        p = pents[pid]
        ia = findfirst(==(ain), p)::Int
        ib = findfirst(==(aout), p)::Int
        wedge_invariants(pent_D(p, esq), ia, ib)
    end
    T = typeof(invs[1][1])
    lorentzian = any(a < 0 || c < 0 || (a * c - b^2) < 0 for (a, b, c) in invs)
    if !lorentzian
        Σ = sum(acos(clamp(b / sqrt(a * c), -one(T), one(T))) for (a, b, c) in invs)
        return (2T(π) - Σ, :euclid, abs(2T(π) - Σ))
    end
    pl = -one(T)                                          # ⟂ plane metric diag(−1,+1)
    α1 = invs[1][1]
    w0 = α1 >= 0 ? SVector(zero(T), sqrt(α1)) : SVector(sqrt(-α1), zero(T))
    w = w0
    @inbounds for k in eachindex(invs)
        (α, β, γ) = invs[k]
        w = _advance(w, α, β, γ, pl)
    end
    nrm = sqrt(abs(α1))
    closure = sqrt((w[1] - w0[1])^2 + (w[2] - w0[2])^2) / nrm
    # Holonomy boost rapidity (sign convention pinned by the boost-identity primitives): for a
    # space-like start (0,√α): Λ(η)w0 = (√α sinh η, √α cosh η) ⇒ det[w0,w]/α = −sinh η.
    dt̂ = (w0[1] * w[2] - w0[2] * w[1]) / abs(α1)
    η = α1 >= 0 ? asinh(-dt̂) : asinh(dt̂)
    return (η, :boost, closure)
end
