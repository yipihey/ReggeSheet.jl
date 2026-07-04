# Homogeneous (FLRW) reduction — the gate-#1 engine.
#
# Flat-slice homogeneous ansatz on the periodic Kuhn lattice: one scale per slice
# (spatial l² = a²dx²·|Δq|², |Δq|² ∈ {1,2,3}), one pole l² = −dτ² per slab (constant lapse =
# synchronous gauge), and three diagonal classes d1,d2,d3 per slab (offsets (1,0,0),(1,1,0),(1,1,1)).
#
# Structure of the tent-move step (per slab n→n+1):
#   * The pole and diagonal edges' hinges are mixed triangles whose pentachora rings close WITHIN
#     the slab (verified by the flat-closure test), so their Regge equations couple only
#     (a_n, pole, d1..d3, a_{n+1}) — the pole equation carries the dust source m/(2τ) and is the
#     discrete Hamiltonian constraint = FIRST-ORDER Friedmann equation. We solve the 4×4 system
#     {pole, d1, d2, d3} for (d1, d2, d3, a_{n+1}²dx²) by Newton (ForwardDiff Jacobian).
#   * The slice-n spatial-edge equations need BOTH adjacent slabs (they are the second-order /
#     Raychaudhuri consistency) — evaluated as CONSTRAINT RESIDUALS each step, never solved.
#
# EdS units: c=1, G=1, H0=1 ⇒ ρ0 = 3/(8π), t0 = 2/3, a(t) = (t/t0)^{2/3}; dust mass per vertex
# m = ρ0·dx³ (one vertex per cube). Action normalization: ∫√−g R → 2·Σ_h A_h ε_h (the classic
# Regge factor), so S_R = (1/8πG)·Σ_h A_h ε_h ⇒ κ = 1/(8π) — verified numerically: with κ=1/(16π)
# the converged expansion rate is exactly √2·H_EdS (gravity term exactly halved).
#
# The Lorentzian action sign conventions (relative signs of boost- vs euclid-hinge terms and the
# dust coupling) are exposed in `conv` and CALIBRATED by this very gate (correct attractive
# Friedmann evolution); the calibrated convention is documented in NOTES.md.

using ForwardDiff: ForwardDiff
using DoubleFloats: Double64

# ── edge classification on an S-slice stack ───────────────────────────────────
@inline function _vertex_sik(v::Int, ns::Int, N::Int)
    s, p = divrem(v - 1, ns)
    return (s, p % N, (p ÷ N) % N, p ÷ (N * N))
end

@inline _mi(d::Int, N::Int) = (d = mod(d, N); d > N ÷ 2 ? d - N : d)

# class key: (:sp, slice, r2) spatial | (:pole, slab) | (:dg, slab, r2) diagonal
function _edge_class(v::Int, w::Int, ns::Int, N::Int)
    sv, iv, jv, kv = _vertex_sik(v, ns, N)
    sw, iw, jw, kw = _vertex_sik(w, ns, N)
    r2 = _mi(iw - iv, N)^2 + _mi(jw - jv, N)^2 + _mi(kw - kv, N)^2
    if sv == sw
        return (:sp, sv, r2)
    else
        slab = min(sv, sw)
        return r2 == 0 ? (:pole, slab, 0) : (:dg, slab, r2)
    end
end

# ── combinatorial precompute: hinges (with rings + slot) per representative edge ──────────────
struct ClassEquation
    rep::Tuple{Int,Int}                                   # representative edge (v, w)
    hinges::Vector{Tuple{NTuple{3,Int},Vector{Tuple{Int,Int,Int}},Int}}  # (h, ring, slot of rep)
end

# triangle edge slots: 1↔(h1,h2), 2↔(h1,h3), 3↔(h2,h3)
@inline function _slot(h::NTuple{3,Int}, v::Int, w::Int)
    (v, w) = v < w ? (v, w) : (w, v)
    (v, w) == (h[1], h[2]) && return 1
    (v, w) == (h[1], h[3]) && return 2
    return 3
end

"""
    build_class_equations(pents, ns, N, wanted) -> Dict{class => ClassEquation}

For each class key in `wanted`, pick a representative edge and precompute all its hinges with
their (metric-independent) pentachora rings and the representative's edge slot. Errors if any
hinge ring fails to close (the class equation would be incomplete on this stack).
"""
function build_class_equations(pents::Vector{NTuple{5,Int}}, ns::Int, N::Int, wanted)
    inc = hinge_incidence(pents)
    # collect hinges by unordered edge
    edge_h = Dict{Tuple{Int,Int},Vector{NTuple{3,Int}}}()
    for h in keys(inc)
        for (a, b) in ((h[1], h[2]), (h[1], h[3]), (h[2], h[3]))
            push!(get!(() -> NTuple{3,Int}[], edge_h, (a, b)), h)
        end
    end
    out = Dict{Any,ClassEquation}()
    for ((v, w), hs) in edge_h
        cls = _edge_class(v, w, ns, N)
        (cls in wanted && !haskey(out, cls)) || continue
        hinges = Tuple{NTuple{3,Int},Vector{Tuple{Int,Int,Int}},Int}[]
        ok = true
        for h in hs
            ring = hinge_ring(h, inc[h], pents)
            ring === nothing && (ok = false; break)
            push!(hinges, (h, ring, _slot(h, v, w)))
        end
        ok || continue                                    # boundary representative: try another
        out[cls] = ClassEquation((v, w), hinges)
    end
    for cls in wanted
        haskey(out, cls) || error("no interior representative for class $cls on this stack")
    end
    return out
end

# ── residual assembly ─────────────────────────────────────────────────────────
# ∂(A²)/∂slot for A² = tri_area_sq(a,b,c) = (2(ab+bc+ca) − a² − b² − c²)/16:
#   ∂A²/∂a = (b + c − a)/8, and cyclic.
@inline function _dA_dslot(a::T, b::T, c::T, slot::Int) where {T}
    A2 = tri_area_sq(a, b, c)
    dA2 = slot == 1 ? (b + c - a) / 8 : slot == 2 ? (a + c - b) / 8 : (a + b - c) / 8
    sA = A2 < 0 ? -one(T) : one(T)
    return sA * dA2 / (2 * sqrt(abs(A2)))                 # ∂√|A²|/∂l²_slot
end

"""
    class_residual(ce, esq, pents, κ, conv; dust=0) -> κ·Σ_h s_kind·(∂A_h/∂l²)·ε_h + dust

The Regge equation residual for the class's representative edge. `conv = (s_boost, s_euclid)`
carries the Lorentzian action sign convention; `dust` adds the matter term (m/(2τ) for poles).
"""
function class_residual(ce::ClassEquation, esq, pents, κ, conv; dust=0.0)
    (v, w) = ce.rep
    T = typeof(esq(v, w))
    G = zero(T) + T(dust)
    for (h, ring, slot) in ce.hinges
        ε, kind, _ = hinge_deficit(h, ring, esq, pents)
        a, b, c = esq(h[1], h[2]), esq(h[1], h[3]), esq(h[2], h[3])
        s = kind == :boost ? T(conv.s_boost) : T(conv.s_euclid)
        G += T(κ) * s * _dA_dslot(a, b, c, slot) * ε
    end
    return G
end

# ── homogeneous esq closures ──────────────────────────────────────────────────
# 2-slice solve window: slice scales (s0, s1) [=a²dx²], slab (pole, d1, d2, d3).
function _esq_window2(vals, ns::Int, N::Int)
    (s0, s1, pole, d1, d2, d3) = vals
    return function (v::Int, w::Int)
        cls = _edge_class(v, w, ns, N)
        if cls[1] === :sp
            return (cls[2] == 0 ? s0 : s1) * cls[3]
        elseif cls[1] === :pole
            return pole
        else
            return cls[3] == 1 ? d1 : cls[3] == 2 ? d2 : d3
        end
    end
end

# 3-slice constraint window: (s0,s1,s2), slabs A=(pole,d1,d2,d3) and B likewise.
function _esq_window3(vals, ns::Int, N::Int)
    (s0, s1, s2, pA, dA1, dA2, dA3, pB, dB1, dB2, dB3) = vals
    return function (v::Int, w::Int)
        cls = _edge_class(v, w, ns, N)
        if cls[1] === :sp
            return (cls[2] == 0 ? s0 : cls[2] == 1 ? s1 : s2) * cls[3]
        elseif cls[1] === :pole
            return cls[2] == 0 ? pA : pB
        else
            d = cls[2] == 0 ? (dA1, dA2, dA3) : (dB1, dB2, dB3)
            return cls[3] == 1 ? d[1] : cls[3] == 2 ? d[2] : d[3]
        end
    end
end

# ── the FLRW model ────────────────────────────────────────────────────────────
struct FLRWModel{T}
    N::Int; ns::Int
    dx::T; dτ::T; m::T; κ::T
    conv::NamedTuple{(:s_boost, :s_euclid),Tuple{T,T}}
    pents2::Vector{NTuple{5,Int}}                          # 2-slice solve stack
    eq_pole::ClassEquation
    eq_dg::NTuple{3,ClassEquation}
    pents3::Vector{NTuple{5,Int}}                          # 3-slice constraint stack
    eq_sp::NTuple{3,ClassEquation}                         # slice-1 spatial classes r²=1,2,3
end

function FLRWModel(; N::Int=3, dx::Float64=1.0, dτ::Float64=1e-3, m::Float64,
                   κ::Float64=1/(8π), s_boost::Float64=1.0, s_euclid::Float64=1.0)
    ns = N^3
    pents2 = build_pentachora_stack(N, 2)
    want2 = [(:pole, 0, 0), (:dg, 0, 1), (:dg, 0, 2), (:dg, 0, 3)]
    ce2 = build_class_equations(pents2, ns, N, want2)
    pents3 = build_pentachora_stack(N, 3)
    want3 = [(:sp, 1, 1), (:sp, 1, 2), (:sp, 1, 3)]
    ce3 = build_class_equations(pents3, ns, N, want3)
    return FLRWModel{Float64}(N, ns, dx, dτ, m, κ,
        (s_boost=s_boost, s_euclid=s_euclid),
        pents2, ce2[(:pole, 0, 0)], (ce2[(:dg, 0, 1)], ce2[(:dg, 0, 2)], ce2[(:dg, 0, 3)]),
        pents3, (ce3[(:sp, 1, 1)], ce3[(:sp, 1, 2)], ce3[(:sp, 1, 3)]))
end

# residual 4-vector of the tent-move step: unknowns x = (d1, d2, d3, s_next), given s_cur, dτ.
function step_residual(M::FLRWModel, s_cur, dτ, x::AbstractVector{T}) where {T}
    pole = -T(dτ)^2
    esq = _esq_window2((T(s_cur), x[4], pole, x[1], x[2], x[3]), M.ns, M.N)
    Gp = class_residual(M.eq_pole, esq, M.pents2, M.κ, M.conv; dust=M.m / (2dτ))
    G1 = class_residual(M.eq_dg[1], esq, M.pents2, M.κ, M.conv)
    G2 = class_residual(M.eq_dg[2], esq, M.pents2, M.κ, M.conv)
    G3 = class_residual(M.eq_dg[3], esq, M.pents2, M.κ, M.conv)
    return [Gp, G1, G2, G3]
end
step_residual(M::FLRWModel, s_cur, x::AbstractVector) = step_residual(M, s_cur, M.dτ, x)

"""
    flrw_step(M, s_cur, dτ; x0=nothing, rtol=1e-24, maxit=40) -> (d1, d2, d3, s_next)

One tent-move slab advance with lapse `dτ` (uniform across the slice = synchronous): Newton-solve
the {pole, d1, d2, d3} Regge equations for (d1, d2, d3, s_next).

Numerics notes (both verified empirically):
  * The Hamiltonian constraint is QUADRATIC in the expansion rate, so `x0` must seed the correct
    (expanding/contracting) branch — at a time-symmetric point the Jacobian is genuinely singular.
  * Newton runs in Double64: the staircase diagonals sweep through NEAR-NULL (l²≈0 by cancellation
    of dτ² against the spatial scale as dτ/(a·dx) crosses 1), where the Float64 deficit chain loses
    ~9 digits and its noise floor exceeds the dust source term.
Convergence is `‖r‖∞ < rtol·(m/2dτ)` — relative to the dust scale of the pole equation (the
Double64 noise floor of the residual sits near 1e-16 relative, so rtol=1e-12 is comfortably
converged while far below any Float64-visible error in the returned lengths).
"""
function flrw_step(M::FLRWModel, s_cur, dτ=M.dτ; x0=nothing, rtol=1e-12, maxit=40)
    D = Double64
    x = D.(x0 === nothing ?
        [-dτ^2 + s_cur, -dτ^2 + 2s_cur, -dτ^2 + 3s_cur, s_cur * (1 + 3dτ)] : x0)
    F = xx -> step_residual(M, D(s_cur), D(dτ), xx)
    tol = D(rtol) * abs(D(M.m) / (2 * D(dτ)))
    for _ in 1:maxit
        r = F(x)
        maximum(abs, r) < tol && return (Float64(x[1]), Float64(x[2]), Float64(x[3]), Float64(x[4]))
        J = ForwardDiff.jacobian(F, x)
        x .-= J \ r
    end
    r = F(x)
    maximum(abs, r) < 1e6 * tol &&
        return (Float64(x[1]), Float64(x[2]), Float64(x[3]), Float64(x[4]))
    error("flrw_step: Newton failed (‖r‖∞ = $(Float64(maximum(abs, r))), tol=$(Float64(tol)))")
end

"""
    spatial_constraint(M, s_prev, slabA, s_cur, slabB, s_next; dτA=M.dτ, dτB=M.dτ) -> (r1, r2, r3)

Residuals of the slice-`cur` spatial-edge Regge equations (the second-order / Raychaudhuri
consistency), given the two adjacent solved slabs (with their lapses). Pure diagnostics — never
solved.
"""
function spatial_constraint(M::FLRWModel, s_prev, slabA, s_cur, slabB, s_next;
                            dτA=M.dτ, dτB=M.dτ)
    D = Double64                       # boost deficits are Float64-garbage (precision law #1)
    esq = _esq_window3(D.((s_prev, s_cur, s_next, -dτA^2, slabA..., -dτB^2, slabB...)), M.ns, M.N)
    return ntuple(i -> Float64(class_residual(M.eq_sp[i], esq, M.pents3, M.κ, M.conv)), 3)
end

"""
    flrw_march(M, a0; a_end=1.0, lna_step=1e-3, H_seed) -> (t, a, cons)

March the homogeneous universe from `a0` to `a_end` with per-slab lapse dτ_n = lna_step / H_n,
where H_n is the LATTICE's own expansion rate from the previous slab (the analytic solution is
never used to drive the evolution — `H_seed` only seeds the very first Newton branch choice).
Returns cosmic times `t` (proper time along comoving worldlines), scale factors `a`, and the
per-step max |spatial constraint| residual (NaN for the first step — needs two slabs).
"""
function flrw_march(M::FLRWModel, a0::Float64; a_end::Float64=1.0, lna_step::Float64=1e-3,
                    H_seed::Float64=a0^(-1.5))
    t = Float64[0.0]; a = Float64[a0]; cons = Float64[NaN]
    s_cur = (a0 * M.dx)^2
    H = H_seed
    slab_prev = nothing
    prev_dτ = NaN
    while a[end] < a_end
        dτ = lna_step / H
        ā2 = ((a[end] * (1 + H * dτ / 2)) * M.dx)^2       # midpoint-scale seed
        x0 = [-dτ^2 + ā2, -dτ^2 + 2ā2, -dτ^2 + 3ā2, (a[end] * (1 + H * dτ) * M.dx)^2]
        d1, d2, d3, s_next = flrw_step(M, s_cur, dτ; x0=x0)
        a_next = sqrt(s_next) / M.dx
        push!(a, a_next); push!(t, t[end] + dτ)
        if slab_prev !== nothing
            r = spatial_constraint(M, slab_prev[1], slab_prev[2], s_cur, (d1, d2, d3), s_next;
                                   dτA=prev_dτ, dτB=dτ)
            push!(cons, maximum(abs, r))
        else
            push!(cons, NaN)
        end
        H = (a_next - a[end-1]) / (dτ * a[end-1])          # lattice's own rate for the next slab
        slab_prev = (s_cur, (d1, d2, d3))
        prev_dτ = dτ
        s_cur = s_next
    end
    return t, a, cons
end
