# Slice embedding: reconstruct vertex POSITIONS from a slice's spatial edge lengths.
#
# The lattice state is intrinsic (edge l²); the M4 comparison needs positions — for Ψ-residuals
# against nLPT and for the SHARED tetrahedral deposition (DiscoDJNative `sheet_deposit`). For a
# quasi-linear slice this is a well-conditioned sparse least-squares multilateration:
#
#     min_x  Σ_edges ( |x_v − x_w + wrap|² − l²_vw )²
#
# with the periodic wrap fixed combinatorially by the comoving offset of each edge (min-image),
# the gauge (global translation) fixed by pinning the MEAN displacement to the seed's mean, and
# Gauss–Newton seeded by x⁰ = q + Ψ_guess (the LPT positions — an excellent starting point, so
# 2–4 iterations suffice). Rotational gauge freedom is broken by the wrap vectors themselves
# (periodicity pins the frame).

using SparseArrays: sparse
using LinearAlgebra: qr as _lqr

"""
    embed_slice(ec_or_dims, ls, x0, dx) -> x (ns×3)

Reconstruct positions of an N³ periodic slice from its spatial edge l² values.
`ls[(v,w)] = l²` for slice-local vertex ids (1..N³, v<w canonical), `x0` the seed positions
(ns×3, e.g. q + Ψ_LPT), `dx` the comoving spacing (box = N·dx). Returns the least-squares
embedding with the mean displacement gauge-pinned to the seed's.
"""
function embed_slice(N::Int, edges::Vector{Tuple{Int,Int}}, lsq::Vector{Float64},
                     x0::Matrix{Float64}, dx::Float64; maxit::Int=8, tol::Float64=1e-24)
    ns = N^3
    box = N * dx
    # comoving min-image wrap vector per edge (fixed combinatorially)
    qco(p) = (dx * ((p - 1) % N), dx * (((p - 1) ÷ N) % N), dx * ((p - 1) ÷ (N * N)))
    mi(d) = (d = mod(d, box); d > box / 2 ? d - box : d)
    wrapv = Vector{NTuple{3,Float64}}(undef, length(edges))
    for (e, (v, w)) in enumerate(edges)
        qv = qco(v); qw = qco(w)
        Δ = (mi(qw[1] - qv[1]), mi(qw[2] - qv[2]), mi(qw[3] - qv[3]))
        # wrap = (unwrapped Δq) − (raw coordinate difference): applied to x_w − x_v
        wrapv[e] = (Δ[1] - (qw[1] - qv[1]), Δ[2] - (qw[2] - qv[2]), Δ[3] - (qw[3] - qv[3]))
    end
    x = copy(x0)
    ne = length(edges)
    nun = 3ns
    for it in 1:maxit
        # residuals + sparse Jacobian triplets (6 nnz per edge row)
        r = zeros(Float64, ne + 3)                       # + 3 gauge rows (mean displacement)
        Is = Int[]; Js = Int[]; Vs = Float64[]
        sizehint!(Is, 6ne + 3ns)
        for (e, (v, w)) in enumerate(edges)
            d1 = x[w, 1] - x[v, 1] + wrapv[e][1]
            d2 = x[w, 2] - x[v, 2] + wrapv[e][2]
            d3 = x[w, 3] - x[v, 3] + wrapv[e][3]
            r[e] = d1^2 + d2^2 + d3^2 - lsq[e]
            for (a, da) in ((1, d1), (2, d2), (3, d3))
                push!(Is, e); push!(Js, (w - 1) * 3 + a); push!(Vs, 2da)
                push!(Is, e); push!(Js, (v - 1) * 3 + a); push!(Vs, -2da)
            end
        end
        # gauge rows: Σ_v (x − x0) = 0 per component, weighted like a typical edge row
        wg = 4 * dx^2 / ns
        for a in 1:3
            row = ne + a
            r[row] = wg * sum(x[v, a] - x0[v, a] for v in 1:ns)
            for v in 1:ns
                push!(Is, row); push!(Js, (v - 1) * 3 + a); push!(Vs, wg)
            end
        end
        maximum(abs, view(r, 1:ne)) < tol && break
        A = sparse(Is, Js, Vs, ne + 3, nun)
        Δu = _lqr(A) \ r
        for v in 1:ns, a in 1:3
            x[v, a] -= Δu[(v - 1) * 3 + a]
        end
    end
    return x
end

"""
    slice_edges(ec, slice) -> (edges, ids)

The spatial edges of window slice `slice` (0/1/2) as slice-local vertex pairs plus their edge ids
in `ec` (for reading l² out of the window state L).
"""
function slice_edges(ec::EdgeComplex, slice::Int)
    ns = ec.ns
    kinds = (:sp0, :sp1, :sp2)
    kk = kinds[slice + 1]
    edges = Tuple{Int,Int}[]
    ids = Int[]
    for (e, (v, w)) in enumerate(ec.edges)
        ec.kind[e] === kk || continue
        push!(edges, (v - slice * ns, w - slice * ns))
        push!(ids, e)
    end
    return edges, ids
end
