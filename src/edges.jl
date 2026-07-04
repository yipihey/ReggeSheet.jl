# General (inhomogeneous) per-edge machinery on a 3-slice tent window.
#
# Window: slices 0,1,2 (spacetime ids v = s·N³ + p), slab A = 0→1 (known when marching),
# slab B = 1→2 (being solved). Edges are ENUMERATED from the pentachora (no offset-pattern
# assumptions — the sorted staircase makes diagonal patterns vary near the periodic wrap) and
# addressed by a canonical id; the state is one flat Vector of signed l² over all window edges.
#
# Per-edge Regge equations are precomputed combinatorially ONCE (hinge list + rings + slots —
# metric-independent). Each hinge's deficit is computed once per sweep into a cache shared by all
# edge equations containing it (a hinge has 3 edges, so this saves ~3× plus ring-recomputation).

"""Edge kinds on the window: spatial per slice, pole/diagonal per slab."""
@inline function edge_kind(v::Int, w::Int, ns::Int)
    sv = (v - 1) ÷ ns; sw = (w - 1) ÷ ns
    if sv == sw
        return sv == 0 ? :sp0 : sv == 1 ? :sp1 : :sp2
    end
    slab = min(sv, sw)
    pole = mod(v - 1, ns) == mod(w - 1, ns)
    return slab == 0 ? (pole ? :poleA : :dgA) : (pole ? :poleB : :dgB)
end

struct EdgeComplex
    N::Int
    ns::Int
    pents::Vector{NTuple{5,Int}}
    edges::Vector{Tuple{Int,Int}}                        # canonical (v<w) list
    eid::Dict{Tuple{Int,Int},Int}
    kind::Vector{Symbol}
    # global hinge table (closed rings only)
    hinges::Vector{NTuple{3,Int}}
    rings::Vector{Vector{Tuple{Int,Int,Int}}}
    # per-edge equation: (hinge index, slot of this edge in that hinge)
    eqs::Vector{Vector{Tuple{Int,Int}}}
    complete::Vector{Bool}                               # all hinges of this edge have closed rings
    # DERIVATIVE INCIDENCE: a hinge's deficit depends on ALL edges of its ring's pentachora
    # (the wedge invariants use full 5-vertex D-matrices), NOT just the hinge triangle's 3 edges.
    ring_edges::Vector{Vector{Int}}                      # per hinge: edge ids of its ring pents
    hinges_touching::Vector{Vector{Int}}                 # per edge: hinges whose RING contains it
end

"""
    EdgeComplex(N) -> the 3-slice window complex with per-edge equation tables.

`complete[e]` marks edges whose Regge equation is fully evaluable on the window (all hinge rings
closed): poles/diagonals of both slabs and slice-1 spatial edges; slice-0/2 spatial edges are
incomplete (their stars extend outside the window) and are never solved here.
"""
function EdgeComplex(N::Int)
    ns = N^3
    pents = build_pentachora_stack(N, 3)
    inc = hinge_incidence(pents)

    eid = Dict{Tuple{Int,Int},Int}()
    edges = Tuple{Int,Int}[]
    function edge_id!(a::Int, b::Int)
        key = a < b ? (a, b) : (b, a)
        get!(eid, key) do
            push!(edges, key)
            length(edges)
        end
    end
    for p in pents, i in 1:4, j in i+1:5
        edge_id!(p[i], p[j])
    end
    kinds = [edge_kind(v, w, ns) for (v, w) in edges]

    hinges = NTuple{3,Int}[]
    rings = Vector{Tuple{Int,Int,Int}}[]
    hid = Dict{NTuple{3,Int},Int}()
    eqs = [Tuple{Int,Int}[] for _ in edges]
    complete = fill(true, length(edges))
    for (h, pids) in inc
        ring = hinge_ring(h, pids, pents)
        hi = 0
        if ring !== nothing
            push!(hinges, h); push!(rings, ring)
            hi = length(hinges); hid[h] = hi
        end
        for (slot, (a, b)) in enumerate(((h[1], h[2]), (h[1], h[3]), (h[2], h[3])))
            e = eid[(a, b)]
            if hi == 0
                complete[e] = false                       # an open hinge ⇒ equation incomplete
            else
                push!(eqs[e], (hi, slot))
            end
        end
    end
    # ring-based derivative incidence
    ring_edges = Vector{Vector{Int}}(undef, length(hinges))
    for hi in eachindex(hinges)
        es = Int[]
        for (pid, _, _) in rings[hi]
            p = pents[pid]
            for i in 1:4, j in i+1:5
                a, b = p[i] < p[j] ? (p[i], p[j]) : (p[j], p[i])
                push!(es, eid[(a, b)])
            end
        end
        ring_edges[hi] = sort(unique(es))
    end
    hinges_touching = [Int[] for _ in edges]
    for hi in eachindex(hinges), e in ring_edges[hi]
        push!(hinges_touching[e], hi)
    end
    return EdgeComplex(N, ns, pents, edges, eid, kinds, hinges, rings, eqs, complete,
                       ring_edges, hinges_touching)
end

# esq closure over the flat state vector L (signed l² per canonical edge).
@inline function state_esq(ec::EdgeComplex, L::AbstractVector)
    return (v::Int, w::Int) -> @inbounds L[ec.eid[v < w ? (v, w) : (w, v)]]
end

"""
    compute_deficits!(ε, kindc, ec, L) -> (ε, kindc)

Deficits for ALL closed-ring hinges from state `L` (one pass; the per-sweep cache). `kindc[i]`
records :euclid/:boost. Threaded over hinges.
"""
function compute_deficits!(ε::Vector{T}, kindc::Vector{Symbol}, ec::EdgeComplex,
                           L::AbstractVector{T}) where {T}
    esq = state_esq(ec, L)
    Threads.@threads :static for i in eachindex(ec.hinges)
        e, k, _ = hinge_deficit(ec.hinges[i], ec.rings[i], esq, ec.pents)
        @inbounds ε[i] = e
        @inbounds kindc[i] = k
    end
    return ε, kindc
end

"""
    edge_residual(ec, e, L, ε, kindc, κ, conv; dust=0) -> the Regge equation residual of edge `e`

Uses the per-sweep deficit cache `ε`. `dust` adds m/(2τ) for pole edges.
"""
function edge_residual(ec::EdgeComplex, e::Int, L::AbstractVector{T},
                       ε::Vector{T}, kindc::Vector{Symbol}, κ, conv; dust=zero(T)) where {T}
    G = T(dust)
    (v, w) = ec.edges[e]
    esq = state_esq(ec, L)
    @inbounds for (hi, slot) in ec.eqs[e]
        h = ec.hinges[hi]
        a, b, c = esq(h[1], h[2]), esq(h[1], h[3]), esq(h[2], h[3])
        s = kindc[hi] == :boost ? T(conv.s_boost) : T(conv.s_euclid)
        G += T(κ) * s * _dA_dslot(a, b, c, slot) * ε[hi]
    end
    return G
end
