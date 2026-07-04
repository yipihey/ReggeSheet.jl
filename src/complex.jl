# 4D simplicial complex of a tent-move slab: two Cauchy slices (N³ periodic vertices each) swept by
# tetrahedral prisms, each prism → 4 pentachora (PRISM_PENTA). Provides the pentachora and the
# hinge → incident-pentachora incidence needed for the deficit angles.
#
# Spacetime vertex id: slice s∈{0,1}, spatial linear index p∈1..N³  →  id = s*N³ + p.

"""    n_spatial(N) = N³ (vertices per slice)."""
@inline n_spatial(N::Int) = N * N * N

"""
    build_pentachora(N) -> Vector{NTuple{5,Int}}

All pentachora of the slab between slice 0 and slice 1 of a periodic N³ lattice, as 5-tuples of
spacetime vertex ids. Bottom (slice-0) id = p; top (slice-1) id = N³+p. 6 tets/cube × 4/prism.
"""
function build_pentachora(N::Int)
    ns = n_spatial(N)
    pents = NTuple{5,Int}[]
    sizehint!(pents, 6 * 4 * ns)
    @inbounds for K in 0:N-1, J in 0:N-1, I in 0:N-1
        # the 8 cube corners' slice-0 linear ids (1-based)
        corner = ntuple(8) do c
            a, b, cc = CUBE_CORNERS[c]
            slice_index(I + a, J + b, K + cc, N)
        end
        for tet in TETS
            # prism local vertices 1..4 = bottom tet (slice 0), 5..8 = top tet (slice 1).
            # CRITICAL (wall consistency): the staircase product triangulation is consistent iff
            # adjacent prisms order their shared wall's vertices identically. Sorting by LOCAL
            # CORNER id (lex in comoving offsets) achieves this — a shared face's relative order
            # depends only on the in-face offsets, which both cubes agree on — AND it is fully
            # translation-invariant (no special wrap cubes; every vertex has exactly 7 incoming
            # diagonals, which the gauge fixing relies on). Sorting by GLOBAL id is also wall-
            # consistent but breaks translation invariance at the periodic wrap (in-diagonal
            # counts 0–14) — do not use it.
            ts = _sort4(tet[1], tet[2], tet[3], tet[4])
            bottom = (corner[ts[1]], corner[ts[2]], corner[ts[3]], corner[ts[4]])
            local8 = (bottom[1], bottom[2], bottom[3], bottom[4],
                      ns + bottom[1], ns + bottom[2], ns + bottom[3], ns + bottom[4])
            for pen in PRISM_PENTA
                push!(pents, ntuple(t -> local8[pen[t]], 5))
            end
        end
    end
    return pents
end

"""
    build_pentachora_stack(N, nslices) -> Vector{NTuple{5,Int}}

Stack `nslices-1` tent slabs: slices s=0..nslices-1, spacetime ids v = s*N³ + p. Interior hinges
(complete pentachora rings) require ≥3 slices for slice-interior spatial triangles.
"""
function build_pentachora_stack(N::Int, nslices::Int)
    ns = n_spatial(N)
    slab = build_pentachora(N)
    all = NTuple{5,Int}[]
    sizehint!(all, length(slab) * (nslices - 1))
    for s in 0:nslices-2
        off = s * ns
        append!(all, [ntuple(t -> p[t] + off, 5) for p in slab])
    end
    return all
end

"""
    hinge_incidence(pents) -> Dict{NTuple{3,Int},Vector{Int}}

Map each triangular hinge (sorted 3-tuple of spacetime vertex ids) to the indices (into `pents`) of
the pentachora containing it. Every pentachoron contributes its C(5,3)=10 triangular faces.
"""
function hinge_incidence(pents::Vector{NTuple{5,Int}})
    inc = Dict{NTuple{3,Int},Vector{Int}}()
    @inbounds for (pi, p) in enumerate(pents)
        for a in 1:3, b in a+1:4, c in b+1:5
            key = _sort3(p[a], p[b], p[c])
            push!(get!(() -> Int[], inc, key), pi)
        end
    end
    return inc
end

@inline function _sort3(a::Int, b::Int, c::Int)
    a > b && ((a, b) = (b, a)); b > c && ((b, c) = (c, b)); a > b && ((a, b) = (b, a))
    return (a, b, c)
end

@inline function _sort4(a::Int, b::Int, c::Int, d::Int)
    a > b && ((a, b) = (b, a)); c > d && ((c, d) = (d, c))
    a > c && ((a, c) = (c, a)); b > d && ((b, d) = (d, b))
    b > c && ((b, c) = (c, b))
    return (a, b, c, d)
end
