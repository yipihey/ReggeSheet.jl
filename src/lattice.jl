# Periodic Lagrangian simplicial lattice (a Cauchy slice) + its 4D sweep connectivity.
#
# Spatial tessellation: the Kuhn 6-tetrahedra decomposition of each cube, shared VERBATIM with
# DiscoDJNative `sheet_deposit.jl` (`_CUBE_CORNERS`/`_TETS`) so the same `sheet_deposit` deposits
# both the Regge and nLPT sheets and the P(k) comparison isolates *dynamics*.

using StaticArrays: SVector

# Cube corners (a,b,c)∈{0,1}³ (1-based id → offset) and the 6 tetrahedra sharing the main diagonal
# corner 1 = (0,0,0) → corner 8 = (1,1,1).  Identical to DiscoDJNative sheet_deposit.jl:137-138.
const CUBE_CORNERS = ((0,0,0),(1,0,0),(0,1,0),(1,1,0),(0,0,1),(1,0,1),(0,1,1),(1,1,1))
const TETS = ((1,2,4,8), (1,4,3,8), (1,3,7,8), (1,7,5,8), (1,5,6,8), (1,6,2,8))

# A tetrahedral prism Δ³×I (bottom tet vertices 1..4 at time n, top 5..8 at time n+1, with
# top[k]=bottom[k]+4) triangulates into 4 pentachora by the standard "staircase": simplex p
# (p=1..4) is {1,2,3,4} with the first p bottom vertices replaced by their top partners along a
# monotone path. Concretely, the p-th 4-simplex vertex set (local ids into the 8-vertex prism):
#   PRISM_PENTA[p] = (top of first p, bottom of the rest)  in a fixed monotone order.
# This yields a consistent global orientation when the 4 bottom vertices are listed in increasing
# global (linear) index — see `slice_index`.
const PRISM_PENTA = ((1,2,3,4,8), (1,2,3,7,8), (1,2,6,7,8), (1,5,6,7,8))

"""    slice_index(i,j,k,N) -> linear 1-based vertex index on an N³ periodic slice (i,j,k 0-based)."""
@inline slice_index(i::Int, j::Int, k::Int, N::Int) =
    1 + mod(i, N) + N * (mod(j, N) + N * mod(k, N))

"""
    lagrangian_vertices(N, boxsize; T=Float64) -> Array{T}(N,N,N,3)

Unperturbed Lagrangian vertex positions of a periodic N³ slice, dx = boxsize/N. Layout matches
DiscoDJNative `lagrangian_grid_3d` / the nLPT ψ array so displacements add elementwise.
"""
function lagrangian_vertices(N::Int, boxsize::Real; T::Type{<:AbstractFloat}=Float64)
    dx = T(boxsize) / N
    q = Array{T,4}(undef, N, N, N, 3)
    @inbounds for k in 1:N, j in 1:N, i in 1:N
        q[i, j, k, 1] = (i - 1) * dx
        q[i, j, k, 2] = (j - 1) * dx
        q[i, j, k, 3] = (k - 1) * dx
    end
    return q
end
