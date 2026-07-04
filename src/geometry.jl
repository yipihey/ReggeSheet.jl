# Geometry primitives for Lorentzian Regge calculus.
#
# All primitives are PURE and type-generic (no globals, no mutation of captured state) so
# ForwardDiff/Enzyme differentiate straight through. The fundamental Regge variables are the
# SIGNED squared edge lengths l² (space-like > 0, time-like < 0); simplex volumes and angles are
# computed intrinsically from them via the Cayley–Menger determinant — no embedding required.
#
# Sign conventions (metric η = diag(−1,+1,+1,+1)):
#   * edge_sq(p,q) = η_{μν}(q−p)^μ(q−p)^ν  → space-like edge l²>0, time-like l²<0.
#   * cm_volume_sq returns the SIGNED squared n-volume V² from the (n+1)-vertex squared-edge
#     matrix. For a Euclidean (all-space-like) simplex V²>0; sign flips track time-like content
#     and are pinned by the unit tests.

using LinearAlgebra: det
using StaticArrays: SVector, SMatrix, MMatrix, MVector

# ── Minkowski edge lengths ────────────────────────────────────────────────────
"""    minkowski_sq(d) -> ⟨d,d⟩ with η=diag(−1,+1,+1,+1)  (d a 4-vector)"""
@inline minkowski_sq(d::SVector{4}) = -d[1]*d[1] + d[2]*d[2] + d[3]*d[3] + d[4]*d[4]

"""    edge_sq(p, q) -> signed squared Lorentzian interval between 4-positions p,q."""
@inline edge_sq(p::SVector{4}, q::SVector{4}) = minkowski_sq(q - p)

# ── Cayley–Menger signed squared volume ───────────────────────────────────────
# For an n-simplex (m = n+1 vertices) with signed squared edge lengths D[i,j] = l²_{ij}:
#   V_n² = (-1)^(n+1) / (2^n (n!)²) · det(B),  B the (m+1)×(m+1) bordered Cayley–Menger matrix
#   ( B[1,1]=0, B[1,2:]=B[2:,1]=1, B[2:,2:]=D ).
# Euclidean checks: unit-edge regular simplex → V²= 3/16 (tri), 1/72 (tet), 5/9216 (4-simplex).
"""
    cm_volume_sq(D) -> signed V²

Signed squared volume of the n-simplex whose (n+1)×(n+1) matrix of signed squared edge lengths is
`D` (symmetric, zero diagonal). Pure & type-generic (ForwardDiff/Enzyme-safe).
"""
function cm_volume_sq(D::AbstractMatrix{T}) where {T}
    m = size(D, 1)                       # vertices = n+1
    @assert size(D, 2) == m "square edge-length matrix required"
    n = m - 1                            # simplex dimension
    B = MMatrix{m + 1, m + 1, T}(undef)
    @inbounds begin
        B[1, 1] = zero(T)
        for j in 1:m
            B[1, j + 1] = one(T); B[j + 1, 1] = one(T)
        end
        for i in 1:m, j in 1:m
            B[i + 1, j + 1] = D[i, j]
        end
    end
    coef = (isodd(n + 1) ? -one(T) : one(T)) / T(2^n * factorial(n)^2)
    return coef * det(SMatrix(B))
end

# Build the symmetric squared-edge matrix from the upper-triangular edge list (row-major:
# (1,2),(1,3),...,(1,m),(2,3),...). Used by the fixed-arity wrappers below.
@inline function _dmatrix(::Val{M}, edges::NTuple{K,T}) where {M,K,T}
    D = MMatrix{M, M, T}(undef)
    @inbounds begin
        for i in 1:M; D[i, i] = zero(T); end
        e = 0
        for i in 1:M-1, j in i+1:M
            e += 1; D[i, j] = edges[e]; D[j, i] = edges[e]
        end
    end
    return SMatrix(D)
end

"""    tri_area_sq(l12², l13², l23²) -> signed A²  (triangle / 2-simplex)."""
@inline tri_area_sq(a::T, b::T, c::T) where {T} = cm_volume_sq(_dmatrix(Val(3), (a, b, c)))

"""    tet_volume_sq(l12²,l13²,l14²,l23²,l24²,l34²) -> signed V²  (tetrahedron / 3-simplex)."""
@inline tet_volume_sq(e::Vararg{T,6}) where {T} = cm_volume_sq(_dmatrix(Val(4), e))

"""    pent_volume_sq(10 edge l²'s) -> signed V²  (pentachoron / 4-simplex).
Edge order: (1,2)(1,3)(1,4)(1,5)(2,3)(2,4)(2,5)(3,4)(3,5)(4,5)."""
@inline pent_volume_sq(e::Vararg{T,10}) where {T} = cm_volume_sq(_dmatrix(Val(5), e))

# ── Intrinsic isometric embedding of a single simplex ─────────────────────────
# A single n-simplex is flat, so its m=n+1 vertices embed isometrically in an n-dim space with
# metric η_local = diag(σ) (σ∈{±1}ⁿ, discovered from the data — Euclidean simplex ⇒ all +1; a
# time-spanning simplex ⇒ one −1). Signed LDLᵀ ("place the simplex") from the squared-edge matrix
# D: closed-form, differentiable (arithmetic + sqrt only — NO eigen), so ForwardDiff/Enzyme-safe.
# Returns (X, σ): X is m×n coordinates (vertex 1 at origin), σ is the length-n metric signature.
"""    embed_simplex(D) -> (X::m×n coords, σ::n signature)  — isometric local embedding from l²."""
function embed_simplex(D::AbstractMatrix{T}) where {T}
    m = size(D, 1); n = m - 1
    # Gram matrix of edge vectors e_a = v_{a+1} − v_1:  G[a,b] = (D[1,a+1]+D[1,b+1]−D[a+1,b+1])/2.
    G = MMatrix{n, n, T}(undef)
    @inbounds for a in 1:n, b in 1:n
        G[a, b] = (D[1, a + 1] + D[1, b + 1] - D[a + 1, b + 1]) / 2
    end
    X = zeros(MMatrix{m, n, T})          # X[1,:]=0; X[a+1,:] = coords of e_a
    σ = zeros(MVector{n, T})
    @inbounds for a in 1:n
        s = G[a, a]
        for d in 1:a-1
            s -= σ[d] * X[a + 1, d]^2
        end
        σ[a] = s < 0 ? -one(T) : one(T)
        xaa = sqrt(abs(s))
        X[a + 1, a] = xaa
        for b in a+1:n
            t = G[a, b]
            for d in 1:a-1
                t -= σ[d] * X[a + 1, d] * X[b + 1, d]
            end
            X[b + 1, a] = t / (σ[a] * xaa)
        end
    end
    return SMatrix(X), SVector(σ)
end

# σ-metric inner product of two length-n coordinate vectors.
@inline _sdot(σ, u, v) = sum(σ .* u .* v)

# Project vectors `us` (tuple of length-n coord vectors) onto the σ-orthogonal complement of the
# span of `basis` (tuple of length-n vectors). Signed Gram–Schmidt (differentiable). Returns the
# projected vectors as a tuple.
function _project_out(σ, basis, us)
    # σ-orthonormalize `basis` (assumed space-like here — the hinge of a Regge triangle spans two
    # spatial edges; time-like character is carried by the apex projections, handled by the caller).
    ortho = ()
    for w in basis
        for o in ortho
            w = w .- (_sdot(σ, w, o) / _sdot(σ, o, o)) .* o
        end
        ortho = (ortho..., w)
    end
    proj(u) = begin
        for o in ortho
            u = u .- (_sdot(σ, u, o) / _sdot(σ, o, o)) .* o
        end
        u
    end
    return map(proj, us)
end

# ── Dihedral angle at a hinge (space-like hinge; Lorentzian boost handled below) ──────────────
# Hinge = the (n−2)-face shared by facets i and j (= all vertices except {i,j}); apexes are vertex j
# (in facet i) and vertex i (in facet j). Project both apex vectors ⟂ the hinge and take the angle
# with the σ metric. If the ⟂ 2-plane is space-like ⇒ ordinary angle; if it contains a time
# direction ⇒ a boost (returned as `(η, :boost)`), see `dihedral`.
"""
    wedge_invariants(D, i, j) -> (α, β, γ)

Metric invariants of the simplex's wedge at the hinge = all vertices except `{i,j}`: the two
off-hinge vertex directions projected ⟂ the hinge, u_i and u_j, returned as
(α=⟨u_i,u_i⟩, β=⟨u_i,u_j⟩, γ=⟨u_j,u_j⟩). Everything about the dihedral/deficit is a function of
these. Pure & type-generic.
"""
function wedge_invariants(D::AbstractMatrix{T}, i::Int, j::Int) where {T}
    m = size(D, 1); n = m - 1
    X, σ = embed_simplex(D)
    row(k) = SVector{n,T}(ntuple(d -> X[k, d], n))
    hinge = Tuple(k for k in 1:m if k != i && k != j)      # n−1 vertices
    h0 = hinge[1]
    xh0 = row(h0)
    basis = Tuple(row(h) - xh0 for h in hinge[2:end])      # spans the hinge affine hull (n−2 dim)
    ui, uj = _project_out(σ, basis, (row(i) - xh0, row(j) - xh0))
    return (_sdot(σ, ui, ui), _sdot(σ, ui, uj), _sdot(σ, uj, uj))
end

"""
    dihedral(D, i, j) -> (θ, kind)

Interior dihedral angle of the n-simplex (given by its m×m signed squared-edge matrix `D`) at the
hinge shared by facets `i` and `j`. `kind` is `:euclid` (θ a real angle, radians) or `:boost`
(θ the boost rapidity η, when the plane ⟂ the hinge is Lorentzian).
"""
function dihedral(D::AbstractMatrix{T}, i::Int, j::Int) where {T}
    ni2, ip, nj2 = wedge_invariants(D, i, j)
    # The two apex directions span the 2-plane ⟂ the hinge. Its induced metric is Euclidean iff
    # their Gram determinant is positive (both norms > 0). Otherwise the plane is Lorentzian and the
    # dihedral is a BOOST (the time-tipped-tetrahedra deficit that carries the extrinsic curvature).
    gdet = ni2 * nj2 - ip^2
    if gdet > 0 && ni2 > 0
        c = clamp(ip / sqrt(ni2 * nj2), -one(T), one(T))
        return (acos(c), :euclid)                          # angle between the two apex directions ⟂ hinge
    else
        ch = abs(ip) / sqrt(abs(ni2 * nj2))                # cosh η = |⟨ui,uj⟩| / sqrt(|ni²·nj²|)
        return (acosh(max(ch, one(T))), :boost)
    end
end
