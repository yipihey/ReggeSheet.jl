# Geometry primitives vs analytic configurations. Every primitive is validated here before use.
using ReggeSheet, Test, StaticArrays

@testset "Minkowski edge lengths" begin
    O  = SVector(0.0, 0.0, 0.0, 0.0)
    ex = SVector(0.0, 1.0, 0.0, 0.0)   # space-like unit
    et = SVector(1.0, 0.0, 0.0, 0.0)   # time-like unit
    @test edge_sq(O, ex) ≈ +1.0        # space-like l² > 0
    @test edge_sq(O, et) ≈ -1.0        # time-like  l² < 0
    @test edge_sq(O, SVector(0.0, 3.0, 4.0, 0.0)) ≈ 25.0

    # Boost invariance of the interval (rapidity η in the t–x plane).
    boost(η, p) = SVector(cosh(η)*p[1] + sinh(η)*p[2], sinh(η)*p[1] + cosh(η)*p[2], p[3], p[4])
    η = 0.7
    p = SVector(0.3, 1.1, -0.4, 0.8); q = SVector(1.7, -0.5, 0.9, 0.2)
    @test edge_sq(p, q) ≈ edge_sq(boost(η, p), boost(η, q))
end

@testset "Cayley–Menger volumes (Euclidean checks)" begin
    # Unit-edge regular simplices: V² = 3/16 (tri), 1/72 (tet), 5/9216 (4-simplex).
    @test tri_area_sq(1.0, 1.0, 1.0)                               ≈ 3/16
    @test tet_volume_sq(1.0,1.0,1.0,1.0,1.0,1.0)                   ≈ 1/72
    @test pent_volume_sq(ntuple(_->1.0, 10)...)                    ≈ 5/9216

    # Right triangle, legs 1 & 1 (hypotenuse² = 2): area = 1/2 ⇒ A² = 1/4.
    @test tri_area_sq(1.0, 1.0, 2.0)                               ≈ 1/4
    # Right tetrahedron on the axes: vertices O,e1,e2,e3 → V = 1/6 ⇒ V² = 1/36.
    #   edges (O-e1,O-e2,O-e3)=1 ; (e1-e2,e1-e3,e2-e3)=2
    @test tet_volume_sq(1.0,1.0,1.0, 2.0,2.0,2.0)                  ≈ 1/36
end

@testset "Volume from coordinates (Lorentzian boost invariance)" begin
    # Build a tetrahedron from 4 Minkowski 4-positions, compute its squared 3-volume via the
    # signed edge lengths, and check it is invariant under a boost (a Lorentzian isometry).
    verts = (SVector(0.0,0.0,0.0,0.0), SVector(0.0,1.0,0.0,0.0),
             SVector(0.2,0.0,1.0,0.0), SVector(0.0,0.3,0.0,1.0))
    esq(vs) = (edge_sq(vs[1],vs[2]), edge_sq(vs[1],vs[3]), edge_sq(vs[1],vs[4]),
               edge_sq(vs[2],vs[3]), edge_sq(vs[2],vs[4]), edge_sq(vs[3],vs[4]))
    V2 = tet_volume_sq(esq(verts)...)
    boost(η, p) = SVector(cosh(η)*p[1] + sinh(η)*p[2], sinh(η)*p[1] + cosh(η)*p[2], p[3], p[4])
    vb = map(v -> boost(0.5, v), verts)
    @test tet_volume_sq(esq(vb)...) ≈ V2
end

# all-off-diagonal-ones squared-edge matrix of a unit regular m-vertex simplex
_reg(m) = [i == j ? 0.0 : 1.0 for i in 1:m, j in 1:m]

@testset "Isometric embedding round-trips edge lengths" begin
    D = _reg(5)                                    # regular 4-simplex
    X, σ = embed_simplex(D)
    @test all(σ .≈ 1.0)                            # all-space-like ⇒ Euclidean signature
    lsq(a, b) = sum(σ[d] * (X[a, d] - X[b, d])^2 for d in 1:length(σ))
    @test all(lsq(a, b) ≈ D[a, b] for a in 1:5, b in 1:5)
end

@testset "Euclidean dihedral angles (regular simplices)" begin
    # Regular tetrahedron: dihedral at an edge = arccos(1/3) ≈ 70.53°.
    θtet, kind = dihedral(_reg(4), 1, 2)
    @test kind == :euclid
    @test θtet ≈ acos(1/3)
    # Regular 4-simplex: dihedral at a triangular hinge = arccos(1/4) ≈ 75.52°.
    θp, kindp = dihedral(_reg(5), 1, 2)
    @test kindp == :euclid
    @test θp ≈ acos(1/4)
end

@testset "Lorentzian dihedral is a boost (boost identity)" begin
    # Tetrahedron with a SPACE-LIKE hinge edge {v3,v4} along x, and two apexes in the (t,y) plane
    # related by a boost of rapidity α: the dihedral about that hinge must be the boost angle α.
    α = 0.6
    v3 = SVector(0.0, 0.0, 0.0, 0.0)
    v4 = SVector(0.0, 1.0, 0.0, 0.0)              # space-like hinge along x
    v1 = SVector(0.0, 0.5, 1.0, 0.0)              # apex (space-like, pure +y)
    v2 = SVector(sinh(α), 0.5, cosh(α), 0.0)      # apex = v1's y-direction boosted by α in t–y
    vs = (v1, v2, v3, v4)
    D  = [edge_sq(vs[a], vs[b]) for a in 1:4, b in 1:4]
    θ, kind = dihedral(D, 1, 2)                    # hinge = {v3,v4}
    @test kind == :boost
    @test θ ≈ α
end
