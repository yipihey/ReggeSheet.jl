# Combinatorial sanity of the 4D tent-slab complex (deficit sign-convention validation is separate).
using ReggeSheet, Test

@testset "pentachora enumeration" begin
    N = 4
    pents = build_pentachora(N)
    ns = n_spatial(N)
    @test ns == 64
    @test length(pents) == N^3 * 6 * 4          # cubes × 6 Kuhn tets × 4 pentachora/prism
    # every pentachoron has 5 distinct spacetime vertices, split across the two slices
    @test all(length(unique(p)) == 5 for p in pents)
    @test all(any(v <= ns for v in p) && any(v > ns for v in p) for p in pents)  # spans both slices
end

@testset "hinge incidence" begin
    N = 4
    pents = build_pentachora(N)
    inc = hinge_incidence(pents)
    @test !isempty(inc)
    # each hinge lists only pentachora that actually contain all 3 of its vertices
    @test all(all(all(v in pents[pi] for v in h) for pi in ps) for (h, ps) in inc)
    # a diagonal face internal to one prism is shared by exactly 2 of its 4 pentachora, so the
    # complex must contain multiply-shared hinges (full interior-hinge rings need the 3-slice stack).
    @test maximum(length(ps) for ps in values(inc)) >= 2
    # sanity: no hinge is listed with duplicate pentachora
    @test all(length(unique(ps)) == length(ps) for ps in values(inc))
end
