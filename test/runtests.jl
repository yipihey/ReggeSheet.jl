using Test

# Deferred-AD probe (ForwardDiff always present in the test env; Enzyme optional).
ad_ok = false
try
    @eval using ForwardDiff
    global ad_ok = true
catch
    global ad_ok = false
end

@testset "ReggeSheet" begin
    @testset "geometry primitives" begin
        include("test_geometry.jl")
    end
    @testset "4D complex connectivity" begin
        include("test_complex.jl")
    end
    @testset "deficit angles (flat closure)" begin
        include("test_deficit.jl")
    end
    @testset "background FLRW (gate #1)" begin
        # lna_step calibrated by the convergence sweep (see NOTES.md); the march is ~O(minutes).
        global GATE1_LNA_STEP = 3e-3
        include("test_background.jl")
    end
    @testset "Zel'dovich plane wave (gate #2)" begin
        # 12 steps ≈ CI-affordable minutes; longer marches + the N=4 spatial-convergence point
        # are documented in NOTES.md.
        global GATE2_STEPS = 12
        include("test_zeldovich.jl")
    end
    if ad_ok
        @testset "differentiability (ForwardDiff)" begin
            include("test_ad.jl")
        end
    end
end
