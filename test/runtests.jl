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
    # GATE #2 (test_zeldovich.jl): NOT yet enabled — the march is blocked by a parasitic root of
    # the two-slab recurrence (|λ_p|≈3/step; NOTES.md #11a); Kreiss–Oliger-style dissipation is
    # the next lever. Machinery validated so far: York initial data (ham → 2.4e-17), CFL regime,
    # constraint damping, first-step wave + emergent plane symmetry.
    # global GATE2_STEPS = 60; include("test_zeldovich.jl")
    if ad_ok
        @testset "differentiability (ForwardDiff)" begin
            include("test_ad.jl")
        end
    end
end
