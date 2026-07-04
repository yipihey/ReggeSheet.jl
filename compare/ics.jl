# M4 ICs bridge: DiscoDJNative → identical Ψ_ini into both pipelines.
#
# Units: DiscoDJNative works in Mpc/h; the Regge engine in c/H0 = 2997.92458 Mpc/h (H0=G=c=1).
# Growth: EdS D₁(a) = a (matches the engine's clock); ΛCDM growth from compute_timetables.
using DiscoDJNative
using DiscoDJNative: Cosmology, compute_timetables, linear_power_spectrum, generate_grf,
                     get_fourier_grid, compute_lpt, evaluate_lpt_psi_at_a, hubble_E, Omega_m
using Random

const CH0 = 2997.92458                     # Mpc/h per c/H0

"""EdS cosmology (Ω_m = 1) with Planck18-ish spectrum shape for the transfer function."""
eds_cosmology() = compute_timetables(Cosmology{Float64}(Omega_c=0.951, Omega_b=0.049, h=0.67742,
                                                        sigma8=0.8105, n_s=0.96822))
lcdm_cosmology() = compute_timetables(Cosmology("Planck18EEBAOSN"))

"""
    make_ics(cosmo, N, box_mpch, seed) -> (lpt, grid)

Gaussian ICs + nLPT shape fields on an N³ grid in a `box_mpch` box (both pipelines share these).
"""
function make_ics(cosmo, N::Int, box_mpch::Float64, seed::Int)
    pk = linear_power_spectrum(cosmo)
    grid = get_fourier_grid(N, box_mpch; T=Float64)
    fphi = generate_grf(:ngenic, 3, pk, N, box_mpch, seed; dtype=Float64, dtype_c=ComplexF64)
    lpt = compute_lpt(fphi, grid; n_order=3, backend=:threads)
    return lpt, grid
end

"""
    psi_at(lpt, cosmo, a, N) -> ns×3 comoving displacement in CODE units (c/H0)

nLPT displacement at scale factor `a`, flattened to the Regge vertex ordering
(x fastest — matches `lagrangian_vertices`/`qco`).
"""
function psi_at(lpt, cosmo, a::Float64, N::Int; n_order::Int=3)
    ψ = evaluate_lpt_psi_at_a(lpt, cosmo, a; n_order=n_order)   # (N,N,N,3), Mpc/h
    ns = N^3
    out = zeros(ns, 3)
    for k in 1:N, j in 1:N, i in 1:N
        p = 1 + (i - 1) + N * ((j - 1) + N * (k - 1))
        for c in 1:3
            out[p, c] = ψ[i, j, k, c] / CH0
        end
    end
    return out
end
