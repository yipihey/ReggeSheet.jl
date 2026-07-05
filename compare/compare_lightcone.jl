# M5 DELIVERABLE: Regge worldlines crossing the past lightcone vs DiscoDJNative
# lightcone_cross_ad on identical ICs — crossing positions, a_cross, RSD velocities, and
# kNN-CDFs (k = 1, 2) in real + redshift space.  EdS only (the lattice's exact model; the ΛCDM
# march is systematics-dominated past a≈0.13 until the Λ volume term lands — NOTES M4).
# Run:  julia --project=compare -t 16 compare/compare_lightcone.jl
using ReggeSheet, Printf, Random, Serialization, Statistics
include(joinpath(@__DIR__, "ics.jl"))
include(joinpath(@__DIR__, "diagnostics.jl"))
include(joinpath(@__DIR__, "lightcone.jl"))

const N = 8
const BOX_MPCH = 250.0
const SEED = 42
const A0 = 0.02
const A_MID = 0.35                       # lightcone sweeps the box around this epoch

cosmo = eds_cosmology()
dx = BOX_MPCH / CH0 / N
lpt, _ = make_ics(cosmo, N, BOX_MPCH, SEED)
E_of_a = a -> hubble_E(cosmo, a)

# ---- lightcone geometry: observer outside the box along +x ----
c0 = BOX_MPCH / 2
χc = comoving_distance(cosmo, A_MID)
obs = (c0 + χc, c0, c0)
a_far = scale_factor_from_chi(cosmo, χc + 230.0)   # box half-diagonal 216.5 + margin
a_near = scale_factor_from_chi(cosmo, χc - 230.0)
@printf("observer at χ(a=%.2f)=%.1f Mpc/h; crossing window a ∈ [%.4f, %.4f]\n",
        A_MID, χc, a_far, a_near)

# ---- march with DENSE snapshots across the crossing window ----
SNAPS = collect(range(0.99 * a_far, 1.01 * a_near, length=18))
dτ = 0.4 * A0 * dx
a1 = A0 * (1 + E_of_a(A0) * dτ)
cache = joinpath(@__DIR__, "march_EdS_lc_N$(N)_s$(SEED).jls")
local snaps, diag
if isfile(cache)
    snaps, diag = deserialize(cache)
    @printf("loaded cached lightcone march (%d steps, %d snaps)\n", length(diag), length(snaps))
else
    Ψ0 = psi_at(lpt, cosmo, A0, N)
    Ψ1 = psi_at(lpt, cosmo, a1, N)
    t = @elapsed snaps, diag = march3d(N, dx, A0, a1, Ψ0, Ψ1;
                                       E_of_a=E_of_a, Ωm=1.0,
                                       snapshots=copy(SNAPS), verbose=true)
    serialize(cache, (snaps, diag))
    @printf("march: %d steps in %.0fs; ham %.2e → %.2e; %d snapshots\n",
            length(diag), t, diag[1].ham, diag[end].ham, length(snaps))
end

# ---- embed each snapshot → comoving positions (Mpc/h) ----
q_code = zeros(N^3, 3)
for k in 1:N, j in 1:N, i in 1:N
    p = 1 + (i - 1) + N * ((j - 1) + N * (k - 1))
    q_code[p, :] .= dx .* (i - 1, j - 1, k - 1)
end
as = Float64[s.a for s in snaps]
xs = Vector{Matrix{Float64}}(undef, length(snaps))
for (i, s) in enumerate(snaps)
    Ψs = psi_at(lpt, cosmo, s.a, N)
    xs[i] = regge_positions(s, N, dx, q_code .+ Ψs) .* CH0
end
@printf("embedded %d snapshots (a = %.4f … %.4f)\n", length(xs), as[1], as[end])

# ---- Regge lightcone ----
lcR = regge_lightcone(xs, as, obs, cosmo)

# ---- reference: SHARED crossing kernel, K=1 Zel'dovich stack (see lightcone.jl header) ----
a_ref = A_MID
Ψshape = psi_at(lpt, cosmo, a_ref, N; n_order=1) .* CH0 ./ growth_D1(cosmo, a_ref)
Psi = reshape(Ψshape, N^3, 3, 1)
q_mpch = q_code .* CH0
ref = lightcone_cross_ad(Psi, q_mpch, cosmo, collect(obs), as[1], as[end]; rsd=true)

# ---- compare ----
both = lcR.valid .& Vector(ref.valid)
nb = count(both)
Δx = [sqrt(sum(abs2, lcR.x_obs[p, :] .- ref.x_obs[p, :])) for p in 1:N^3 if both[p]]
Δa = [abs(lcR.a_cross[p] - ref.a_cross[p]) for p in 1:N^3 if both[p]]
Δv = [abs(lcR.v_r[p] - ref.v_r[p]) for p in 1:N^3 if both[p]]
@printf("crossings: Regge %d / ref %d / both %d of %d\n",
        count(lcR.valid), count(ref.valid), nb, N^3)
@printf("|Δx_obs|  rms %.4f  max %.4f  Mpc/h  (dx = %.2f)\n",
        sqrt(mean(Δx .^ 2)), maximum(Δx), BOX_MPCH / N)
@printf("|Δa_cross| rms %.2e  max %.2e\n", sqrt(mean(Δa .^ 2)), maximum(Δa))
@printf("|Δv_r|    rms %.4f  max %.4f  Mpc/h per e-fold (Regge: finite-diff slope)\n",
        sqrt(mean(Δv .^ 2)), maximum(Δv))

# ---- kNN-CDFs, real + redshift space ----
sR = redshift_space(lcR.x_obs, lcR.v_r, obs)
sF = redshift_space(Matrix{Float64}(ref.x_obs), Vector{Float64}(ref.v_r), obs)
idx = findall(both)
Q = uniform_queries(8192, BOX_MPCH)
kn = Dict()
for (tag, XR, XF) in (("real", lcR.x_obs[idx, :], Matrix{Float64}(ref.x_obs)[idx, :]),
                      ("rsd", sR[idx, :], sF[idx, :]))
    for k in (1, 2)
        dR = knn_cdf(XR, Q, k)
        dF = knn_cdf(XF, Q, k)
        # KS-style sup |ΔCDF| on the merged support
        grid = sort(vcat(dR, dF))
        cdf(d, g) = searchsortedlast(d, g) / length(d)
        ks = maximum(abs(cdf(dR, g) - cdf(dF, g)) for g in grid)
        kn[(tag, k)] = (dR, dF, ks)
        @printf("kNN-CDF %s k=%d: sup|ΔCDF| = %.4f  (median d: Regge %.2f vs ref %.2f Mpc/h)\n",
                tag, k, ks, dR[end÷2], dF[end÷2])
    end
end

# ---- figure ----
using CairoMakie
fig = Figure(size=(1150, 420))
ax1 = Axis(fig[1, 1]; title="lightcone kNN-CDF (EdS, N=8, a≈$(A_MID))",
           xlabel="distance [Mpc/h]", ylabel="CDF")
for (k, ls) in ((1, :solid), (2, :dash))
    dR, dF, _ = kn[("rsd", k)]
    lines!(ax1, dR, (1:length(dR)) ./ length(dR); linestyle=ls, color=:firebrick,
           label="Regge k=$k (rsd)")
    lines!(ax1, dF, (1:length(dF)) ./ length(dF); linestyle=ls, color=:navy,
           label="nLPT k=$k (rsd)")
end
axislegend(ax1, position=:rb)
ax2 = Axis(fig[1, 2]; title="crossing-position residual", xlabel="|Δx_obs| [Mpc/h]",
           ylabel="count")
hist!(ax2, Δx; bins=30)
ax3 = Axis(fig[1, 3]; title="a_cross: Regge vs reference", xlabel="a_cross (ref)",
           ylabel="Δa_cross")
scatter!(ax3, [ref.a_cross[p] for p in idx], [lcR.a_cross[p] - ref.a_cross[p] for p in idx];
         markersize=4)
outpdf = joinpath(@__DIR__, "..", "output", "compare_lightcone.pdf")
mkpath(dirname(outpdf))
save(outpdf, fig)
println("saved: ", outpdf)
