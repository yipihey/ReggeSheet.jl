# M4 DELIVERABLE: ReggeSheet vs DISCO-DJ nLPT on identical ICs — Ψ residuals, P(k) ratios,
# constraint drift; EdS + Planck18 ΛCDM.  Run:  julia --project=compare -t 16 compare/compare_nlpt.jl
using ReggeSheet, Printf, Random, Serialization
include(joinpath(@__DIR__, "ics.jl"))
include(joinpath(@__DIR__, "diagnostics.jl"))

const N = 8
const BOX_MPCH = 250.0
const SEED = 42
const A0 = 0.02
const SNAPS = [0.03, 0.045, 0.06]

function run_case(tag, cosmo)
    @printf("=== %s ===\n", tag)
    dx = BOX_MPCH / CH0 / N
    lpt, _ = make_ics(cosmo, N, BOX_MPCH, SEED)
    E_of_a = a -> hubble_E(cosmo, a)
    # seeded slice pair: a1 from the engine's CFL gap
    dτ = 0.4 * A0 * dx
    a1 = A0 * (1 + E_of_a(A0) * dτ)
    Ψ0 = psi_at(lpt, cosmo, A0, N)
    Ψ1 = psi_at(lpt, cosmo, a1, N)
    cache = joinpath(@__DIR__, "march_$(tag)_N$(N)_s$(SEED).jls")
    local snaps, diag
    if isfile(cache)
        snaps, diag = deserialize(cache)
        @printf("%s: loaded cached march (%d steps)\n", tag, length(diag))
    else
        t = @elapsed snaps, diag = march3d(N, dx, A0, a1, Ψ0, Ψ1;
                                           E_of_a=E_of_a,
                                           Ωm=Omega_m(cosmo) > 0.99 ? 1.0 : Omega_m(cosmo),
                                           snapshots=copy(SNAPS), verbose=true)
        serialize(cache, (snaps, diag))
        @printf("%s: %d steps in %.0fs; ham drift %.2e → %.2e\n", tag, length(diag), t,
                diag[1].ham, diag[end].ham)
    end
    q = zeros(N^3, 3)
    for k in 1:N, j in 1:N, i in 1:N
        p = 1 + (i - 1) + N * ((j - 1) + N * (k - 1))
        q[p, :] .= dx .* (i - 1, j - 1, k - 1)
    end
    out = NamedTuple[]
    for s in snaps
        Ψ3 = psi_at(lpt, cosmo, s.a, N; n_order=3)
        xr = regge_positions(s, N, dx, q .+ Ψ3)
        row = (a=s.a, ξdx=maximum(sqrt.(sum(abs2, Ψ3, dims=2))) / dx)
        resids = Dict{Int,Tuple{Float64,Float64}}()
        for no in 1:3
            Ψn = psi_at(lpt, cosmo, s.a, N; n_order=no)
            resids[no] = psi_residual(xr, Ψn, q)
        end
        kk, Pr = pk_of_positions(xr, N, N * dx)
        _, P3 = pk_of_positions(q .+ Ψ3, N, N * dx)
        push!(out, (a=s.a, ξdx=row.ξdx, resids=resids, k=kk, Pratio=Pr ./ P3))
        env = 0.03 * row.ξdx
        @printf("  a=%.3f (ξ/dx=%.3f, sys≲%.1e): |Ψ_R−Ψ_nLPT| rms (code) 1LPT=%.2e 2LPT=%.2e 3LPT=%.2e\n",
                s.a, row.ξdx, env * s.a * dx,
                resids[1][1], resids[2][1], resids[3][1])
        @printf("    P_Regge/P_3LPT: %s\n", join([@sprintf("%.4f", x) for x in Pr ./ P3], " "))
    end
    return out, diag
end

results = Dict()
for (tag, cosmo) in (("EdS", eds_cosmology()), ("LCDM", lcdm_cosmology()))
    results[tag] = run_case(tag, cosmo)
end

# ---- plots (CairoMakie → PDF) ----
using CairoMakie
fig = Figure(size=(1150, 780))
for (col, tag) in enumerate(("EdS", "LCDM"))
    out, diag = results[tag]
    ax1 = Axis(fig[1, col]; title="$tag: |Ψ_Regge − Ψ_nLPT| rms", xlabel="a",
               ylabel="rms [code]", yscale=log10)
    for no in 1:3
        lines!(ax1, [r.a for r in out], [r.resids[no][1] for r in out], label="$(no)LPT")
    end
    lines!(ax1, [r.a for r in out],
           [0.03 * r.ξdx * r.a * BOX_MPCH / CH0 / N for r in out],
           linestyle=:dash, color=:gray, label="ξ/dx systematics")
    axislegend(ax1, position=:lt)
    ax2 = Axis(fig[2, col]; title="$tag: P_Regge / P_3LPT (shared deposit)", xlabel="k [h/Mpc]",
               ylabel="ratio", xscale=log10)
    for r in out
        lines!(ax2, r.k, r.Pratio, label=@sprintf("a=%.3f", r.a))
    end
    hlines!(ax2, [1.0], color=:gray, linestyle=:dot)
    axislegend(ax2, position=:lb)
    ax3 = Axis(fig[3, col]; title="$tag: constraint drift", xlabel="step",
               ylabel="|ham|", yscale=log10)
    lines!(ax3, [d.step for d in diag], [max(d.ham, 1e-30) for d in diag])
end
outpdf = joinpath(@__DIR__, "..", "output", "compare_nlpt.pdf")
mkpath(dirname(outpdf))
save(outpdf, fig)
println("saved: ", outpdf)
