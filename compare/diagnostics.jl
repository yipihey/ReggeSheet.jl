# M4 diagnostics: Ψ-residuals + P(k) via the SHARED deposition (the comparison isolates dynamics).
using DiscoDJNative: sheet_deposit, evaluate_power_spectrum, lagrangian_grid_3d
using Statistics: mean

"""
    regge_positions(snap, N, dx, x_seed) -> ns×3 comoving positions (code units)

Embed the snapshot slice (scaled/physical lengths) and return COMOVING positions x/a.
`x_seed` = the LPT-position guess at the snapshot scale (also fixes the translation gauge).
"""
function regge_positions(snap, N::Int, dx::Float64, x_seed::Matrix{Float64})
    xphys = embed_slice(N, snap.edges, snap.lsq, x_seed .* snap.a, snap.a * dx)
    return xphys ./ snap.a
end

"""
    psi_residual(x_regge, Ψ_lpt, q) -> (rms, max)  — |Ψ_regge − Ψ_lpt| stats (code units)
"""
function psi_residual(x_regge::Matrix{Float64}, Ψ_lpt::Matrix{Float64}, q::Matrix{Float64})
    # marginalize the TRANSLATION gauge and a GLOBAL SCALE: the background-clock truncation of
    # the coarse-CFL march appears as δa/a·(q-spread) and swamps the perturbation signal at low
    # amplitude — fit it and report separately.
    qc = q .- mean(q, dims=1)
    d0 = (x_regge .- q) .- Ψ_lpt
    d0 .-= mean(d0, dims=1)
    s = sum(d0 .* qc) / sum(qc .* qc)             # LSQ global scale (δa/a)
    d = d0 .- s .* qc
    r = sqrt.(sum(abs2, d, dims=2))
    return sqrt(mean(r .^ 2)), maximum(r), s
end

"""
    pk_of_positions(x_com, N, box_code; ngrid=2N) -> (k, Pk)

P(k) of a point set via the SHARED tetrahedral deposit (positions comoving, code units →
converted to Mpc/h for DiscoDJNative). Same code path for Regge and nLPT.
"""
function pk_of_positions(x_com::Matrix{Float64}, N::Int, box_code::Float64; ngrid::Int=N)
    # NOTE: DiscoDJNative sheet_deposit requires the deposit mesh res == the Lagrangian grid N
    # (its internal reshape assumes res³ sample points); P(k) then reaches the N-grid Nyquist,
    # which covers every mode the N³ IC grid carries anyway.
    box = box_code * CH0
    ns = N^3
    xg = zeros(Float64, N, N, N, 3)
    for k in 1:N, j in 1:N, i in 1:N
        p = 1 + (i - 1) + N * ((j - 1) + N * (k - 1))
        for c in 1:3
            xg[i, j, k, c] = mod(x_com[p, c] * CH0, box)
        end
    end
    w = ones(Float64, N, N, N)
    @assert ngrid == N "sheet_deposit requires deposit res == Lagrangian N"
    mesh = sheet_deposit(xg, w, ngrid, box; n_sub=1)
    δ = mesh ./ mean(mesh) .- 1
    res = evaluate_power_spectrum(δ, box; bins=8)
    return res.k, res.Pk
end
