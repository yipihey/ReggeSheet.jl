# M5: past-lightcone crossing of Regge worldlines + kNN-CDF comparison against the SHARED
# DiscoDJNative crossing machinery (`lightcone_cross_ad`).
#
# Geometry: observer OUTSIDE the box along +x at comoving distance χ(a_mid) from the box
# center, so the past lightcone sweeps the whole box over a narrow scale-factor window
# (δa ≈ a²E/CH0 · L_box). The Regge side interpolates embedded snapshot positions linearly in
# a and bisects |x(a) − obs| = χ(a) per vertex; the reference runs the SAME root-find through
# the fused lightcone_cross_ad kernel (K=1 Zel'dovich stack — the standard-nLPT ψ₂ carries D₁²
# growth, incompatible with the kernel's exact-growth D₂plus column; at M4's measured
# amplitudes LPT orders are identical to 4 digits, so K=1 loses nothing).
#
# Conventions shared by both sides: positions/χ in comoving Mpc/h; "velocity" = dx/dlna
# (comoving Mpc/h per e-fold — lightcone_cross_ad's v_r = Σ f₁D_kΨ_k·r̂ is exactly d(D₁Ψ₁)/dlna
# projected radially for K=1); redshift-space offset s = x + (dx/dlna·r̂)·r̂.
using DiscoDJNative: comoving_distance, scale_factor_from_chi, lightcone_cross_ad, growth_D1
using Random: MersenneTwister

"""
    regge_lightcone(xs, as, obs, cosmo) -> (x_obs, a_cross, v_r, valid)

Per-vertex lightcone crossing of Regge worldlines given embedded snapshot positions
`xs[i]::ns×3` (comoving Mpc/h) at scale factors `as[i]` (ascending): linear-in-a worldline
interpolation, 60-step bisection of |x(a)−obs| − χ(a) = 0 on [as[1], as[end]].
`v_r` = dx/dlna·r̂ from the local snapshot slope at the crossing.
"""
function regge_lightcone(xs::Vector{Matrix{Float64}}, as::Vector{Float64},
                         obs::NTuple{3,Float64}, cosmo)
    ns = size(xs[1], 1)
    x_obs = zeros(ns, 3); a_cross = zeros(ns); v_r = zeros(ns); valid = falses(ns)
    seg(a) = clamp(searchsortedlast(as, a), 1, length(as) - 1)
    for p in 1:ns
        xp = a -> begin
            i = seg(a)
            t = (a - as[i]) / (as[i+1] - as[i])
            ntuple(c -> xs[i][p, c] * (1 - t) + xs[i+1][p, c] * t, 3)
        end
        F = a -> begin
            x = xp(a)
            sqrt((x[1] - obs[1])^2 + (x[2] - obs[2])^2 + (x[3] - obs[3])^2) -
                comoving_distance(cosmo, a)
        end
        lo, hi = as[1], as[end]
        Flo = F(lo)
        (Flo < 0) == (F(hi) < 0) && continue          # no crossing in the snapshot window
        for _ in 1:60
            mid = (lo + hi) / 2
            if (F(mid) < 0) == (Flo < 0)
                lo = mid; Flo = F(mid)
            else
                hi = mid
            end
        end
        a = (lo + hi) / 2
        x = xp(a)
        i = seg(a)
        dlna = log(as[i+1]) - log(as[i])
        slope = ntuple(c -> (xs[i+1][p, c] - xs[i][p, c]) / dlna, 3)
        d = (x[1] - obs[1], x[2] - obs[2], x[3] - obs[3])
        r = sqrt(d[1]^2 + d[2]^2 + d[3]^2)
        x_obs[p, :] .= x
        a_cross[p] = a
        v_r[p] = (slope[1] * d[1] + slope[2] * d[2] + slope[3] * d[3]) / r
        valid[p] = true
    end
    return (x_obs=x_obs, a_cross=a_cross, v_r=v_r, valid=valid)
end

"""
    redshift_space(x, v_r, obs) -> ns×3 positions with the radial dx/dlna offset applied
"""
function redshift_space(x::Matrix{Float64}, v_r::Vector{Float64}, obs::NTuple{3,Float64})
    s = copy(x)
    for p in 1:size(x, 1)
        d = (x[p, 1] - obs[1], x[p, 2] - obs[2], x[p, 3] - obs[3])
        r = sqrt(d[1]^2 + d[2]^2 + d[3]^2)
        for c in 1:3
            s[p, c] += v_r[p] * d[c] / r
        end
    end
    return s
end

"""
    knn_cdf(data, queries, k) -> sorted k-th-NN distances (CDF support; CDF value = (1:nq)/nq)

Banerjee–Abel kNN-CDF: distances from volume-filling query points to the k-th nearest data
point. Brute force (fine for ns ≲ 10³).
"""
function knn_cdf(data::Matrix{Float64}, queries::Matrix{Float64}, k::Int)
    nq, nd = size(queries, 1), size(data, 1)
    dk = zeros(nq)
    ds = zeros(nd)
    for i in 1:nq
        for j in 1:nd
            ds[j] = sqrt((data[j, 1] - queries[i, 1])^2 + (data[j, 2] - queries[i, 2])^2 +
                         (data[j, 3] - queries[i, 3])^2)
        end
        dk[i] = partialsort!(ds, k)
    end
    return sort!(dk)
end

"""uniform query points in [0,L]³ (fixed seed — both point sets see identical queries)"""
uniform_queries(M::Int, L::Float64; seed::Int=7) = L .* rand(MersenneTwister(seed), M, 3)
