# General 3D march (M4): seed from arbitrary displacement fields (e.g. DiscoDJNative nLPT),
# march with the validated scheme (Sorkin split + gauge pins + wc + wd + CFL + York), and record
# slice snapshots for the comparison harness. Cosmology enters ONLY through E(a) = H/H0 (the
# clock/lapse grading) and the dust mass m = (3Ω_m/8π)·dx³ — the Regge core is cosmology-free.
#
# Units: c = G = H0 = 1 (lengths in c/H0; DiscoDJNative Mpc/h ÷ 2997.92458).
#
# Seeding: slice l² = a²·|Δq + ΔΨ(a)|² (comoving displacement Ψ, X = q + Ψ; exact for Zel'dovich,
# LPT-accurate generally — the standard Lagrangian-code IC treatment); slab-A diagonals at the
# midpoint; the discrete York solve then repairs O(dτ³)+LPT constraint violations.

"""
    march3d(N, dx, a_grid, Ψ0, Ψ1; E_of_a, Ωm, snapshots, lna_cfl=0.4, wd=1.0, verbose)
        -> (snaps, diag)

March from two seeded slices at `a_grid[1] = (a0, a1)` with displacements `Ψ0, Ψ1` (ns×3,
comoving, in code units). `E_of_a(a)` = H/H0. Records the slice-1 spatial edge state + scale
whenever `a` crosses an entry of `snapshots`; returns those plus per-step diagnostics.
Each snapshot: (a, edges, lsq, x_seed) ready for `embed_slice`.
"""
function march3d(N::Int, dx::Float64, a0::Float64, a1::Float64,
                 Ψ0::Matrix{Float64}, Ψ1::Matrix{Float64};
                 E_of_a, Ωm::Float64=1.0, ΩΛ::Float64=0.0, snapshots::Vector{Float64},
                 lna_cfl::Float64=0.4, wd::Float64=1.0, n_max::Int=100_000,
                 verbose::Bool=false, sparse_threshold::Int=800)
    # Λ enters the ACTION via conv.lam = Λ = 3Ω_Λ (code units) — without it a ΛCDM march is a
    # pure-dust universe fed ΛCDM seeds and drifts once Ω_Λ(a) ≳ 0.5% (NOTES M4, a ≈ 0.13).
    conv = (s_boost=1.0, s_euclid=1.0, lam=3ΩΛ)
    ec = EdgeComplex(N)
    pins = gauge_pins(ec)
    pinsA = gauge_pins(ec; kind=:dgA)
    shiftmap = window_shift_map(ec)
    ns = ec.ns
    box = N * dx
    ρ0 = 3Ωm / (8π)
    m = ρ0 * dx^3

    qco(p) = (dx * ((p - 1) % N), dx * (((p - 1) ÷ N) % N), dx * ((p - 1) ÷ (N * N)))
    mi(d) = (d = mod(d, box); d > box / 2 ? d - box : d)

    # chord l² between slice-local vertices at scale a with displacement field Ψ
    function sp_lsq(v::Int, w::Int, a::Float64, Ψ::Matrix{Float64})
        qv = qco(v); qw = qco(w)
        d1 = mi(qw[1] - qv[1]) + Ψ[w, 1] - Ψ[v, 1]
        d2 = mi(qw[2] - qv[2]) + Ψ[w, 2] - Ψ[v, 2]
        d3 = mi(qw[3] - qv[3]) + Ψ[w, 3] - Ψ[v, 3]
        return a^2 * (d1^2 + d2^2 + d3^2)
    end

    # ---- seed the window ----
    dτA = lna_cfl * a0 * dx * 1.0                       # CFL lapse (per-slab constant)
    # keep the seeded (a0→a1) gap consistent with the caller's a-grid: use their dτ implied gap
    L = zeros(Float64, length(ec.edges))
    Ψmid = (Ψ0 .+ Ψ1) ./ 2
    amid = (a0 + a1) / 2
    for (e, (v, w)) in enumerate(ec.edges)
        kd = ec.kind[e]
        vv = mod(v - 1, ns) + 1; ww = mod(w - 1, ns) + 1
        if kd === :sp0
            L[e] = sp_lsq(vv, ww, a0, Ψ0)
        elseif kd === :sp1
            L[e] = sp_lsq(vv, ww, a1, Ψ1)
        elseif kd === :poleA || kd === :poleB
            L[e] = -dτA^2
        elseif kd === :dgA
            L[e] = -dτA^2 + (vv == ww ? 0.0 : sp_lsq(vv, ww, amid, Ψmid))
        end
    end
    r0, ham0 = initial_slab_solve!(L, ec; κ=1 / (8π), conv=conv,
                                   dust_pole=m / (2dτA), tol=1e-16, pins=pinsA)
    verbose && @printf("york: |r|=%.2e ham=%.2e\n", r0, ham0)

    # EdS-consistent clock for the seeded gap is the caller's job (a1 from a0 + dτA); track time
    acur = a1
    a_prev = a0                                          # scale of the window's slice-0 (for the
                                                         # de-trended KO target's gr_prev)
    snaps = NamedTuple[]
    diag = NamedTuple[]
    snapq = sort(copy(snapshots))
    step = 0
    while !isempty(snapq) && step < n_max
        step += 1
        H = E_of_a(acur)
        dτB = lna_cfl * acur * dx                        # CFL number lna_cfl in a·dx units
        # analytic-free scale advance estimate for seeding (lattice will refine): da = a·H·dτ
        a2_est = acur * (1 + H * dτB)
        gr = (a2_est / acur)^2
        for (e, (v, w)) in enumerate(ec.edges)
            kd = ec.kind[e]
            if kd === :poleB
                L[e] = -dτB^2
            elseif kd === :sp2
                L[e] = L[ec.eid[(v - ns, w - ns)]] * gr
            end
        end
        for (e, (v, w)) in enumerate(ec.edges)
            ec.kind[e] === :dgB || continue
            L[e] = -dτB^2 + L[_diag_sp1_partner(ec, e)]
        end
        # DE-TRENDED (comoving-linear) KO target: extrapolate l²/a² linearly, rescale to the new
        # epoch → gr·(2·l1 − l0·gr_prev). Exact for the homogeneous background (comoving squared
        # length constant ⇒ zero dissipation bias) and = the original stable linear extrapolation
        # for perturbations. The plain linear target 2·l1−l0 carries an O(dlna²) bias against the
        # a²-background that fed a weakly unstable Ω_m<1 marching mode; the geometric target
        # sp1²/sp0 removed it for homogeneous but OVERSHOT perturbations (measured: EdS full-amp
        # ham 6.7e-4→8e-2). This de-trended form fixes both.
        gr_prev = (acur / a_prev)^2
        diss = Tuple{Int,Float64}[]
        for (e, (v, w)) in enumerate(ec.edges)
            ec.kind[e] === :sp2 || continue
            l1 = L[ec.eid[(v - ns, w - ns)]]
            l0 = L[ec.eid[(v - 2ns, w - 2ns)]]
            push!(diss, (e, gr * (2l1 - l0 * gr_prev)))
        end
        dust = m / (2dτB)
        # tol ∝ dust ∝ dx²/(cfl·a): it SHRINKS as the march advances, and the constraint-drift
        # onset tracks tol(a) crossing the scheme's floor (measured a_on ≈ 290·dx²/cfl —
        # clean-exit → stall-exit transition). REGGE_TOLSCALE (default 1) is the A/B knob.
        tolscale = parse(Float64, get(ENV, "REGGE_TOLSCALE", "1.0"))
        maxr, cons = solve_slab!(L, ec; κ=1 / (8π), conv=conv,
                                 dust_pole=dust, tol=1e-5 * dust * tolscale, pins=pins,
                                 diss=diss, wd=wd, sparse_threshold=sparse_threshold)
        # lattice's own new scale: mean of sp2/sp1 ratios (the engine drives its own clock)
        sr = 0.0; nr = 0
        for (e, (v, w)) in enumerate(ec.edges)
            ec.kind[e] === :sp2 || continue
            sr += L[e] / L[ec.eid[(v - ns, w - ns)]]; nr += 1
        end
        anew = acur * sqrt(sr / nr)
        push!(diag, (step=step, a=anew, maxr=maxr, ham=cons.ham, mom=cons.mom))
        verbose && step % 10 == 0 &&
            @printf("step %4d: a=%.5f |r|=%.2e ham=%.2e\n", step, anew, maxr, cons.ham)
        # snapshot?
        if !isempty(snapq) && anew >= snapq[1]
            edges, ids = slice_edges(ec, 2)
            push!(snaps, (a=anew, edges=edges, lsq=Float64[L[i] for i in ids]))
            popfirst!(snapq)
        end
        a_prev = acur                                    # window shifts: next slice-0 = this s-1
        # shift window
        Lold = copy(L)
        for e in eachindex(L)
            shiftmap[e] != 0 && (L[e] = Lold[shiftmap[e]])
        end
        acur = anew
    end
    return snaps, diag
end
