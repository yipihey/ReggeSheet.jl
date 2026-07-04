# Zel'dovich plane wave (gate #2): analytic solution, lattice seeding, marching driver.
#
# EdS, plane-symmetric dust in synchronous-comoving gauge — EXACT GR solution pre-crossing:
#   X(q,t) = q + D(a)·s(q),   s(q) = (A/k)·sin(k q),   D(a) = a/a_norm (growing mode)
#   3-metric = a(t)²·δ_ab dX^a dX^b  (Euclidean in the displaced coordinates X — so every spatial
#   chord is CLOSED FORM: ℓ² = a²·|ΔX|²),  poles: l² = −dτ² exactly (comoving observers).
# Density: 1+δ = 1/(1 + D s′(q)),  s′(q) = A cos(k q); first crossing when D·A → 1.
#
# Lattice marching (the general engine, gauge-fixed):
#   * lapse: poleB := −dτ² (synchronous);
#   * shift: the 3 axis-offset incoming diagonals per vertex pinned to pole + sp1 (the discrete
#     zero-shift/orthogonality choice — the FLRW ansatz's silent gauge);
#   * all other slab-B edges solved by solve_slab! (global LM Gauss–Newton, Double64 geometry).
#
# The window (slices 0,1,2) SHIFTS after each solve: slice1→slice0, slice2→slice1, slabB→slabA;
# the edge-id shift map is combinatorial (computed once).

using Printf: @printf

"""    zeldovich_X(q, a; A, k, anorm) — displaced comoving coordinate X = q + D(a)s(q)."""
@inline zel_D(a; anorm=1.0) = a / anorm
@inline zel_s(q; A, k) = (A / k) * sin(k * q)
@inline zel_X(q, a; A, k, anorm=1.0) = q + zel_D(a; anorm=anorm) * zel_s(q; A=A, k=k)

"""
    zel_spatial_lsq(qv, qw, a; A, k, anorm) -> exact spatial chord l² between comoving points

`qv, qw` are 3-vectors of comoving coordinates (x displaced by the wave; y,z unaffected).
"""
function zel_spatial_lsq(qv, qw, a; A, k, anorm=1.0)
    dX = zel_X(qw[1], a; A=A, k=k, anorm=anorm) - zel_X(qv[1], a; A=A, k=k, anorm=anorm)
    return a^2 * (dX^2 + (qw[2] - qv[2])^2 + (qw[3] - qv[3])^2)
end

"""
    zel_seed_window!(L, ec, dx, a0, a1, dτA; A, k, anorm) -> nothing

Seed slices 0,1 (exact chords at a0, a1), poles (−dτ²), and slab-A diagonals (midpoint-metric
chords, O(dτ³)) of the window state `L` from the analytic Zel'dovich solution. Slice 2 / slab B
are left for the solver.
"""
function zel_seed_window!(L::Vector{Float64}, ec::EdgeComplex, dx::Float64,
                          a0::Float64, a1::Float64, dτA::Float64; A, k, anorm=1.0)
    ns = ec.ns; N = ec.N
    qco(v) = begin                                        # comoving coords of spacetime vertex
        p = mod(v - 1, ns)
        (dx * (p % N), dx * ((p ÷ N) % N), dx * (p ÷ (N * N)))
    end
    # minimum-image comoving displacement (periodic; box = N·dx)
    box = N * dx
    mi(d) = (d = mod(d, box); d > box / 2 ? d - box : d)
    amid = (a0 + a1) / 2
    for (e, (v, w)) in enumerate(ec.edges)
        kind = ec.kind[e]
        qv = qco(v); qw = qco(w)
        Δ = (mi(qw[1] - qv[1]), mi(qw[2] - qv[2]), mi(qw[3] - qv[3]))
        qw_near = (qv[1] + Δ[1], qv[2] + Δ[2], qv[3] + Δ[3])   # unwrapped neighbor position
        if kind === :sp0
            L[e] = zel_spatial_lsq(qv, qw_near, a0; A=A, k=k, anorm=anorm)
        elseif kind === :sp1
            L[e] = zel_spatial_lsq(qv, qw_near, a1; A=A, k=k, anorm=anorm)
        elseif kind === :poleA || kind === :poleB
            L[e] = -dτA^2
        elseif kind === :dgA
            L[e] = -dτA^2 + zel_spatial_lsq(qv, qw_near, amid; A=A, k=k, anorm=anorm)
        end
        # sp2 / dgB left to the solver (seeded per step by the march driver)
    end
    return nothing
end

"""
    window_shift_map(ec) -> Vector{Int}

`map[e_new] = e_old` such that the slice-s data of the new window (s=0,1 and slab A) comes from
slice s+1 (slab B) of the old window. Entries are 0 where no source exists (slice-2/slab-B edges —
reseeded by the march).
"""
function window_shift_map(ec::EdgeComplex)
    ns = ec.ns
    m = zeros(Int, length(ec.edges))
    for (e, (v, w)) in enumerate(ec.edges)
        k = ec.kind[e]
        (k === :sp0 || k === :sp1 || k === :poleA || k === :dgA) || continue
        src = (v + ns, w + ns)
        m[e] = ec.eid[src]
    end
    return m
end

"""
    gauge_pins(ec) -> Vector{Int}  — the slab-B axis-offset (|Δq|²=1) diagonals (3 per vertex).
"""
function gauge_pins(ec::EdgeComplex)
    ns = ec.ns; N = ec.N
    mi(d) = (d = mod(d, N); d > N ÷ 2 ? d - N : d)
    pins = Int[]
    for (e, (v, w)) in enumerate(ec.edges)
        ec.kind[e] === :dgB || continue
        pv = mod(v - 1, ns); pw = mod(w - 1, ns)
        di = mi(pw % N - pv % N); dj = mi((pw ÷ N) % N - (pv ÷ N) % N)
        dk = mi(pw ÷ (N * N) - pv ÷ (N * N))
        (di^2 + dj^2 + dk^2 == 1) && push!(pins, e)
    end
    return pins
end

# spatial-edge partner (u,w in slice 1) of a slab-B diagonal edge (b_u, t_w)
function _diag_sp1_partner(ec::EdgeComplex, e::Int)
    ns = ec.ns
    (v, w) = ec.edges[e]
    b = (v - 1) ÷ ns == 1 ? v : w                        # slice-1 endpoint
    t = (v - 1) ÷ ns == 2 ? v : w                        # slice-2 endpoint
    u1 = b                                               # slice-1 id
    w1 = t - ns                                          # slice-1 id of the top's column
    key = u1 < w1 ? (u1, w1) : (w1, u1)
    return ec.eid[key]
end

"""
    march_zeldovich(N, dx, a_start, n_steps; A, k, lna_step, m, verbose) -> results

March the plane wave with the gauge-fixed general engine. Per step: pin poles (−dτ²) and axis
diagonals (pole + sp1), seed the rest by expansion-extrapolation, solve_slab!, measure against
the analytic solution, shift the window. Returns per-step diagnostics:
(t, a_mean, err_x, err_perp, maxr, bianchi) where err_x = max rel deviation of slice-2 x-edge
proper lengths vs analytic, err_perp = same for transverse edges (plane-symmetry emergence).
"""
function march_zeldovich(N::Int, dx::Float64, a_start::Float64, n_steps::Int;
                         A::Float64, k::Float64, lna_step::Float64=3e-3,
                         ρ0::Float64=3 / (8π), anorm::Float64=1.0, verbose::Bool=false)
    ec = EdgeComplex(N)
    tb = TentBlocks(ec)
    pins = gauge_pins(ec)
    shiftmap = window_shift_map(ec)
    m = ρ0 * dx^3
    ns = ec.ns
    box = N * dx

    # EdS clock: t(a) = t0 a^{3/2}, t0 = 2/3 (H0 = 1)
    t0 = 2 / 3
    a0 = a_start
    H0 = a0^(-1.5)
    dτ = lna_step / H0
    a1 = ((t0 * a0^1.5 + dτ) / t0)^(2 / 3)               # exact EdS advance for the seed slab

    L = zeros(Float64, length(ec.edges))
    zel_seed_window!(L, ec, dx, a0, a1, dτ; A=A, k=k, anorm=anorm)

    # per-edge analytic reference for slice 2 (recomputed each step)
    qco(v) = begin
        p = mod(v - 1, ns)
        (dx * (p % N), dx * ((p ÷ N) % N), dx * (p ÷ (N * N)))
    end
    mi(d) = (d = mod(d, box); d > box / 2 ? d - box : d)

    tcur = t0 * a1^1.5                                    # cosmic time at slice 1
    acur = a1
    out = NamedTuple[]
    for step in 1:n_steps
        H = acur^(-1.5)
        dτB = lna_step / H
        a2_an = ((tcur + dτB) / t0)^(2 / 3)               # analytic scale at the new slice
        # ---- seed slab B ----
        gr = (a2_an / acur)^2                             # expansion-extrapolation factor
        for (e, (v, w)) in enumerate(ec.edges)
            kd = ec.kind[e]
            if kd === :poleB
                L[e] = -dτB^2
            elseif kd === :sp2
                # extrapolate from the corresponding sp1 edge
                src = ec.eid[(v - ns, w - ns)]
                L[e] = L[src] * gr
            end
        end
        for e in pins                                     # shift gauge: pole + sp1 (zero-shift)
            L[e] = -dτB^2 + L[_diag_sp1_partner(ec, e)]
        end
        for (e, (v, w)) in enumerate(ec.edges)            # non-pinned diagonals: orthogonal seed
            (ec.kind[e] === :dgB && !(e in Set(pins))) || continue
            L[e] = -dτB^2 + L[_diag_sp1_partner(ec, e)]
        end
        # ---- solve ----
        dust = m / (2dτB)
        maxr = solve_slab!(L, ec, tb; κ=1 / (8π), conv=(s_boost=1.0, s_euclid=1.0),
                           dust_pole=dust, tol=1e-5 * dust, verbose=false, pinned=pins)
        # Bianchi check: residuals of the PINNED edges' equations
        LD = Double64.(L)
        ε = zeros(Double64, length(ec.hinges)); kc = fill(:euclid, length(ec.hinges))
        compute_deficits!(ε, kc, ec, LD)
        bianchi = maximum(pins) do e
            Float64(abs(edge_residual(ec, e, LD, ε, kc, 1 / (8π),
                                      (s_boost=1.0, s_euclid=1.0))))
        end
        # ---- measure slice 2 vs analytic ----
        err_x = 0.0; err_perp = 0.0
        for (e, (v, w)) in enumerate(ec.edges)
            ec.kind[e] === :sp2 || continue
            qv = qco(v); qw = qco(w)
            Δ = (mi(qw[1] - qv[1]), mi(qw[2] - qv[2]), mi(qw[3] - qv[3]))
            qw_near = (qv[1] + Δ[1], qv[2] + Δ[2], qv[3] + Δ[3])
            lan = zel_spatial_lsq(qv, qw_near, a2_an; A=A, k=k, anorm=anorm)
            rel = abs(L[e] / lan - 1)
            if Δ[2] == 0 && Δ[3] == 0
                err_x = max(err_x, rel)                  # pure x-edges: the wave
            elseif Δ[1] == 0
                err_perp = max(err_perp, rel)            # pure transverse: plane symmetry
            end
        end
        tcur += dτB
        acur = a2_an                                      # clock bookkeeping only (analytic time)
        push!(out, (t=tcur, a=acur, err_x=err_x, err_perp=err_perp, maxr=maxr, bianchi=bianchi))
        verbose && @printf("step %3d: a=%.4f  err_x=%.3e  err_perp=%.3e  |r|=%.2e  bianchi=%.2e\n",
                           step, acur, err_x, err_perp, maxr, bianchi)
        # ---- shift the window ----
        Lold = copy(L)
        for e in eachindex(L)
            shiftmap[e] != 0 && (L[e] = Lold[shiftmap[e]])
        end
    end
    return out
end
