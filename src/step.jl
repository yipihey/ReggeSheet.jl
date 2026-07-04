# Inhomogeneous tent-slab advance: per-vertex Gauss–Newton least-squares blocks swept
# Gauss–Seidel style until the slab's Regge equations hold everywhere.
#
# Per vertex v (slab B = slice1→slice2 of the window):
#   unknowns x_v : slab-B diagonals ENDING at t_v  +  slice-2 spatial edges OWNED by v
#                  (owner = the endpoint with the smaller spacetime id; poles are GAUGE, pinned
#                  to −dτ² before the sweep)
#   equations    : poleB(v)  +  those same dgB edges  +  the slice-1 spatial edges owned by v
#                  (~15 eqs × ~14 unknowns; the 1 redundancy is the time-Bianchi/lapse gauge —
#                  solved least-squares, residual → 0 at convergence like the FLRW split)
# The two-slab window supplies the momentum: slab A (known) enters the slice-1 spatial equations.
#
# Deficits: a GLOBAL cache is computed once per sweep; each local solve recomputes only the
# hinges touched by its unknowns (override esq), so blocks stay O(local).

using LinearAlgebra: qr, ColumnNorm, Diagonal, lu, Symmetric

struct TentBlocks
    unknowns::Vector{Vector{Int}}                 # per spatial vertex: unknown edge ids
    equations::Vector{Vector{Int}}                # per spatial vertex: equation edge ids
    touched::Vector{Vector{Int}}                  # per spatial vertex: hinge ids to recompute
    poleB::Vector{Int}                            # per spatial vertex: its slab-B pole edge id
    colors::Vector{Vector{Int}}                   # vertex classes with disjoint touched sets
end

"""
    TentBlocks(ec) -> per-vertex unknown/equation/touched-hinge tables (combinatorial, once).
"""
function TentBlocks(ec::EdgeComplex)
    ns = ec.ns
    nv = ns
    unknowns = [Int[] for _ in 1:nv]
    poleB = zeros(Int, nv)
    spidx(v) = mod(v - 1, ns) + 1                 # spacetime id -> spatial vertex index
    for (e, (v, w)) in enumerate(ec.edges)
        k = ec.kind[e]
        if k === :dgB
            top = (v - 1) ÷ ns == 2 ? v : w       # the slice-2 endpoint
            push!(unknowns[spidx(top)], e)
        elseif k === :sp2
            push!(unknowns[spidx(min(v, w))], e)
        elseif k === :poleB
            poleB[spidx(v)] = e
        end
    end
    # touched hinges = hinges whose RING contains any unknown (a deficit depends on all edges of
    # its ring pentachora — hinge-triangle-only incidence truncates the derivative pattern)
    touched = [sort(unique(vcat(Int[], (ec.hinges_touching[e] for e in unknowns[v])...)))
               for v in 1:nv]
    # OVERLAPPING (Schwarz) equations: every COMPLETE, solvable-kind edge equation that references
    # any touched hinge — i.e. every equation this block's unknowns actually influence. Unknown
    # updates stay ownership-disjoint; only the local LSQ sees overlapping rows. (The disjoint
    # ownership pairing diverges: extreme blocks own many sp2 edges whose true constraints are
    # NEIGHBORS' dgB equations.)
    edge_of_hinge = [Int[] for _ in eachindex(ec.hinges)]
    for (e, eql) in enumerate(ec.eqs)
        (ec.complete[e] && ec.kind[e] in (:sp1, :dgB, :poleB)) || continue
        for (hi, _) in eql
            push!(edge_of_hinge[hi], e)
        end
    end
    equations = Vector{Vector{Int}}(undef, nv)
    for v in 1:nv
        eqs = Int[]
        for hi in touched[v]
            append!(eqs, edge_of_hinge[hi])
        end
        equations[v] = sort(unique(eqs))
    end
    # Greedy graph coloring so same-color blocks can run in parallel. The conflict set must be the
    # UNION of a block's unknown-touched hinges AND its equations' hinges: equation residuals READ
    # edges of all their hinges while another block may be WRITING its unknowns there (Double64 is
    # 128-bit ⇒ torn reads) — touched-only coloring races.
    hingeset = Vector{Vector{Int}}(undef, nv)
    for v in 1:nv
        hs = copy(touched[v])
        for eeq in equations[v], (hi, _) in ec.eqs[eeq]
            push!(hs, hi)
        end
        hingeset[v] = sort(unique(hs))
    end
    color = zeros(Int, nv)
    hinge_color = Dict{Int,Set{Int}}()                    # hinge -> colors already using it
    for v in 1:nv
        used = Set{Int}()
        for hi in hingeset[v]
            haskey(hinge_color, hi) && union!(used, hinge_color[hi])
        end
        c = 1
        while c in used; c += 1; end
        color[v] = c
        for hi in hingeset[v]
            push!(get!(() -> Set{Int}(), hinge_color, hi), c)
        end
    end
    ncol = maximum(color; init=1)
    colors = [findall(==(c), color) for c in 1:ncol]
    return TentBlocks(unknowns, equations, touched, poleB, colors)
end

# esq with a local override of the unknown edges (values from x) on top of the global state L.
@inline function _esq_override(ec::EdgeComplex, L::AbstractVector, ov::Dict{Int,T}) where {T}
    return (v::Int, w::Int) -> begin
        e = ec.eid[v < w ? (v, w) : (w, v)]
        ho = get(ov, e, zero(T))
        haskey(ov, e) ? ho : T(@inbounds L[e])
    end
end

# residuals of the block's equations for unknown values x (type-generic for ForwardDiff)
function _block_residual(ec::EdgeComplex, tb::TentBlocks, v::Int, L::AbstractVector,
                         ε::Vector, kindc::Vector{Symbol}, x::AbstractVector{T},
                         κ, conv, dust_pole) where {T}
    ov = Dict{Int,T}(tb.unknowns[v][i] => x[i] for i in eachindex(x))
    esq = _esq_override(ec, L, ov)
    # recompute touched hinge deficits with the override
    εloc = Dict{Int,T}()
    for hi in tb.touched[v]
        e, _, _ = hinge_deficit(ec.hinges[hi], ec.rings[hi], esq, ec.pents)
        εloc[hi] = e
    end
    out = Vector{T}(undef, length(tb.equations[v]))
    for (i, eeq) in enumerate(tb.equations[v])
        G = ec.kind[eeq] === :poleB ? T(dust_pole) : zero(T)   # every pole eq carries its dust
        for (hi, slot) in ec.eqs[eeq]
            h = ec.hinges[hi]
            a, b, c = esq(h[1], h[2]), esq(h[1], h[3]), esq(h[2], h[3])
            s = kindc[hi] == :boost ? T(conv.s_boost) : T(conv.s_euclid)
            εh = get(εloc, hi, T(ε[hi]))
            G += T(κ) * s * _dA_dslot(a, b, c, slot) * εh
        end
        out[i] = G
    end
    return out
end

"""
    solve_slab!(L, ec, tb; κ, conv, dust_pole, tol, maxit=25, verbose=false) -> max residual

GLOBAL damped Gauss–Newton for the whole slab: all slab-B unknowns solved together against all
complete {sp1, dgB, poleB} equations. The per-vertex Gauss–Seidel/Schwarz splitting DIVERGES here:
the equations are stiff (∂ε/∂l² ~ η/l², with O(|ln lna|) rapidities) and strongly cross-coupled,
so local block updates overshoot; the global Newton converges quadratically. The Jacobian is
assembled per-equation from LOCAL duals (an equation depends only on the edges of its hinges),
threaded, so assembly is O(#eqs · local-size), never global-dense differentiation.
"""
function solve_slab!(L::Vector{Float64}, ec::EdgeComplex, tb::TentBlocks;
                     κ, conv, dust_pole, tol::Float64, maxit::Int=25, verbose::Bool=false,
                     pinned::Vector{Int}=Int[], soft_gauge::Bool=true, wgauge::Float64=1e-3)
    D = Double64
    pinset = soft_gauge ? Set{Int}() : Set(pinned)
    # SOFT gauge (default): the shift-gauge edges STAY unknowns; their gauge conditions
    # (L[e] = caller-seeded pin value) enter as weighted least-squares rows that steer the
    # near-null shift space. HARD pinning (soft_gauge=false) excludes them — but the discrete
    # Bianchi identities are only approximate, so hard pins make successive windows exactly
    # unsolvable (measured: sp1 floors growing ×1e3/step). Soft gauge never conflicts with the
    # physics equations.
    gauge_edges = soft_gauge ? pinned : Int[]
    gauge_vals = Float64[L[e] for e in gauge_edges]
    U = Int[]
    for v in eachindex(tb.unknowns)
        append!(U, tb.unknowns[v])
    end
    sort!(U)
    filter!(e -> !(e in pinset), U)
    uidx = Dict{Int,Int}(e => i for (i, e) in enumerate(U))
    E = [e for e in eachindex(ec.edges)
         if ec.complete[e] && ec.kind[e] in (:sp1, :dgB, :poleB) && !(e in pinset)]
    # per-equation unknown support: eq e depends on the RING edges of its hinges (a deficit is a
    # function of ALL edges of its ring pentachora, not just the hinge triangle) ∩ unknowns.
    uset = Set(U)
    support = Vector{Vector{Int}}(undef, length(E))
    for (k, e) in enumerate(E)
        s = Int[]
        for (hi, _) in ec.eqs[e]
            for re in ec.ring_edges[hi]
                re in uset && push!(s, re)
            end
        end
        support[k] = sort(unique(s))
    end

    LD = D.(L)
    ε = zeros(D, length(ec.hinges))
    kindc = fill(:euclid, length(ec.hinges))
    nu = length(U); ne = length(E)
    r = zeros(D, ne)
    J = zeros(D, ne, nu)                                     # dense is fine at gate sizes
    maxr = Inf
    lm_λ = 1e-10                                             # LM damping (relative), adaptive
    for it in 1:maxit
        compute_deficits!(ε, kindc, ec, LD)
        # residuals (cheap, cached deficits)
        Threads.@threads :static for k in 1:ne
            e = E[k]
            de = ec.kind[e] === :poleB ? D(dust_pole) : zero(D)
            r[k] = edge_residual(ec, e, LD, ε, kindc, κ, conv; dust=de)
        end
        maxr = Float64(maximum(abs, r))
        verbose && println("  GN it $it: max|r| = $maxr")
        maxr < tol && break
        # Jacobian: per-equation local ForwardDiff over its support
        fill!(J, zero(D))
        Threads.@threads :static for k in 1:ne
            e = E[k]
            S = support[k]
            isempty(S) && continue
            de = ec.kind[e] === :poleB ? D(dust_pole) : zero(D)
            f = xx -> begin
                T = eltype(xx)
                ov = Dict{Int,T}(S[i] => xx[i] for i in eachindex(S))
                esq = _esq_override(ec, LD, ov)
                G = T(de)
                for (hi, slot) in ec.eqs[e]
                    h = ec.hinges[hi]
                    εh, kh, _ = hinge_deficit(h, ec.rings[hi], esq, ec.pents)
                    a, b, c = esq(h[1], h[2]), esq(h[1], h[3]), esq(h[2], h[3])
                    s = kh == :boost ? T(conv.s_boost) : T(conv.s_euclid)
                    G += T(κ) * s * _dA_dslot(a, b, c, slot) * εh
                end
                G
            end
            g = ForwardDiff.gradient(f, D[LD[s] for s in S])
            for (i, s) in enumerate(S)
                J[k, uidx[s]] = g[i]
            end
        end
        # Levenberg–Marquardt: adaptive damping instead of a line search — the system has a
        # near-null space (the un-gauge-fixed SHIFT freedom, ~3 modes/vertex), so small fixed λ
        # produces noise-dominated steps that a line search just rejects. LM raises λ until the
        # step reduces ‖r‖∞, then relaxes it.
        JtJ = transpose(J) * J
        dmax = maximum(i -> JtJ[i, i], 1:nu)
        Jtr = transpose(J) * r
        # soft gauge rows: w²·(x_g − pin) steering the shift null space (scaled to the physics rows)
        if soft_gauge && !isempty(gauge_edges)
            wg2 = D(wgauge)^2 * dmax
            for (gi, ge) in enumerate(gauge_edges)
                i = uidx[ge]
                JtJ[i, i] += wg2
                Jtr[i] += wg2 * (LD[ge] - D(gauge_vals[gi]))
            end
        end
        λ = D(lm_λ) * dmax
        accepted = false
        for _ in 1:12
            Δ = lu(JtJ + λ * Diagonal(ones(D, nu))) \ Jtr
            Lt = copy(LD)
            for (i, e) in enumerate(U)
                Lt[e] = LD[e] - Δ[i]
            end
            compute_deficits!(ε, kindc, ec, Lt)
            rt = 0.0
            for k in 1:ne
                e = E[k]
                de = ec.kind[e] === :poleB ? D(dust_pole) : zero(D)
                rt = max(rt, Float64(abs(edge_residual(ec, e, Lt, ε, kindc, κ, conv; dust=de))))
            end
            if rt < maxr
                LD = Lt
                lm_λ = max(lm_λ / 4, 1e-14)
                accepted = true
                break
            end
            λ *= 10
            lm_λ = min(lm_λ * 10, 1e2)
        end
        accepted || break                                    # LM exhausted: report current maxr
    end
    for i in eachindex(L)
        L[i] = Float64(LD[i])
    end
    return maxr
end

"""
    sweep_slab!(L, ec, tb; κ, conv, dust_pole, nsweep=30, tol, verbose=false) -> max residual

Gauss–Seidel–Newton sweeps over all vertex blocks, updating the slab-B unknowns in `L` in place.
`dust_pole` = m/(2dτ_B). Poles must already be pinned (gauge). Converges when the max block
residual < tol.

PRECISION: all geometry runs in Double64 — in the cosmological weak-field regime each boost-hinge
wedge rapidity is O(|ln lna_step|) and the deficit is their ~1e-8-relative cancellation; SO(1,1)
is abelian so no reformulation avoids the sum, Float64 boost deficits are garbage (measured O(10)
errors). Blocks within a color class have disjoint touched-hinge sets and run threaded.
"""
function sweep_slab!(L::Vector{Float64}, ec::EdgeComplex, tb::TentBlocks;
                     κ, conv, dust_pole, nsweep::Int=30, tol::Float64, verbose::Bool=false)
    D = Double64
    LD = D.(L)
    ε = zeros(D, length(ec.hinges))
    kindc = fill(:euclid, length(ec.hinges))
    maxr = Inf
    for sweep in 1:nsweep
        compute_deficits!(ε, kindc, ec, LD)
        maxr_atom = Threads.Atomic{Float64}(0.0)
        for cls in tb.colors
            Threads.@threads :static for v in cls
                isempty(tb.unknowns[v]) && continue
                x = D[LD[e] for e in tb.unknowns[v]]
                F = xx -> _block_residual(ec, tb, v, LD, ε, kindc, xx, κ, conv, dust_pole)
                r = F(x)
                Threads.atomic_max!(maxr_atom, Float64(maximum(abs, r)))
                J = ForwardDiff.jacobian(F, x)
                # Tikhonov-regularized Gauss–Newton with a trust-region cap: block Jacobians can
                # be strongly rank-deficient (extreme blocks own many spatial edges and no
                # diagonals — ownership by id is unbalanced), so null-space components must be
                # damped (λ) AND the step length capped — Gauss–Seidel supplies the missing
                # directions on later sweeps.
                JtJ = transpose(J) * J
                λ = D(1e-10) * maximum(i -> JtJ[i, i], 1:size(JtJ, 1))
                Δ = lu(JtJ + λ * Diagonal(ones(D, size(JtJ, 1)))) \ (transpose(J) * r)
                xscale = maximum(abs, x)
                cap = D(0.2) * xscale
                for i in eachindex(Δ)
                    Δ[i] = clamp(Δ[i], -cap, cap)
                end
                x .-= Δ
                for (i, e) in enumerate(tb.unknowns[v])
                    LD[e] = x[i]
                end
            end
        end
        maxr = maxr_atom[]
        verbose && println("  sweep $sweep: max|r| = $maxr")
        maxr < tol && break
    end
    for i in eachindex(L)
        L[i] = Float64(LD[i])
    end
    return maxr
end
