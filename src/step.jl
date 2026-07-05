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

using LinearAlgebra: qr, ColumnNorm, Diagonal, lu, Symmetric, I, cholesky, cholesky!
using SparseArrays: sparse, spdiagm, SparseMatrixCSC

const _DUMP_A = Ref{Any}(nothing)   # debug hook (REGGE_TIMING=1): last assembled (Aall, rhsF)

# NOTE (measured, then REMOVED): warm-starting lm_λ across march solves — from either the
# end-of-solve (decayed) value or the last-accepted (pre-decay) value — made marches SLOWER
# (212s → 297s/347s for the N=5 8-step march): the ∞-norm LM acceptance rides a knife edge in
# λ, and a different starting rung changes which marginal steps get accepted, wandering the
# shift null space (solves exploding from n_it=2 to the maxit cap). Cold lm_λ=1e-10 per solve
# is the validated best; the ladder waste is addressed by the minimal-progress bar + ladder-
# plateau exit below instead.

@inline _ck(a::Int, b::Int) = a < b ? (a, b) : (b, a)

# sparse row accumulator: per-thread dense acc + mark (last row that touched the column) + the
# touched-column list — O(1) per contribution, no per-row Dict.
@inline function _acc!(acc::Vector{Float64}, mark::Vector{Int}, cols::Vector{Int},
                       k::Int, c::Int, v::Float64)
    @inbounds if mark[c] == k
        acc[c] += v
    else
        mark[c] = k
        acc[c] = v
        push!(cols, c)
    end
    return nothing
end

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
    solve_edges!(L, ec; unknowns, equations, dust_of, κ, conv, tol, ...) -> max residual

The general LM Gauss–Newton core: solve the Regge equations of `equations` (edge ids) for the
`unknowns` (edge ids), everything else frozen. `dust_of(e)` returns the matter term for edge e
(m/(2τ) for pole edges, 0 otherwise). Geometry in Double64 (the boost-deficit precision law).
The per-vertex Gauss–Seidel/Schwarz splittings DIVERGE on these stiff, strongly cross-coupled
equations (∂ε/∂l² ~ η/l²); the global Newton converges. The Jacobian is assembled per-equation
from LOCAL duals over the equation's ring-edge support, threaded.
"""
function solve_edges!(L::Vector{Float64}, ec::EdgeComplex;
                      unknowns::Vector{Int}, equations::Vector{Int}, dust_of,
                      κ, conv, tol::Float64, maxit::Int=25, verbose::Bool=false,
                      soft_equations::Vector{Int}=Int[], soft_weight::Float64=1.0,
                      reg_rows::Vector{Tuple{Int,Float64}}=Tuple{Int,Float64}[],
                      wreg::Float64=0.0, sparse_threshold::Int=800, hinge_jac::Bool=true,
                      jacT::DataType=Double64)
    # `reg_rows`: (unknown edge id, target value) — Kreiss–Oliger-style DISSIPATION rows
    # wreg·√dmax·(L[e] − target), penalizing the parasitic (time-oscillating) root of the
    # marching recurrence toward the smooth extrapolation `target` (its own recurrence has a
    # double root at λ=1). One unknown per row ⇒ diagonal contributions to the normal equations.
    D = Double64
    U = sort(unknowns)
    uidx = Dict{Int,Int}(e => i for (i, e) in enumerate(U))
    # CONSTRAINT DAMPING: `soft_equations` (the Hamiltonian/momentum constraints) join the
    # least-squares system with weight `soft_weight`. Free evolution of the hard (square) set
    # alone admits an exponentially growing constraint-violating mode (measured ×50–1e3/step);
    # the weighted rows pull the trajectory back onto the constraint surface each step. With an
    # on-shell past the combined minimum sits at the (tiny) approximate-Bianchi floor.
    nh = length(equations)
    E = vcat(equations, soft_equations)
    wrow = vcat(ones(D, nh), fill(D(soft_weight), length(soft_equations)))
    nu = length(U); ne = length(E)
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
    # ── per-hinge Jacobian machinery (fast assembly path): differentiate each hinge's deficit
    # ONCE over its ring unknowns, then chain-rule into every equation referencing the hinge:
    #   ∂G_e/∂u = κ·s_h·[ ∂(∂A/∂slot)/∂u · ε_h  +  (∂A/∂slot) · ∂ε_h/∂u ]
    # term 1 lives on the 3 hinge-triangle edges, term 2 on the ring edges; ε_h comes from the
    # Double64 cache. vs the per-equation path this evaluates each hinge once instead of ~3×
    # (its 3 edge equations + soft rows) and replaces the per-call Dict override with a slot
    # array. NOTE the dual arithmetic must stay Double64 (see pass 1) — the win here is purely
    # structural, not precision.
    ntid0 = Threads.maxthreadid()
    local Hn, hpos, ring_unk, hedges, gH, slotbufs, accbufs, markbufs, colbufs
    if hinge_jac
        Hn = Int[]
        let seen = falses(length(ec.hinges))
            for e in E, (hi, _) in ec.eqs[e]
                seen[hi] || (seen[hi] = true; push!(Hn, hi))
            end
        end
        sort!(Hn)
        hpos = zeros(Int, length(ec.hinges))
        for (i, hi) in enumerate(Hn); hpos[hi] = i; end
        ring_unk = [[e for e in ec.ring_edges[hi] if e in uset] for hi in Hn]
        hedges = [(ec.eid[_ck(h[1], h[2])], ec.eid[_ck(h[1], h[3])], ec.eid[_ck(h[2], h[3])])
                  for h in (ec.hinges[hi] for hi in Hn)]
        gH = Vector{Vector{Float64}}(undef, length(Hn))
        slotbufs = [zeros(Int32, length(ec.edges)) for _ in 1:ntid0]
        accbufs = [zeros(Float64, nu) for _ in 1:ntid0]
        markbufs = [zeros(Int, nu) for _ in 1:ntid0]
        colbufs = [Int[] for _ in 1:ntid0]
    end

    LD = D.(L)
    ε = zeros(D, length(ec.hinges))
    kindc = fill(:euclid, length(ec.hinges))
    r = zeros(D, ne)
    # section timers (REGGE_TIMING=1): where does the wall-clock go?
    timing = get(ENV, "REGGE_TIMING", "0") == "1"
    t_def = 0.0; t_res = 0.0; t_asm = 0.0; t_fac = 0.0; t_trial = 0.0
    n_it = 0; n_asm = 0; n_trials = 0
    # dense Double64 normal equations up to ~N=4; beyond that assemble a Float64 sparse J and
    # solve the damped least squares by sparse QR — the STEP DIRECTION only needs a few digits
    # (LM acceptance uses full-precision Double64 residuals, so an imprecise direction merely
    # costs iterations), while residual EVALUATION keeps the boost-precision law intact.
    sparse_path = nu > sparse_threshold
    J = sparse_path ? zeros(D, 0, 0) : zeros(D, ne, nu)
    maxr = Inf
    lm_λ = 1e-10                                             # LM damping (relative), adaptive
    dmax_prev = one(D)
    # CHORD Gauss–Newton: J is REUSED across iterations (one march step moves the geometry very
    # little, so J barely changes; LM acceptance on full-precision Double64 residuals makes stale
    # directions safe — they cost iterations, never correctness), refreshed only when LM stalls
    # on a stale J. The sparse QR is CACHED by (J version, λ): with both unchanged only the
    # backsolve runs. Measured (N=5 slab, nu=1375): assembly was 23×3.3s + QR 30×1.7s of a 140s
    # solve — chord+cache collapse both to O(1) occurrences.
    need_J = true
    Jver = 0
    eps_current = false                                      # ε already computed at current LD?
    local A, Aall, Cs, JtJ0, dmax, sdm
    chol_fact = nothing; chol_λ = NaN; chol_ver = 0          # CHOLMOD numeric-refactor cache
    lowprog = 0                                              # consecutive low-progress steps
    it = 0
    while it < maxit                   # `it` counts ACCEPTED steps; failed stale-J retries and
        n_it = it + 1                  # the final convergence pass are free
        eps_current || (t_def += @elapsed compute_deficits!(ε, kindc, ec, LD))
        eps_current = false
        # residuals (cheap, cached deficits); rows weighted (soft constraints < hard)
        t_res += @elapsed Threads.@threads :static for k in 1:ne
            e = E[k]
            r[k] = wrow[k] * edge_residual(ec, e, LD, ε, kindc, κ, conv; dust=D(dust_of(e)))
        end
        maxr = Float64(maximum(abs, view(r, 1:nh)))          # convergence on the HARD rows
        obj = Float64(maximum(abs, r))
        if wreg > 0 && n_asm > 0                              # dmax from the last assembly
            sdm0 = sqrt(Float64(dmax_prev))
            for (ge, tgt) in reg_rows
                obj = max(obj, abs(wreg * sdm0 * Float64(LD[ge] - D(tgt))))
            end
        end
        verbose && println("  GN it $n_it: max|r_hard| = $maxr  obj = $obj")
        # with soft rows present, keep iterating until LM stalls (soft rows have a nonzero
        # floor and must be given the iterations to pull; hard-only early exit skips them)
        maxr < tol && isempty(soft_equations) && break
        maxr < tol && obj < 1e3 * tol && break
        # Jacobian: per-equation local ForwardDiff over its support (only when marked stale)
        fresh = false
        if need_J
            ntid = Threads.maxthreadid()                 # interactive pool: threadid > nthreads
            trip_i = [Int[] for _ in 1:ntid]
            trip_j = [Int[] for _ in 1:ntid]
            trip_v = [Float64[] for _ in 1:ntid]
            sparse_path || fill!(J, zero(D))
            n_asm += 1
            t_asm += @elapsed if hinge_jac
                # pass 1: per-hinge deficit gradients over ring unknowns (jacT duals, threaded).
                # jacT MUST default to Double64: the boost-deficit precision law EXTENDS TO THE
                # DERIVATIVES — ∂ε/∂l² is the same deep cancellation as ε itself, and Float64
                # duals give O(40)-relative garbage (measured frob 3.9e1 vs the per-eq reference,
                # while Double64 duals agree to 8.8e-13). Results are STORED as Float64 (storage
                # rounding is safe — the validated old path stored F64 triplets too).
                LF = Float64.(LD)
                LJ = jacT === Double64 ? LD : jacT.(LD)
                Threads.@threads :static for i in eachindex(Hn)
                    RU = ring_unk[i]
                    if isempty(RU)
                        gH[i] = Float64[]
                        continue
                    end
                    hi = Hn[i]
                    slotb = slotbufs[Threads.threadid()]
                    for (j, e) in enumerate(RU)
                        slotb[e] = j
                    end
                    f = xx -> begin
                        T = eltype(xx)
                        esq = (v, w) -> begin
                            e = ec.eid[v < w ? (v, w) : (w, v)]
                            sj = slotb[e]
                            sj == 0 ? T(LJ[e]) : xx[sj]
                        end
                        first(hinge_deficit(ec.hinges[hi], ec.rings[hi], esq, ec.pents))
                    end
                    gH[i] = Float64.(ForwardDiff.gradient(f, jacT[LJ[e] for e in RU]))
                    for e in RU
                        slotb[e] = 0
                    end
                end
                # pass 2: chain-rule rows (threaded over equations, no dual arithmetic)
                Threads.@threads :static for k in 1:ne
                    e = E[k]
                    isempty(support[k]) && continue
                    tid = Threads.threadid()
                    acc = accbufs[tid]; mark = markbufs[tid]; cols = colbufs[tid]
                    empty!(cols)
                    kk = n_asm * ne + k              # mark key unique ACROSS assemblies too
                    for (hi, slot) in ec.eqs[e]
                        i = hpos[hi]
                        e12, e13, e23 = hedges[i]
                        a = LF[e12]; b = LF[e13]; c = LF[e23]
                        s = kindc[hi] === :boost ? Float64(conv.s_boost) : Float64(conv.s_euclid)
                        κs = Float64(κ) * s
                        # term 2: (∂A/∂slot) · ∂ε/∂u over the hinge's ring unknowns
                        coef = κs * _dA_dslot(a, b, c, slot)
                        g = gH[i]; RU = ring_unk[i]
                        for (j, u) in enumerate(RU)
                            _acc!(acc, mark, cols, kk, uidx[u], coef * g[j])
                        end
                        # term 1: ∂(∂A/∂slot)/∂(a,b,c) · ε on the (unknown) triangle edges
                        εκs = κs * Float64(ε[hi])
                        gv = ForwardDiff.gradient(v -> _dA_dslot(v[1], v[2], v[3], slot),
                                                  SVector(a, b, c))
                        e12 in uset && _acc!(acc, mark, cols, kk, uidx[e12], εκs * gv[1])
                        e13 in uset && _acc!(acc, mark, cols, kk, uidx[e13], εκs * gv[2])
                        e23 in uset && _acc!(acc, mark, cols, kk, uidx[e23], εκs * gv[3])
                    end
                    wk = Float64(wrow[k])
                    if sparse_path
                        for c in cols
                            push!(trip_i[tid], k); push!(trip_j[tid], c)
                            push!(trip_v[tid], wk * acc[c])
                        end
                    else
                        for c in cols
                            J[k, c] = wrow[k] * D(acc[c])
                        end
                    end
                end
            else
                Threads.@threads :static for k in 1:ne
                    e = E[k]
                    S = support[k]
                    isempty(S) && continue
                    de = D(dust_of(e))
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
                    if sparse_path
                        tid = Threads.threadid()
                        for (i, s) in enumerate(S)
                            push!(trip_i[tid], k); push!(trip_j[tid], uidx[s])
                            push!(trip_v[tid], Float64(wrow[k] * g[i]))
                        end
                    else
                        for (i, s) in enumerate(S)
                            J[k, uidx[s]] = wrow[k] * g[i]
                        end
                    end
                end
            end
            A = sparse_path ?
                sparse(vcat(trip_i...), vcat(trip_j...), vcat(trip_v...), ne, nu) :
                sparse(Int[], Int[], Float64[], 1, 1)
            if sparse_path
                # column-norm² for scaling; reg rows appended as sparse rows
                cn = zeros(Float64, nu)
                for col in 1:nu
                    for idx in A.colptr[col]:(A.colptr[col+1]-1)
                        cn[col] += A.nzval[idx]^2
                    end
                end
                dmax = D(maximum(cn))
                sdm = sqrt(Float64(dmax))
                nreg = length(reg_rows)
                if wreg > 0 && nreg > 0
                    ri = collect(ne+1:ne+nreg)
                    rj = [uidx[ge] for (ge, _) in reg_rows]
                    rv = fill(wreg * sdm, nreg)
                    Aall = vcat(A, sparse(ri .- ne, rj, rv, nreg, nu))
                else
                    Aall = A
                end
                # normal equations for CHOLMOD (pattern constant across refreshes: the supports
                # are combinatorial) — measured 30-100× cheaper than SPQR of the augmented
                # system at identical direction (Δdir ~1e-14); LM guards the squared condition.
                Cs = Symmetric(Aall' * Aall)
            else
                JtJ0 = transpose(J) * J                       # pristine (reg applied per-iter)
                dmax = maximum(i -> JtJ0[i, i], 1:nu)
                sdm = sqrt(Float64(dmax))
            end
            dmax_prev = dmax
            Jver += 1
            need_J = false
            fresh = true
        end
        # per-iteration RHS (depends on the CURRENT residuals/state; J factors reused)
        local rhsF, Atb, JtJr, Jtr
        if sparse_path
            if wreg > 0 && !isempty(reg_rows)
                rhsF = vcat(Float64.(r), [wreg * sdm * Float64(LD[ge] - D(tgt))
                                          for (ge, tgt) in reg_rows])
            else
                rhsF = Float64.(r)
            end
            Atb = Aall' * rhsF
        else
            Jtr = transpose(J) * r
            JtJr = copy(JtJ0)
            # dissipation rows: wreg²·dmax on the diagonal, pulling toward the extrapolation
            if wreg > 0
                wr2 = D(wreg)^2 * dmax
                for (ge, tgt) in reg_rows
                    i = uidx[ge]
                    JtJr[i, i] += wr2
                    Jtr[i] += wr2 * (LD[ge] - D(tgt))
                end
            end
        end
        # Levenberg–Marquardt: adaptive damping instead of a line search — the system has a
        # near-null space (the un-gauge-fixed SHIFT freedom, ~3 modes/vertex), so small fixed λ
        # produces noise-dominated steps that a line search just rejects. LM raises λ until the
        # step reduces ‖r‖∞, then relaxes it.
        timing && sparse_path && (_DUMP_A[] = (Aall, rhsF))
        λ = D(lm_λ) * dmax
        lm_λ_in = lm_λ
        # ACCEPTANCE BARS. Stale (chord) J: the step must contract STRONGLY (rt<0.3·obj) and
        # gets a single trial — anything weaker refreshes J and redoes the iteration (whose
        # fresh-J behavior is the validated original). Pure chord under the plain rt<obj bar
        # crawls: λ escalates, tiny steps keep passing, maxit exhausts far from the solution
        # (measured recovery 4.6e-2 vs 5e-7). Fresh J keeps the PLAIN bar: a per-step
        # minimal-progress bar (rt<0.98·obj) stalls tol-converging solves — the ∞-norm
        # genuinely plateaus for an iteration while the residual field reorganizes, then drops
        # (measured: FLRW recovery 3e-3 vs 5e-7). The floor-grind is handled at the SOLVE level
        # by the consecutive-low-progress detector after the ladder.
        bar = fresh ? 1.0 : 0.3
        accepted = false
        rt_acc = Inf
        rt_best = Inf
        nworse = 0
        for _ in 1:12
            n_trials += 1
            local Δ
            if sparse_path
                λf = Float64(λ)
                if chol_fact === nothing || chol_ver != Jver || chol_λ != λf
                    okf = true
                    t_fac += @elapsed begin
                        okf = try
                            if chol_fact === nothing
                                chol_fact = cholesky(Cs; shift=λf)
                            else
                                cholesky!(chol_fact, Cs; shift=λf)   # numeric-only refactor
                            end
                            true
                        catch
                            false
                        end
                    end
                    if !okf                                   # not PD at this λ (rank-deficient
                        chol_fact = nothing                   # floor): escalate λ / refresh
                        fresh || break
                        λ *= 10; lm_λ = min(lm_λ * 10, 1e2)
                        continue
                    end
                    chol_ver = Jver; chol_λ = λf
                end
                ΔF = chol_fact \ Atb
                Δ = D.(ΔF)
            else
                t_fac += @elapsed Δ = lu(JtJr + λ * Diagonal(ones(D, nu))) \ Jtr
            end
            Lt = copy(LD)
            for (i, e) in enumerate(U)
                Lt[e] = LD[e] - Δ[i]
            end
            local rt
            t_trial += @elapsed begin
                compute_deficits!(ε, kindc, ec, Lt)
                rtt = zeros(Float64, Threads.maxthreadid())
                Threads.@threads :static for k in 1:ne
                    e = E[k]
                    v = Float64(abs(wrow[k] * edge_residual(ec, e, Lt, ε, kindc, κ, conv;
                                                            dust=D(dust_of(e)))))
                    tid = Threads.threadid()
                    v > rtt[tid] && (rtt[tid] = v)
                end
                rt = maximum(rtt)
                if wreg > 0
                    sdm0 = sqrt(Float64(dmax))
                    for (ge, tgt) in reg_rows
                        rt = max(rt, abs(wreg * sdm0 * Float64(Lt[ge] - D(tgt))))
                    end
                end
            end
            if rt < bar * obj                             # LM acceptance on the full objective
                LD = Lt
                lm_λ = max(lm_λ / 4, 1e-14)
                accepted = true
                rt_acc = rt
                eps_current = true                            # ε just computed at the accepted LD
                break
            end
            fresh || break                # stale J underperforms: refresh, retry this iteration
            # LADDER ECONOMICS — every rung costs a full deficit sweep. rt(λ) is FLAT for many
            # rungs (the trial rt saturates while λ ≪ dmax) and only improves near λ ~ dmax, so
            # flat rungs FAR from the objective are normal mid-ladder behavior, not a stall:
            # after 2 of them, jump ×100 instead of ×10 (an unconditional 3-flat-rungs break
            # froze cold starts at their first iteration — measured rt = 8.7·obj constant over
            # 3 decades of λ, acceptance many rungs higher). Flat rungs ALREADY AT the
            # objective floor (rt_best < 1.5·obj) are the opposite — the state is converged on
            # the KO-dissipation floor and further rungs are polish — exit after 3 (measured:
            # floor-stall ladders burned all 12 rungs, ~2× the whole march cost).
            if isfinite(rt) && rt < rt_best - 0.02 * obj
                rt_best = rt
                nworse = 0
            else
                nworse += 1
                nworse >= 3 && rt_best < 1.5 * obj && break
            end
            fac = nworse >= 2 ? 100 : 10
            λ *= fac
            lm_λ = min(lm_λ * fac, 1e2)
        end
        if accepted
            it += 1
            # FLOOR-STALL detector: march solves sit on the KO-dissipation objective floor far
            # above tol and can only exit by stall — without this, LM grinds hair-thin (<2%)
            # improvements for 20+ iterations (measured n_it 24-25 at the maxit cap, ~30s of
            # pure floor-polish per solve). Three CONSECUTIVE low-progress accepted iterations
            # ⇒ converged-at-floor; isolated ∞-norm plateaus (the tol-converging cold-start
            # pattern) reset the counter and survive.
            if rt_acc > 0.98 * obj
                lowprog += 1
                lowprog >= 3 && break
            else
                lowprog = 0
            end
        else
            fresh && break                # LM exhausted on a FRESH J: report current maxr
            need_J = true                 # refresh and RETRY this iteration (no budget consumed)
            lm_λ = lm_λ_in
        end
    end
    timing && @info "solve_edges! timing" nu ne n_it n_asm n_trials t_def t_res t_asm t_fac t_trial
    for i in eachindex(L)
        L[i] = Float64(LD[i])
    end
    return maxr
end

"""
    solve_slab!(L, ec; κ, conv, dust_pole, tol, pins, ...) -> (maxr, constraints)

The SORKIN-SPLIT evolution step (the square, integrable scheme):
  solve  {unpinned dgB eqs (4/vtx) + sp1 eqs (7/vtx)}  for  {unpinned dgB + sp2} (11/vtx),
with gauge pins EXCLUDED as unknowns (poleB := −dτ² lapse; 3 axis dgB := shift — caller sets
their values in L). The pole equation is NEVER solved: comoving dust has T_ij = 0, so the
spatial/diagonal equations are dust-free and the pole equation is the (dust-sourced) Hamiltonian
constraint — returned in `constraints = (ham, mom)` as the monitored drift, alongside the pinned
diagonals' (momentum-like) residuals. Solving it too over-determines the system by 1/vertex and
the approximate discrete Bianchi identities amplify the inconsistency ×1e3/step (measured).
"""
function solve_slab!(L::Vector{Float64}, ec::EdgeComplex;
                     κ, conv, dust_pole, tol::Float64, pins::Vector{Int},
                     maxit::Int=25, verbose::Bool=false, wc::Float64=1.0,
                     diss::Vector{Tuple{Int,Float64}}=Tuple{Int,Float64}[], wd::Float64=0.0)
    pinset = Set(pins)
    U = [e for e in eachindex(ec.edges)
         if (ec.kind[e] === :dgB && !(e in pinset)) || ec.kind[e] === :sp2]
    E = [e for e in eachindex(ec.edges)
         if ec.complete[e] &&
            ((ec.kind[e] === :dgB && !(e in pinset)) || ec.kind[e] === :sp1)]
    # constraint-damping rows: Hamiltonian (poleB, dust-sourced) + momentum (pinned dgB), weight wc
    S = [e for e in eachindex(ec.edges)
         if ec.complete[e] && (ec.kind[e] === :poleB || (e in pinset))]
    dust_of = e -> ec.kind[e] === :poleB ? dust_pole : 0.0
    maxr = solve_edges!(L, ec; unknowns=U, equations=E, dust_of=dust_of,
                        κ=κ, conv=conv, tol=tol, maxit=maxit, verbose=verbose,
                        soft_equations=S, soft_weight=wc, reg_rows=diss, wreg=wd)
    # monitored constraints: Hamiltonian (poleB eqs, WITH dust) + momentum (pinned dgB eqs)
    LD = Double64.(L)
    ε = zeros(Double64, length(ec.hinges)); kc = fill(:euclid, length(ec.hinges))
    compute_deficits!(ε, kc, ec, LD)
    ham = 0.0; mom = 0.0
    for e in eachindex(ec.edges)
        ec.complete[e] || continue
        if ec.kind[e] === :poleB
            ham = max(ham, Float64(abs(edge_residual(ec, e, LD, ε, kc, κ, conv;
                                                     dust=Double64(dust_pole)))))
        elseif e in pinset
            mom = max(mom, Float64(abs(edge_residual(ec, e, LD, ε, kc, κ, conv))))
        end
    end
    return maxr, (ham=ham, mom=mom)
end

"""
    initial_slab_solve!(L, ec; κ, conv, dust_pole, tol, pins, ...) -> (maxr, ham)

The DISCRETE INITIAL-DATA solve: given slice-0 and slice-1 (intrinsic geometry at two times —
they encode the expansion rate) and poleA pinned to −dτ², solve the slab-A diagonal equations
{unpinned dgA eqs} for the {unpinned dgA} diagonals (square, 4/vertex), with the 3 axis dgA per
vertex gauge-pinned by the caller. Returns the max solved residual and the Hamiltonian residual
(poleA eqs WITH dust) — the latter measures the consistency of the (slice0, slice1, dτ) triplet
and is the true initial constraint violation.
"""
function initial_slab_solve!(L::Vector{Float64}, ec::EdgeComplex;
                             κ, conv, dust_pole, tol::Float64, pins::Vector{Int},
                             maxit::Int=25, verbose::Bool=false, york::Bool=true)
    pinset = Set(pins)
    U = [e for e in eachindex(ec.edges) if ec.kind[e] === :dgA && !(e in pinset)]
    E = copy(U)                                           # square: each unpinned dgA's own eq
    dust_of = e -> ec.kind[e] === :poleA ? dust_pole : 0.0
    if york
        # DISCRETE YORK PROCEDURE: for inhomogeneous data the analytic chords violate the
        # Hamiltonian constraint at O((k·dx)²·D·A) (measured 14% of the dust scale at 3 pts per
        # wavelength) and the violation destabilizes the march. Add the slice-1 spatial edges as
        # unknowns (a minimal LSQ adjustment of the intrinsic slice geometry) and require the
        # FULL slab-A equation set: unpinned dgA (hard) + Hamiltonian/momentum as soft rows.
        append!(U, [e for e in eachindex(ec.edges) if ec.kind[e] === :sp1])
        S = [e for e in eachindex(ec.edges)
             if ec.complete[e] && (ec.kind[e] === :poleA || (e in pinset))]
        maxr = solve_edges!(L, ec; unknowns=U, equations=E, dust_of=dust_of,
                            κ=κ, conv=conv, tol=tol, maxit=maxit, verbose=verbose,
                            soft_equations=S, soft_weight=1.0)
    else
        maxr = solve_edges!(L, ec; unknowns=U, equations=E, dust_of=dust_of,
                            κ=κ, conv=conv, tol=tol, maxit=maxit, verbose=verbose)
    end
    LD = Double64.(L)
    ε = zeros(Double64, length(ec.hinges)); kc = fill(:euclid, length(ec.hinges))
    compute_deficits!(ε, kc, ec, LD)
    ham = 0.0
    for e in eachindex(ec.edges)
        (ec.complete[e] && ec.kind[e] === :poleA) || continue
        ham = max(ham, Float64(abs(edge_residual(ec, e, LD, ε, kc, κ, conv;
                                                 dust=Double64(dust_pole)))))
    end
    return maxr, ham
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
