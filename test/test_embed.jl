# embed_slice: position reconstruction from edge lengths must recover the analytic Zel'dovich
# slice to machine precision (mod translation gauge) from a perturbed seed.
using ReggeSheet, Test, Random
using Statistics: mean

@testset "embed_slice recovers analytic positions" begin
    Random.seed!(1)
    N = 4; dx = 1e-4; box = N * dx; k = 2π / box; a = 0.05; A = 10.0
    ec = EdgeComplex(N); ns = N^3
    edges, _ = slice_edges(ec, 0)
    qco(p) = (dx * ((p - 1) % N), dx * (((p - 1) ÷ N) % N), dx * ((p - 1) ÷ (N * N)))
    mi(d) = (d = mod(d, box); d > box / 2 ? d - box : d)
    lsq = Float64[]
    for (v, w) in edges
        qv = qco(v); qw = qco(w)
        Δ = (mi(qw[1] - qv[1]), mi(qw[2] - qv[2]), mi(qw[3] - qv[3]))
        push!(lsq, zel_spatial_lsq(qv, (qv[1] + Δ[1], qv[2] + Δ[2], qv[3] + Δ[3]), a; A=A, k=k))
    end
    Xan = zeros(ns, 3)
    for p in 1:ns
        q = qco(p)
        Xan[p, 1] = a * zel_X(q[1], a; A=A, k=k)
        Xan[p, 2] = a * q[2]; Xan[p, 3] = a * q[3]
    end
    x0 = Xan .+ 0.02 * a * dx .* randn(ns, 3)
    x = embed_slice(N, edges, lsq, x0, a * dx)
    d = x .- Xan
    d .-= mean(d, dims=1)
    @test maximum(abs.(d)) / (a * dx) < 1e-10
end
