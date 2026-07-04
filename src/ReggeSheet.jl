"""
ReggeSheet.jl — Lorentzian Regge-calculus evolution of cold dust on a perturbed-FLRW simplicial
lattice, for head-to-head comparison against DISCO-DJ's nLPT (pre-shell-crossing only).

The core (this module) is intrinsic Regge calculus in the signed squared-edge-length variables and
depends on NOTHING cosmological beyond an analytic FLRW background; the DISCO-DJ comparison lives in
a separate `compare/` environment.
"""
module ReggeSheet

include("geometry.jl")
include("types.jl")
include("lattice.jl")
include("complex.jl")
include("deficit.jl")
include("background.jl")
include("edges.jl")
include("step.jl")
include("zeldovich.jl")
include("embed_slice.jl")
include("march3d.jl")

# geometry primitives
export minkowski_sq, edge_sq, cm_volume_sq, tri_area_sq, tet_volume_sq, pent_volume_sq
export embed_simplex, dihedral, wedge_invariants
# element hook
export AbstractReggeElement, LinearTet, nvertices
# lattice + 4D complex
export CUBE_CORNERS, TETS, PRISM_PENTA, slice_index, lagrangian_vertices
export n_spatial, build_pentachora, build_pentachora_stack, hinge_incidence
# deficits
export hinge_ring, hinge_deficit, pent_D
# homogeneous FLRW (gate #1)
export FLRWModel, flrw_step, flrw_march, spatial_constraint, build_class_equations, class_residual
# general per-edge machinery (M2+)
export EdgeComplex, edge_kind, state_esq, compute_deficits!, edge_residual
export TentBlocks, sweep_slab!, solve_slab!, solve_edges!, initial_slab_solve!
export zel_spatial_lsq, zel_X, zel_seed_window!, window_shift_map, gauge_pins, march_zeldovich
export embed_slice, slice_edges, march3d

end # module
