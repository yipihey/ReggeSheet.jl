# Element-type hook + evolution state.
#
# The time-stepper is written against the AbstractReggeElement interface ONLY, so a future
# `QuadraticTet <: AbstractReggeElement` (a NON-GOAL here — hook only) slots in without touching
# step.jl / evolve.jl. Only the linear tetrahedron is implemented in this cycle.

"""
    AbstractReggeElement

Interface for a Regge-calculus element. Required methods (all PURE, allocation-free, type-generic
for ForwardDiff/Enzyme):

    nvertices(::Type{E})  -> Int    # vertices per spatial element (4 for the linear tetrahedron)

Geometry (volumes, deficit contributions) is provided by `geometry.jl` against the element's
squared edge lengths; higher-order elements would add their own curvature rule here.
"""
abstract type AbstractReggeElement end

"""    LinearTet — the linear (flat-faced) tetrahedron; the one element implemented this cycle."""
struct LinearTet <: AbstractReggeElement end

nvertices(::Type{LinearTet}) = 4
nvertices(::Type{E}) where {E<:AbstractReggeElement} = error("nvertices not implemented for $E")
