module Inputs

export AbstractInput

"""
    AbstractInput{T<:Real, N, L}

An abstract type for iterators producing SNN input data, where
`T` is the type of input produced, `N` is the number of neurons
to make input for, and `L` is the length of the input sequence.
As such, an AbstractInput produces a length-`N` `Vector{T}` each
iteration.
"""
abstract type AbstractInput{T<:Real, N, L} end

function Base.length(iter::AbstractInput{T, N, L}) where {T, N, L} return L end
function Base.size(iter::AbstractInput{T, N, L}) where {T, N, L} return (N, L) end
function Base.eltype(t::Type{AbstractInput{T, N, L}}) where {T, N, L} return T end

include("PoissonInputs.jl")
include("ConstantPoissonInputs.jl")
include("VariablePoissonInputs.jl")
include("SummedInputs.jl")
end
