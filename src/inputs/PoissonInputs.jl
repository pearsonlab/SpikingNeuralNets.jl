using Distributions
export AbstractPoissonInput, ConstantPoissonInput, VariablePoissonInput

"""
    AbstractPoissonInput{N, L} <: AbstractInput{Int, N, L}

An abstract type for Poisson inputs. Subtypes must include the field 'buffer'
to store intermittent results, 'start' which defines the iteration before which
input is zero, and 'stop' defines the iteration after which input is zero.
Subtypes should also define the function 'rates', which retrieves the rates
for all neurons at a given valid iteration.
"""
abstract type AbstractPoissonInput{N, L} <: AbstractInput{Int, N, L} end

"""
    rates(iter::AbstractPoissonInput{N, L}, i::Integer)::Vector{Float64} where {N, L}

Obtain the firing rate for each neuron in `iter` for iteration `i`.
"""
function rates(iter::AbstractPoissonInput{N, L}, i::Integer)::Vector{Float64} where {N, L}
    return zeros(Float64, N)
end

function Base.iterate(iter::AbstractPoissonInput{N, L}, i=1) where {N, L}
    if i > L
        return nothing
    elseif i < iter.start || i > iter.stop
        iter.buffer .= zero(Float64)
    else
        iter.buffer .= rand.(Poisson.(rates(iter, i)))
    end

    return iter.buffer, i+1
end
