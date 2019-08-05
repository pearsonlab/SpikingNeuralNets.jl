function sample(μs::Vector{<:Real}, σs::Vector{<:Real})::Vector{Float64}
    return max.(rand.(Normal.(μs, σs)), zero(Float64))
end

"""
    VariablePoissonInput{N, L}

Iterate over a Poisson input for `N` neurons with rate `N(λs[n], σs[n])`
for each neuron `n` `L` times, where the rate is resampled every
`resample` iterations.
"""
struct VariablePoissonInput{N, L} <: AbstractPoissonInput{N, L}
    μs::Vector{Float64}
    σs::Vector{Float64}
    λs::Vector{Float64}
    buffer::Vector{Int64}
    start::Int64
    stop::Int64
    resample::Int64

    function VariablePoissonInput{N, L}(μs, σs, λs, buffer, start, stop, resample) where {N, L}
        if !(typeof(N) <: Integer)
            error("Number of neurons N must be an integer")
        elseif !(typeof(L) <: Integer)
            error("Input length L must be an integer")
        elseif length(μs) != N
            error("μs must have length N.")
        elseif length(σs) != N
            error("σs must have length N.")
        elseif length(λs) != N
            error("λs must have length N.")
        elseif length(buffer) != N
            error("buffer must have length N.")
        end

        new(μs, σs, λs, buffer, start, stop, resample)
    end
end

VariablePoissonInput(μs::Vector{<:Real}, σs::Vector{<:Real}, L::Integer; resample=one(Int64), start=one(Int64), stop=L) = VariablePoissonInput{length(μs), L}(μs, σs, sample(μs, σs), zeros(Int64, length(μs)), start, stop, resample)
VariablePoissonInput(μ::Real, σ::Real, N::Integer, L::Integer; resample=one(Int64), start=one(Int64), stop=L) = VariablePoissonInput(fill(μ, N), fill(σ, N), N, L; resample=resample, start=start, stop=stop)

"""
    rates(iter::VariablePoissonInput{N, L}, i::Integer)::Vector{Float64} where {N, L}

Obtain the firing rate for each neuron in `iter` for iteration `i`.
"""
function rates(iter::VariablePoissonInput{N, L}, i::Integer)::Vector{Float64} where {N, L}
    if i % iter.resample == 0
        iter.λs .= sample(iter.μs, iter.σs)
    end
    return iter.λs
end
