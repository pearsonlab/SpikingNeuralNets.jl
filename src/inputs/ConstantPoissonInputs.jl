"""
    ConstantPoissonInput{N, L}

Iterate over a Poisson input for `N` neurons with rate `λ[n]` for each neuron `n` `L` times.
"""
struct ConstantPoissonInput{N, L} <: AbstractPoissonInput{N, L}
    λs::Vector{Float64}
    buffer::Vector{Int64}
    start::Int64
    stop::Int64

    function ConstantPoissonInput{N, L}(λs, buffer, start, stop) where {N, L}
        if !(typeof(N) <: Integer)
            error("Number of neurons N must be an integer")
        elseif !(typeof(L) <: Integer)
            error("Input length L must be an integer")
        elseif length(λs) != N
            error("λs must have length N.")
        elseif length(buffer) != N
            error("buffer must have length N.")
        end

        new(λs, buffer, start, stop)
    end
end

ConstantPoissonInput(λs::Vector{<:Real}, L::Integer; start::Integer=one(Int64), stop::Integer=L) = ConstantPoissonInput{length(λs), L}(λs, zeros(Int64, N), start, stop)
ConstantPoissonInput(λ::Real, N::Integer, L::Integer; start::Integer=one(Int64), stop::Integer=L) = ConstantPoissonInput{N, L}(fill(λ, N), zeros(Int64, N), start, stop)

"""
    rates(iter::ConstantPoissonInput{N, L}, i::Integer)::Vector{Float64} where {N, L}

Obtain the firing rate for each neuron in `iter` for iteration `i`.
"""
function rates(iter::ConstantPoissonInput{N, L}, i::Integer)::Vector{Float64} where {N, L}
    return iter.λs
end
