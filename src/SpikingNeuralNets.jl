module SpikingNeuralNets
using LightGraphs, SimpleWeightedGraphs, Statistics, SparseArrays
import Base.size, Base.length, Base.Broadcast.broadcastable

export AbstractSNN,  # data-types
    size, neurons, memory, bias, spikes,
    spike, spiketimes, spiketime, spiketrain,
    voltages, voltage, configuration, input, output, inputs, outputs,
    is_input, is_output, excitors, inhibitors,  # accessor functions
    potential, # neuron potential function
    sigmoid, thresh, ramp, fire, # transfer functions
    step!  # used to iterate the SNN

"""
    AbstractSNN{VT<:Real, ST<:Integer, G<:AbstractGraph}

An abstract SNN type where VT is the voltage type,
ST is the spike train type, and G is the graph type.

An `AbstractSNN{VT, ST, G}` must have the following elements:
    - `graph::G`
    - `I::Vector{Bool}`
    - `O::Vector{Bool}`
    - `S::Vector{Vector{ST}}`
    - `V::Vector{Vector{VT}}`
"""
abstract type AbstractSNN{VT<:Real, ST<:Integer, G<:AbstractGraph} end

Broadcast.broadcastable(snn::AbstractSNN) = Ref(snn)

"""
    size(snn::AbstractSNN)::Integer

Return the number of neurons in the network `snn`.
"""
size(snn::AbstractSNN) = nv(snn.graph)

"""
    length(snn::AbstractSNN)::Integer

Return the number of neurons in the network `snn`.
"""
length(snn::AbstractSNN) = nv(snn.graph)

"""
    neurons(snn::AbstractSNN)::UnitRange{Integer}

Return the a range of indices for all neurons in the network `snn`.
"""
neurons(snn::AbstractSNN) = vertices(snn.graph)

"""
    memory(snn::AbstractSNN)::Integer

Return the size of the network `snn`'s voltage memory.
"""
memory(snn::AbstractSNN) = length(snn.V)

"""
    bias(snn::AbstractSNN)::Vector{<:Real}

Return the bias for all neurons in the network `snn`.
"""
function bias(snn::AbstractSNN{VT})::Vector{VT} where {VT<:Real}
    return zeros(VT, neurons(snn))
end

"""
    bias(snn::AbstractSNN, n::Integer)::Real

Return the bias of neuron `n` in the network `snn`.
"""
@views function bias(snn::AbstractSNN{VT}, n::Integer)::VT where {VT<:Real}
    return bias(snn)[n]
end

"""
    voltages(snn::AbstractSNN{VT})::Vector{VT} where {VT<:Real}

Get the current voltage of all neurons in the network `snn`.
"""
@views function voltages(snn::AbstractSNN{VT})::Vector{VT} where {VT<:Real}
    return snn.V[end]
end

"""
    voltage(snn::AbstractSNN{VT}, n::Integer)::VT where {VT<:Real}

Get the current voltage of neuron `n` in the network `snn`.
"""
@views function voltage(snn::AbstractSNN{VT}, n::Integer)::VT where {VT<:Real}
    return snn.V[end][n]
end

"""
    spike(snn::AbstractSNN, n::Integer; t::Integer=0)::Bool

Calculate whether neuron `n` in the network `snn` was spiking `t` timesteps ago.
By default, calculates whether `n` is currently spiking.
"""
@views spike(snn::AbstractSNN, n::Integer; t::Integer=0)::Bool = (iszero(t) && !isempty(snn.S[n]) && snn.S[n][end] == 0) ||
                                                                 !isempty(searchsorted(snn.S[n], t, rev=true))

"""
    spikes(snn::AbstractSNN, neurons::AbstractVector{<:Integer}; t::Integer=0)::BitArray{1}

Calculate whether `neurons` in the network `snn` are currently spiking.
"""
function spikes(snn::AbstractSNN, neurons::AbstractVector{<:Integer}=neurons(snn); t::Integer=0)::BitArray{1}
    S = BitArray(undef, length(neurons))
    @inbounds for i in eachindex(neurons)
        S[i] = spike(snn, neurons[i]; t=t)
    end
    return S
end

"""
    spiketimes(snn::AbstractSNN{VT, ST}, n::Integer)::Vector{ST} where {VT<:Real, ST<:Integer}

Return the number of timesteps into the past of all spikes for neuron `n` in the network `snn`.
"""
@views function spiketimes(snn::AbstractSNN{VT, ST}, n::Integer)::Vector{ST} where {VT<:Real, ST<:Integer}
    return snn.S[n]
end

"""
    spiketimes(snn::AbstractSNN{VT, ST})::Vector{Vector{ST}} where {VT<:Real, ST<:Integer}

Return the number of timesteps into the past of all spikes for all neurons in the network `snn`.
"""
function spiketimes(snn::AbstractSNN{VT,ST})::Vector{Vector{ST}} where {VT<:Real, ST<:Integer}
    snn.S
end

"""
    spiketime(snn::AbstractSNN{VT, ST}, n::Integer)::ST where {VT<:Real, ST<:Integer}

Return the number of timesteps into the past of the most recent spike for neuron
    `n` in the network `snn`, or typemax(ST) if `n` has not yet spiked.
"""
@views function spiketime(snn::AbstractSNN{VT,ST}, n::Integer)::ST where {VT<:Real, ST<:Integer}
    if isempty(snn.S[n])
        return typemax(ST)
    end
    return snn.S[n][end]
end

"""
    spiketrain(snn::AbstractSNN, t::T)::SparseMatrixCSC{Bool, T}

Return the full matrix S[neuron, time] (with t columns) of binary spike data
for all neurons in the network `snn` for the last `t` time-steps.
"""
function spiketrain(snn::AbstractSNN, t::T)::SparseMatrixCSC{Bool, T} where T<:Integer
    if t <= 0
        return spzeros(Bool, size(snn), 0)
    end
    nspikes = sum(map(length, snn.S))
    I = Vector{T}(undef, nspikes)
    J = Vector{T}(undef, nspikes)
    S = trues(nspikes)
    s = 1
    @inbounds @views for n in eachindex(snn.S)
        for spike in eachindex(snn.S[n])
            I[s] = n
            J[s] = t - snn.S[n][spike]
            s += 1
        end
    end

    i = J .>= 1
    return sparse(I[i], J[i], S[i], size(snn), t)
end

"""
    input(snn::AbstractSNN{VT, ST})::Vector{ST} where {VT<:Real, ST<:Integer}

Get the current spike activity of `snn`'s input neurons.
"""
function input(snn::AbstractSNN{VT, ST})::Vector{ST} where {VT<:Real, ST<:Integer}
    return spikes(snn, findall(snn.I))
end

"""
    inputs(snn::AbstractSNN{VT, ST})::Vector{ST} where {VT<:Real, ST<:Integer}

Get the indices of all input neurons
"""
function inputs(snn::AbstractSNN{VT, ST})::Vector{ST} where {VT<:Real, ST<:Integer}
    return findall(snn.I)
end

"""
    inputs(snn::AbstractSNN{VT, ST}, n::Integer)::Vector{ST} where {VT<:Real, ST<:Integer}

Get the indices of all input neurons with projections to neuron `n`
"""
function inputs(snn::AbstractSNN{VT, ST}, n::Integer)::Vector{ST} where {VT<:Real, ST<:Integer}
    i = inneighbors(snn.graph, n)
    return i[snn.I[i]]
end

"""
    outputs(snn::AbstractSNN{VT, ST})::Vector{ST} where {VT<:Real, ST<:Integer}

Get the indices of all output neurons
"""
function outputs(snn::AbstractSNN{VT, ST})::Vector{ST} where {VT<:Real, ST<:Integer}
    return findall(snn.O)
end

"""
    outputs(snn::AbstractSNN{VT, ST}, n::Integer)::Vector{ST} where {VT<:Real, ST<:Integer}

Get the indices of all output neurons with projections from neuron `n`
"""
function outputs(snn::AbstractSNN{VT, ST}, n::Integer)::Vector{ST} where {VT<:Real, ST<:Integer}
    o = outneighbors(snn.graph, n)
    return o[snn.O[o]]
end

"""
    output(snn::AbstractSNN{VT, ST})::Vector{ST} where {VT<:Real, ST<:Integer}

Get the current spike activity of `snn`'s output neurons.
"""
function output(snn::AbstractSNN{VT, ST})::Vector{ST} where {VT<:Real, ST<:Integer}
    return spikes(snn, findall(snn.O))
end

"""
    is_input(snn::AbstractSNN, n::Integer)::Bool

Return true if neuron `n` is an input neuron in `snn`.
"""
is_input(snn::AbstractSNN, n::Integer)::Bool = snn.I[n]

"""
    is_output(snn::AbstractSNN, n::Integer)::Bool

Return true if neuron `n` is an output neuron in `snn`.
"""
is_output(snn::AbstractSNN, n::Integer)::Bool = snn.O[n]

"""
    excitors(snn::AbstractSNN, n::integer)::Vector{<:Integer}

Return the indices of all neurons in the network `snn` with
excitatory connections to neuron `n`.
"""
function excitors(snn::AbstractSNN, n::Integer)::Vector{<:Integer}
    i = inneighbors(snn.graph, n)
    w = view(weights(snn.graph), i, n)
    return i[@. w > 0 & !snn.I[i]]
end

"""
    inhibitors(snn::AbstractSNN, n::integer)::Vector{<:Integer}

Return the indices of all neurons in the network `snn` with
inhibitory connections to neuron `n`.
"""
function inhibitors(snn::AbstractSNN, n::Integer)::Vector{<:Integer}
    i = inneighbors(snn.graph, n)
    w = view(weights(snn.graph), i, n)
    return i[@. w > 0 & !snn.I[i]]
end

"""
    potential(snn::AbstractSNN, n::Integer)::Real

Calculate the mean of a neuron `n`'s inputs in the
current configuration of the network `snn`.
"""
function potential(snn::AbstractSNN, n::Integer)::Real
    # ensure the vertex is valid
    if snn.I[n]
        throw("Neuron $n is an input neuron in the SNN, so its potential must be provided.")
    end

    return mean(spikes(snn, inneighbors(snn.graph, n))) + bias(snn, n)
end

"""
    potential(snn::AbstractSNN{VT, ST ,G}, n::Integer)::Real where {VT<:Real, ST<:Integer, G<:AbstractSimpleWeightedGraph}

Calculate the weighted average of a neuron `n`'s inputs in the
current configuration of the weighted network `snn`.
"""
function potential(snn::AbstractSNN{VT, ST ,G}, n::Integer)::Real where {VT<:Real, ST<:Integer, G<:AbstractSimpleWeightedGraph}
    # ensure the vertex is valid
    if  snn.I[n]
        throw("Neuron $n is an input neuron in the SNN, so its potential must be provided.")
    end

    neighbors = inneighbors(snn.graph, n)
    return sum(spikes(snn, neighbors) .* weights(snn.graph)[neighbors, n]) + bias(snn, n)
end

"""
    sigmoid(x::Real)::Real

The sigmoid spike probability function.
"""
sigmoid(x::Real)::Real = 1 / (1 + exp(-x))

"""
    thresh(x::Real, θ::Real)::Real

The threshold/step probability function.
"""
thresh(x::Real, θ::Real)::Real = x >= θ ? 1 : 0

"""
    ramp(x::Real)::Real

The ramp activation function (a rectifier).
"""
ramp(x::Real)::Real = max(0, x)

"""
    fire(prob::Real, on=1, off=0)::Real

Return `true` with probability `prob`, otherwise return `false`.
"""
fire(prob::Real)::Bool = rand() <= prob

"""
    cycle!(snn::AbstractSNN{VT, ST, G}, s::Vector{<:ST}, v::Vector{<:VT})::AbstractSNN{VT,ST,G} where {VT<:Real, ST<:Integer, G<:AbstractGraph}

    Push `v` to the front of `snn`'s voltage memory, deleting any excess
    voltages at the end of `snn`'s memory.
    Add current spike times (zeros) to `snn`'s spike train memory if `s[n] = true`,
    and increment previous spike times by 1.
"""
function cycle!(snn::AbstractSNN{VT, ST, G}, s::AbstractVector{<:ST}, v::AbstractVector{<:VT})::AbstractSNN{VT,ST,G} where {VT<:Real, ST<:Integer, G<:AbstractGraph}
    # add the new voltages
    if memory(snn) == 1
        @inbounds snn.V[1] = v
    elseif memory(snn) > 1
        push!(snn.V, v)
        while length(snn.V) > memory(snn)
            popfirst!(snn.V)
        end
    end

    # increment spike times and add the new spikes
    @inbounds for n in neurons(snn)
        snn.S[n] .+= oneunit(ST)
        if !iszero(s[n])
            push!(snn.S[n], zero(ST))
        end
    end
    return snn
end

"""
    step!(snn::AbstractSNN{VT,ST,G}, input::AbstractVector{<:ST}=zeros(ST, sum(snn.I)); pot::Function=potential, transfer::Function=sigmoid, f=fire)::AbstractSNN{VT,ST,G} where {VT<:Real, ST<:Integer, G<:AbstractGraph}

Calculate `snn`'s new firing configuration as `f(p(pot(snn, input)))`
and cycle `snn`'s memory accordingly.
"""
function step!(snn::AbstractSNN{VT,ST,G}, input::AbstractVector{<:ST}=falses(sum(snn.I)); pot::Function=potential, transfer::Function=sigmoid, f=fire)::AbstractSNN{VT,ST,G} where {VT<:Real, ST<:Integer, G<:AbstractGraph}
    s = Vector{ST}(undef, size(snn))
    v = Vector{VT}(undef, size(snn))

    # retrieve the new input configuration
    I = findall(snn.I)
    @inbounds v[I] .= zero(VT)
    @inbounds s[I] = input

    # compute the new configuration
    I = findall(.!snn.I)
    @inbounds @. v[I] = pot(snn, I)
    @inbounds @. s[I] = f(transfer(v[I]))

    # add the new configuration and remove excess configuration memory
    return cycle!(snn, s, v)
end

"""
    step!(snn::AbstractSNN{VT,ST,G}, input::AbstractMatrix{ST}; pot::Function=potential, transfer::Function=sigmoid, f=fire)::Tuple{AbstractSNN{VT,ST,G}, Matrix{VT}} where {VT<:Real, ST<:Integer, G<:AbstractGraph}

For each column in `input[neuron, time]`, step! through `snn`. Return the updated
SNN and the entire history of its voltages `V[neuron, time]` through each step.
"""
function step!(snn::AbstractSNN{VT,ST,G}, input::AbstractMatrix{<:ST}, stateFns::Vararg{Function, N}; pot::Function=potential, transfer::Function=sigmoid, f=fire)::Tuple{AbstractSNN{VT,ST,G}, Vararg{Vector{Any}, N}} where {VT<:Real, ST<:Integer, G<:AbstractGraph, N}
    T = size(input, 2)
    state = Tuple{Vararg{Vector{Any}, N}}(Vector{Any}(undef,T) for i in 1:length(stateFns))
    @inbounds @views for t in 1:T
        step!(snn, input[:,t], pot=pot, transfer=transfer, f=f)
        for i in eachindex(state)
            state[i][t] = stateFns[i](snn)
        end
    end

    return snn, state...
end

"""
    step!(snn::AbstractSNN{VT,ST,G}, iterations::Integer; pot::Function=potential, transfer::Function=sigmoid, f=fire)::Tuple{AbstractSNN{VT,ST,G}, Matrix{VT}} where {VT<:Real, ST<:Integer, G<:AbstractGraph}

For each `iteration` iterations, step! through `snn` with zero input. Return the updated
SNN and the entire history of its voltages `V[neuron, time]` through each step.
"""
function step!(snn::AbstractSNN{VT,ST,G}, iterations::Integer, stateFns::Vararg{Function, N}; pot::Function=potential, transfer::Function=sigmoid, f=fire)::Tuple{AbstractSNN{VT,ST,G}, Vararg{Vector{Any}, N}} where {VT<:Real, ST<:Integer, G<:AbstractGraph, N}
    return step!(snn, zeros(ST, sum(snn.I), iterations), stateFns...; pot=pot, transfer=transfer, f=f)
end

include("SimpleSNN.jl")
include("LIF.jl")
include("GraphUtils.jl")
include("SNNPlot.jl")

end
