module SpikingNeuralNets
using LightGraphs, SimpleWeightedGraphs, Statistics, Plots, RecursiveArrayTools, SparseArrays
import Base.size, Base.length, Base.push!

export AbstractSNN,  # data-types
    size, neurons, memory, bias, spikes,
    spike, spiketimes, spiketime, spiketrain,
    voltages, voltage, configuration, input, output,
    is_input, is_output, excitors, inhibitors,  # accessor functions
    potential, # neuron potential function
    sigmoid, thresh, ramp, fire, # transfer functions
    step!, stepv!,  # used to iterate the SNN
    rasterplot, rateplot, vplot  # plotting functions

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

"""
    size(snn::AbstractSNN)::Integer

Return the number of neurons in the network `snn`.
"""
size(snn::AbstractSNN) = nv(snn.graph)

"""
    length(snn::AbstractSNN)::Integer

Return the number of neurons in the network `snn`.
"""
length(snn::AbstractSNN)::Integer = nv(snn.graph)

"""
    neurons(snn::AbstractSNN)::UnitRange{Integer}

Return the a range of indices for all neurons in the network `snn`.
"""
neurons(snn::AbstractSNN)::UnitRange{<:Integer} = vertices(snn.graph)

"""
    memory(snn::AbstractSNN)::Integer

Return the size of the network `snn`'s voltage memory.
"""
memory(snn::AbstractSNN)::Integer = length(snn.V)

"""
    bias(snn::AbstractSNN)::Vector{<:Real}

Return the bias for all neurons in the network `snn`.
"""
bias(snn::AbstractSNN)::Vector{<:Real} = zeros(neurons(snn))

"""
    bias(snn::AbstractSNN, n::Integer)::Real

Return the bias of neuron `n` in the network `snn`.
"""
bias(snn::AbstractSNN, n::Integer)::Real = n <= neurons(snn) ? bias(snn)[n] : throw("Neuron index out of bounds.")

"""
    voltages(snn::AbstractSNN{VT})::Vector{VT} where {VT<:Real}

Get the current voltage of all neurons in the network `snn`.
"""
function voltages(snn::AbstractSNN{VT})::Vector{VT} where {VT<:Real}
    return snn.V[end]
end

"""
    voltage(snn::AbstractSNN{VT}, n::Integer)::VT where {VT<:Real}

Get the current voltage of neuron `n` in the network `snn`.
"""
function voltage(snn::AbstractSNN{VT}, n::Integer)::VT where {VT<:Real}
    return snn.V[end][n]
end

"""
    spike(snn::AbstractSNN, n::Integer; t::Integer=0)::Bool

Calculate whether neuron `n` in the network `snn` was spiking `t` timesteps ago.
By default, calculates whether `n` is currently spiking.
"""
spike(snn::AbstractSNN, n::Integer; t::Integer=0)::Bool = (iszero(t) && !isempty(snn.S[n]) && snn.S[n][end] == 0) ||
                                                          !isempty(searchsorted(snn.S[n], t, rev=true))

"""
    spikes(snn::AbstractSNN, neurons::AbstractVector{<:Integer}; t::Integer=0)::BitArray{1}

Calculate whether `neurons` in the network `snn` are currently spiking.
"""
function spikes(snn::AbstractSNN, neurons::AbstractVector{<:Integer}=neurons(snn); t::Integer=0)::BitArray{1}
    S = BitArray(undef, length(neurons))
    Threads.@threads for i in eachindex(neurons)
        S[i] = spike(snn, neurons[i]; t=t)
    end
    return S
end

"""
    spiketimes(snn::AbstractSNN{VT, ST}, n::Integer)::Vector{ST} where {VT<:Real, ST<:Integer}

Return the number of timesteps into the past of all spikes for neuron `n` in the network `snn`.
"""
function spiketimes(snn::AbstractSNN{VT, ST}, n::Integer)::Vector{ST} where {VT<:Real, ST<:Integer}
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
function spiketime(snn::AbstractSNN{VT,ST}, n::Integer)::Integer where {VT<:Real, ST<:Integer}
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
    for n in eachindex(snn.S)
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
excitors(snn::AbstractSNN, n::Integer)::Vector{<:Integer} =
    filter(x->(weights(snn.graph)[x,n] > 0), inneighbors(snn.graph, n))

"""
    inhibitors(snn::AbstractSNN, n::integer)::Vector{<:Integer}

Return the indices of all neurons in the network `snn` with
inhibitory connections to neuron `n`.
"""
inhibitors(snn::AbstractSNN, n::Integer)::Vector{<:Integer} =
    filter(x->weights(snn.graph)[x,n] < 0, inneighbors(snn.graph, n))

"""
    potential(snn::AbstractSNN, n::Integer)::Real

Calculate the mean of a neuron `n`'s inputs in the
current configuration of the network `snn`.
"""
function potential(snn::AbstractSNN, n::Integer)::Real
    # ensure the vertex is valid
    if !has_vertex(snn.graph, n)
        throw("Vertex $(n) does not exist in graph $(snn.graph)")
    elseif snn.I[n]
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
    if !has_vertex(snn.graph, n)
        throw("Vertex $(n) does not exist in graph $(snn.graph)")
    elseif snn.I[n]
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

Return `1` with probability `prob`, otherwise return `0`.
"""
fire(prob::Real)::Integer = rand() <= prob ? 1 : 0

"""
    cycle!(snn::AbstractSNN{VT, ST, G}, s::Vector{<:ST}, v::Vector{<:VT})::AbstractSNN{VT,ST,G} where {VT<:Real, ST<:Integer, G<:AbstractGraph}

    Push `v` to the front of `snn`'s voltage memory, deleting any excess
    voltages at the end of `snn`'s memory.
    Add current spike times (zeros) to `snn`'s spike train memory, and
    increment previous spike times by 1.
"""
function cycle!(snn::AbstractSNN{VT, ST, G}, s::Vector{<:ST}, v::Vector{<:VT})::AbstractSNN{VT,ST,G} where {VT<:Real, ST<:Integer, G<:AbstractGraph}
    # add the new voltages
    if memory(snn) == 1
        snn.V[1] = v
    elseif memory(snn) > 1
        push!(snn.V, v)
        while length(snn.V) > memory(snn)
            popfirst!(snn.V)
        end
    end

    # increment spike times and add the new spikes
    Threads.@threads for n in neurons(snn)
        snn.S[n] .+= ST(1)
        if Bool(s[n])
            push!(snn.S[n], ST(0))
        end
    end
    return snn
end

"""
    step!(snn::AbstractSNN{VT,ST,G}, input::AbstractVector{<:ST}=zeros(ST, sum(snn.I)); pot::Function=potential, transfer::Function=sigmoid, f=fire)::AbstractSNN{VT,ST,G} where {VT<:Real, ST<:Integer, G<:AbstractGraph}

Calculate `snn`'s new firing configuration as `f(p(pot(snn, input)))`
and cycle `snn`'s memory accordingly.
"""
function step!(snn::AbstractSNN{VT,ST,G}, input::AbstractVector{<:ST}=zeros(ST, sum(snn.I)); pot::Function=potential, transfer::Function=sigmoid, f=fire)::AbstractSNN{VT,ST,G} where {VT<:Real, ST<:Integer, G<:AbstractGraph}
    s = Vector{ST}(undef, size(snn))
    v = Vector{VT}(undef, size(snn))

    # retrieve the new input configuration
    s[findall(snn.I)] = input
    v[findall(snn.I)] .= VT(0)

    # compute the new configuration in parallel
    Threads.@threads for i in findall(.!snn.I)
        v[i] = VT(pot(snn, i))
        s[i] = ST(f(transfer(v[i])))
    end

    # add the new configuration and remove excess configuration memory
    return cycle!(snn, s, v)
end

"""
    step!(snn::AbstractSNN{VT,ST,G}, input::AbstractMatrix{ST}; pot::Function=potential, transfer::Function=sigmoid, f=fire)::Tuple{AbstractSNN{VT,ST,G}, Matrix{VT}} where {VT<:Real, ST<:Integer, G<:AbstractGraph}

For each column in `input[neuron, time]`, step! through `snn`. Return the updated
SNN and the entire history of its voltages `V[neuron, time]` through each step.
"""
function step!(snn::AbstractSNN{VT,ST,G}, input::AbstractMatrix{ST}; pot::Function=potential, transfer::Function=sigmoid, f=fire)::Tuple{AbstractSNN{VT,ST,G}, Matrix{VT}} where {VT<:Real, ST<:Integer, G<:AbstractGraph}
    T = size(input, 2)
    V = Matrix{VT}(undef, size(snn), T)

    for i in 1:T
        step!(snn, input[:,i], pot=pot, transfer=transfer, f=f)
        V[:,i] = voltages(snn)
    end

    return snn, V
end

"""
    step!(snn::AbstractSNN{VT,ST,G}, iterations::Integer; pot::Function=potential, transfer::Function=sigmoid, f=fire)::Tuple{AbstractSNN{VT,ST,G}, Matrix{VT}} where {VT<:Real, ST<:Integer, G<:AbstractGraph}

For each `iteration` iterations, step! through `snn` with zero input. Return the updated
SNN and the entire history of its voltages `V[neuron, time]` through each step.
"""
function step!(snn::AbstractSNN{VT,ST,G}, iterations::Integer; pot::Function=potential, transfer::Function=sigmoid, f=fire)::Tuple{AbstractSNN{VT,ST,G}, Matrix{VT}} where {VT<:Real, ST<:Integer, G<:AbstractGraph}
    V = Matrix{VT}(undef, size(snn), iterations)
    input = zeros(ST, sum(snn.I))

    for i in 1:iterations
        step!(snn, input, pot=pot, transfer=transfer, f=f)
        V[:,i] = voltages(snn)
    end
    return snn, V
end

"""
    rasterplot(S::AbstractMatrix{<:Real}; timestep=1, kwargs...)

Generate a raster plot of neuronal spiking given the matrix `S[neuron, time]`
of neuronal spike trains, where each time-step in `C` corresponds
to `timestep=1` seconds.
"""
function rasterplot(S::AbstractMatrix{<:Real}; timestep=1, kwargs...)
    spikes = findall(Bool.(S))
    N = getindex.(spikes, 1)
    T = getindex.(spikes, 2) * timestep
    return scatter(T, N; leg=false, kwargs...)
end

"""
    rateplot(S::AbstractMatrix{<:Real}; window=0, dt=1, timestep=1, average=true, kwargs...)

Generate a population rate plot of neuronal spiking given the matrix
`S[neuron, time]` of neuronal spike trains. Rates are averaged
over a window of `window=1` time-steps which slides in strides of
 `dt=1` time-steps, where a single time-step corresponds to `timestep=1` seconds.
 If `average` is `true`, then average over all neurons in each window. Otherwise,
 plot a separate line for each neuron.
"""
function rateplot(S::AbstractMatrix{<:Real}; window=0, dt=1, timestep=1, average=true, kwargs...)
    N, T = size(S)
    bins = floor(Int, (T - window) / dt)
    if bins <= 0
        return plot([]; leg=!average, kwargs...)
    end
    rates = Matrix{Float64}(undef, bins, average ? 1 : N)
    time = range(window*timestep/2, length=bins, step=timestep*dt)
    tohz = 1 / (timestep*dt)

    Threads.@threads for t in 0:bins-1
        if average
            rates[t+1] = mean(S[:,(dt*t + 1):(dt*t + window + 1)]) * tohz
        else
            for n in 1:N
                rates[t+1, n] = mean(S[n, (dt*t + 1):(dt*t + window + 1)]) * tohz
            end
        end
    end

    return plot(time, rates; kwargs...)
end

"""
    vplot(V::AbstractArray{<:Real}; timestep=1, kwargs...)

Generate a plot of neuronal voltage over time from the matrix V[neuron, time],
where each column of V is separated by `timestep=1` seconds.
"""
function vplot(V::AbstractArray{<:Real}; timestep=1, kwargs...)
    time = range(0, length=size(V, 2), step=timestep)
    return plot(time, V'; kwargs...)
end

include("SimpleSNN.jl")
include("LIF.jl")
include("GraphUtils.jl")

end
