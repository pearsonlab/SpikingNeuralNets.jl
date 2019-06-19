module SimpleSNN
import SpikingNeuralNets: AbstractSNN, memory, bias
using LightGraphs

export SNN, memory, bias

"""
A simple data-type for generalized spiking neural networks.

# Arguments
- `graph::G`: the graph structure of neurons (vertices) and weights (edges)
- `bias::Vector{Real}`: the activation (potential) bias of each neuron
- `I::Vector{Bool}`: specifies whether each neuron is an input neuron
- `O::Vector{Bool}`: specifies whether each neuron is an output neuron
- `m::Integer`: the length of configuration memory
- `S::Vector{Vector{ST}}`: a memory S[neuron][spike] of spike times
- `V::Vector{Vector{VT}}`: a length m voltage memory V[neuron][time],
    where V[1] is the least recent voltage and V[end] is the current recent voltage
"""
struct SNN{VT, ST, G} <: AbstractSNN{VT, ST, G}
    graph::G
    bias::Vector{VT}
    I::Vector{Bool}
    O::Vector{Bool}
    m::Integer
    S::Vector{Vector{ST}}
    V::Vector{Vector{VT}}

    """
        SNN{VT,ST,G}(graph::G, bias::Vector{<:VT}=zeros(nv(graph)), m::Integer=1,
                     S::Vector{Vector{<:ST}}=[Vector{ST}() for _ in vertices(g)],
                     V::Vector{Vector{<:VT}}=[zeros(VT, nv(graph)) for _ in 1:m])

    Create a new SNN{VT,ST,G} with the given graph and starting configuration.
    """
    function SNN{VT,ST,G}(graph::G; bias::Vector{<:VT}=zeros(VT, nv(graph)), m::Integer=1,
                          S::Vector{<:Vector{<:ST}}=[Vector{ST}() for _ in vertices(g)],
                          V::Vector{<:Vector{<:VT}}=[zeros(VT, nv(graph)) for _ in 1:m]) where {VT<:Real, ST<:Integer, G<:AbstractGraph}
        if m < 1
            error("Configuration memory length (m) must be <= 1.")
        end

        # Input neurons are those without incoming connections
        I = map(n -> length(inneighbors(graph, n)) == 0, vertices(graph))
        # Output neurons are those without outgoing connections
        O = map(n -> length(outneighbors(graph, n)) == 0, vertices(graph))
        return new(graph, bias, I, O, m, S[1:m], V[1:m])
    end
end

"""
    SNN{VT,ST}(graph::G, bias::Vector{Real}=zeros(nv(graph)), m::Integer=1,
               S::Vector{<:Vector{<:ST}}=[Vector{ST}() for _ in vertices(g)],
               VT::Vector{<:Vector{<:VT}}=[zeros(VT, nv(graph)) for _ in 1:m])

Create a new SNN{VT,ST,G} with the given graph and starting configuration.
Shorthand for the more explicit call SNN{VT,G}(graph, m, S, V)
"""
function SNN{VT, ST}(graph::G; bias::Vector{VT}=zeros(VT, nv(graph)), m::Integer=1,
                     S::Vector{<:Vector{<:ST}}=[Vector{ST}() for _ in vertices(g)],
                     V::Vector{<:Vector{<:VT}}=[zeros(VT, nv(graph)) for _ in 1:m]) where {VT<:Real, ST<:Integer, G<:AbstractGraph}
    return SNN{VT, ST, G}(graph; bias=bias, m=m, S=S, V=V)
end

"""
    memory(snn::AbstractSNN)::Integer

Return the size of the network `snn`'s voltage memory.
"""
memory(snn::SNN)::Real = snn.m

"""
    bias(snn::AbstractSNN)::Vector{<:Real}

Return the bias for all neurons in the network `snn`.
"""
bias(snn::SNN)::Vector{<:Real} = snn.bias

"""
    bias(snn::AbstractSNN, n::Integer)::Real

Return the bias of neuron `n` in the network `snn`.
"""
bias(snn::SNN, n::Integer)::Real = snn.bias[n]

end
