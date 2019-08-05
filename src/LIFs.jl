module LIFs
using LightGraphs
using SpikingNeuralNets: voltage, spike, spiketime
import SpikingNeuralNets: AbstractSNN, potential, thresh
export LIF, memory, channels, haschannel, currentFn, varFn, vars, add_channel!, remove_channel!, I_syn, potential, thresh

"""
A data-type for LIF spiking neural networks.

# Arguments
- `graph::G`: the graph structure of neurons (vertices) and weights (edges)
- `I::Vector{Bool}`: specifies whether each neuron is an input neuron
- `O::Vector{Bool}`: specifies whether each neuron is an output neuron
- `S::Vector{Dict{ST, ST}}`: a memory S[neuron][time] of spike counts
- `V::Vector{Vector{VT}}`: a length m voltage memory V[neuron][time],
    where V[1] is the least recent voltage and V[end] is the current recent voltage
- `m::Integer`: the length of configuration memory
- `dt::VT`: the integration time-step of the network
- `Vrest::Vector{VT}`: the resting voltage of each neuron in the network
- `Vθ::Vector{VT}`: the voltage threshold of each neuron in the network
- `Vreset::Vector{VT}`: the reset voltage of each neuron in the network
- `τ::Vector{VT}`: the membrane time constant of each neuron in the network
- `R::Vector{VT}`: the membrane resistance of each neuron in the network
- `τref::Vector{VT}`: the refractory period of each neuron in the network
- `currentFns::Dict{String, Function}`: the mapping of channel names to the current
    functions I(lif::LIF, n::Integer, s::Vector{ST})::Real for each channel
- `varFns::Dict{String, Function}`: the mapping of channel names to the variable
    update functions s(lif::LIF, n::Integer, s::Vector{ST})::Vector{ST} for each channel
- `vars::Dict{Tuple{String, Integer}, Array{VT}}`: the mapping of channel names
    and neuron index to the (gating) parameters Array{VT}
"""
struct LIF{VT, ST, G} <: AbstractSNN{VT, ST, G}
    graph::G
    I::Vector{Bool}
    O::Vector{Bool}
    S::Vector{Dict{ST, ST}}
    V::Vector{Vector{VT}}
    m::ST
    dt::VT
    Vrest::Vector{VT}
    Vθ::Vector{VT}
    Vreset::Vector{VT}
    τ::Vector{VT}
    R::Vector{VT}
    τref::Vector{VT}
    currentFns::Dict{String, Function}
    varFns::Dict{String, Function}
    vars::Dict{Tuple{String, Integer}, Array{VT}}

    """
        LIF{VT,ST,G}(graph::G; m::ST=ST(1), dt::VT=VT(1), Vrest::Vector{<:VT}=zeros(VT, nv(graph)),
                     Vθ::Vector{<:VT}=zeros(VT, nv(graph)),
                     Vreset::Vector{VT}=zeros(VT, nv(graph)),
                     τ::Vector{<:VT}=ones(VT, nv(graph)),
                     R::Vector{<:VT}=ones(VT, nv(graph)),
                     τref::Vector{<:VT}=zeros(VT, nv(graph)),
                     currentFns::Dict{String, Function}=Dict{String, Function}(),
                     varFns::Dict{String, Function}=Dict{String, Function}(),
                     vars::Dict{<:Tuple{String, <:Integer}, <:Array{<:VT}}=Dict{Tuple{String, Int}, Vector{VT}}(),
                     S::Vector{<:Dict{<:ST, <:ST}}=[Dict{ST, ST}() for _ in vertices(graph)],
                     V::Vector{<:Vector{<:VT}}=[rand(nv(graph)).*(Vθ.-Vrest).+Vrest]) where {VT<:Real, ST<:Integer, G<:AbstractGraph{ST}}

    Create a new LIF{VT,ST,G} with the given graph, parameters, and initial configuration.
    """
    function LIF{VT,ST,G}(graph::G; m::ST=ST(1), dt::VT=VT(1), Vrest::Vector{<:VT}=zeros(VT, nv(graph)),
                          Vθ::Vector{<:VT}=zeros(VT, nv(graph)),
                          Vreset::Vector{VT}=zeros(VT, nv(graph)),
                          τ::Vector{<:VT}=ones(VT, nv(graph)),
                          R::Vector{<:VT}=ones(VT, nv(graph)),
                          τref::Vector{<:VT}=zeros(VT, nv(graph)),
                          currentFns::Dict{String, Function}=Dict{String, Function}(),
                          varFns::Dict{String, Function}=Dict{String, Function}(),
                          vars::Dict{<:Tuple{String, <:Integer}, <:Array{<:VT}}=Dict{Tuple{String, Int}, Vector{VT}}(),
                          S::Vector{<:Dict{<:ST, <:ST}}=[Dict{ST, ST}() for _ in vertices(graph)],
                          V::Vector{<:Vector{<:VT}}=[rand(nv(graph)).*(Vθ.-Vrest).+Vrest]) where {VT<:Real, ST<:Integer, G<:AbstractGraph{ST}}
        # Input neurons are those without incoming connections
        I = map(n -> length(inneighbors(graph, n)) == 0, vertices(graph))
        # Output neurons are those without outgoing connections
        O = map(n -> length(outneighbors(graph, n)) == 0, vertices(graph))
        return new(graph, I, O, S, V, m, dt, Vrest, Vθ, Vreset, τ, R, τref, currentFns, varFns, vars)
    end
end

"""
    LIF{VT,ST}(graph::G; m::ST=ST(1), dt::VT=VT(1), Vrest::Vector{<:VT}=zeros(VT, nv(graph)),
               Vθ::Vector{<:VT}=zeros(VT, nv(graph)),
               Vreset::Vector{VT}=zeros(VT, nv(graph)),
               τ::Vector{<:VT}=ones(VT, nv(graph)),
               R::Vector{<:VT}=ones(VT, nv(graph)),
               τref::Vector{<:VT}=zeros(VT, nv(graph)),
               currentFns::Dict{String, Function}=Dict{String, Function}(),
               varFns::Dict{String, Function}=Dict{String, Function}(),
               vars::Dict{<:Tuple{String, <:Integer}, <:Array{<:VT}}=Dict{Tuple{String, Int}, Vector{VT}}(),
               S::Vector{<:Dict{<:ST, <:ST}}=[Dict{ST, ST}() for _ in vertices(graph)],
               V::Vector{<:Vector{<:VT}}=[rand(nv(graph)).*(Vθ.-Vrest).+Vrest]) where {VT<:Real, ST<:Integer, G<:AbstractGraph{ST}}

Create a new LIF{VT,ST,G} with the given graph and starting configuration.
Shorthand for the more explicit call LIF{VT,ST,G}(graph...)
"""

function LIF{VT,ST}(graph::G; m::ST=ST(1), dt::VT=VT(1), Vrest::Vector{<:VT}=zeros(VT, nv(graph)),
                    Vθ::Vector{<:VT}=zeros(VT, nv(graph)),
                    Vreset::Vector{VT}=zeros(VT, nv(graph)),
                    τ::Vector{<:VT}=ones(VT, nv(graph)),
                    R::Vector{<:VT}=ones(VT, nv(graph)),
                    τref::Vector{<:VT}=zeros(VT, nv(graph)),
                    currentFns::Dict{String, Function}=Dict{String, Function}(),
                    varFns::Dict{String, Function}=Dict{String, Function}(),
                    vars::Dict{<:Tuple{String, <:Integer}, <:Array{<:VT}}=Dict{Tuple{String, Int}, Vector{VT}}(),
                    S::Vector{<:Dict{<:ST, <:ST}}=[Dict{ST, ST}() for _ in vertices(graph)],
                    V::Vector{<:Vector{<:VT}}=[rand(nv(graph)).*(Vreset.-Vrest).+Vrest]) where {VT<:Real, ST<:Integer, G<:AbstractGraph{ST}}
    return LIF{VT,ST,G}(graph; S=S, V=V, m=m, dt=dt, Vrest=Vrest, Vθ=Vθ, Vreset=Vreset,
                        τ=τ, R=R, τref=τref, currentFns=currentFns, varFns=varFns, vars=vars)
end

"""
    memory(lif::LIF)::Integer

Return the size of the network `lif`'s voltage memory.
"""
function memory(lif::LIF{VT, ST})::ST where {VT<:Real, ST<:Integer}
    return lif.m
end

"""
    channels(lif::LIF)::Base.KeySet{String, Dict{String, Function}}

Return a set of channel labels for all channels in the network `lif`.
"""
channels(lif::LIF)::Base.KeySet{String, Dict{String, Function}} = keys(lif.currentFns)

"""
    haschannel(lif::LIF, channel::String)::Bool

Return true if the network `lif` has a channel with the label `channel`.
"""
haschannel(lif::LIF, channel::String)::Bool = haskey(lif.currents, channel)

"""
    haschannel(lif::LIF, channel::String, n::Integer)::Bool

Return true if the neuron `n` in the network `lif` has a channel with the label `channel`.
"""
haschannel(lif::LIF, channel::String, n::Integer)::Bool = haskey(lif.vars, (channel, n))

"""
    currentFn(lif::LIF, channel::String)::Union{Function, Nothing}

Return the current function for the channel `channel` in the network `lif`.
"""
currentFn(lif::LIF, channel::String)::Union{Function, Nothing} = get(lif.currentFns, channel, nothing)

"""
    varFn(lif::LIF, channel::String)::Union{Function, Nothing}

Return the variable update function for the channel `channel` in the network `lif`.
"""
varFn(lif::LIF, channel::String)::Union{Function, Nothing} = get(lif.varFns, channel, nothing)

"""
    vars(lif::LIF{VT}, channel::String, n::Integer)::Union{Array{VT}, Nothing} where {VT<:Real}

Return the (gating) variables for the channel `channel` of neuron `n` in the network `lif`.
"""
function vars(lif::LIF{VT}, channel::String, n::Integer)::Union{Array{VT}, Nothing} where {VT<:Real}
    return get(lif.vars, (channel, n), nothing)
end

"""
    vars(lif::LIF{VT}, channel::String)::Vector{Union{Array{VT}, Nothing}} where {VT<:Real}

Return the (gating) variables for the channel `channel` of all neurons in the network `lif`.
"""
function vars(lif::LIF{VT}, channel::String)::Vector{Array{VT}} where {VT<:Real}
    v = Vector{Array{VT}}(undef, size(lif))
    @inbounds for n in neurons(lif)
        v[n] = vars(lif, channel, n)
    end
    return v
 end

"""
    add_channel!(lif::LIF{VT,ST,G}, name::String, currentFn::Function, varFn::Function,
                 vars::Vector{<:Array{<:VT}}=fill(zeros(VT, 0), size(lif)),
                 neurons::AbstractVector{Bool}=trues(size(lif)))::LIF{VT,ST,G} where {VT<:Real,ST<:Integer,G<:AbstractGraph}

Add a new channel type to the network `lif` with the label `channel`, the current
function `currentFn`, the variable update function `varFn`, the initial variables
`vars`, for every neuron in `neurons`.
"""
function add_channel!(lif::LIF{VT,ST,G}, name::String, currentFn::Function, varFn::Function,
                      vars::Vector{<:Array{<:VT}}=fill(zeros(VT, 0), size(lif)),
                      neurons::AbstractVector{<:Integer}=findall(.!lif.I))::LIF{VT,ST,G} where {VT<:Real,ST<:Integer,G<:AbstractGraph}
    lif.currentFns[name] = currentFn
    lif.varFns[name] = varFn
    for n in neurons
        lif.vars[name, n] = vars[n]
    end
    return lif
end

"""
    remove_channel!(lif::LIF{VT,ST,G}, name::String)::LIF{VT,ST,G} where {VT<:Real,ST<:Integer,G<:AbstractGraph}

Remove all channels from the network `lif` with the label `channel`.
"""
function remove_channel!(lif::LIF{VT,ST,G}, name::String)::LIF{VT,ST,G} where {VT<:Real,ST<:Integer,G<:AbstractGraph}
    delete!(lif.currentFns, name)
    delete!(lif.varFns, name)
    for n in neurons(lif)
        delete!(lif.vars, (name, n))
    end
    return lif
end

"""
    Isyn!(lif::LIF, n::Integer)::Real

Calculate the the total synaptic input to neuron `n` in the network `lif`,
calculating and updating the (gating) variables `lif.vars[channel, n]` for
each `channel` of the neuron.
"""
function Isyn!(lif::LIF, n::Integer)::Real
    I = 0

    @inbounds for channel in channels(lif)
        if haskey(lif.vars, (channel, n))
            # calculate the gating variables
            lif.vars[channel, n] .+= lif.varFns[channel](lif, n, lif.vars[channel, n])
            # calculate the synaptic current
            I += lif.currentFns[channel](lif, n, lif.vars[channel, n])
        end
    end
    return I
end


"""
    potential(lif::LIF, n::Integer)::Real

Calculate the potential of neuron `n` in the network `lif` according
to a leaky integrate-and-fire rule. Mutates the (gating) variables for neuron `n`
while calculating the synaptic input (see `Isyn!`).
"""
function potential(lif::LIF, n::Integer)::Real
    v = voltage(lif, n)
    I = Isyn!(lif, n)  # make sure to update the gating variables
    if !iszero(spike(lif, n)) || spiketime(lif, n) * lif.dt < lif.τref[n]
        return lif.Vreset[n]
    else
        return v + lif.dt * (lif.Vrest[n] - v + (lif.R[n] * I)) / lif.τ[n]
    end
end

"""
    thresh(lif::LIF, n::Integer, x::Real)::Real

The threshold/step transfer function.
"""
thresh(lif::LIF, n::Integer, x::Real)::Real = thresh(x, lif.Vθ[n])
end
