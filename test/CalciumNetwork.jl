module CalciumNetwork

using SpikingNeuralNets, SpikingNeuralNets.LIFs,
        SpikingNeuralNets.Inputs, SpikingNeuralNets.GraphUtils,
        SpikingNeuralNets.SNNPlot
using LightGraphs, SimpleWeightedGraphs, Plots


#--------------------------------Graph Definition-------------------------------
# number of neurons in excitatory and inhibitory populations
const Nex = 25
const Nin = 25

# Build a meta-graph with excitatory and inhibitory populations and probabilistic connections
metaGraph = SimpleWeightedDiGraph(2)
probabilityGraph = SimpleWeightedDiGraph(2)
add_edge!(metaGraph, 1, 1, 1.0);   add_edge!(probabilityGraph, 1, 1, 0.9)
add_edge!(metaGraph, 1, 2, 1.0);   add_edge!(probabilityGraph, 1, 2, 0.75)
add_edge!(metaGraph, 2, 1, -1.0);  add_edge!(probabilityGraph, 2, 1, 0.8)
add_edge!(metaGraph, 2, 2, -1.0);  add_edge!(probabilityGraph, 2, 2, 0.66)

# Boolean arrays to serve as accessors for different populations
const excitatory = [trues(Nex); falses(Nin + Nex + Nin)]
const inhibitory = [falses(Nex); trues(Nin); falses(Nex + Nin)]
const inputsex = [falses(Nex+Nin); trues(Nex); falses(Nin)]
const inputsin = [falses(Nex+Nin+Nex); trues(Nin)]

# LIF Parameters
const dt = 0.02e-3      # Timestep
const Vrest = -70e-3     # Resting potential (-70 mV)
const Vθ = -50e-3        # Firing threshold (-50 mV)
const Vreset = -55e-3   # Reset potential (-55 mv)
const Vi = -70e-3       # inhibitory reversal potential (-70 mV)
const Mg = 1e-3         # Mg concentration (1 mM)
const λext = 2.4e3      # Rate of spontaneous external inputs (2.4 kHz)

Cm(n::Integer)::Real = excitatory[n] ? 0.5e-9 : 0.2e-9  # membrane capacitance (nF)
Rm(n::Integer)::Real = 1.0 / Cm(n)          # membrane resistance
gL(n::Integer)::Real = excitatory[n] ? 25e-9 : 20e-9    # membrane leak conductance (nS)
τref(n::Integer)::Real = excitatory[n] ? 2e-3 : 1e-3        # refractory period (mS)
τ(n::Integer)::Real = Cm(n) / gL(n)

# synaptic conductances (nS)
gext(n::Integer)::Real = excitatory[n] ? 0.25e-9 : 0.2e-9
gNMDA(n::Integer)::Real = excitatory[n] ? 0.15e-9 : 0.13e-9
gGABA(n::Integer)::Real = excitatory[n] ? 1.0e-9 : 1.25e-9

# synaptic decay times
const τext = 2e-3/dt
const α = 500.0*dt
const τNMDAd = 100e-3/dt # decay
const τNMDAr = 2e-3/dt # rise
const τGABA = 5e-3/dt

# synaptic current and gating functions
function Iext(lif::LIF, n::Integer, s::AbstractVector{<:Real})::Real
    return -gext(n) * voltage(lif, n) * s[1]
end
function dsext(lif::LIF{VT}, n::Integer, s::AbstractVector{VT})::Vector{VT} where {VT <: Real}
    input = spikes(lif, inputs(lif, n))
    return @. -(s/τext) + input
end

function INMDA(lif::LIF{VT}, n::Integer, vars::Matrix{VT}; inputs=false)::Real where {VT <: Real}
    v = voltage(lif, n)
    input = sum(weights(lif.graph)[excitors(lif, n; inputs=inputs), n] .* view(vars, :, 1))
    return -gNMDA(n) * v * input / (1 + Mg * exp(-0.062v) / 3.57)
end
function dvarsNMDA(lif::LIF{VT}, n::Integer, vars::Matrix{VT}; inputs=false)::Matrix{VT} where {VT <: Real}
    dvars = similar(vars)
    s, x = eachcol(vars)
    input = spikes(lif, excitors(lif, n; inputs=inputs))
    dvars[:, 1] = @. -(s/τNMDAd) + α*x*(1 - s)
    dvars[:, 2] = @. -(x/τNMDAr) + input
    return dvars
end

function IGABA(lif::LIF{VT}, n::Integer, s::AbstractVector{VT}; inputs=false)::Real where {VT <: Real}
    return -gGABA(n) * (voltage(lif, n) - Vi) * sum(s)
end
function dsGABA(lif::LIF{VT}, n::Integer, s::AbstractVector{VT}; inputs=false)::Vector{VT} where {VT <: Real}
    input = spikes(lif, inhibitors(lif, n; inputs=inputs))
    return @. -(s/τGABA) + input
end









function run(; iterations=100, window=100)
    # randomly generate a graph
    graph = ComplexGraph(metaGraph, [Nex, Nin], probabilityGraph; selfloops=false)
    # add external input neurons for all neurons, whose spikes will be pre-defined by a ConstantPoissonInput
    add_inputs!(graph, vertices(graph), fill(1.0, nv(graph)))

    # input iterator: randomly draw from a Poisson distribution every timestep
    I = ConstantPoissonInput(λext*dt, Nex+Nin, iterations)

    # construct an LIF with the above parameters
    lif = LIF{Float64, Int}(graph; dt=dt, Vrest=fill(Vrest, nv(graph)),
                            Vθ=fill(Vθ, nv(graph)),
                            Vreset=fill(Vreset, nv(graph)),
                            τ=τ.(vertices(graph)),
                            R=Rm.(vertices(graph)),
                            τref=τref.(vertices(graph)))
    add_channel!(lif, "ext", Iext, dsext, [zeros(1) for n in neurons(lif)])
    add_channel!(lif, "NMDA", INMDA, dvarsNMDA, [zeros(length(excitors(lif, n)), 2) for n in neurons(lif)])
    add_channel!(lif, "GABA", IGABA, dsGABA, [zeros(length(inhibitors(lif, n))) for n in neurons(lif)])

    # run the simulation and make some plots
    step!(lif, I; threaded=true)
    S = spiketrain(lif, iterations)[1:(Nex+Nin),:]

    Rex, t = smooth(S[1:Nex, :]; window=window, timestep=dt)
    Rin, t = smooth(S[(Nex+1):end, :]; window=window, timestep=dt)

    display(plot(rasterplot(S; timestep=dt),
                 plot(t, [Rex Rin], ylab="Firing Rate (Hz)", label=["Excitatory", "Inhibitory"]),
                 xlim=[0.0, iterations*dt], layout=(2,1), xlab="Time (s)"))
end

end
