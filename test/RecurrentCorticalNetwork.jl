module RecurrentCorticalNetwork
using LightGraphs, SimpleWeightedGraphs, Distributions, Plots
using SpikingNeuralNets, SpikingNeuralNets.LIFSNN, SpikingNeuralNets.GraphUtils

# Set up the network
const nA = 100
const nB = 100
const nI = 50
const f = 0.5   # percentage of A/B neurons recieving input
const nAi = ceil(Int, f * nA)
const nBi = ceil(Int, f * nB)
const excitatoryP = 1.7   # the weight of strong excitatory connections
const excitatoryD = 1 - f*(excitatoryP-1)/(1-f)    # the weight of weak excitatory connections

# Vertices:
#   1- A neurons
#   2- B neurons
#   3- Interneurons
const metagraph = SimpleWeightedDiGraph(3)
add_edge!(metagraph, 1, 1, excitatoryP)  # A and B neurons are fully connected
add_edge!(metagraph, 2, 2, excitatoryP)
add_edge!(metagraph, 1, 2, excitatoryD)
add_edge!(metagraph, 2, 1, excitatoryD)
add_edge!(metagraph, 1, 3, 1.0)  # interneurons have excitatory inputs
add_edge!(metagraph, 2, 3, 1.0)
add_edge!(metagraph, 3, 1, -1.0)  # and inhibitory outputs
add_edge!(metagraph, 3, 2, -1.0)
add_edge!(metagraph, 3, 3, -1.0)
const graph = ComplexGraph(metagraph, [nA, nB, nI], selfloops=false) # neurons do not excite themselves

# Predicates for neuronal populations
const aneurons = map(x -> x <= nA, 1:nv(graph))
const ainputs = map(x -> x <= nAi, 1:nv(graph))
const bneurons = map(x -> x > nA && x <= nA+nB, 1:nv(graph))
const binputs = map(x -> x > nA && x <= nA+nBi, 1:nv(graph))
const pyramidalneurons = aneurons .| bneurons
const interneurons = .!pyramidalneurons

# Define input parameters
const μ0 = 40 # Hz
const ρa = μ0 / 100
const ρb = μ0 / 100
const σ = 4

# Define Leaky Integrate-and-Fire Parameters
const dt = 0.02e-3 # Timestep (0.02ms)
const Vrest = -0.07 # Resting potential (-70 mV)
const Vθ = -0.05 # Firing threshold (-50 mV)
const Vreset = -0.055 # Reset potential (-55 mv)
const Ve = 0.0      # excitatory reversal potential (0 mV)
const Vi = -70e-3   # inhibitory reversal potential (-70 mV)
const Mg = 1e-3    # Mg concentration (1 mM)
const Vext = 2.4e3  # Rate of spontaneous external inputs (2.4 kHz)

Cm(pyramidal::Bool)::Real = pyramidal ? 0.5e-9 : 0.2e-9  # membrane capacitance (nF)
Rm(pyramidal::Bool)::Real = 1.0 / Cm(pyramidal)
gL(pyramidal::Bool)::Real = pyramidal ? 25e-9 : 20e-9    # membrane leak conductance (nS)
τref(pyramidal::Bool)::Real = pyramidal ? 0.002 : 0.001        # refractory period (mS)
τ(pyramidal::Bool)::Real = Cm(pyramidal) / gL(pyramidal)

# synaptic conductances (nS)
gextAMPA(pyramidal::Bool)::Real = pyramidal ? 2.1e-9 : 1.62e-9
grecAMPA(pyramidal::Bool)::Real = pyramidal ? 0.05e-9 : 0.04e-9
gNMDA(pyramidal::Bool)::Real = pyramidal ? 0.165e-9 : 0.13e-9
gGABA(pyramidal::Bool)::Real = pyramidal ? 1.3e-9 : 1.0e-9
# synaptic decay times
const τAMPA = 2e-3
const α = 1e-3 / 0.5   # 0.5 (1/ms)
const τNMDAd = 100e-3 # decay
const τNMDAr = 2e-3 # rise
const τGABA = 5e-3

# Synaptic latency
const synl = 0.5e-3  # delay in seconds
const synt = round(Int, synl / dt)  # delay in timesteps

# synaptic current and gating functions
function IextAMPA(lif::LIF, n::Integer, s::Vector{<:Real})::Real
    # there's only one gating variable, so just index it
    return gextAMPA(pyramidalneurons[n]) * (voltage(lif, n) - Ve) * s[1]
end
function sextAMPA(inputA, inputB)
    # return a function which pulls inputs to A and B populations from inputA and inputB
    return function sextAMPA(lif::LIF{VT}, n::Integer, s::Vector{VT})::Vector{VT} where {VT <: Real}
        return -(s / τAMPA) + (rand(Poisson(Vext*lif.dt), 1) +
                [ainputs[n] ? popfirst!(inputA[n]) : 0.0] +
                [binputs[n] ? popfirst!(inputB[n-nA]) : 0.0]) / lif.dt

    end
end

function IrecAMPA(lif::LIF{VT}, n::Integer, s::Vector{VT})::Real where {VT <: Real}
    return grecAMPA(pyramidalneurons[n]) * (voltage(lif, n) - Ve) *
            sum(weights(lif.graph)[excitors(lif, n), n] .* s)
end
function srecAMPA(lif::LIF{VT}, n::Integer, s::Vector{VT})::Vector{VT} where {VT <: Real}
    return -(s / τAMPA) + spikes(lif, excitors(lif, n); t=synt) / lif.dt
end

function INMDA(lif::LIF{VT}, n::Integer, vars::Matrix{VT})::Real where {VT <: Real}
    v = voltage(lif, n)
    s = vars[:, 1]
    return gNMDA(pyramidalneurons[n]) * (v - Ve) * sum(weights(lif.graph)[excitors(lif, n), n] .* s) /
            (1 + Mg * exp(-0.062*v) / 3.57)
end
function xNMDA(lif::LIF{VT}, n::Integer, x::Vector{VT})::Vector{VT} where {VT <: Real}
    return -(x / τNMDAr) + spikes(lif, excitors(lif, n); t=synt) / lif.dt
end
function sNMDA(lif::LIF{VT}, s::Vector{VT}, x::Vector{VT})::Vector{VT} where {VT <: Real}
    return -(s / τNMDAd) + @. α * x * (1 - s)
end
function varsNMDA(lif::LIF{VT}, n::Integer, vars::Matrix{VT})::Matrix{VT} where {VT <: Real}
    x = xNMDA(lif, n, vars[:, 2])
    return [sNMDA(lif, vars[:, 1], x) x]
end

function IGABA(lif::LIF{VT}, n::Integer, s::Vector{VT})::Real where {VT <: Real}
    return gGABA(pyramidalneurons[n]) * (voltage(lif, n) - Vi) * sum(s)
end
function sGABA(lif::LIF{VT}, n::Integer, s::Vector{VT})::Vector{VT} where {VT <: Real}
    return -(s / τGABA) + spikes(lif, inhibitors(lif, n); t=synt) / lif.dt
end


function inputs(c::Real, length::Integer)::Tuple{Vector{Vector{Int}}, Vector{Vector{Int}}}
    μa = μ0 + ρa * c
    μb = μ0 - ρb * c
    sa = rand(Normal(μa, σ))  # input rates (Hz)
    sb = rand(Normal(μb, σ))
    Ainputs = [Vector{Int}(undef, length) for i in 1:nAi]
    Binputs = [Vector{Int}(undef, length) for i in 1:nBi]
    for t in 1:length
        Ain = Poisson(dt*sa)
        Bin = Poisson(max(dt*sb, 0.0))
        for n in 1:nAi
            Ainputs[n][t] = rand(Ain)
        end
        for n in 1:nBi
            Binputs[n][t] = rand(Bin)
        end
        # resample the rates every 50 ms.
        if t % (50e-3 / dt) == 0
            sa = rand(Normal(μa, σ))
            sb = rand(Normal(μb, σ))
        end
    end
    return Ainputs, Binputs
end

function run(;c::Real=0.0, iterations::Integer=10, window=100)
    GC.gc()
    Ainputs, Binputs = inputs(c, iterations)

    # create the LIF and add all of the channel types
    lif = LIF{Float64, Int}(graph, V=[[rand()*(Vθ-Vrest)+Vrest for n in vertices(graph)]],
                            m=1, dt=dt, Vrest=fill(Vrest, nv(graph)),
                            Vθ=fill(Vθ, nv(graph)),
                            Vreset=fill(Vreset, nv(graph)),
                            τ=τ.(pyramidalneurons),
                            R=Rm.(pyramidalneurons),
                            τref=τref.(pyramidalneurons))

    # Add excitatory and inhibitory channels
    add_channel!(lif, "extAMPA", IextAMPA, sextAMPA(Ainputs, Binputs), [rand(1)/100 for n in neurons(lif)])
    add_channel!(lif, "recAMPA", IrecAMPA, srecAMPA, [rand(length(excitors(lif, n)))/100 for n in neurons(lif)])
    add_channel!(lif, "NMDA", INMDA, varsNMDA, [rand(length(excitors(lif, n)), 2)/100 for n in neurons(lif)])
    add_channel!(lif, "GABA", IGABA, sGABA, [rand(length(inhibitors(lif, n)))/100 for n in neurons(lif)])

    lif, V = step!(lif, iterations, transfer=(x->thresh(x, Vθ)))
    S = spiketrain(lif, iterations)

    display(plot(rasterplot(S[aneurons,:]; ylab="Neuron", timestep=dt),
                 rateplot(S[aneurons,:]; ylab="Firing Rate", timestep=dt, window=window),
                 vplot(V[aneurons,:]; timestep=dt, ylab="Voltage", ylim=[Vrest, Vθ+0.005]);
                 title="A Neurons", xlim=[0.0, iterations*dt], leg=false, layout=(3,1)))
    display(plot(rasterplot(S[bneurons,:]; ylab="Neuron", timestep=dt),
                 rateplot(S[bneurons,:]; ylab="Firing Rate", timestep=dt, window=window),
                 vplot(V[bneurons,:]; timestep=dt, ylab="Voltage", ylim=[Vrest, Vθ+0.005]);
                 title="B Neurons", xlim=[0.0, iterations*dt], leg=false, layout=(3,1)))
    display(plot(rasterplot(S[interneurons,:]; ylab="Neuron", timestep=dt),
                 rateplot(S[interneurons,:]; ylab="Firing Rate", timestep=dt, window=window),
                 vplot(V[interneurons,:]; timestep=dt, ylab="Voltage", ylim=[Vrest, Vθ+0.005]);
                 title="Interneurons", xlim=[0.0, iterations*dt], leg=false, layout=(3,1)))
    GC.gc()
end

end
