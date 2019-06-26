module RecurrentCorticalNetwork
using LightGraphs, SimpleWeightedGraphs, Distributions, Plots
using SpikingNeuralNets, SpikingNeuralNets.LIFSNN, SpikingNeuralNets.GraphUtils, SpikingNeuralNets.SNNPlot

# Set up the network
const N = 1000
const fE = 0.8
const fI = 1.0 - fE
const nE = round(Int, fE*N)
const nI = round(Int, fI*N)
const f = 0.15  # percentage of excitatoy neurons selective to input
const nA = round(Int, f*nE)
const nB = nA
const nNS = nE - nA - nB
const excitatoryP = 1.05   # the weight of strong excitatory connections
const excitatoryD = 1 - f*(excitatoryP-1)/(1-f)    # the weight of weak excitatory connections
const inhibitory = -1.0
const neutral = 1.0

# Vertices:
#   1- A neurons
#   2- B neurons
#   3- Non-selective neurons
#   4- Interneurons
const metagraph = SimpleWeightedDiGraph(4)
add_edge!(metagraph, 1, 1, excitatoryP)  # A connections
add_edge!(metagraph, 1, 2, excitatoryD)
add_edge!(metagraph, 1, 3, neutral)
add_edge!(metagraph, 2, 1, excitatoryD)  # B connections
add_edge!(metagraph, 2, 2, excitatoryP)
add_edge!(metagraph, 2, 3, neutral)
add_edge!(metagraph, 3, 1, excitatoryD)  # non-selective connections
add_edge!(metagraph, 3, 2, excitatoryD)
add_edge!(metagraph, 3, 3, neutral)
add_edge!(metagraph, 1, 4, neutral)      # interneurons have excitatory inputs
add_edge!(metagraph, 2, 4, neutral)
add_edge!(metagraph, 3, 4, neutral)
add_edge!(metagraph, 4, 1, inhibitory)   # and inhibitory outputs
add_edge!(metagraph, 4, 2, inhibitory)
add_edge!(metagraph, 4, 3, inhibitory)
add_edge!(metagraph, 4, 4, inhibitory)
const graph = ComplexGraph(metagraph, [nA, nB, nNS, nI], selfloops=false) # neurons do not excite themselves
add_inputs!(graph, vertices(graph), fill(neutral, nv(graph)))        # add external inputs for all neurons

# Predicates for neuronal populations
const aneurons = map(x -> x <= nA, vertices(graph))
const bneurons = map(x -> x > nA && x <= nA+nB, vertices(graph))
const nsneurons = map(x -> x > nA+nB && x <= nE, vertices(graph))
const pyramidalneurons = @. aneurons | bneurons | nsneurons
const interneurons = map(x -> x > nE && x <= nE+nI, vertices(graph))
const inputneurons = map(x -> x > N, vertices(graph))
const ainputneurons = map(x -> x > N && x <= N+nA+nB, vertices(graph))
const binputneurons = map(x -> x > N && x <= N+nA+nB, vertices(graph))
const nsinputneurons = map(x -> x > N && x <= N+nE, vertices(graph))
const iinputneurons = map(x -> x > N+nE, vertices(graph))

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
const Wext = [1.0]   # weight of spontaneous external inputs

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
const synt = 0 # round(Int, synl / dt)  # delay in timesteps

# synaptic current and gating functions
function IextAMPA(lif::LIF, n::Integer, s::AbstractVector{<:Real})::Real
    return -gextAMPA(pyramidalneurons[n]) * (voltage(lif, n) - Ve) * sum(Wext .* s)
end
return function sextAMPA(lif::LIF{VT}, n::Integer, s::AbstractVector{VT})::Vector{VT} where {VT <: Real}
    input = spikes(lif, inputs(lif, n); t=synt)
    return @. -(s/τAMPA) + (input/lif.dt)
end

function IrecAMPA(lif::LIF{VT}, n::Integer, s::AbstractVector{VT})::Real where {VT <: Real}
    return -grecAMPA(pyramidalneurons[n]) * (voltage(lif, n) - Ve) *
            sum(weights(lif.graph)[excitors(lif, n), n] .* s)
end
function srecAMPA(lif::LIF{VT}, n::Integer, s::AbstractVector{VT})::Vector{VT} where {VT <: Real}
    input = spikes(lif, excitors(lif, n); t=synt)
    return @. -(s/τAMPA) + (input/lif.dt)
end

function INMDA(lif::LIF{VT}, n::Integer, vars::Matrix{VT})::Real where {VT <: Real}
    v = voltage(lif, n)
    return -gNMDA(pyramidalneurons[n]) * (v - Ve) * sum(weights(lif.graph)[excitors(lif, n), n] .* view(vars, :, 1)) /
            (1 + Mg * exp(-0.062*v) / 3.57)
end
function xNMDA(lif::LIF{VT}, n::Integer, x::AbstractVector{VT})::Vector{VT} where {VT <: Real}
    input = spikes(lif, excitors(lif, n); t=synt)
    return @. -(x/τNMDAr) + (input/lif.dt)
end
function sNMDA(lif::LIF{VT}, s::AbstractVector{VT}, x::AbstractVector{VT})::Vector{VT} where {VT <: Real}
    return @. -(s/τNMDAd) + α*x*(1 - s)
end
@views function varsNMDA(lif::LIF{VT}, n::Integer, vars::Matrix{VT})::Matrix{VT} where {VT <: Real}
    dvars = similar(vars)
    dvars[:, 2] = xNMDA(lif, n, vars[:, 2])
    dvars[:, 1] = sNMDA(lif, vars[:, 1], dvars[:, 2])
    return dvars
end

function IGABA(lif::LIF{VT}, n::Integer, s::AbstractVector{VT})::Real where {VT <: Real}
    return -gGABA(pyramidalneurons[n]) * (voltage(lif, n) - Vi) * sum(s)
end
function sGABA(lif::LIF{VT}, n::Integer, s::AbstractVector{VT})::Vector{VT} where {VT <: Real}
    input = spikes(lif, inhibitors(lif, n); t=synt)
    return @. -(s/τGABA) + (input/lif.dt)
end


function singleneurontest(;pyramidal=true, x=x=0:-1e-13:-2e-11, iterations=10000)
    rates = Vector{Float64}(undef, length(x))
    # Create a network of a single neuron
    g = SimpleWeightedDiGraph(1)
    add_edge!(g, 1, 1, 1.0)
    for i in eachindex(x)
        lif = LIF{Float64, Int}(g; m=1, dt=dt, Vrest=[Vrest],
                                Vθ=[Vθ], Vreset=[Vreset],
                                τ=[τ(pyramidal)], R=[Rm(pyramidal)],
                                τref=[τref(pyramidal)])
        I(lif, n, s) = x[i]
        s(lif, n, s) = zeros(size(s))
        add_channel!(lif, "input", I, s, [zeros(1)])
        step!(lif, iterations; transfer=Base.Fix2(thresh, Vθ))
        rates[i] = mean(spiketrain(lif, iterations)) / dt
    end
    display(scatter(x, rates, xflip=true, xlab="Input Current (V)", ylab="Firing Rate (Hz)"))
end

function synapsetest(;channel="AMPA", x=x=0:100, iterations=10000)
    gating = Vector{Float64}(undef, length(x))
    rates = Vector{Float64}(undef, length(x))
    pyramidal = channel!="GABA"
    I = if channel=="AMPA" IrecAMPA elseif channel=="NMDA" INMDA else IGABA end
    s = if channel=="AMPA" srecAMPA elseif channel=="NMDA" varsNMDA else sGABA end

    gate(lif) = vars(lif, channel, 2)

    # Create a network of two neurons
    g = SimpleWeightedDiGraph(2)
    add_edge!(g, 1, 2, (pyramidal ? 1.0 : -1.0))
    for i in eachindex(x)
        lif = LIF{Float64, Int}(g; m=1, dt=dt, Vrest=fill(Vrest, 2),
                                Vθ=fill(Vθ, 2), Vreset=fill(Vreset,2),
                                τ=fill(τ(pyramidal), 2), R=fill(Rm(pyramidal), 2),
                                τref=fill(τref(pyramidal), 2))
        add_channel!(lif, channel, I, s, channel=="NMDA" ? [ones(1,2), ones(1,2)] : [zeros(1), zeros(1)])
        input = rand(Poisson(x[i]*dt), 1, iterations) .> 0
        lif, V, G = step!(lif, input, voltages, gate; transfer=Base.Fix2(thresh, Vθ))
        V = reduce(hcat, V)
        G = reduce(vcat, G)
        #gating[i] = mean(G)
        #rates[i] = mean(spiketrain(lif, iterations)) / dt

        display(plot(V', xlab="Iteration", ylab="Voltage (V)"))
        display(plot(G, xlab="Iteration", ylab="Gating Variable"))
        display(rasterplot(spiketrain(lif, iterations)))
    end
    #display(scatter(x, gating, xlab="Input Rate (Hz)", ylab="Average Gating Variable"))
    #display(scatter(x, rates, xlab="Input Rate (Hz)", ylab="Output Firing Rate (Hz)"))
end



"""
    geninput(c::Real, length::Integer, extStart=1, extStop=0)::Matrix{Int}

Generate an `N`x`length` `Matrix` of background inputs given by `Poission(Vext)`,
and additional external inputs to `aneurons` and `bneurons` given by
`Poisson(Normal(μ0 + ρa*c))` and `Poisson(Normal(μ0 - ρb*c))` respectively.
"""
function geninput(;c::Real, length::Integer, extStart=1, extStop=length)::Matrix{Int}
    μa = μ0 + ρa * c
    μb = μ0 - ρb * c
    sa = rand(Normal(μa, σ))  # input rates (Hz)
    sb = rand(Normal(μb, σ))

    # initialize with background noise at rate Vext
    I = rand(Poisson(Vext*dt), N, length)

    # add external inputs for iterations
    for t in extStart:extStop
        I[1:nA, t] += rand(Poisson(dt*sa), nA)
        I[(nA+1):(nA+nB), t] += rand(Poisson(max(dt*sb, 0.0)), nB)

        # resample the rates every 50 ms.
        if t % (50e-3 / dt) == 0
            sa = rand(Normal(μa, σ))
            sb = rand(Normal(μb, σ))
        end
    end
    return I
end

function run(;c::Real=0.0, iterations::Integer=10, window::Integer=100)
    GC.gc()
    I = geninput(;c=c, length=iterations)

    # create the LIF and add all of the channel types
    lif = LIF{Float64, Int}(graph; m=1, dt=dt, Vrest=fill(Vrest, nv(graph)),
                            Vθ=fill(Vθ, nv(graph)),
                            Vreset=fill(Vreset, nv(graph)),
                            τ=τ.(pyramidalneurons),
                            R=Rm.(pyramidalneurons),
                            τref=τref.(pyramidalneurons))

    # Add excitatory and inhibitory channels
    add_channel!(lif, "extAMPA", IextAMPA, sextAMPA, [rand(1)/1000 for n in neurons(lif)])
    add_channel!(lif, "recAMPA", IrecAMPA, srecAMPA, [rand(length(excitors(lif, n)))/1000 for n in neurons(lif)])
    add_channel!(lif, "NMDA", INMDA, varsNMDA, [rand(length(excitors(lif, n)), 2)/1000 for n in neurons(lif)])
    add_channel!(lif, "GABA", IGABA, sGABA, [rand(length(inhibitors(lif, n)))/1000 for n in neurons(lif)])

    lif, V = step!(lif, I, voltages; transfer=Base.Fix2(thresh, Vθ))
    V = reduce(hcat, V)
    S = spiketrain(lif, iterations)

    #println(lif.S)
    #println(S)

    display(plot(rasterplot(S[aneurons,:]; title="A Neurons", timestep=dt),
                 rateplot(S[aneurons,:]; timestep=dt, window=window),
                 voltageplot(V[aneurons,:]; timestep=dt, ylim=[Vrest, Vθ+0.0025]);
                 xlim=[0.0, iterations*dt], layout=(3,1)))
    display(plot(rasterplot(S[bneurons,:]; title="B Neurons", timestep=dt),
                 rateplot(S[bneurons,:]; timestep=dt, window=window),
                 voltageplot(V[bneurons,:]; timestep=dt, ylim=[Vrest, Vθ+0.0025]);
                 xlim=[0.0, iterations*dt], layout=(3,1)))
    display(plot(rasterplot(S[nsneurons,:]; title="Non-selective Neurons", timestep=dt),
                 rateplot(S[nsneurons,:]; timestep=dt, window=window),
                 voltageplot(V[nsneurons,:]; timestep=dt, ylim=[Vrest, Vθ+0.0025]);
                 xlim=[0.0, iterations*dt], layout=(3,1)))
    display(plot(rasterplot(S[interneurons,:]; title="Interneurons", timestep=dt),
                 rateplot(S[interneurons,:]; timestep=dt, window=window),
                 voltageplot(V[interneurons,:]; timestep=dt, ylim=[Vrest, Vθ+0.0025]);
                 xlim=[0.0, iterations*dt], layout=(3,1)))
    #display(rArBplot(S[aneurons,:], S[bneurons,:], window=window, timestep=dt,
    #                 xlim=[0.0, 50], ylim=[0.0, 50], aspect_ratio=:equal,
    #                 xlab="Firing Rate r_A (Hz)", ylab="Firing Rate r_B (Hz)", leg=false))
    #display(rArBgif(S[aneurons,:], S[bneurons,:], window=window, timestep=dt, tail=100, fps=10, skip=50,
    #                  xlim=[0.0, 1/τref(true)], ylim=[0.0, 1/τref(true)], aspect_ratio=:equal,
    #                  xlab="Firing Rate r_A (Hz)", ylab="Firing Rate r_B (Hz)", leg=false))
    GC.gc()
end

end
