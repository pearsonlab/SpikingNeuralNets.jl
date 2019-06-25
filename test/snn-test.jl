module SNNTest
using LightGraphs, SimpleWeightedGraphs, PyPlot
using SpikingNeuralNets, SpikingNeuralNets.LIFSNN, SpikingNeuralNets.GraphUtils, SpikingNeuralNets.SNNPlot

const iterations = 1000
const inputs = 1
const outputs = 1
const dt = 1.0 #* 1e-3 # timestep

const Vrest = 0.0
const Vθ = 1.5
const Vreset = 0.0
const gL = 25.0 * 10e-9
const Cm = 0.5
const Rm = 1 / Cm
const τ = 100.0 #Cm / gL * dt
const τref = 5.0

# continuous input of 50mV
function I(snn, n, s)
    return 1.00
end

function s(snn, n, s)
    return zeros(length(s))
end

metagraph = SimpleWeightedDiGraph(3)
add_edge!(metagraph, 1, 2, 1.0)
add_edge!(metagraph, 1, 3, 1.0)
graph = ComplexGraph(metagraph, [inputs, outputs, outputs])

snn = LIF{Float64, Int}(graph; V=[[Vrest, Vrest, 1.0]], dt=dt,
                        Vrest=fill(Vrest, 3), Vθ=fill(Vθ, 3),
                        Vreset=fill(Vreset, 3), τref=fill(τref, 3),
                        τ=[τ, τ, τ], R=fill(Rm, 3))
add_channel!(snn, "input", I, s, fill([0.001], 3))

#input = [inputAct(i,t) for i in 1:inputs, t in 1:iterations]
snn, V = step!(snn, iterations, voltages; transfer=Base.Fix2(thresh, Vθ))
V = reduce(hcat, V) # convert V from a Vector{Vector} to a Matrix
S = spiketrain(snn, iterations)

# Make a raster plot of the output
display(rasterplot(S; xlim=[0, 1000], timestep=dt))
display(rateplot(S; xlim=[0, 1000], timestep=dt, average=false, window=250, dt=1))
display(voltageplot(V; xlim=[0, 1000], timestep=dt))

end
