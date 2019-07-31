module SNNTest
using LightGraphs
using SpikingNeuralNets, SpikingNeuralNets.LIFSNN, SpikingNeuralNets.SNNPlot

const iterations = 1000
const dt = 1.0 # timestep

I(snn, n, s) = 1.00
s(snn, n, s) = s

graph = Graph(1)
add_edge!(graph, 1, 1)
lif = LIF{Float64, Int}(graph; V=[[0.0]], dt=dt, Vrest=[0.0], Vθ=[1.5],
                        Vreset=[0.0], τref=[10.0], τ=[100.0], R=[1.75])
add_channel!(lif, "input", I, s, [Float64[]])
lif, V = step!(lif, iterations, voltages; threaded=false)
V = reduce(hcat, V) # convert V from a Vector{Vector} to a Matrix
S = spiketrain(lif, iterations)

# Make a raster plot of the output
display(rasterplot(S; xlim=[0, iterations], timestep=dt))
display(voltageplot(V; xlim=[0, iterations], timestep=dt))

end
