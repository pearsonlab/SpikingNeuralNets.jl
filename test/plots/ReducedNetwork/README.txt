test/plots/ReducedNetwork/README.txt
This file contains code used to generate plots contained in this folder.

--------------------------------------------------------------------------------------------
                                  Single Population
--------------------------------------------------------------------------------------------
include("test/ReducedNetwork.jl")
using LightGraphs, SpikingNeuralNets.SNNPlot
g = Graph(1)
g, t, sa, sb, Inoisea, Inoiseb, Ineta, Inetb, xa, xb, ra, rb = ReducedNetwork.run(g; c=[50.0], tstop=1.5)
plot!(dpi=1000)
savefig("test/plots/ReducedNetwork/single-firing-rate.png")
rArBgif(sa, sb; window=100, dt=1, timestep=1, tohz=1.0, aspect_ratio=:equal, xlim=[0.0, 0.75], ylim=[0.0, 0.75], fps=250, skip=100, tail=1000, filename="test/plots/ReducedNetwork/single.gif")

--------------------------------------------------------------------------------------------
                                      Bias Model
--------------------------------------------------------------------------------------------
include("test/ReducedNetwork.jl")
g, t, sa, sb, Inoisea, Inoiseb, Ineta, Inetb, xa, xb, ra, rb = ReducedNetwork.run((4, 4); c=50.0, tstop=3.0, weight=0.003, Jex=1.0, Jin=1.0)
plot!(dpi=1000)
savefig("test/plots/ReducedNetwork/bias-firing-rate.png")
ReducedNetwork.RBgif(g, sa, sb, layout=((g) -> grid_layout(g, (4, 4))), skip=100, fps=250, filename="test/plots/ReducedNetwork/bias-4x4-0.003.gif")


--------------------------------------------------------------------------------------------
                                    Inhibition Model
--------------------------------------------------------------------------------------------
include("test/ReducedNetwork.jl")
g, t, sa, sb, Inoisea, Inoiseb, Ineta, Inetb, xa, xb, ra, rb = ReducedNetwork.run((4, 4); c=50.0, tstop=3.0, weight=0.005, Jex=1.0, Jin=0.0)
plot!(dpi=1000)
savefig("test/plots/ReducedNetwork/inhibition-firing-rate.png")
ReducedNetwork.RBgif(g, sa, sb, layout=((g) -> grid_layout(g, (4, 4))), skip=100, fps=250, filename="test/plots/ReducedNetwork/inhibition-4x4-0.005.gif")


--------------------------------------------------------------------------------------------
                                      Phase plots
--------------------------------------------------------------------------------------------
include("test/ReducedNetwork.jl")
using LightGraphs, SimpleWeightedGraphs
Jex = -1:0.05:1
Jin = -1:0.05:1
gb = SimpleWeightedDiGraph(LightGraphs.SimpleGraphs.barbell_graph(1, 1), 0.01)
p = ReducedNetwork.phaseplot(gb, [1], c=50.0, Jex=Jex, Jin=Jin, iterations=100, tstop=3.0)
heatmap(Jex, Jin, p[:,:,1]', xlab="Wex", ylab="Win", dpi=1000)
savefig("test/plots/ReducedNetwork/phase-1.png")
heatmap(Jex, Jin, p[:,:,2]', xlab="Wex", ylab="Win", dpi=1000)
savefig("test/plots/ReducedNetwork/phase-2.png")
