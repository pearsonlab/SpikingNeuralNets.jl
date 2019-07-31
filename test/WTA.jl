module WTA
using SpikingNeuralNets, SpikingNeuralNets.SimpleSNN, SpikingNeuralNets.GraphUtils, SpikingNeuralNets.SNNPlot
using LightGraphs, SimpleWeightedGraphs

function wta_layout(g::AbstractGraph)
    N = round(Int, (nv(g) - 2) / 2) # number of inputs/outputs
    xloc = Float64[fill(1, N)..., fill(2, 2)..., fill(3, N)...]
    yloc = Float64[((1:N) .- N/2)..., -1, 1, ((1:N) .- N/2)...]
    return xloc, yloc
end

function two_inhibitor_snn(n::Integer, γ::Real)
    graph = SimpleWeightedDiGraph(2n+2)
    for i in 1:n
        yi = n+2+i
        add_edge!(graph, i, yi, 3γ)      # inputs -> outputs
        add_edge!(graph, yi, yi, 2γ)
        add_edge!(graph, yi, n+1, γ)
        add_edge!(graph, yi, n+2, γ)
        add_edge!(graph, n+1, yi, -γ)
        add_edge!(graph, n+2, yi, -γ)
    end

    bias = [fill(0, n)..., -γ/2, -3γ/2, fill(-3γ, n)...]
    return SNN{Float64, Int}(graph; bias=bias)
end

function run(input::Vector{T}, γ::Real, args...; iterations=100, kwargs...) where {T<:Integer}
    N = length(input)
    snn = two_inhibitor_snn(N, γ)
    step!(snn, iterations, args...; input=input, transfer=sigmoid, f=fire, kwargs...)
    display(rasterplot(spiketrain(snn, iterations)))
end
end
