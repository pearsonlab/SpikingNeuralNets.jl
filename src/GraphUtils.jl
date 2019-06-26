# Requires LightGraphs and SimpleWeightedGraphs
module GraphUtils
using LightGraphs, SimpleWeightedGraphs

export CompleteBipartiteDiGraph, CompleteMultipartiteDiGraph, ComplexGraph, add_inputs!

"""
    CompleteBipartiteDiGraph(n1, n2)

Create a directed [complete bipartite graph](https://en.wikipedia.org/wiki/Complete_bipartite_graph)
with `n1 + n2` vertices.
"""
function CompleteBipartiteDiGraph(n1::T, n2::T) where {T <: Integer}
    (n1 < 0 || n2 < 0) && return SimpleDiGraph{T}(0)
    Tw = widen(T)
    n = T(Tw(n1) + Tw(n2))  # checks if T is large enough for n1 + n2

    adjmat = falses(n, n)
    range1 = 1:n1
    range2 = (n1 + 1):n
    @inbounds for u in range1, v in range2
        adjmat[u, v] = true
    end
    return SimpleDiGraph{T}(adjmat)
end

"""
    CompleteMultipartiteDiGraph(partitions)

Create a directed [complete multipartite graph](https://en.wikipedia.org/wiki/Multipartite_graph)
with `sum(partitions)` vertices. A partition with `0` vertices is skipped.
### Implementation Notes
Preserves the eltype of the partitions vector. Will error if the required number of vertices
exceeds the eltype.
"""
function CompleteMultipartiteDiGraph(partitions::AbstractVector{T}) where {T<:Integer}
    any(x -> x < 0, partitions) && return SimpleDiGraph{T}(0)
    length(partitions) == 1 && return SimpleDiGraph{T}(partitions[1])
    length(partitions) == 2 && return CompleteBipartiteDiGraph(partitions[1], partitions[2])

    n = sum(partitions)
    adjmat = falses(n, n)
    cur = 1
    for i in 1:length(partitions)
        curp = partitions[i]
        nextp = i < length(partitions) ? partitions[i+1] : 0
        currange = cur:(cur+curp-1) # all vertices in the current partition
        upperrange = (cur+curp):(cur+curp+nextp-1)   # all vertices higher than the current partition

        @inbounds for u in currange, v in upperrange
            adjmat[u,v] = true
        end
        cur += curp
    end

    return SimpleDiGraph{T}(adjmat)
end

"""
    ComplexGraph(metagraph, layers)

Construct a complex graph from a metagraph, a graph where each vertex vi represents a set of layers[vi] vertices in the complex graph which have identical edges. If selfloops is true, a self-loop in the metagraph will produce self-loops in all of the vertices in the resulting set. If completeloops is true, a self-loop in the metagraph will produce full connectivity between all vertices in the resulting set.
"""
function ComplexGraph(metagraph::G, layers::AbstractVector{<:Integer}; selfloops=true, completeloops=true) where {G <: AbstractGraph}
    nlayers = length(layers)
    if nlayers != nv(metagraph)
        error("$layers must have the same length as the number of vertices in $metagraph.")
    end

    # construct ranges for each layer
    cur = 1
    layerrange = Vector{Vector{Int}}(undef, nlayers)
    for i in 1:nlayers
        layerrange[i] = cur:(cur+layers[i] - 1)
        cur += layers[i]
    end

    # fill in the adjacency matrix
    n = sum(layers)
    adjmat = falses(n, n)
    for i in 1:nlayers
        for j in outneighbors(metagraph, i)
            for u in layerrange[i]
                for v in layerrange[j]
                    if i != j || (u == v && selfloops) || (u != v && completeloops)
                        adjmat[u,v] = true
                    end
                end
            end
        end

    end

    return G(adjmat)
end

"""
    ComplexGraph(metagraph, layers)

Construct a complex graph from a metagraph, a graph where each vertex vi represents a set of layers[vi] vertices in the complex graph which have identical edges. If selfloops is true, a self-loop in the metagraph will produce self-loops in all of the vertices in the resulting set. If completeloops is true, a self-loop in the metagraph will produce full connectivity between all vertices in the resulting set.
"""
function ComplexGraph(metagraph::G, layers::AbstractVector{<:Integer}; selfloops=true, completeloops=true) where G<:AbstractSimpleWeightedGraph{T, U} where {T,U}
    nlayers = length(layers)
    if nlayers != nv(metagraph)
        error("$layers must have the same length as the number of vertices in $metagraph.")
    end

    # construct ranges for each layer
    cur = 1
    layerrange = Vector{Vector{Int}}(undef, nlayers)
    for i in 1:nlayers
        layerrange[i] = cur:(cur+layers[i] - 1)
        cur += layers[i]
    end

    # fill in the adjacency matrix
    n = sum(layers)
    w = weights(metagraph)
    adjmat = zeros(U, n, n)
    for i in 1:nlayers
        for j in outneighbors(metagraph, i)
            for u in layerrange[i]
                for v in layerrange[j]
                    if i != j || (u == v && selfloops) || (u != v && completeloops)
                        adjmat[u,v] = w[i, j]
                    end
                end
            end
        end
    end

    return G(adjmat)
end

"""
    add_inputs!(g::AbstractGraph{T}, vertices::AbstractVector{<:T}=vertices(g))

For every vertex `vi` in `vertices` a new vertex `vinput` having a single edge `vinput->vi`.
Return the indices of all new input vertices.
"""
function add_inputs!(g::AbstractGraph{T}, vertices::AbstractVector{<:T}=vertices(g)) where {T<:Integer}
    vstart = size(g)+1
    vstop = vstart + length(vertices)
    add_vertices!(g, length(vertices))
    for (vi, vinput) in zip(vertices, vstart:vstop)
        add_edge!(g, vinput, vi)
    end
    return vstart:vstop
end

"""
    add_inputs!(g::AbstractSimpleWeightedGraph{T,U}, vertices::AbstractVector{<:T}=vertices(g), w::AbstractVector{<:U}=ones(U, length(vertices)))

For every vertex `vi` in `vertices` a new vertex `vinput` having a single edge `vinput->vi` with weight `weights[i]`.
Return the indices of all new input vertices.
"""
function add_inputs!(g::AbstractSimpleWeightedGraph{T,U}, vertices::AbstractVector{<:T}=vertices(g), weights::AbstractVector{<:U}=ones(U, length(vertices))) where {T<:Integer, U<:Real}
    vstart = nv(g)+1
    vstop = vstart + length(vertices)
    add_vertices!(g, length(vertices))
    for (vi, vinput, w) in zip(vertices, vstart:vstop, weights)
        add_edge!(g, vinput, vi, w)
    end
    return vstart:vstop
end

end
