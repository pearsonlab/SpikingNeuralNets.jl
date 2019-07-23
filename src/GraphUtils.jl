# Requires LightGraphs and SimpleWeightedGraphs
module GraphUtils
using LightGraphs, SimpleWeightedGraphs
using Base.Cartesian
export CompleteBipartiteDiGraph, CompleteMultipartiteDiGraph, ComplexGraph, mesh_graph, grid_layout, add_inputs!

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
    _neighbors(ind::CartesianIndex{N}; forward::Vector{<:Integer}=ones(Int, N), backward::Vector{<:Integer}=ones(Int, N),
               self::Bool=true, diagonals::Bool=true, horizontals::Bool=true)::Vector{CartesianIndex{N}} where N

For any `N`-dimensional index `ind`, generate a vector of all indices neighboring
`ind`, including all indices within `backward` indices behind ind to `forward` in
front of ind. `ind` is included in the return value iff `self` is true. Indices
diagonal to `ind` are included iff `diagonals` is true, and indices horizontal to
`ind` are included iff `horizontals` is true.
"""
function _neighbors(ind::CartesianIndex{N}; forward::Vector{<:Integer}=ones(Int, N), backward::Vector{<:Integer}=ones(Int, N), self::Bool=true, diagonals::Bool=true, horizontals::Bool=true)::Vector{CartesianIndex{N}} where N
    if length(forward) != N || length(backward) != N
        error("Forward and backward strides must have length $N.")
    end

    return [ind+i for i in -CartesianIndex(backward...):CartesianIndex(forward...) if (self && i == zero(i)) || (horizontals && any(iszero.(Tuple(i)))&& any(.!iszero.(Tuple(i)))) || (diagonals && !any(iszero.(Tuple(i))))]
end

"""
    _mesh(G::Type{<:AbstractGraph}, dims::Tuple{Vararg{<:Integer, N}}, args...; kwargs...) where N

Generate an `N` dimensional mesh graph of type `G` with size `dims` in each dimension.
Optional arguments are passed to LightGraphs.add_edge!, and keyword arguments are
passed to `SpikingNeuralNets.GraphUtils_neighbors`.
"""
function _mesh(G::Type{<:AbstractGraph}, dims::Tuple{Vararg{<:Integer, N}}, args...; kwargs...) where N
    if iszero(N)
        return G()
    end

    g = G(reduce(*, dims))
    verts = reshape(vertices(g), dims)    # store the linear index of each vertex
    inds = CartesianIndices(verts)
    Ifirst, Ilast = Tuple(first(inds)), Tuple(last(inds))
    for u in inds
        for v in _neighbors(u; self=false, kwargs...)
            if all(Tuple(v) .>= Ifirst) && all(Tuple(v) .<= Ilast)
                add_edge!(g, verts[u], verts[v], args...)
            end
        end
    end

    return g
end

"""
    mesh_graph(dims::Tuple{Vararg{<:Integer, N}}; directed=true, weight=nothing, kwargs...) where N

Generate an `N`-dimensional mesh graph of size `dims` in each dimension.
Keyword arguments:
  `directed`: if true (default), return a directed graph. Else return an undirected graph.
  `weight`: if nothing (default), return an `AbstractSimpleGraph`. Else return an
            `AbstractSimpleWeightedGraph` where each edge has weight `weight`.
  `backward`: the number of vertices (in each dimension) behind each vertex `v`
              with incoming edges from `v`. By default, `ones(Int, N)`.
  `forward`: the number of vertices (in each dimension) in front of each vertex
             `v` with incoming edges from `v`. By default, `ones(Int, N)`.
  `horizontals`: if true (default), vertices are connected horizontally (where vertices
                 are aligned on any single dimension). Else these edges are omitted.
  `diagonals`: if true (default), vertices are connected diagonally, where vertices are not
               aligned on any dimension. Else these edges are omitted.
  `self`: if true (default), edges `(v, v)` are included for all vertices `v`.
          Else these edges are omitted.
"""
function mesh_graph(dims::Tuple{Vararg{<:Integer, N}}; directed=true, weight=nothing, kwargs...) where N
    if isnothing(weight)
        if directed
            return _mesh(SimpleDiGraph, dims; kwargs...)
        else
            return _mesh(SimpleGraph, dims; kwargs...)
        end
    else
        if directed
            return _mesh(SimpleWeightedDiGraph, dims, weight; kwargs...)
        else
            return _mesh(SimpleWeightedGraph, dims, weight; kwargs...)
        end
    end
end

"""
    mesh_graph(dims::Vararg{<:Integer, N}; directed=true, weight=nothing, kwargs...) where N

Generate an `N`-dimensional mesh graph of size `dims` in each dimension.
Keyword arguments:
  `directed`: if true (default), return a directed graph. Else return an undirected graph.
  `weight`: if nothing (default), return an `AbstractSimpleGraph`. Else return an
            `AbstractSimpleWeightedGraph` where each edge has weight `weight`.
  `backward`: the number of vertices (in each dimension) behind each vertex `v`
              with incoming edges from `v`. By default, `ones(Int, N)`.
  `forward`: the number of vertices (in each dimension) in front of each vertex
             `v` with incoming edges from `v`. By default, `ones(Int, N)`.
  `horizontals`: if true (default), vertices are connected horizontally (where vertices
                 are aligned on any single dimension). Else these edges are omitted.
  `diagonals`: if true (default), vertices are connected diagonally, where vertices are not
               aligned on any dimension. Else these edges are omitted.
  `self`: if true (default), edges `(v, v)` are included for all vertices `v`.
          Else these edges are omitted.
"""
function mesh_graph(dims::Vararg{<:Integer, N}; kwargs...) where N
    return mesh_graph(dims; kwargs...)
end

"""
    grid_layout(g::AbstractGraph; reverse_x::Bool=false, reverse_y::Bool=false, columnwise=false)

Return a tuple of x locations and y locations for a graph, laying out
each vertex in a single row (if `columnwise == false`, default) or column
(if `columnwise == true`). The row/column order can be reversed using the boolean
flags `reverse_x` and `reverse_y`, respectively.
"""
function grid_layout(g::AbstractGraph; reverse_x::Bool=false, reverse_y::Bool=false, columnwise=false)
    if columnwise
        locs_x, locs_y = zeros(nv(g)), vertices(g)
    else
        locs_x, locs_y = vertices(g), zeros(nv(g))
    end

    if reverse_x
        reverse!(locs_x)
    end
    if reverse_y
        reverse!(locs_y)
    end
    return locs_x, locs_y
end

"""
    grid_layout(g::AbstractGraph, dims::Tuple{<:Integer, <:Integer};
                reverse_x::Bool=false, reverse_y::Bool=false, columnwise::Bool=true)

Return a tuple of x locations and y locations for all vertices in a graph, laying out
each vertex an a grid of size `dims` in row-major order (if `columnwise == false`)
or column-major order (if `columnwise == true`, default). The row/column order can
be reversed using the boolean flags `reverse_x` and `reverse_y`, respectively.
"""
function grid_layout(g::AbstractGraph, dims::Tuple{<:Integer, <:Integer};
                     reverse_x::Bool=false, reverse_y::Bool=false, columnwise::Bool=true)
    if reduce(*, dims) != nv(g)
        error("dims ($(dims)) does not match the number of vertices in g ($(nv(g))).")
    end

    inds = vec([(Float64(x), Float64(y)) for x in Base.OneTo(dims[1]), y in Base.OneTo(dims[2])])
    if columnwise
        locs_x = getindex.(inds, 2)
        locs_y = getindex.(inds, 1)
    else
        locs_x = getindex.(inds, 1)
        locs_y = getindex.(inds, 2)
    end

    if reverse_x
        reverse!(locs_x)
    end
    if reverse_y
        reverse!(locs_y)
    end

    return locs_x, locs_y
end

"""
    add_inputs!(g::AbstractGraph{T}, vertices::AbstractVector{<:T}=vertices(g))

For every vertex `vi` in `vertices` a new vertex `vinput` having a single edge `vinput->vi`.
Return the indices of all new input vertices.
"""
function add_inputs!(g::AbstractGraph{T}, vertices::AbstractVector{<:T}=vertices(g)) where {T<:Integer}
    vstart = nv(g)+1
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
