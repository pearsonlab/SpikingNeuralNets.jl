module ReducedNetwork
using DifferentialEquations, LightGraphs, SimpleWeightedGraphs,
    Plots, GraphPlot, Compose, Cairo, Fontconfig, Printf # for plotting
using SpikingNeuralNets.GraphUtils: mesh_graph

# Effective transfer function
x(S1, S2, Inet, I, Inoise, t, p) = p.JNex*S1 + p.JNin*S2 + Inet + p.Iext + I*(t >= 0.0) + Inoise
function H(x1, x2, p)
    v = p.a*x1 - p.fA(x2) - p.b
    return v / (1 - exp(p.d * v))
end

"""
    Inetwork(S::AbstractVector{T}, g::AbstractGraph)::Vector{T} where {T <: Number}

For each vertex in `g` calculate a weighted average of S from all inneighbors of `v`.
"""
function Inetwork(S::AbstractVector{T}, g::AbstractGraph)::Vector{T} where {T <: Number}
    I = zeros(T, size(S))
    @inbounds for v in vertices(g)
        in = inneighbors(g, v)
        if !iszero(length(in))
            I[v] = sum(weights(g)[in,v] .* S[in])
        end
    end
    return I
end

"""
    ReducedModel(du, u, p, t, W)

Calculate the change mean gating variables `du` given the current gating
variables `u`, named parameters `p`, time `t`, and noise currents `W`.
"""
function ReducedModel(du, u, p, t, W)
    Sa, Sb = eachcol(u)
    Inoisea, Inoiseb = eachcol(W)

    Ineta = Inetwork(p.Jex*Sa + p.Jin*Sb, p.g)          # network inputs
    Inetb = Inetwork(p.Jex*Sb + p.Jin*Sa, p.g)
    xa = x.(Sa, Sb, Ineta, p.Ia, Inoisea, t, Ref(p))    # intermediate variables
    xb = x.(Sb, Sa, Inetb, p.Ib, Inoiseb, t, Ref(p))
    du[:,1] .= .-(Sa/p.τs) .+ (1 .- Sa) .* p.γ .* H.(xa, xb, Ref(p))
    du[:,2] .= .-(Sb/p.τs) .+ (1 .- Sb) .* p.γ .* H.(xb, xa, Ref(p))
end

"""
    run(g::AbstractGraph; c=fill(0, nv(g)), μ0=fill(30.0, nv(g)), S0=zeros(nv(g), 2),
        Inoise0=zeros(nv(g), 2), tstart=-0.5, tstop=1.0, reltol=1e-15,
        abstol=1e-15, dt=0.1e-3, Jex=1.0, Jin=0.0, Jext=0.2243e-3, JNex=0.1561,
        JNin=-0.0264, JAex=9.9026e-4, JAin=-6.5177e-5, τs=100e-3, τAMPA=2e-3,
        γ=0.641, Iext=0.35, σnoise=0.007, color1=:red, color2=:blue, kwargs...)

Run a reduced 2-variable model of neural populations during a random dot motion task.
Return a tuple (g, t, Sa, Sb, Inoisea, Inoiseb, Ineta, Inetb, xa, xb, ra, rb).

`g`: a graph of neural populations, where each vertex represents a population of
     neurons, and each edge represents a connection between neural populations.
     Weighted graphs (e.g. `SimpleWeightedDiGraph`) a.
`c`: the coherence of external input to each node.
`μ0`: the firing rate (in Hz) of external input to each node,
`S0`: the initial average NMDA gating variables for each node,
`Inoise0`: are the initial noise currents for each node,
`tstart`: the starting time of the simulation (negative times having no stimulus)
`tstop`: is the stopping time of the simulation
`dt`: is the timestep of the simulation.
`Jex`: the weight of recurrent input from excitatory sub-populations
`Jin`: the weight of recurrent input from inhibitory sub-populations
`Jext`: the weight of external inputs
`JNex`: the excitatory (> 0) NMDA weight
`JNin`: the inhibitory (< 0) NMDA weight
`JAex`: the excitatory (> 0) AMPA weight
`JAin`: the excitatory (> 0) AMPA weight
`τs`: the time constant of NMDA channels
`τAMPA`: the time constant of AMPA channels
`γ`: a factor applied to firing rate to update gating variables
`Iext`: baseline external input current (nV)
`σnoise`: standard deviation of noise current (nV).
`color1`: the line color of the correct direction in the output plot
`color2`: the line color of the incorrect direction in the output plot
Additional keyword arguments are passed to DifferentialEquations.solve.
"""
function run(g::AbstractGraph; c=fill(0, nv(g)), μ0=fill(30.0, nv(g)), S0=zeros(nv(g), 2),
             Inoise0=zeros(nv(g), 2), tstart=-0.5, tstop=1.0, reltol=1e-15,
             abstol=1e-15, dt=0.1e-3, Jex=1.0, Jin=0.0, Jext=0.2243e-3, JNex=0.1561,
             JNin=-0.0264, JAex=9.9026e-4, JAin=-6.5177e-5, τs=100e-3, τAMPA=2e-3,
             γ=0.641, Iext=0.35, σnoise=0.007, color1=:red, color2=:blue, kwargs...)
    # set up parameters
    Ia = @. Jext*μ0*(1 + c/100.0)
    Ib = @. Jext*μ0*(1 - c/100.0)
    a = 239400JAex + 270
    b = 97000JAex + 108
    d = 30JAex - 0.154
    fA(x) = JAin*(276x - 106) * (x >= 0.4)
    params = (g=g, a=a, b=b, c=c, d=d, fA=fA, μ0=μ0, Ia=Ia, Ib=Ib, Jex=Jex, Jin=Jin,
              Jext=Jext, JNex=JNex, JNin=JNin, JAex=JAex, JAin=JAin,
              τs=τs, τAMPA=τAMPA, γ=γ, Iext=Iext)

    # noise process: τAMPA dI(t)/dt = -I(t) + W(t)*σnoise*√τAMPA
    W = OrnsteinUhlenbeckProcess(τAMPA, 0.0, σnoise*√τAMPA, tstart, Inoise0)
    prob = RODEProblem(ReducedModel, S0, (tstart, tstop), params, noise=W)
    println(prob)
    sol = solve(prob, reltol=reltol, abstol=abstol, dt=dt, kwargs...)

    # Reconstruct the variables
    Sa, Sb = sol[:,1,:], sol[:,2,:]
    Inoisea, Inoiseb = sol.W[:,1,1:length(sol.t)], sol.W[:,2,1:length(sol.t)]
    Ineta = mapslices((s)->Inetwork(s, g), Jex*Sa + Jin*Sb; dims=1)
    Inetb = mapslices((s)->Inetwork(s, g), Jex*Sb + Jin*Sa; dims=1)

    # reconstruct the firing rates
    xa, xb = similar(Sa), similar(Sb)
    @views for v in vertices(g)
        xa[v,:] .= x.(Sa[v,:], Sb[v,:], Ineta[v,:], Ia[v], Inoisea[v,:], sol.t, Ref(params))
        xb[v,:] .= x.(Sb[v,:], Sa[v,:], Inetb[v,:], Ib[v], Inoiseb[v,:], sol.t, Ref(params))
    end
    ra = H.(xa, xb, Ref(params))
    rb = H.(xb, xa, Ref(params))

    display(plot(plot(sol.t, [Sa' Sb'], title="Gating Variables"),
                 plot(sol.t, [ra' rb'], title="Firing Rates (Hz)"),
                 leg=false, linecolor=[fill(:red, 1, nv(g)) fill(:blue, 1, nv(g))], layout=(2, 1)))
    return (g, sol.t, Sa, Sb, Inoisea, Inoiseb, Ineta, Inetb, xa, xb, ra, rb)
end


"""
    run(g::AbstractGraph, inputs; c=0.0, μ0=30.0, kwargs...)

Run a reduced 2-variable model of neural populations during a random dot motion task,
where `inputs` contains all of the vertices of `g` receiving external input, and
`c` and `μ0` are the coherence and strength of external input. Return a tuple
(g, t, Sa, Sb, Inoisea, Inoiseb, Ineta, Inetb, xa, xb, ra, rb).
"""
function run(g, inputs; c=0.0, μ0=30.0, kwargs...)
    C = fill(c, nv(g))
    μ = zeros(nv(g))
    μ[inputs] .= μ0
    return run(g; c=C, μ0=μ, kwargs...)
end

"""
    run(dims::Tuple{Vararg{<:Integer, N}}; inputs=ifelse(N==1, 1, 1:first(dims)), weight=0.015, c=0, μ0=30.0, kwargs...) where N

Run a reduced 2-variable model of a mesh_graph of neural populations during a
random dot motion task, where `dims` is the size of the mesh graph, `inputs`
contains all of the vertices of `g` receiving external input, and `c` and `μ0`
are the coherence strength of external input. Return a tuple
(g, t, Sa, Sb, Inoisea, Inoiseb, Ineta, Inetb, xa, xb, ra, rb).
"""
function run(dims::Tuple{Vararg{<:Integer, N}}; inputs=ifelse(N==1, 1, 1:first(dims)), weight=0.015, c=0, μ0=30.0, kwargs...) where N
    return run(mesh_graph(dims; self=false, weight=weight), inputs; kwargs...)
end

"""
    run(dims::Vararg{<:Integer, N}; kwargs...) where N

Run a reduced 2-variable model of a mesh_graph of neural populations during a
random dot motion task, where `dims` is the size of the mesh graph, `inputs`
contains all of the vertices of `g` receiving external input, and `c` and `μ0`
are the coherence strength of external input. Return a tuple
(g, t, Sa, Sb, Inoisea, Inoiseb, Ineta, Inetb, xa, xb, ra, rb).
"""
function run(dims::Vararg{<:Integer, N}; kwargs...) where N
    return run(dims; kwargs...)
end

"""
    RBplot(g::AbstractGraph, Sa::Vector{<:Real}, Sb::Vector{<:Real};
           color1=colorant"red", color2=colorant"yellow",
           locs_x=nothing, locs_y=nothing, kwargs...)

Draw a plot of the graph g, where each vertex `v` is colored by the values
`Sa[v]` and `Sb[v]` on a scale from `color1` to `color2`.
"""
function RBplot(g::AbstractGraph, Sa::Vector{<:Real}, Sb::Vector{<:Real};
                color1=colorant"red", color2=colorant"yellow",
                locs_x=nothing, locs_y=nothing, kwargs...)
    nodefillc = @. weighted_color_mean((Sa-Sb+1)/2, color1, color2)
    if locs_x != nothing && locs_y != nothing
        return gplot(DiGraph(weights(g)), locs_x, locs_y; nodefillc=nodefillc, kwargs...)
    end
    return gplot(DiGraph(weights(g)); nodefillc=nodefillc, kwargs...)
end

"""
    RBgif(g::AbstractGraph, Sa::Matrix{<:Real}, Sb::Matrix{<:Real};
          layout::Function=spring_layout,fps:Integer=100, skip::Integer=1,
          width::Integer=800, height::Integer=600, kwargs...)

Draw a gif of the graph g, where each vertex `v` is colored over time by the
values `Sa[v, t]` and `Sb[v, t]` on a scale from `color1` to `color2`.
"""
function RBgif(g::AbstractGraph, Sa::Matrix{<:Real}, Sb::Matrix{<:Real};
               layout::Function=spring_layout,fps::Integer=100, skip::Integer=1,
               width::Integer=800, height::Integer=600, kwargs...)
    # use the Animation object manually since frame() doesn't work on Compose objects
    anim = Animation()
    locs_x, locs_y = layout(g)
    for t in 1:skip:size(Sa, 2)
        fname = @sprintf("%06d.png", length(anim.frames)+1)
        draw(PNG(joinpath(anim.dir, fname), width, height),
             RBplot(g, Sa[:,t], Sb[:,t]; locs_x=locs_x, locs_y=locs_y, kwargs...))
        push!(anim.frames, fname)
    end
    println(anim)
    return gif(anim, fps=fps)
end

end
