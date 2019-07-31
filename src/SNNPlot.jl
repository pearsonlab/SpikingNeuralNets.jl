module SNNPlot
using Statistics, Plots, GraphPlot

export rasterplot, voltageplot, smooth, rateplot, rArBplot, rArBgif

"""
    rasterplot(S::AbstractMatrix{<:Real}; timestep=1, kwargs...)

Generate a raster plot of neuronal spiking given the matrix `S[neuron, time]`
of neuronal spike trains, where each time-step in `C` corresponds
to `timestep=1` seconds.
"""
function rasterplot(S::AbstractMatrix{Bool}; timestep=1, kwargs...)
    spikes = findall(S)
    N = getindex.(spikes, 1)
    T = getindex.(spikes, 2) * timestep
    return scatter(T, N; leg=false, marker=:vline, ylab="Neuron", kwargs...)
end

"""
    voltageplot(V::AbstractArray{<:Real}; timestep=1, kwargs...)

Generate a plot of neuronal voltage over time from the matrix V[neuron, time],
where each column of V is separated by `timestep=1` seconds.
"""
function voltageplot(V::AbstractArray{<:Real}; timestep=1, kwargs...)
    return plot(range(0, length=size(V, 2), step=timestep), V'; ylab="Voltage (V)", leg=false, kwargs...)
end


"""
    smooth(S::AbstractArray{<:Real}; window=0, dt=1, timestep=1, average=true)

    Calculate the average of S for all rows (if `average==true`) or each
individual row (if `average==false`) over a window of size `window`
which slides in strides of `dt` columns, where each column corresponds
to `timestep` seconds. Multiplies the values by `tohz`, which by default converts
boolean spikes into firing rates.  Return the tuple of `smoothed[index, neuron]`
and `time[index]` where `index` is the number of strides, `smoothed`
contains the smoothed value at stride `index`, and `time` contains the
continuous timepoint of that stride in seconds.
"""
function smooth(S::AbstractArray{<:Real}; window=0, dt=1, timestep=1, average=true, tohz=1/(timestep*dt))
    if ndims(S) == 1
        S = reshape(S, 1, length(S))  # Reshape to 1 Neuron x length(S) timesteps
        average = false
    end

    N, T = size(S)
    bins = floor(Int, (T - window) / dt)
    if bins <= 0
        return [], []
    end

    R = Matrix{Float64}(undef, bins, average ? 1 : N)
    t = range(window*timestep/2, length=bins, step=timestep*dt)

    @inbounds for t in 0:bins-1
        if average
            R[t+1, 1] = mean(S[:,(dt*t + 1):(dt*t + window + 1)]) * tohz
        else
            for n in 1:N
                R[t+1, n] = mean(S[n, (dt*t + 1):(dt*t + window + 1)]) * tohz
            end
        end
    end

    return R, t
end

"""
    rateplot(S::AbstractArray{<:Real}; window=0, dt=1, timestep=1, average=true, tohz=1/(timestep*dt), kwargs...)

Generate a population rate plot of neuronal spiking given the matrix
`S[neuron, time]` of neuronal spike trains. Rates are averaged
over a window of `window=1` time-steps which slides in strides of
 `dt=1` time-steps, where a single time-step corresponds to `timestep=1` seconds.
 If `average` is `true`, then average over all neurons in each window. Otherwise,
 plot a separate line for each neuron.
"""
function rateplot(S::AbstractArray{<:Real}; window=0, dt=1, timestep=1, average=true, tohz=1/(timestep*dt), kwargs...)
    R, t = smooth(S; window=window, dt=dt, timestep=timestep, average=average, tohz=tohz)
    return plot(t, R; ylab="Firing Rate (Hz)", leg=false, kwargs...)
end

"""
    rArBplot(SA::AbstractArray{<:Real}, SB::AbstractArray{<:Real};
             window=0, dt=1, timestep=1, average=true, tohz=1/(timestep*dt), kwargs...)

Generate a plot of the rate of two neural populations over time, where
each axis corresponds to the rate of a population, and the line is a traversal
through this rate space over time.
"""
function rArBplot(SA::AbstractArray{<:Real}, SB::AbstractArray{<:Real}; window=0, dt=1, timestep=1, average=true, tohz=1/(timestep*dt), kwargs...)
    rA, _ = smooth(SA; window=window, dt=dt, timestep=timestep, average=average, tohz=tohz)
    rB, _ = smooth(SB; window=window, dt=dt, timestep=timestep, average=average, tohz=tohz)
    return plot(rA, rB; leg=false, kwargs...)
end

"""
    rArBgif(SA::AbstractArray{<:Real}, SB::AbstractArray{<:Real};
            window=0, dt=1, timestep=1, average=true, tohz=1/(timestep*dt), fps=1, tail=0, skip=1, kwargs...)

Generate a gif of the rate of two neural populations over time, where
each axis corresponds to the rate of a population, and the line is a traversal
through this rate space over time.
"""
function rArBgif(SA::AbstractArray{<:Real}, SB::AbstractArray{<:Real};
                 window=0, dt=1, timestep=1, average=true, tohz=1/(timestep*dt), fps=1, tail=0, skip=1, filename=nothing, kwargs...)
    rA, _ = smooth(SA; window=window, dt=dt, timestep=timestep, average=average, tohz=tohz)
    rB, _ = smooth(SB; window=window, dt=dt, timestep=timestep, average=average, tohz=tohz)

    anim = @animate for i in 1:size(rA, 1)
        scatter(rA[i,:], rB[i,:]; kwargs...)
        if !iszero(tail)
            range = max(1, i-tail):i
            plot!(rA[range,:], rB[range,:]; leg=false, kwargs...)
        end
    end every skip
    if isnothing(filename)
        return gif(anim; fps=fps)
    end
    return gif(anim, filename; fps=fps)
end

function snnplot(snn; kwargs...)
    gplot(snn.graph; kwargs...)
end

end
