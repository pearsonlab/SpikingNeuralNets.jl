export SummedInput

"""
    SummedInput{T, N, L, NI} <: AbstractInput{T, N, L}

An AbstractInput iterator which simply sums the results
of several component AbstractInput iterators.
"""
struct SummedInput{T, N, L, NI} <: AbstractInput{T, N, L}
    iters::NTuple{NI, AbstractInput{T}}
end

function SummedInput(iters::Vararg{AbstractInput{<:T, N}, NI}) where {T<:Real, N, NI}
    L = maximum(length.(iters))
    return SummedInput{T, N, L, NI}(iters)
end

function Base.iterate(iter::SummedInput{T, N, L, NI}, i=1) where {T, N, L, NI}
    if i > L
        return nothing
    end

    out = zeros(T, N)
    for itr in iter.iters
        arr, _ = iterate(itr, i)
        if arr != nothing
            out .+= arr
        end
    end
    return out, i+1
end
