import Distributions.sample
import Base.append!
using Flux: Zygote.@adjoint

function test(f, env, maxt=100)
    R = 0f0
    s = reset!(env)
    for _ in 1:maxt
        s, r, done, _ = step!(env, f(s))
        R += r
        done && break
    end
    return R
end

gather(y, a) = y[CartesianIndex.(a, axes(a, 1))]
gather(y, a::AbstractMatrix) = gather(y, vec(a))

@adjoint gather(y, a) = gather(y, a), 
    ȳ -> begin 
        o = zeros(eltype(ȳ), size(y))
        @inbounds for (i, j) in enumerate(a)
            o[j, i] = ȳ[i]
        end
        return (o, nothing)
    end

function softupdate!(target::T, model::T, τ=1f-2) where T
    for f in fieldnames(T)
        softupdate!(getfield(target, f), getfield(model, f), τ)
    end
end

function softupdate!(dst::A, src::A, τ=T(1f-2)) where {T, A<:AbstractArray{T}}
    dst .= τ .* src .+ (one(T) - τ) .* dst
end

mutable struct ReplayMemory{S<:Real, A<:Real, R<:Real}
    n::Int64
    maxsize::Int64
    batchsize::Int64
    s::Matrix{S}
    a::Matrix{A}
    r::Vector{R}
    d::Vector{Bool}
    function ReplayMemory{S, A, R}(obssize, actiondim, maxsize, batchsize) where {S<:Real, A<:Real, R<:Real}
        new(
            0, 
            maxsize, 
            batchsize,
            zeros(S, obssize, maxsize), 
            zeros(A, actiondim, maxsize), 
            zeros(R, maxsize), 
            zeros(Bool, maxsize)
        )
    end
end

function append!(memory::ReplayMemory, s, a, r, d)
    ind = memory.n % memory.maxsize + 1
    memory.n += 1
    memory.s[:, ind] .= s
    memory.a[:, ind] .= a
    memory.r[ind] = r
    memory.d[ind] = d
end

function sample(memory::ReplayMemory)
    inds = sample(1:(min(memory.n, memory.maxsize)-1), min(memory.n-1, memory.batchsize), replace=false)
    return (
        memory.s[:, inds],
        memory.a[:, inds],
        memory.r[inds],
        memory.s[:, inds .+ 1],
        memory.d[inds]
    )
end
