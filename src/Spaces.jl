using Distributions: Uniform
import Distributions.sample

struct ActionSpace{A, N}
    actions::Array{A, N}
end

function ActionSpace(pyspace::PyObject)
    type = pyspace.__class__.__name__
    if type == "Discrete"
        return ActionSpace([pyspace.n])
    elseif type == "MultiDiscrete"
        return ActionSpace(pyspace.nvec)
    elseif type == "Box"
        return ActionSpace(map((l, h) -> (l, h), pyspace.low, pyspace.high))
    else
        error("Unknown python action_space type")
    end
end

Base.size(a::ActionSpace) = size(a.actions)
Base.length(a::ActionSpace) = length(a.actions)
Base.eltype(::Type{ActionSpace{A, N}}) where {A, N} = A
Base.eltype(::Type{ActionSpace{Tuple{A, A}, N}}) where {A, N} = A

sample(a::ActionSpace{A}) where A <: Integer = map(x -> rand(one(A):x), a.actions)
function sample(a::ActionSpace{Tuple{T, T}}) where T <: AbstractFloat
    return map(x -> T(rand(Uniform(x[1], x[2]))), a.actions)
end

struct ObservationSpace{S, N}
    vals::Array{Tuple{S, S}, N}
end

Base.size(o::ObservationSpace) = size(o.vals)
Base.length(o::ObservationSpace) = length(o.vals)
Base.eltype(::Type{ObservationSpace{S, N}}) where {S, N} = S

function ObservationSpace(pyspace::PyObject)
    type = pyspace.__class__.__name__
    if type == "Box"
        return ObservationSpace(map((l, h) -> (l, h), pyspace.low, pyspace.high))
    elseif type == "Tuple"
        return [ObservationSpace(space) for space in pyspace."spaces"]
    else
        error("Unknown python observation_space type")
    end
end
