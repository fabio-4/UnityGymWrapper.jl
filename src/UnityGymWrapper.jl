__precompile__() 
module UnityGymWrapper
using PyCall
include("Spaces.jl")

export Environment, step!, reset!, close!

const pymlagents = PyNULL()
const pygym = PyNULL()

function __init__()
    copy!(pymlagents, pyimport("mlagents_envs.environment"))
    copy!(pygym, pyimport("gym_unity.envs"))
end

struct Environment{A}
    pyenv::PyObject
    pyreset::PyObject
    pystep::PyObject
    pyclose::PyObject
    actionspace::ActionSpace
    observationspace::Union{ObservationSpace, Vector{ObservationSpace}}

    function Environment(path; 
            nographics=true, usevisual=false, uint8visual=false, multipleobs=false, kwargs...)
        args = collect(Iterators.flatten(Tuple{String,String}[("-$(arg[1])", "$(arg[2])") for arg in kwargs]))
        env = pymlagents.UnityEnvironment(path, no_graphics=nographics, args=args)
        try
            gym = pygym.UnityToGymWrapper(
                env, use_visual=usevisual, uint8_visual=uint8visual, allow_multiple_visual_obs=multipleobs
            )
            as = ActionSpace(gym."action_space")
            os = ObservationSpace(gym."observation_space")
            return new{eltype(as)}(gym, gym."reset", gym."step", gym."close", as, os)
        catch
            pycall(env."close", Nothing)
            rethrow()
        end
    end
end

reset!(env) = pycall(env.pyreset, PyArray)

step!(env) = step!(env, sample(env.actionspace))
step!(env, action::Integer) = step!(env, [action])
function step!(env::Environment{A}, action::AbstractArray{A}) where A
    if A <: Integer
        action -= ones(A, size(action)) # "Python conversion"
    end
    return pycall(env.pystep, Tuple{PyArray, Float32, Bool, PyObject}, PyReverseDims(action))
end

close!(env) = pycall(env.pyclose, Nothing)

end
