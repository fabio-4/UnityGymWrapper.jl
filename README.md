# UnityGymWrapper.jl
[Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents) ([Gym](https://github.com/Unity-Technologies/ml-agents/tree/master/gym-unity)) wrapper using [PyCall](https://github.com/JuliaPy/PyCall.jl).  
#### [JuliaRL Implementations](https://github.com/fabio-4/JuliaRL) 

## Requirements (Python Environment)
* [Unity ML-Agents - Release 2](https://github.com/Unity-Technologies/ml-agents/releases/tag/release_2)
* [Unity ML-Agents Python Interface (Envs) - v0.16.1](https://github.com/Unity-Technologies/ml-agents/tree/master/ml-agents-envs)
* [Unity ML-Agents Gym Wrapper - v0.16.1](https://github.com/Unity-Technologies/ml-agents/tree/master/gym-unity) (same limitations)
* [Environment Executables](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Executable.md)  

## API
* Create/ close environment:
```julia
Environment(path; nographics=true, usevisual=false, uint8visual=false, multipleobs=false, kwargs...)

# logFile path to prevent unity terminal output
env = Environment("envs/Basic", logFile=pwd()*"/envs/logs/Basic.log")
close!(env)
```
* Environment interaction
```julia
s = reset!(env) # needs to be called after (done == true) step
s, r, done, info = step!(env) # random action
s, r, done, info = step!(env, action) # eltype(action) == eltype(env.actionspace)
```
* Environment data (check examples & tests)
```julia
# Required action array size & eltype (model output)
size(env.actionspace)
length(env.actionspace)
eltype(env.actionspace)

sample(env.actionspace) # random action

# Discrete/ MultiDiscrete actionspace
# Max ranges for each dim
actiondim1 = env.actionspace.actions[1]
actiondim2 = env.actionspace.actions[2]
...
# Box actionspace
# Range tuples for each dim
l, h = env.actionspace.actions[1]
...

# State/ Observation size & eltype (model input, single or multiple observations)
env.observationspace::Union{ObservationSpace, Vector{ObservationSpace}}
size(observationspace)
length(observationspace)
eltype(observationspace)
l, h = env.actionspace.vals[1]
```
