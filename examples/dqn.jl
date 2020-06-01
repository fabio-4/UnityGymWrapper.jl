include("helpers.jl")
include("../src/UnityGymWrapper.jl")
using .UnityGymWrapper
using Plots
using Flux
using Flux: Optimise.update!

function run!(model, opt, env; 
        epochs=30, steps=300, maxt=100, batchsize=128, 
        trainiters=100, ϵ=0.07, γ=99f-2, τ=1f-2)
    rewards = zeros(Float32, epochs)
    target = deepcopy(model)
    memory = ReplayMemory{eltype(env.observationspace), eltype(env.actionspace), Float32}(
        length(env.observationspace), length(env.actionspace), 5*steps, batchsize
    )
    ps = params(model)

    for i in 1:epochs
        j = 0
        while j < steps
            s1 = reset!(env)
            for _ in 1:(min(maxt, steps-j))
                a = ifelse(rand() <= ϵ, sample(env.actionspace), Flux.onecold(model(s1)))
                s2, r, done, _ = step!(env, a)
                append!(memory, s1, a, r, done)
                s1 = s2
                j += 1
                done && break
            end
        end

        for _ in 1:trainiters
            s1, a, r, s2, d = sample(memory)
            Qtar = r .+ (1 .- d) .* γ .* vec(maximum(target(s2), dims=1))
            gs = gradient(ps) do
                Qs1a = gather(model(s1), a)
                return Flux.mse(Qs1a, Qtar)
            end
            update!(opt, ps, gs)
            softupdate!(target, model, τ)
        end

        ϵ = max(ϵ * 0.99, 0.01)
        rewards[i] = test(s -> Flux.onecold(model(s)), env)
        println("$(i): r = $(rewards[i])")
    end
    return rewards
end

env = Environment("envs/Basic", logFile=pwd()*"/envs/logs/Basic.log")

model = Chain(
    Dense(length(env.observationspace), 64, relu), 
    Dense(64, 64, relu), 
    Dense(64, env.actionspace.actions[1])
)

r = run!(model, ADAM(), env)
close!(env)

plt = plot(r, labels="Reward")
display(plt)
