using UnityGymWrapper
using PyCall
using PyCall: PyError
using Test
include("speedtest.jl")

Basic = Dict(
    "name" => "Basic",
    "path" => "../envs/Basic",
    "kwargs" => (),
    "actiontype" => Int64,
    "actionsize" => (1,),
    "actionrange" => 3,
    "testaction" => 3,
    "observationtype" => Float32,
    "observationsize" => (20,),
    "observationrange" => (-Inf32, Inf32),
    "rewardtype" => Float32,
    "donetype" => Bool
)

BouncerSingleAgent = Dict(
    "name" => "BouncerSingleAgent",
    "path" => "../envs/BouncerSingleAgent",
    "kwargs" => (),
    "actiontype" => Float32,
    "actionsize" => (3,),
    "actionrange" => (-1f0, 1f0),
    "testaction" => [-1f0, 0f0, 1f0],
    "observationtype" => Float32,
    "observationsize" => (18,),
    "observationrange" => (-Inf32, Inf32),
    "rewardtype" => Float32,
    "donetype" => Bool
)

VisualPushBlock = Dict(
    "name" => "VisualPushBlock",
    "path" => "../envs/VisualPushBlock",
    "kwargs" => (nographics=false, usevisual=true),
    "actiontype" => Int64,
    "actionsize" => (1,),
    "actionrange" => 7,
    "testaction" => 7,
    "observationtype" => Float32,
    "observationsize" => (84, 84, 3),
    "observationrange" => (0.0f0, 1.0f0),
    "rewardtype" => Float32,
    "donetype" => Bool
)

function testenv(data)
    @testset "$(data["name"])" begin
        env = Environment(data["path"]; data["kwargs"]...)

        @test eltype(env.actionspace) == data["actiontype"]
        @test size(env.actionspace) == data["actionsize"]
        @test env.actionspace.actions[1] == data["actionrange"]

        @test eltype(env.observationspace) == data["observationtype"]
        @test size(env.observationspace) == data["observationsize"]
        @test env.observationspace.vals[1] == data["observationrange"]

        obs, reward, done, info = step!(env, data["testaction"])
        @test size(obs) == data["observationsize"]
        @test eltype(obs) == eltype(env.observationspace)
        @test typeof(reward) == data["rewardtype"]
        @test typeof(done) == data["donetype"]

        obs = reset!(env)
        @test size(obs) == data["observationsize"]
        @test eltype(obs) == eltype(env.observationspace)

        obs, _, _, _ = step!(env)
        @test size(obs) == data["observationsize"]

        close!(env)
    end;
end

testenv.([Basic, BouncerSingleAgent, VisualPushBlock])

@testset "3DBall" begin
    @test_throws PyError Environment("../envs/3DBall")
end;

@testset "VisualPushBlockUInt8" begin
    env = Environment("../envs/VisualPushBlock", nographics=false, usevisual=true, uint8visual=true)

    @test eltype(env.observationspace) == UInt8
    @test size(env.observationspace) == (84, 84, 3)
    @test env.observationspace.vals[1] == (0x00, 0xff)
    # @test convert(Tuple{Int64, Int64}, env.observationspace.vals[1]) == (0, 255)

    close!(env)
end;

@testset "PushBlock" begin
    env = Environment("../envs/PushBlock")

    @test eltype(env.observationspace) == Float32
    @test size(env.observationspace) == (210,)
    @test env.observationspace.vals[1] == (-Inf32, Inf32)

    close!(env)
end;

speedtest(3000)
