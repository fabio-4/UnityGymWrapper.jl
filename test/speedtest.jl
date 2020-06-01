function timejulia(i)
    function steps(env, j)
        for _ in 1:j
            s, r, d, info = step!(env, rand(1:3))
            d && (s = reset!(env))
        end
    end

    env = Environment("../envs/Basic", logFile=pwd()*"/../envs/logs/BasicTestJl.log")
    steps(env, 10)
    t = @elapsed steps(env, i)
    close!(env)
    return t
end

function timepython(i)
    py"""
    from mlagents_envs.environment import UnityEnvironment
    from gym_unity.envs import UnityToGymWrapper
    import time
    import random
    import os

    def pysteps(i):
        def steps(env, j):
            start = time.time()
            for _ in range(0, j):
                s, r, d, info = env.step(random.randint(0, 2))
                if d == True:
                    s = env.reset()
            return time.time() - start

        unity_env = UnityEnvironment(
            "../envs/Basic", no_graphics=True, args=["-logFile", os.getcwd()+"/../envs/logs/BasicTestPy.log"]
        )
        env = UnityToGymWrapper(unity_env)
        t = steps(env, i)
        env.close()
        return t
    """
    return py"pysteps"(i)
end

function speedtest(steps)
    @testset "Speed test" begin
        t1 = timejulia(steps)
        t2 = timepython(steps)
        @test t1 < t2 * 1.05
    end;
end
