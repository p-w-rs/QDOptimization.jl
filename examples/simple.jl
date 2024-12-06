include("QDOptimization.jl")
using .QDOptimization

archive = GridArchive{Float64, Float64}(
    10, (10, 10), [(-1.0, 1.0), (-1.0, 1.0)]
)
emitter = GaussianEmitter{Float64, Float64}(archive)
scheduler = RoundRobinScheduler(emitter)

function objective_fn(solution)
    return sum(solution)
end

function measure_fn(solution)
    return solution[1:2]
end

run!(scheduler, objective_fn, measure_fn, 1000)
