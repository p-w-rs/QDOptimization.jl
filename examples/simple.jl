# examples/simple.jl

include("../src/QDOptimization.jl")
using .QDOptimization
"""
# Create and archive where
- solutions are defined by 10 elements
- measures are defined by 2 elements
- we bin the measures in a 50x50 grid
- the first grid cell is at (-Inf, -1.0] the last at (1.0, Inf) for each dimension
"""
archive = GridArchive{Float64, Float64}(
    10, (50, 50), [(-1.0, 1.0), (-1.0, 1.0)]
)

# Create a Gaussian emitter
emitter = GaussianEmitter{Float64, Float64}(archive)

# Create a scheduler with the emitter
scheduler = RoundRobinScheduler(emitter)

# Define the objective function
# The objective function must return a NamedTuple with fields `objective` and `measure`
# The objective field is a scalar value
# The measure field is a vector of values
function objective_fn(solution)
    return (objective = sum(solution), measure = solution[1:2])
end

# Run the optimization for 1000 evaluations of the objective function
run!(scheduler, objective_fn, 1000)
