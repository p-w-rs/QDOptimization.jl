# examples/simple.jl

include("../src/QDOptimization.jl")
using .QDOptimization
"""
# Create and archive where
- solutions are defined by 10 elements
- measures are defined by 2 elements
- we bin the measures in a 4x4x4 grid
- the first grid cell is at (-Inf, 0.0] the last at (1.0, Inf) for each dimension
"""
archive = GridArchive{Float32,Float64}(
    25, Tuple(10 for _ in 1:3), [(0.0, 1.0) for _ in 1:3]
)

# Create a Gaussian emitter
emitters = [CMAESEmitter{Float32,Float64}(archive) for _ in 1:2]
emitters = vcat(emitters..., [CMAESEmitter{Float32,Float64}(archive) for _ in 1:2])
emitters = vcat(emitters..., [CMAESEmitter{Float32,Float64}(archive) for _ in 1:2])

# Create a scheduler with the emitter
scheduler = RoundRobinScheduler(emitters, batch_size=100, stats_frequency=100)

# Define the objective function
# The objective function must return a NamedTuple with fields `objective` and `measure`
# The objective field is a scalar value
# The measure field is a vector of values
function rosenbrock(x::AbstractVector{<:Real})
    n = length(x)
    if n < 2
        throw(ArgumentError("Input vector must have at least 2 dimensions"))
    end

    return (
        objective=-sum(100 * (x[i+1] - x[i]^2)^2 + (1 - x[i])^2 for i in 1:(n-1)),
        measure=x[rand(1:n, 3)]
    )
end

# Run the optimization for 100000000 evaluations of the objective function
run!(scheduler, rosenbrock, 100000000, parallel=true)
