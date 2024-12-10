# Emitters/GaussianEmitter.jl

"""
    GaussianEmitter{S,M} <: Emitter{S,M}

Emits solutions by adding Gaussian noise to existing archive solutions.

# Type parameters
- S: Solution type (Float or Integer)
- M: Measure type (Float)

# Fields
- `archive::Archive{S,M}`: Archive to sample from and add solutions to
- `σ::Vector{M}`: Standard deviation of Gaussian noise per dimension
- `x0::Vector{S}`: Initial solution when archive is empty
- `lower_bounds::Vector{S}`: Lower bounds for solution values
- `upper_bounds::Vector{S}`: Upper bounds for solution values
"""
struct GaussianEmitter{S<:SolutionType,M<:MeasureType} <: Emitter{S,M}
    archive::Archive{S,M}
    σ::Vector{S}
    x0::Vector{S}
    lower_bounds::Vector{S}
    upper_bounds::Vector{S}
    rng::AbstractRNG

    function GaussianEmitter{S,M}(
        archive::Archive{S,M};
        σ::Union{S,AbstractVector{S}} = one(S),
        x0::Union{S,AbstractVector{S}} = zero(S),
        bounds::Union{Nothing,Tuple{S,S},AbstractVector{Tuple{S,S}}} = nothing,
        seed::Union{Nothing,Integer} = nothing
    ) where {S<:SolutionType,M<:MeasureType}
        sol_dim = solution_dim(archive)

        # Convert scalar to vector if needed
        σ_vec = σ isa S ? fill(σ, sol_dim) : convert(Vector{S}, σ)
        length(σ_vec) == sol_dim ||
            throw(ArgumentError("σ must have length equal to solution dimension"))

        x0_vec = x0 isa S ? fill(x0, sol_dim) : convert(Vector{S}, x0)
        length(x0_vec) == sol_dim ||
            throw(ArgumentError("x0 must have length equal to solution dimension"))

        # Set up bounds
        if bounds === nothing
            lower = fill(typemin(S), sol_dim)
            upper = fill(typemax(S), sol_dim)
        elseif bounds isa Tuple{S,S}
            bounds[1] >= bounds[2] &&
                throw(ArgumentError("lower bound must be less than upper bound"))
            lower = fill(bounds[1], sol_dim)
            upper = fill(bounds[2], sol_dim)
        else
            length(bounds) == sol_dim ||
                throw(ArgumentError("bounds must have length equal to solution dimension"))
            lower = S[b[1] for b in bounds]
            upper = S[b[2] for b in bounds]
            all(lower .< upper) ||
                throw(ArgumentError("lower bounds must be less than upper bounds"))
        end

        new{S,M}(archive, σ_vec, x0_vec, lower, upper,
            seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed))
    end
end

function ask!(emitter::GaussianEmitter{S,M}, n::Integer)::Matrix{S} where {S,M}
    sol_dim = solution_dim(emitter.archive)

    # Generate parent solutions
    if isempty(emitter.archive)
        parents = repeat(emitter.x0, 1, n)
    else
        parents, _ = sample(emitter.rng, emitter.archive, n)
    end

    # Add Gaussian noise
    noise = randn(emitter.rng, sol_dim, n) .* emitter.σ
    solutions = parents .+ convert(Matrix{S}, noise)

    # Clip to bounds
    return clamp.(solutions, emitter.lower_bounds, emitter.upper_bounds)
end

function tell!(
    emitter::GaussianEmitter{S,M},
    solutions::AbstractMatrix{S},
    objectives::AbstractVector{M},
    measures::AbstractMatrix{M}
)::Nothing where {S,M}
    size(solutions, 2) == length(objectives) ||
        throw(ArgumentError("number of objectives must match number of solutions"))
    size(measures, 2) == size(solutions, 2) ||
        throw(ArgumentError("number of measures must match number of solutions"))

    for i in 1:size(solutions, 2)
        add!(emitter.archive,
            view(solutions, :, i),
            objectives[i],
            view(measures, :, i))
    end
    return nothing
end
