# Emitters/IsoLineEmitter.jl

"""
    IsoLineEmitter{S,M} <: Emitter{S,M}

Emits solutions using the Iso+LineDD operator that combines isotropic Gaussian
mutation with directional variation based on correlations between solutions.

# Type parameters
- S: Solution type (Float or Integer)
- M: Measure type (Float)

# Fields
- `archive::Archive{S,M}`: Archive to sample from and add solutions to
- `σ1::M`: Standard deviation for isotropic Gaussian distribution
- `σ2::M`: Standard deviation for directional variation
- `x0::Vector{S}`: Initial solution when archive is empty
- `lower_bounds::Vector{S}`: Lower bounds for solution values
- `upper_bounds::Vector{S}`: Upper bounds for solution values
"""
struct IsoLineEmitter{S<:SolutionType,M<:MeasureType} <: Emitter{S,M}
    archive::Archive{S,M}
    σ1::S  # Isotropic std dev
    σ2::S  # Directional std dev
    x0::Vector{S}
    lower_bounds::Vector{S}
    upper_bounds::Vector{S}
    rng::AbstractRNG

    function IsoLineEmitter{S,M}(
        archive::Archive{S,M};
        σ1::S = convert(S, 0.01),  # Default from paper
        σ2::S = convert(S, 0.2),   # Default from paper
        x0::Union{S,AbstractVector{S}} = zero(S),
        bounds::Union{Nothing,Tuple{S,S},AbstractVector{Tuple{S,S}}} = nothing,
        seed::Union{Nothing,Integer} = nothing
    ) where {S<:SolutionType,M<:MeasureType}
        sol_dim = solution_dim(archive)

        # Convert scalar x0 to vector if needed
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

        new{S,M}(archive, σ1, σ2, x0_vec, lower, upper,
            seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed))
    end
end

function ask!(emitter::IsoLineEmitter{S,M}, n::Integer)::Matrix{S} where {S,M}
    sol_dim = solution_dim(emitter.archive)
    solutions = Matrix{S}(undef, sol_dim, n)

    for i in 1:n
        # Select parent solution
        if isempty(emitter.archive)
            x1 = emitter.x0
            x2 = emitter.x0
        else
            xs, _ = sample(emitter.rng, emitter.archive, 2)
            x1 = view(xs, :, 1)
            x2 = view(xs, :, 2)
        end

        # Generate offspring using Iso+LineDD operator
        # x' = x1 + σ1 * N(0,I) + σ2 * (x2-x1) * N(0,1)
        iso_noise = randn(emitter.rng, sol_dim)
        dir_noise = randn(emitter.rng)

        offspring = x1 .+
                   emitter.σ1 .* iso_noise .+
                   emitter.σ2 .* (x2 .- x1) .* dir_noise

        # Clip to bounds
        solutions[:, i] .= clamp.(offspring, emitter.lower_bounds, emitter.upper_bounds)
    end

    return solutions
end

function tell!(
    emitter::IsoLineEmitter{S,M},
    solutions::AbstractMatrix{S},
    objectives::AbstractVector{M},
    measures::AbstractMatrix{M}
)::Nothing where {S,M}
    size(solutions, 2) == length(objectives) ||
        throw(ArgumentError("number of objectives must match number of solutions"))
    size(measures, 2) == size(solutions, 2) ||
        throw(ArgumentError("number of measures must match number of solutions"))

    # Add solutions to archive
    for i in 1:size(solutions, 2)
        add!(emitter.archive,
            view(solutions, :, i),
            objectives[i],
            view(measures, :, i))
    end
    return nothing
end
