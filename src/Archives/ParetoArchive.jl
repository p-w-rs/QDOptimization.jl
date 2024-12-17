"""
    ParetoArchive{S,M} <: Archive{S,M}

Archive storing non-dominated solutions based on multiple measures and objective.
All measures and objective are assumed to be maximized.

# Type parameters
- S: Solution type (Float or Integer)
- M: Measure type (Float)

# Fields
- `solution_dim::Int`: Dimension of solution space
- `measure_dim::Int`: Number of measures
- `solutions::Vector{Vector{S}}`: List of solution vectors
- `objectives::Vector{M}`: List of objective values
- `measures::Vector{Vector{M}}`: List of measure vectors
"""
mutable struct ParetoArchive{S<:SolutionType,M<:MeasureType} <: Archive{S,M}
    solution_dim::Int
    measure_dim::Int
    solutions::Vector{Vector{S}}
    objectives::Vector{M}
    measures::Vector{Vector{M}}

    function ParetoArchive{S,M}(solution_dim::Integer, measure_dim::Integer) where {S<:SolutionType,M<:MeasureType}
        new{S,M}(
            solution_dim,
            measure_dim,
            Vector{Vector{S}}(),
            Vector{M}(),
            Vector{Vector{M}}()
        )
    end
end

# Helper function to check if solution a dominates solution b
function dominates(a_obj::M, a_meas::Vector{M}, b_obj::M, b_meas::Vector{M})::Bool where {M}
    # Check if at least one value is better (greater)
    better = false

    # Compare objective
    if a_obj > b_obj
        better = true
    elseif a_obj < b_obj
        return false
    end

    # Compare measures
    for i in eachindex(a_meas)
        if a_meas[i] > b_meas[i]
            better = true
        elseif a_meas[i] < b_meas[i]
            return false
        end
    end

    return better
end

function add!(archive::ParetoArchive{S,M}, solution::AbstractVector{S}, objective::M, measure::AbstractVector{M})::NamedTuple{(:status, :value),Tuple{STATUS,M}} where {S,M}
    length(solution) == archive.solution_dim ||
        throw(DimensionMismatch("solution dimension mismatch"))
    length(measure) == archive.measure_dim ||
        throw(DimensionMismatch("measure dimension mismatch"))

    # Check if the new solution is dominated by any existing solution
    for i in eachindex(archive.solutions)
        if dominates(archive.objectives[i], archive.measures[i], objective, measure)
            return (status=NOT_ADDED, value=objective)
        end
    end

    # Remove solutions that are dominated by the new solution
    to_remove = Int[]
    for i in eachindex(archive.solutions)
        if dominates(objective, measure, archive.objectives[i], archive.measures[i])
            push!(to_remove, i)
        end
    end

    # If we're removing solutions, it's an improvement
    is_improvement = !isempty(to_remove)

    # Remove dominated solutions in reverse order
    for i in reverse(to_remove)
        deleteat!(archive.solutions, i)
        deleteat!(archive.objectives, i)
        deleteat!(archive.measures, i)
    end

    # Add the new solution
    push!(archive.solutions, copy(solution))
    push!(archive.objectives, objective)
    push!(archive.measures, copy(measure))

    return (status=is_improvement ? IMPROVE : NEW, value=objective)
end

# Implement required interface
Base.length(archive::ParetoArchive) = length(archive.solutions)
Base.isempty(archive::ParetoArchive) = isempty(archive.solutions)
solution_dim(archive::ParetoArchive) = archive.solution_dim
measure_dim(archive::ParetoArchive) = archive.measure_dim
cells(archive::ParetoArchive) = length(archive.solutions)  # Each solution is effectively its own cell
coverage(archive::ParetoArchive) = one(eltype(archive.objectives))  # Always 100% coverage
obj_max(archive::ParetoArchive) = maximum(archive.objectives)
obj_mean(archive::ParetoArchive) = mean(archive.objectives)
qd_score(archive::ParetoArchive) = sum(archive.objectives)
norm_qd_score(archive::ParetoArchive) = mean(archive.objectives)

function clear!(archive::ParetoArchive)::Nothing
    empty!(archive.solutions)
    empty!(archive.objectives)
    empty!(archive.measures)
    empty!(archive.elites_history)
    return nothing
end

function Base.get(archive::ParetoArchive{S,M}, measure::AbstractVector{M})::Tuple{Bool,Union{Elite{S,M},Nothing}} where {S,M}
    # Find the closest solution by Euclidean distance
    if isempty(archive)
        return (false, nothing)
    end

    min_dist = Inf
    min_idx = 1
    for (i, m) in enumerate(archive.measures)
        dist = sum((measure .- m) .^ 2)
        if dist < min_dist
            min_dist = dist
            min_idx = i
        end
    end

    return (true, Elite{S,M}(
        min_idx,
        archive.solutions[min_idx],
        archive.objectives[min_idx],
        archive.measures[min_idx]
    ))
end

get_elite(archive::ParetoArchive, measure::AbstractVector) = get(archive, measure)
elites(archive::ParetoArchive{S,M}) where {S,M} = collect(values(archive.elites_history))

function sample(rng::AbstractRNG, archive::ParetoArchive{S,M}, n::Integer)::Tuple{Matrix{S},Vector{Int}} where {S,M}
    isempty(archive) && throw(ArgumentError("Archive is empty"))
    indices = rand(rng, 1:length(archive), n)
    solutions = hcat([archive.solutions[i] for i in indices]...)
    return solutions, indices
end

sample(archive::ParetoArchive, n::Integer) = StatsBase.sample(Random.default_rng(), archive, n)
