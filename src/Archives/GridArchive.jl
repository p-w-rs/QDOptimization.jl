# Archives/GridArchive.jl

"""
    GridArchive{S,M} <: Archive{S,M}

Grid-based archive dividing measure space into uniform cells.

# Type parameters
- S: Solution type (Float or Integer)
- M: Measure type (Float)

# Fields
- `solution_dim::Int`: Dimension of solution space
- `cells_per_measure::Tuple{Vararg{Int}}`: Number of cells per dimension
- `measure_ranges::Vector{Tuple{M,M}}`: Min/max ranges per dimension
- `learning_rate::M`: Learning rate for threshold updates
- `threshold_min::M`: Minimum threshold value
"""
mutable struct GridArchive{S<:SolutionType,M<:MeasureType} <: Archive{S,M}
    # Archive configuration
    solution_dim::Int
    cells_per_measure::Tuple{Vararg{Int}}
    measure_ranges::Vector{Tuple{M,M}}
    learning_rate::M
    threshold_min::M

    # Measure space data
    lower_bounds::Vector{M}
    upper_bounds::Vector{M}
    interval_size::Vector{M}
    boundaries::Vector{Vector{M}}

    # Archive data
    occupied::Set{Int}
    solutions::Matrix{S}
    objectives::Vector{M}
    measures::Matrix{M}
    thresholds::Vector{M}

    # Runtime data
    elites::Dict{Int,Elite{S,M}}
    qd_score_offset::M

    function GridArchive{S,M}(
        solution_dim::Integer,
        cells_per_measure::Tuple{Vararg{Integer}},
        measure_ranges::Vector{Tuple{M,M}};
        learning_rate::M = one(M),
        threshold_min::M = M(-Inf)
    ) where {S<:SolutionType,M<:MeasureType}
        length(cells_per_measure) == length(measure_ranges) ||
            throw(ArgumentError("cells_per_measure must have same length as measure_ranges"))
        all(r -> r[1] < r[2], measure_ranges) ||
            throw(ArgumentError("range lower bounds must be less than upper bounds"))

        lower_bounds = M[r[1] for r in measure_ranges]
        upper_bounds = M[r[2] for r in measure_ranges]
        interval_size = upper_bounds .- lower_bounds

        boundaries = [range(lower_bounds[i], upper_bounds[i], length=cells_per_measure[i]-1) |> collect
                     for i in eachindex(cells_per_measure)]

        ncells = prod(cells_per_measure)
        new{S,M}(
            solution_dim, cells_per_measure, measure_ranges, learning_rate, threshold_min,
            lower_bounds, upper_bounds, interval_size, boundaries,
            Set{Int}(),
            zeros(S, solution_dim, ncells),
            fill(M(-Inf), ncells),
            zeros(M, length(cells_per_measure), ncells),
            fill(threshold_min, ncells),
            Dict{Int,Elite{S,M}}(),
            zero(M)
        )
    end
end

Base.length(archive::GridArchive) = length(archive.occupied)
Base.isempty(archive::GridArchive) = isempty(archive.occupied)
solution_dim(archive::GridArchive) = archive.solution_dim
measure_dim(archive::GridArchive) = length(archive.cells_per_measure)
cells(archive::GridArchive) = prod(archive.cells_per_measure)
coverage(archive::GridArchive) = length(archive.occupied) / cells(archive)
obj_max(archive::GridArchive) = maximum(archive.objectives[collect(archive.occupied)])
obj_mean(archive::GridArchive) = sum(archive.objectives[collect(archive.occupied)]) / length(archive)
qd_score(archive::GridArchive) = sum(archive.objectives[collect(archive.occupied)] .- archive.qd_score_offset)
norm_qd_score(archive::GridArchive) = qd_score(archive) / cells(archive)

function index_of(archive::GridArchive{S,M}, measure::AbstractVector{M})::Int where {S,M}
    length(measure) == length(archive.cells_per_measure) ||
        throw(ArgumentError("measure must have same length as cells_per_measure"))

    cell_idxs = [searchsortedfirst(b, m) for (b, m) in zip(archive.boundaries, measure)]

    idx = cell_idxs[1]
    for i in 2:length(cell_idxs)
        idx += (cell_idxs[i] - 1) * prod(archive.cells_per_measure[1:i-1])
    end

    return idx
end

function add!(archive::GridArchive{S,M}, solution::AbstractVector{S}, objective::M, measure::AbstractVector{M})::NamedTuple{(:status,:value),Tuple{STATUS,M}} where {S,M}
    length(solution) == archive.solution_dim ||
        throw(DimensionMismatch("solution dimension mismatch"))
    length(measure) == length(archive.cells_per_measure) ||
        throw(DimensionMismatch("measure dimension mismatch"))

    archive.qd_score_offset = min(archive.qd_score_offset, objective)
    idx = index_of(archive, measure)

    if idx ∉ archive.occupied
        push!(archive.occupied, idx)
        archive.solutions[:, idx] .= solution
        archive.objectives[idx] = objective
        archive.measures[:, idx] .= measure
        archive.thresholds[idx] = max(archive.threshold_min, objective)

        archive.elites[idx] = Elite{S,M}(
            idx,
            deepcopy(solution),
            objective,
            deepcopy(measure)
        )

        return (status = NEW, value = objective)
    else
        curr_threshold = archive.thresholds[idx]
        if objective > curr_threshold
            improvement = objective - archive.objectives[idx]

            archive.solutions[:, idx] .= solution
            archive.objectives[idx] = objective
            archive.measures[:, idx] .= measure

            archive.thresholds[idx] = max(
                archive.threshold_min,
                (one(M) - archive.learning_rate) * curr_threshold +
                archive.learning_rate * objective
            )

            if objective > archive.elites[idx].objective
                archive.elites[idx] = Elite{S,M}(
                    idx,
                    deepcopy(solution),
                    objective,
                    deepcopy(measure)
                )
            end

            return (status = IMPROVE, value = improvement)
        end

        return (status = NOT_ADDED, value = objective - curr_threshold)
    end
end

function clear!(archive::GridArchive{S,M})::Nothing where {S,M}
    empty!(archive.occupied)
    fill!(archive.solutions, zero(S))
    fill!(archive.objectives, M(-Inf))
    fill!(archive.measures, zero(M))
    fill!(archive.thresholds, archive.threshold_min)
    empty!(archive.elites)
    archive.qd_score_offset = zero(M)
    return nothing
end

function get(archive::GridArchive{S,M}, measure::AbstractVector{M})::Tuple{Bool,Union{Elite{S,M},Nothing}} where {S,M}
    idx = index_of(archive, measure)
    if idx ∉ archive.occupied
        return (false, nothing)
    else
        return (true, Elite{S,M}(
            idx,
            archive.solutions[:, idx],
            archive.objectives[idx],
            archive.measures[:, idx]
        ))
    end
end

function get_elite(archive::GridArchive{S,M}, measure::AbstractVector{M})::Tuple{Bool,Union{Elite{S,M},Nothing}} where {S,M}
    idx = index_of(archive, measure)
    if idx ∉ archive.occupied
        return (false, nothing)
    else
        return (true, archive.elites[idx])
    end
end

function elites(archive::GridArchive{S,M})::Vector{Elite{S,M}} where {S,M}
    collect(values(archive.elites))
end

function sample(archive::GridArchive{S,M}, n::Integer)::Tuple{Matrix{S},Vector{Int}} where {S,M}
    isempty(archive) && throw(ArgumentError("Archive is empty"))
    indices = rand(collect(archive.occupied), n)
    return archive.solutions[:, indices], indices
end

function sample(rng::AbstractRNG, archive::GridArchive{S,M}, n::Integer)::Tuple{Matrix{S},Vector{Int}} where {S,M}
    isempty(archive) && throw(ArgumentError("Archive is empty"))
    indices = rand(rng, collect(archive.occupied), n)
    return archive.solutions[:, indices], indices
end
