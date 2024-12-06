# Archives/Archive.jl

"""
Abstract type for Quality-Diversity archives.

Type parameters:
- S: Solution type (Float or Integer)
- M: Measure/Objective type (Float)
"""
abstract type Archive{S<:SolutionType, M<:MeasureType} end

"""
    Elite{S,M}

Represents an elite solution in the archive.

# Fields
- `index::Int`: Index in the archive
- `solution::Vector{S}`: Solution vector
- `objective::M`: Objective value
- `measure::Vector{M}`: Measure vector
"""
struct Elite{S<:SolutionType, M<:MeasureType}
    index::Int
    solution::Vector{S}
    objective::M
    measure::Vector{M}
end

# Status enum for add! operation
@enum STATUS NEW IMPROVE NOT_ADDED

"""
    add!(archive::Archive{S,M}, solution::AbstractVector{S}, objective::M, measure::AbstractVector{M}) where {S,M}

Add a solution to the archive.

Returns a NamedTuple{(:status,:value),Tuple{STATUS,M}} containing:
- status: NEW if added to empty cell, IMPROVE if improved existing cell, NOT_ADDED otherwise
- value: The improvement value if improved, else the objective value
"""
function add! end

"""
    clear!(archive::Archive{S,M}) where {S,M}

Remove all solutions from the archive.
"""
function clear! end

"""
    get(archive::Archive{S,M}, measure::AbstractVector{M}) where {S,M}

Get the current archive solution at the specified measure point.
Returns Tuple{Bool,Union{Elite{S,M},Nothing}}.
"""
function Base.get(archive::Archive{S,M}, measure::AbstractVector{M}) where {S,M}
    error("get not implemented")
end

"""
    get_elite(archive::Archive{S,M}, measure::AbstractVector{M}) where {S,M}

Get the best archive solution that has ever existed at the specified measure point.
Returns Tuple{Bool,Union{Elite{S,M},Nothing}}.
"""
function get_elite end

"""
    elites(archive::Archive{S,M}) where {S,M}

Get all elite solutions that have ever existed in the archive.
Returns Vector{Elite{S,M}}.
"""
function elites end

"""
    sample([rng::AbstractRNG,] archive::Archive{S,M}, n::Integer) where {S,M}

Sample n solutions from the archive randomly using the optionally provided random number generator.
Returns Tuple{Matrix{S},Vector{Int}}.
"""
function sample end

# Required interface properties
"""
    length(archive::Archive{S,M}) where {S,M}

Get the number of occupied cells in the archive.
Returns Int.
"""
function Base.length(::Archive)
    error("length not implemented")
end

"""
    isempty(archive::Archive{S,M}) where {S,M}

Check if the archive has any solutions.
Returns Bool.
"""
function Base.isempty(::Archive)
    error("isempty not implemented")
end

"""
    solution_dim(archive::Archive{S,M}) where {S,M}

Get the dimension of the solution space.
Returns Int.
"""
function solution_dim(::Archive)
    error("solution_dim not implemented")
end

"""
    measure_dims(archive::Archive{S,M}) where {S,M}

Get the dimensions of the measure space.
Returns Tuple{Vararg{Int}}.
"""
function measure_dims(::Archive)
    error("measure_dims not implemented")
end

"""
    cells(archive::Archive{S,M}) where {S,M}

Get the total number of cells in the archive.
Returns Int.
"""
function cells(::Archive)
    error("cells not implemented")
end

"""
    coverage(archive::Archive{S,M}) where {S,M}

Get the proportion of cells that are occupied.
Returns M.
"""
function coverage(::Archive)
    error("coverage not implemented")
end

"""
    obj_max(archive::Archive{S,M}) where {S,M}

Get the maximum objective value in the archive.
Returns M.
"""
function obj_max(::Archive)
    error("obj_max not implemented")
end

"""
    obj_mean(archive::Archive{S,M}) where {S,M}

Get the mean objective value across all solutions in the archive.
Returns M.
"""
function obj_mean(::Archive)
    error("obj_mean not implemented")
end

"""
    qd_score(archive::Archive{S,M}) where {S,M}

Get the Quality-Diversity score of the archive (sum of objectives minus offset).
Returns M.
"""
function qd_score(::Archive)
    error("qd_score not implemented")
end

"""
    norm_qd_score(archive::Archive{S,M}) where {S,M}

Get the normalized Quality-Diversity score (QD score divided by number of cells).
Returns M.
"""
function norm_qd_score(::Archive)
    error("norm_qd_score not implemented")
end
