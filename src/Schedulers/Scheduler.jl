# Schedulers/Scheduler.jl

"""
Abstract type for optimization schedulers.

Type parameters:
- S: Solution type (Float or Integer)
- M: Measure type (Float)
"""
abstract type Scheduler{S<:SolutionType,M<:MeasureType} end

"""
    ReportMode

Enum specifying the verbosity level of scheduler reporting:
- VERBOSE: Detailed statistics including all available metrics
- COMPACT: Basic statistics with only essential metrics
"""
@enum ReportMode VERBOSE COMPACT

"""
    run!(
        scheduler::RoundRobinScheduler{S,M},
        objective_fn::Function,
        n_evaluations::Integer;
        parallel::Bool=false,
        show_progress::Bool=true
    )::Nothing where {S,M}

Run the scheduler for a specified number of evaluations.
"""
function run! end

"""
    generate_stats_report(
        archives::Vector{<:Archive{S,M}},
        mode::ReportMode,
        total_evals::Int,
        batch::Int
    ) where {S,M}

Generate a statistics report for the given archives based on the reporting mode.
Returns a Dict with the computed statistics.
"""
function generate_stats_report(
    archives::Vector{<:Archive{S,M}},
    mode::ReportMode,
    total_evals::Int,
    batch::Int
) where {S,M}
    stats = Dict{Symbol,Any}(
        :batch => batch,
        :total_evaluations => total_evals,
        :best_objective => maximum(obj_max(arch) for arch in archives)
    )

    if mode == VERBOSE
        # Aggregate stats across archives
        stats[:coverage] = mean(coverage(arch) for arch in archives)
        stats[:mean_objective] = mean(obj_mean(arch) for arch in archives)
        stats[:total_qd_score] = sum(qd_score(arch) for arch in archives)
        stats[:normalized_qd_score] = mean(norm_qd_score(arch) for arch in archives)
        stats[:total_cells] = sum(cells(arch) for arch in archives)
        stats[:filled_cells] = sum(length(arch) for arch in archives)
    else # COMPACT
        stats[:coverage] = mean(coverage(arch) for arch in archives)
        stats[:total_qd_score] = sum(qd_score(arch) for arch in archives)
    end

    return stats
end
