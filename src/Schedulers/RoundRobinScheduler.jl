# Schedulers/RoundRobinScheduler.jl

"""
    RoundRobinScheduler{S,M} <: Scheduler{S,M}

Scheduler that alternates between multiple emitters in round-robin fashion.

# Type parameters
- S: Solution type (Float or Integer)
- M: Measure type (Float)

# Fields
- `emitters::Vector{Emitter{S,M}}`: List of emitters to alternate between
- `batch_size::Int`: Number of solutions to evaluate per batch
- `stats_frequency::Int`: How often to print statistics (in batches)
- `report_mode::ReportMode`: Level of detail in progress reporting
- `report_archives::Vector{Archive{S,M}}`: Archives to track for reporting statistics
"""
struct RoundRobinScheduler{S<:SolutionType,M<:MeasureType} <: Scheduler{S,M}
    emitters::Vector{<:Emitter{S,M}}
    batch_size::Int
    stats_frequency::Int
    report_mode::ReportMode
    report_archives::Vector{Archive{S,M}}

    """
        RoundRobinScheduler{S,M}(
            emitters::Vector{<:Emitter{S,M}};
            batch_size::Integer=Threads.nthreads(),
            stats_frequency::Integer=1,
            report_mode::ReportMode=VERBOSE,
            report_archives::Union{Nothing,Archive{S,M},Vector{<:Archive{S,M}}}=nothing
        ) where {S<:SolutionType,M<:MeasureType}

    Construct a RoundRobinScheduler with multiple emitters.
    """
    function RoundRobinScheduler{S,M}(
        emitters::Vector{<:Emitter{S,M}};
        batch_size::Integer=Threads.nthreads(),
        stats_frequency::Integer=1,
        report_mode::ReportMode=VERBOSE,
        report_archives::Union{Nothing,Archive{S,M},Vector{<:Archive{S,M}}}=nothing
    ) where {S<:SolutionType,M<:MeasureType}
        n = solution_dim(first(emitters).archive)
        all(solution_dim(emitter.archive) == n for emitter in emitters) ||
            throw(ArgumentError("All emitters must have the same solution dimension"))

        archives = if report_archives === nothing
            unique([e.archive for e in emitters])
        elseif report_archives isa Archive{S,M}
            [report_archives]
        else
            collect(report_archives)
        end
        new{S,M}(emitters, batch_size, stats_frequency, report_mode, archives)
    end
end

"""
    RoundRobinScheduler(
        emitter::Emitter{S,M};
        kwargs...
    ) where {S,M}

Convenience constructor for single emitter that infers types.
"""
function RoundRobinScheduler(
    emitter::Emitter{S,M};
    kwargs...
) where {S,M}
    RoundRobinScheduler{S,M}([emitter]; kwargs...)
end

"""
    RoundRobinScheduler(
        emitters::Vector{<:Emitter{S,M}};
        kwargs...
    ) where {S,M}

Convenience constructor for multiple emitters that infers types.
"""
function RoundRobinScheduler(
    emitters::Vector{<:Emitter{S,M}};
    kwargs...
) where {S,M}
    RoundRobinScheduler{S,M}(emitters; kwargs...)
end

"""
    run!(
        scheduler::RoundRobinScheduler{S,M},
        objective_fn::Function,
        n_evaluations::Integer;
        parallel::Bool=false,
        show_progress::Bool=true
    )::Nothing where {S,M}

Run the scheduler for a specified number of evaluations.

# Arguments
- `objective_fn`: Function that takes a solution vector and returns a NamedTuple with fields:
  - `objective::AbstractFloat`: Objective value
  - `measure::AbstractVector{<:AbstractFloat}`: Measure vector
- `n_evaluations`: Total number of evaluations to perform
- `parallel`: Whether to evaluate solutions in parallel
- `show_progress`: Whether to show progress reports
"""
function run!(
    scheduler::RoundRobinScheduler{S,M},
    objective_fn::Function,
    n_evaluations::Integer;
    parallel::Bool=false,
    show_progress::Bool=true
)::Nothing where {S,M}
    # Check return type
    test_sol = zeros(S, solution_dim(first(scheduler.emitters).archive))
    ret_type = Base.return_types(objective_fn, (typeof(test_sol),))[1]
    if !(ret_type <: NamedTuple{(:objective, :measure),<:Tuple{AbstractFloat,AbstractVector{<:AbstractFloat}}})
        throw(ArgumentError("objective_fn must return a NamedTuple with fields (objective::AbstractFloat, measure::AbstractVector{<:AbstractFloat})"))
    end

    n_batches = ceil(Int, n_evaluations / scheduler.batch_size)
    n_emitters = length(scheduler.emitters)
    total_evals = 0

    # Pre-allocate arrays for objectives and measures
    objectives = Vector{M}(undef, scheduler.batch_size)
    measures = Matrix{M}(undef, length(measure_dims(first(scheduler.emitters).archive)), scheduler.batch_size)

    for batch in 1:n_batches
        emitter = scheduler.emitters[mod1(batch, n_emitters)]
        solutions = ask!(emitter, scheduler.batch_size)

        if parallel
            Threads.@threads for i in 1:scheduler.batch_size
                result = objective_fn(view(solutions, :, i))
                objectives[i] = result.objective
                measures[:, i] .= result.measure
            end
        else
            for i in 1:scheduler.batch_size
                result = objective_fn(view(solutions, :, i))
                objectives[i] = result.objective
                measures[:, i] .= result.measure
            end
        end

        tell!(emitter, solutions, objectives, measures)
        total_evals += scheduler.batch_size

        if show_progress && batch % scheduler.stats_frequency == 0
            stats = generate_stats_report(
                scheduler.report_archives,
                scheduler.report_mode,
                total_evals,
                batch
            )
            @info "Progress Report" pairs(stats)...
        end
    end
    return nothing
end
