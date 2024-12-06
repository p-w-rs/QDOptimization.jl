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
"""
struct RoundRobinScheduler{S<:SolutionType,M<:MeasureType} <: Scheduler{S,M}
    emitters::Vector{<:Emitter{S,M}}
    batch_size::Int
    stats_frequency::Int

    function RoundRobinScheduler{S,M}(
        emitter::Emitter{S,M};
        batch_size::Integer=Threads.nthreads(),
        stats_frequency::Integer=1
    ) where {S<:SolutionType,M<:MeasureType}
        new{S,M}([emitter], batch_size, stats_frequency)
    end

    function RoundRobinScheduler{S,M}(
        emitters::Vector{<:Emitter{S,M}};
        batch_size::Integer=Threads.nthreads(),
        stats_frequency::Integer=1
    ) where {S<:SolutionType,M<:MeasureType}
        new{S,M}(emitters, batch_size, stats_frequency)
    end
end

# Convenience constructors that infer types from emitters
function RoundRobinScheduler(
    emitter::Emitter{S,M};
    kwargs...
) where {S,M}
    RoundRobinScheduler{S,M}(emitter; kwargs...)
end

function RoundRobinScheduler(
    emitters::Vector{<:Emitter{S,M}};
    kwargs...
) where {S,M}
    RoundRobinScheduler{S,M}(emitters; kwargs...)
end

function run!(
    scheduler::RoundRobinScheduler{S,M},
    objective_fn::Function,
    measure_fn::Function,
    n_evaluations::Integer;
    parallel::Bool=false,
    show_progress::Bool=true
)::Nothing where {S,M}
    n_batches = ceil(Int, n_evaluations / scheduler.batch_size)
    n_emitters = length(scheduler.emitters)

    for batch in 1:n_batches
        # Round robin through emitters
        emitter = scheduler.emitters[mod1(batch, n_emitters)]

        # Generate and evaluate solutions
        solutions = ask!(emitter, scheduler.batch_size)

        # Evaluate solutions
        if parallel
            batch_size = div(scheduler.batch_size, Threads.nthreads())
            objectives = Vector{M}(pmap(objective_fn, eachcol(solutions), batch_size=batch_size))
            measures = hcat(pmap(measure_fn, eachcol(solutions), batch_size=batch_size)...)
        else
            objectives = Vector{M}(map(objective_fn, eachcol(solutions)))
            measures = hcat(map(measure_fn, eachcol(solutions))...)
        end

        # Update emitter and archive
        tell!(emitter, solutions, objectives, measures)

        # Print statistics if needed
        if show_progress && batch % scheduler.stats_frequency == 0
            archive = first(scheduler.emitters).archive
            @info "Batch $batch" coverage=coverage(archive) qd_score=qd_score(archive)
        end
    end
    return nothing
end
