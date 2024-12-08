# Schedulers/BanditScheduler.jl

"""
    BanditScheduler{S,M} <: Scheduler{S,M}

Scheduler that uses UCB1 algorithm to select emitters.

# Type parameters
- S: Solution type (Float or Integer)
- M: Measure type (Float)

# Fields
- `emitters::Vector{<:Emitter{S,M}}`: Pool of available emitters
- `num_active::Int`: Number of active emitters at a time
- `zeta::M`: UCB1 exploration parameter
- `batch_size::Int`: Number of solutions per batch
- `stats_frequency::Int`: How often to print statistics
- `report_mode::ReportMode`: Level of detail in reporting
- `report_archives::Vector{Archive{S,M}}`: Archives to track for reporting
"""
struct BanditScheduler{S<:SolutionType,M<:MeasureType} <: Scheduler{S,M}
    emitters::Vector{<:Emitter{S,M}}
    num_active::Int
    zeta::M
    batch_size::Int
    stats_frequency::Int
    report_mode::ReportMode
    report_archives::Vector{Archive{S,M}}

    function BanditScheduler{S,M}(
        emitters::Vector{<:Emitter{S,M}},
        num_active::Int;
        zeta::M = convert(M, 0.05),
        batch_size::Int = Threads.nthreads(),
        stats_frequency::Int = 1,
        report_mode::ReportMode = VERBOSE,
        report_archives = nothing
    ) where {S<:SolutionType,M<:MeasureType}

        # Validation
        num_active > 0 || throw(ArgumentError("num_active must be positive"))
        num_active <= length(emitters) ||
            throw(ArgumentError("num_active cannot exceed number of emitters"))

        # Check solution dimensions match
        n = solution_dim(first(emitters).archive)
        all(solution_dim(e.archive) == n for e in emitters) ||
            throw(ArgumentError("All emitters must have same solution dimension"))

        # Set up report archives
        archives = if report_archives === nothing
            unique([e.archive for e in emitters])
        elseif report_archives isa Archive{S,M}
            [report_archives]
        else
            collect(report_archives)
        end

        new{S,M}(emitters, num_active, zeta, batch_size,
                 stats_frequency, report_mode, archives)
    end
end

"""
    BanditScheduler(
        emitters::Vector{<:Emitter{S,M}},
        num_active::Int;
        kwargs...
    ) where {S,M}

Convenience constructor that infers types.
"""
function BanditScheduler(
    emitters::Vector{<:Emitter{S,M}},
    num_active::Int;
    kwargs...
) where {S,M}
    BanditScheduler{S,M}(emitters, num_active; kwargs...)
end

"""
Helper function to select active emitters using UCB1 algorithm
"""
function select_active_emitters(
    scheduler::BanditScheduler{S,M},
    emitter_counts::Vector{Int},
    emitter_rewards::Vector{M}
) where {S,M}
    if any(iszero, emitter_counts)
        # Select unused emitters first
        never_used = findall(iszero, emitter_counts)
        return never_used[1:min(length(never_used), scheduler.num_active)]
    else
        # Calculate UCB1 scores
        total_counts = sum(emitter_counts)
        scores = map(enumerate(emitter_counts)) do (i, count)
            avg_reward = emitter_rewards[i] / count
            exploration = sqrt((2 * log(total_counts)) / count)
            avg_reward + scheduler.zeta * exploration
        end

        # Return indices of top scoring emitters
        return partialsort(1:length(scores), 1:scheduler.num_active, by=i->scores[i], rev=true)
    end
end

"""
Helper function to evaluate solutions
"""
function evaluate_solutions(
    objective_fn::Function,
    solutions::Matrix{S},
    measure_dim::Int,
    parallel::Bool
) where {S}
    n_solutions = size(solutions, 2)
    objectives = Vector{Float64}(undef, n_solutions)
    measures = Matrix{Float64}(undef, measure_dim, n_solutions)

    if parallel
        Threads.@threads for i in 1:n_solutions
            result = objective_fn(view(solutions, :, i))
            objectives[i] = result.objective
            measures[:, i] .= result.measure
        end
    else
        for i in 1:n_solutions
            result = objective_fn(view(solutions, :, i))
            objectives[i] = result.objective
            measures[:, i] .= result.measure
        end
    end

    return objectives, measures
end

"""
    run!(scheduler::BanditScheduler{S,M}, objective_fn::Function, n_evaluations::Integer;
         parallel::Bool=false, show_progress::Bool=true)::Nothing where {S,M}

Run optimization using UCB1 bandit algorithm to select emitters.
"""
function run!(
    scheduler::BanditScheduler{S,M},
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

    # Initialize tracking variables
    emitter_counts = zeros(Int, length(scheduler.emitters))
    emitter_rewards = zeros(M, length(scheduler.emitters))
    n_batches = ceil(Int, n_evaluations / scheduler.batch_size)
    total_evals = 0

    # Get solution and measure dimensions
    sol_dim = solution_dim(first(scheduler.emitters).archive)
    measure_dim = length(measure_dims(first(scheduler.emitters).archive))

    for batch in 1:n_batches
        # Select active emitters
        active_emitters = select_active_emitters(scheduler, emitter_counts, emitter_rewards)

        # Get solutions from active emitters
        solutions_per_emitter = ceil(Int, scheduler.batch_size / length(active_emitters))
        solutions = Matrix{S}(undef, sol_dim, scheduler.batch_size)
        emitter_indices = Vector{Int}(undef, scheduler.batch_size)

        # Collect solutions from emitters
        current_idx = 1
        for emitter_idx in active_emitters
            n_solutions = min(solutions_per_emitter,
                            scheduler.batch_size - current_idx + 1)
            if n_solutions <= 0
                break
            end

            emitter_sols = ask!(scheduler.emitters[emitter_idx], n_solutions)
            solutions[:, current_idx:(current_idx + n_solutions - 1)] .= emitter_sols
            emitter_indices[current_idx:(current_idx + n_solutions - 1)] .= emitter_idx
            current_idx += n_solutions
        end

        # Evaluate solutions
        objectives = Vector{M}(undef, size(solutions, 2))
        measures = Matrix{M}(undef, measure_dim, size(solutions, 2))

        if parallel
            Threads.@threads for i in 1:size(solutions, 2)
                result = objective_fn(view(solutions, :, i))
                objectives[i] = result.objective
                measures[:, i] .= result.measure
            end
        else
            for i in 1:size(solutions, 2)
                result = objective_fn(view(solutions, :, i))
                objectives[i] = result.objective
                measures[:, i] .= result.measure
            end
        end

        # Update emitters with results
        for emitter_idx in unique(emitter_indices)
            mask = emitter_indices .== emitter_idx
            if any(mask)
                sols = view(solutions, :, mask)
                objs = view(objectives, mask)
                meas = view(measures, :, mask)

                # Update statistics
                emitter_counts[emitter_idx] += length(objs)
                emitter_rewards[emitter_idx] += sum(objs)

                # Tell emitter results
                tell!(scheduler.emitters[emitter_idx], sols, objs, meas)
            end
        end

        total_evals += size(solutions, 2)

        # Report progress if needed
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
