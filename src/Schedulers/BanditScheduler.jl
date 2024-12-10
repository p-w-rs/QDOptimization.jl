# Schedulers/BanditScheduler.jl

"""
    BanditScheduler{S,M} <: Scheduler{S,M}

Scheduler that uses Thompson Sampling to select emitters.

# Type parameters
- S: Solution type (Float or Integer)
- M: Measure type (Float)

# Fields
- `emitters::Vector{<:Emitter{S,M}}`: Pool of available emitters
- `num_active::Int`: Number of active emitters at a time
- `batch_size::Int`: Number of solutions per batch
- `stats_frequency::Int`: How often to print statistics
- `report_mode::ReportMode`: Level of detail in reporting
- `report_archives::Vector{Archive{S,M}}`: Archives to track for reporting
"""
struct BanditScheduler{S<:SolutionType,M<:MeasureType} <: Scheduler{S,M}
    emitters::Vector{<:Emitter{S,M}}
    num_active::Int
    batch_size::Int
    stats_frequency::Int
    report_mode::ReportMode
    report_archives::Vector{Archive{S,M}}

    function BanditScheduler{S,M}(
        emitters::Vector{<:Emitter{S,M}};
        num_active::Int=1,
        batch_size::Int=Threads.nthreads(),
        stats_frequency::Int=1,
        report_mode::ReportMode=VERBOSE,
        report_archives=nothing
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

        new{S,M}(emitters, num_active, batch_size,
            stats_frequency, report_mode, archives)
    end
end

"""
    BanditScheduler(
        emitters::Vector{<:Emitter{S,M}};
        kwargs...
    ) where {S,M}

Convenience constructor that infers types.
"""
function BanditScheduler(
    emitters::Vector{<:Emitter{S,M}};
    kwargs...
) where {S,M}
    BanditScheduler{S,M}(emitters; kwargs...)
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
    emitter_n = length(scheduler.emitters)
    emitter_means = zeros(M, emitter_n)
    emitter_vars = zeros(M, emitter_n)
    emitter_steps = zeros(M, emitter_n)
    n_batches = ceil(Int, n_evaluations / scheduler.batch_size)
    total_evals = 0

    # Get solution and measure dimensions
    sol_dim = solution_dim(first(scheduler.emitters).archive)
    mea_dim = measure_dim(first(scheduler.emitters).archive)

    for batch in 1:n_batches
        # Select active emitters based on Thompson Sampling
        emitters_idxs = sortperm(emitter_means .+ randn(emitter_n) .* sqrt.(emitter_vars))
        emitters_idxs = emitters_idxs[1:scheduler.num_active]

        # Get solutions from active emitters
        solutions_per_emitter = max(1, ceil(Int, scheduler.batch_size / scheduler.num_active))
        solutions = Matrix{S}(undef, sol_dim, solutions_per_emitter*scheduler.num_active)
        idx = 1
        for (i, emitter_idx) in enumerate(emitters_idxs)
            solutions[:, idx:i*solutions_per_emitter] .= ask!(scheduler.emitters[emitter_idx], solutions_per_emitter)
            idx += solutions_per_emitter
        end

        # Evaluate solutions
        objectives = Vector{M}(undef, size(solutions, 2))
        measures = Matrix{M}(undef, mea_dim, size(solutions, 2))
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

        idx = 1
        for (i, emitter_idx) in enumerate(emitters_idxs)
            sols = view(solutions, :, idx:i*solutions_per_emitter)
            objs = view(objectives, idx:i*solutions_per_emitter)
            meas = view(measures, :, idx:i*solutions_per_emitter)
            idx += solutions_per_emitter
            tell!(scheduler.emitters[emitter_idx], sols, objs, meas)

            n = emitter_steps[emitter_idx] + 1
            x = StatsBase.mean(objs)
            μ_prev = emitter_means[emitter_idx]
            v_prev = emitter_vars[emitter_idx]

            μ = μ_prev + ((x - μ_prev) / n)
            v = ((n - 1) * v_prev + (x - μ_prev) * (x - μ)) / n

            emitter_means[emitter_idx] = μ
            emitter_vars[emitter_idx] = v
            emitter_steps[emitter_idx] = n
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
