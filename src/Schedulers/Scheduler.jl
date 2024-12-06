# Schedulers/Scheduler.jl

"""
Abstract type for optimization schedulers.

Type parameters:
- S: Solution type (Float or Integer)
- M: Measure type (Float)
"""
abstract type Scheduler{S<:SolutionType, M<:MeasureType} end

"""
    run!(scheduler::Scheduler{S,M},
         objective_fn::Function,
         measure_fn::Function,
         n_evaluations::Integer;
         parallel::Bool=false,
         show_progress::Bool=true) where {S,M}

Run the scheduler for a specified number of evaluations.
"""
function run! end
