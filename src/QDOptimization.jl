# QDOptimization.jl

"""
    QDOptimization

A Julia package for Quality-Diversity Optimization algorithms.

This package provides implementations of:
- Archives (Grid-based storage of solutions)
- Emitters (Solution generators)
- Schedulers (Optimization controllers)

Main types:
- GridArchive: Grid-based archive for storing solutions
- GaussianEmitter: Generates solutions using Gaussian perturbation
- RoundRobinScheduler: Manages multiple emitters in round-robin fashion
"""
module QDOptimization

using Random, Distributed
import Base: get, length, isempty

const SolutionType = Union{AbstractFloat, Integer}
const MeasureType = AbstractFloat

include("Archives/Archive.jl")
include("Archives/GridArchive.jl")
export
    # Archive Types
    Archive, GridArchive,

    # Status enum
    STATUS, NEW, IMPROVE, NOT_ADDED,

    # Archive functions
    add!, clear!, get, get_elite, elites, sample,
    length, isempty, solution_dim, measure_dims, cells,
    coverage, obj_max, obj_mean, qd_score, norm_qd_score

include("Emitters/Emitter.jl")
include("Emitters/GaussianEmitter.jl")
export
    # Emitter Types
    Emitter, GaussianEmitter,

    # Emitter functions
    ask!, tell!

include("Schedulers/Scheduler.jl")
include("Schedulers/RoundRobinScheduler.jl")
export
    # Scheduler Types
    Scheduler, RoundRobinScheduler,

    # Scheduler functions
    run!

end