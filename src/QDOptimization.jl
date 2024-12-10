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

using Random
import Base: get, length, isempty
import StatsBase

const SolutionType = Union{AbstractFloat,Integer}
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
    length, isempty, solution_dim, measure_dim, cells,
    coverage, obj_max, obj_mean, qd_score, norm_qd_score

include("Emitters/Emitter.jl")
include("Emitters/GaussianEmitter.jl")
include("Emitters/IsoLineEmitter.jl")
export
    # Emitter Types
    Emitter, GaussianEmitter, IsoLineEmitter,

    # Emitter functions
    ask!, tell!

include("Schedulers/Scheduler.jl")
include("Schedulers/RoundRobinScheduler.jl")
include("Schedulers/BanditScheduler.jl")
export
    # Scheduler Types
    Scheduler, RoundRobinScheduler, BanditScheduler,

    # Report enum for run! operation
    ReportMode, VERBOSE, COMPACT,

    # Scheduler functions
    run!, generate_stats_report

end
