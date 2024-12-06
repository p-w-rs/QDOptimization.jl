# Emitters/Emitter.jl

"""
Abstract type for solution emitters.

Type parameters:
- S: Solution type (Float or Integer)
- M: Measure type (Float)
"""
abstract type Emitter{S<:SolutionType, M<:MeasureType} end

"""
    ask!(emitter::Emitter{S,M}, n::Integer) where {S,M}

Get n candidate solutions from the emitter.
Returns Matrix{S} where each column is a candidate solution.
"""
function ask! end

"""
    tell!(emitter::Emitter{S,M}, solutions::AbstractMatrix{S},
          objectives::AbstractVector{M}, measures::AbstractMatrix{M}) where {S,M}

Provide the emitter with the objectives and measures of the last solutions it emitted.
"""
function tell! end
