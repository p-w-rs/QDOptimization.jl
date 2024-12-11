# Emitters/CMAESEmitter.jl

"""
    RankingType

Enum specifying the ranking method for CMA-ES:
- IMPROVEMENT: Ranks based on improvement over current solutions
- TWO_STAGE_IMPROVEMENT: Two-stage ranking using improvement
- RANDOM_DIRECTION: Ranks based on random directions
- TWO_STAGE_RANDOM_DIRECTION: Two-stage ranking using random directions
- OBJECTIVE: Ranks based on objective values only
- TWO_STAGE_OBJECTIVE: Two-stage ranking using objectives
"""
@enum RankingType begin
    IMPROVEMENT
    TWO_STAGE_IMPROVEMENT
    RANDOM_DIRECTION
    TWO_STAGE_RANDOM_DIRECTION
    OBJECTIVE
    TWO_STAGE_OBJECTIVE
end

"""
    SelectionRule

Enum specifying the selection rule for CMA-ES:
- MU: Select top μ individuals
- FILTER: Filter-based selection
"""
@enum SelectionRule MU FILTER

"""
    CMAESEmitter{S,M} <: Emitter{S,M}

Emits solutions using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
"""
mutable struct CMAESEmitter{S<:SolutionType,M<:MeasureType} <: Emitter{S,M}
    archive::Archive{S,M}
    x0::Vector{S}
    σ0::S
    lower_bounds::Vector{S}
    upper_bounds::Vector{S}
    ranking::RankingType
    selection_rule::SelectionRule
    restart_rule::Int
    rng::AbstractRNG

    # CMA-ES parameters
    λ::Int  # Population size
    μ::Int  # Parent number
    weights::Vector{S}  # Recombination weights
    μeff::S  # Effective selection mass
    cc::S  # Cumulation factor for C
    c1::S  # Learning rate for rank-one update
    cμ::S  # Learning rate for rank-μ update
    cσ::S  # Cumulation factor for σ
    dσ::S  # Damping for σ
    chiN::S  # Expected norm of N(0,I)

    # CMA-ES state
    C::Matrix{S}  # Covariance matrix
    B::Matrix{S}  # Eigenvectors of C
    D::Vector{S}  # Sqrt of eigenvalues of C
    pc::Vector{S}  # Evolution path for C
    pσ::Vector{S}  # Evolution path for σ
    mean::Vector{S}  # Current mean
    σ::S  # Current step size
    generation::Int
    last_improvement::Int
    random_direction::Union{Nothing,Vector{S}}  # For random direction ranking

    function CMAESEmitter{S,M}(
        archive::Archive{S,M};
        x0::Union{S,AbstractVector{S}}=zero(S),
        σ0::S=one(S),
        bounds::Union{Nothing,Tuple{S,S},AbstractVector{Tuple{S,S}}}=nothing,
        ranking::RankingType=IMPROVEMENT,
        selection_rule::SelectionRule=MU,
        restart_rule::Int=100,
        seed::Union{Nothing,Integer}=nothing
    ) where {S<:SolutionType,M<:MeasureType}
        sol_dim = solution_dim(archive)

        # Initialize x0 and bounds
        x0_vec = x0 isa S ? fill(x0, sol_dim) : convert(Vector{S}, x0)
        length(x0_vec) == sol_dim ||
            throw(ArgumentError("x0 must have length equal to solution dimension"))

        if bounds === nothing
            lower = fill(typemin(S), sol_dim)
            upper = fill(typemax(S), sol_dim)
        elseif bounds isa Tuple{S,S}
            bounds[1] >= bounds[2] &&
                throw(ArgumentError("lower bound must be less than upper bound"))
            lower = fill(bounds[1], sol_dim)
            upper = fill(bounds[2], sol_dim)
        else
            length(bounds) == sol_dim ||
                throw(ArgumentError("bounds must have length equal to solution dimension"))
            lower = S[b[1] for b in bounds]
            upper = S[b[2] for b in bounds]
            all(lower .< upper) ||
                throw(ArgumentError("lower bounds must be less than upper bounds"))
        end

        # Initialize CMA-ES parameters
        λ = 4 + floor(Int, 3 * log(sol_dim))  # Default population size
        μ = λ ÷ 2  # Default parent number

        # Initialize weights logarithmically
        weights = [log((λ + 1) / 2) - log(i) for i in 1:μ]
        weights ./= sum(weights)

        μeff = 1 / sum(weights .^ 2)

        # Learning rates and constants
        cc = 4 / (sol_dim + 4)
        c1 = 2 / ((sol_dim + 1.3)^2 + μeff)
        cμ = min(1 - c1, 2 * (μeff - 2 + 1 / μeff) / ((sol_dim + 2)^2 + μeff))
        cσ = (μeff + 2) / (sol_dim + μeff + 5)
        dσ = 1 + 2 * max(0, sqrt((μeff - 1) / (sol_dim + 1)) - 1) + cσ
        chiN = sqrt(sol_dim) * (1 - 1 / (4 * sol_dim) + 1 / (21 * sol_dim^2))

        # Initialize state variables
        C = Matrix{S}(I, sol_dim, sol_dim)
        B = Matrix{S}(I, sol_dim, sol_dim)
        D = ones(S, sol_dim)
        pc = zeros(S, sol_dim)
        pσ = zeros(S, sol_dim)

        new{S,M}(
            archive, x0_vec, σ0, lower, upper, ranking, selection_rule, restart_rule,
            seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed),
            λ, μ, weights, μeff, cc, c1, cμ, cσ, dσ, chiN,
            C, B, D, pc, pσ, copy(x0_vec), σ0, 0, 0, nothing
        )
    end
end

function rank_solutions(
    emitter::CMAESEmitter{S,M},
    solutions::AbstractMatrix{S},
    objectives::AbstractVector{M},
    measures::AbstractMatrix{M}
)::Vector{Int} where {S,M}
    n = length(objectives)
    ranks = collect(1:n)

    if emitter.ranking == IMPROVEMENT
        # Rank by improvement over current archive solutions
        improvements = Vector{M}(undef, n)
        for i in 1:n
            result = add!(emitter.archive, view(solutions, :, i), objectives[i], view(measures, :, i))
            improvements[i] = result.value
        end
        sort!(ranks, by=i -> improvements[i], rev=true)

    elseif emitter.ranking == TWO_STAGE_IMPROVEMENT
        # First stage: feasibility, Second stage: improvement
        feasible = fill(true, n)
        improvements = Vector{M}(undef, n)
        for i in 1:n
            result = add!(emitter.archive, view(solutions, :, i), objectives[i], view(measures, :, i))
            feasible[i] = result.status != NOT_ADDED
            improvements[i] = result.value
        end
        sort!(ranks, by=i -> (feasible[i], improvements[i]), rev=true)

    elseif emitter.ranking in (RANDOM_DIRECTION, TWO_STAGE_RANDOM_DIRECTION)
        # Generate random direction if needed
        if isnothing(emitter.random_direction)
            emitter.random_direction = randn(emitter.rng, size(solutions, 1))
            emitter.random_direction ./= norm(emitter.random_direction)
        end

        # Project solutions onto random direction
        projections = [dot(view(solutions, :, i), emitter.random_direction) for i in 1:n]

        if emitter.ranking == RANDOM_DIRECTION
            sort!(ranks, by=i -> projections[i], rev=true)
        else  # TWO_STAGE_RANDOM_DIRECTION
            feasible = fill(true, n)
            for i in 1:n
                result = add!(emitter.archive, view(solutions, :, i), objectives[i], view(measures, :, i))
                feasible[i] = result.status != NOT_ADDED
            end
            sort!(ranks, by=i -> (feasible[i], projections[i]), rev=true)
        end

    else  # OBJECTIVE or TWO_STAGE_OBJECTIVE
        if emitter.ranking == OBJECTIVE
            sort!(ranks, by=i -> objectives[i], rev=true)
        else  # TWO_STAGE_OBJECTIVE
            feasible = fill(true, n)
            for i in 1:n
                result = add!(emitter.archive, view(solutions, :, i), objectives[i], view(measures, :, i))
                feasible[i] = result.status != NOT_ADDED
            end
            sort!(ranks, by=i -> (feasible[i], objectives[i]), rev=true)
        end
    end

    return ranks
end

function select_parents(
    emitter::CMAESEmitter{S,M},
    solutions::AbstractMatrix{S},
    ranks::Vector{Int}
)::Vector{Int} where {S,M}
    if emitter.selection_rule == MU
        return ranks[1:emitter.μ]
    else  # FILTER
        # Select solutions that are not dominated by any other solution
        n = length(ranks)
        is_selected = fill(true, n)

        for i in 1:n
            if !is_selected[i]
                continue
            end
            xi = view(solutions, :, ranks[i])

            for j in (i+1):n
                if !is_selected[j]
                    continue
                end
                xj = view(solutions, :, ranks[j])

                # Check if xi dominates xj or vice versa
                if all(xi .<= xj) && any(xi .< xj)
                    is_selected[j] = false
                elseif all(xj .<= xi) && any(xj .< xi)
                    is_selected[i] = false
                    break
                end
            end
        end

        return ranks[is_selected]
    end
end

function ask!(emitter::CMAESEmitter{S,M}, n::Integer)::Matrix{S} where {S,M}
    sol_dim = solution_dim(emitter.archive)
    solutions = Matrix{S}(undef, sol_dim, n)

    # Generate samples
    for i in 1:n
        z = randn(emitter.rng, sol_dim)
        x = emitter.mean + emitter.σ * (emitter.B * (emitter.D .* z))
        solutions[:, i] .= clamp.(x, emitter.lower_bounds, emitter.upper_bounds)
    end

    return solutions
end

function tell!(
    emitter::CMAESEmitter{S,M},
    solutions::AbstractMatrix{S},
    objectives::AbstractVector{M},
    measures::AbstractMatrix{M}
)::Nothing where {S,M}
    emitter.generation += 1
    sol_dim = solution_dim(emitter.archive)

    # Rank solutions and select parents
    ranks = rank_solutions(emitter, solutions, objectives, measures)
    selected = select_parents(emitter, solutions, ranks)

    # Calculate weighted mean of selected solutions
    old_mean = copy(emitter.mean)
    emitter.mean .= 0
    for (w, i) in zip(emitter.weights, selected)
        emitter.mean .+= w .* view(solutions, :, i)
    end

    # Update evolution paths
    y = (emitter.mean - old_mean) / emitter.σ
    C_2 = emitter.B * Diagonal(emitter.D) * emitter.B'
    C_2_inv = emitter.B * Diagonal(1 ./ emitter.D) * emitter.B'

    # Update pσ
    emitter.pσ = (1 - emitter.cσ) .* emitter.pσ +
                 sqrt(emitter.cσ * (2 - emitter.cσ) * emitter.μeff) .* (C_2_inv * y)

    # Update pc
    hsig = norm(emitter.pσ) / sqrt(1 - (1 - emitter.cσ)^(2 * emitter.generation)) <
           (1.4 + 2 / (sol_dim + 1)) * emitter.chiN
    emitter.pc = (1 - emitter.cc) .* emitter.pc +
                 hsig * sqrt(emitter.cc * (2 - emitter.cc) * emitter.μeff) .* y

    # Update covariance matrix
    w_io = [i <= length(emitter.weights) ? emitter.weights[i] :
            -emitter.c1 / emitter.cμ / (sol_dim + 2) for i in 1:length(selected)]

    delta_h = hsig ? 0 : emitter.cc * (2 - emitter.cc)
    rank_one = emitter.pc * emitter.pc'
    rank_μ = sum(w * (x - old_mean) * (x - old_mean)' / emitter.σ^2
                 for (w, i, x) in zip(w_io, selected, eachcol(solutions)))

    emitter.C = (1 - emitter.c1 - emitter.cμ + (1 - hsig) * emitter.c1) * emitter.C +
                emitter.c1 * rank_one +
                emitter.cμ * rank_μ

    # Update step size
    emitter.σ *= exp((emitter.cσ / emitter.dσ) * (norm(emitter.pσ) / emitter.chiN - 1))

    # Update B and D through eigendecomposition of C
    F = eigen(Symmetric(emitter.C))
    emitter.D .= sqrt.(max.(real.(F.values), 0))
    emitter.B .= real.(F.vectors)

    # Check for improvement and restart if needed
    if any(i -> add!(emitter.archive, view(solutions, :, i), objectives[i],
            view(measures, :, i)).status in (NEW, IMPROVE), 1:size(solutions, 2))
        emitter.last_improvement = emitter.generation
    elseif emitter.generation - emitter.last_improvement >= emitter.restart_rule
        # Reset to initial state
        emitter.C .= I(sol_dim)
        emitter.B .= I(sol_dim)
        emitter.D .= 1
        emitter.pc .= 0
        emitter.pσ .= 0
        emitter.mean .= emitter.x0
        emitter.σ = emitter.σ0
        emitter.random_direction = nothing
    end

    return nothing
end
