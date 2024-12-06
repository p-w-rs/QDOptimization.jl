using Test
using Random
using QDOptimization

@testset "QDOptimization Tests" begin
    @testset "GridArchive" begin
        # Test construction and basic properties
        @testset "Construction" begin
            archive = GridArchive{Float64,Float64}(
                2,                              # solution_dim
                (10, 10),                       # measure_dims
                [(0.0, 1.0), (0.0, 1.0)],      # measure_ranges
                learning_rate=0.5,
                threshold_min=-1.0
            )

            @test solution_dim(archive) == 2
            @test measure_dims(archive) == (10, 10)
            @test cells(archive) == 100
            @test isempty(archive)
            @test length(archive) == 0
            @test coverage(archive) == 0.0

            # Test invalid construction
            @test_throws ArgumentError GridArchive{Float64,Float64}(
                2,
                (10,),  # Mismatched dimensions
                [(0.0, 1.0), (0.0, 1.0)]
            )

            @test_throws ArgumentError GridArchive{Float64,Float64}(
                2,
                (10, 10),
                [(1.0, 0.0), (0.0, 1.0)]  # Invalid range
            )
        end

        @testset "Add and Retrieve" begin
            archive = GridArchive{Float64,Float64}(
                2,
                (10, 10),
                [(0.0, 1.0), (0.0, 1.0)]
            )

            solution = [0.5, 0.5]
            measure = [0.3, 0.3]
            objective = 1.0

            # Test adding solution
            result = add!(archive, solution, objective, measure)
            @test result.status == NEW
            @test !isempty(archive)
            @test length(archive) == 1

            # Test retrieving solution
            found, elite = get(archive, measure)
            @test found
            @test elite.solution == solution
            @test elite.objective == objective
            @test elite.measure == measure

            # Test invalid additions
            @test_throws DimensionMismatch add!(archive, [0.5], objective, measure)  # Wrong solution dim
            @test_throws DimensionMismatch add!(archive, solution, objective, [0.3]) # Wrong measure dim
        end

        @testset "Clear and Sample" begin
            archive = GridArchive{Float64,Float64}(2, (10, 10), [(0.0, 1.0), (0.0, 1.0)])
            add!(archive, [0.5, 0.5], 1.0, [0.3, 0.3])

            clear!(archive)
            @test isempty(archive)
            @test length(archive) == 0

            # Test sampling from empty archive
            @test_throws ArgumentError sample(archive, 1)
        end
    end

    @testset "GaussianEmitter" begin
        @testset "Construction" begin
            archive = GridArchive{Float64,Float64}(2, (10, 10), [(0.0, 1.0), (0.0, 1.0)])

            # Test valid constructions
            emitter = GaussianEmitter{Float64,Float64}(
                archive,
                σ=[0.1, 0.1],
                x0=[0.5, 0.5],
                bounds=(0.0, 1.0)
            )
            @test isa(emitter, GaussianEmitter)

            # Test invalid constructions
            @test_throws ArgumentError GaussianEmitter{Float64,Float64}(
                archive,
                σ=[0.1],  # Wrong σ dimension
                x0=[0.5, 0.5]
            )

            @test_throws ArgumentError GaussianEmitter{Float64,Float64}(
                archive,
                σ=[0.1, 0.1],
                x0=[0.5]  # Wrong x0 dimension
            )
        end

        @testset "Ask and Tell" begin
            archive = GridArchive{Float64,Float64}(2, (10, 10), [(0.0, 1.0), (0.0, 1.0)])
            emitter = GaussianEmitter{Float64,Float64}(
                archive,
                σ=[0.1, 0.1],
                x0=[0.5, 0.5],
                bounds=(0.0, 1.0)
            )

            # Test ask
            solutions = ask!(emitter, 5)
            @test size(solutions) == (2, 5)
            @test all(0.0 .<= solutions .<= 1.0)  # Check bounds

            # Test tell
            objectives = ones(5)
            measures = repeat([0.5, 0.5], 1, 5)
            @test nothing === tell!(emitter, solutions, objectives, measures)

            # Test invalid tell
            @test_throws ArgumentError tell!(
                emitter,
                solutions,
                ones(4),  # Wrong number of objectives
                measures
            )
        end
    end

    @testset "RoundRobinScheduler" begin
        @testset "Construction" begin
            archive = GridArchive{Float64,Float64}(2, (10, 10), [(0.0, 1.0), (0.0, 1.0)])
            emitter = GaussianEmitter{Float64,Float64}(
                archive,
                σ=[0.1, 0.1],
                x0=[0.5, 0.5],
                bounds=(0.0, 1.0)
            )

            # Test single emitter
            scheduler = RoundRobinScheduler(emitter, batch_size=10)
            @test isa(scheduler, RoundRobinScheduler)

            # Test multiple emitters
            scheduler = RoundRobinScheduler([emitter, emitter], batch_size=10)
            @test isa(scheduler, RoundRobinScheduler)
        end

        @testset "Run" begin
            archive = GridArchive{Float64,Float64}(2, (10, 10), [(0.0, 1.0), (0.0, 1.0)])
            emitter = GaussianEmitter{Float64,Float64}(
                archive,
                σ=[0.1, 0.1],
                x0=[0.5, 0.5],
                bounds=(0.0, 1.0)
            )
            scheduler = RoundRobinScheduler(emitter, batch_size=10)

            # Define simple objective and measure functions
            objective_fn(x) = sum(x)
            measure_fn(x) = x

            # Test run
            @test nothing === run!(
                scheduler,
                objective_fn,
                measure_fn,
                100,
                parallel=false,
                show_progress=false
            )

            # Test parallel run
            @test nothing === run!(
                scheduler,
                objective_fn,
                measure_fn,
                100,
                parallel=true,
                show_progress=false
            )
        end
    end
end