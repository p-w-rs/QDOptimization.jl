# QDOptimization.jl

A Julia package for Quality-Diversity Optimization algorithms, providing flexible components for building QD optimization systems. While inspired by pyribs, it offers its own unique implementation and features.

## Features

- **Grid-based Archives**: Store and manage solutions in discretized measure spaces
- **Flexible Emitters**: Generate candidate solutions using various strategies
- **Customizable Schedulers**: Control optimization flow and emitter coordination

## Installation

```julia
using Pkg
Pkg.add("QDOptimization")
```

## Quick Start

```julia
using QDOptimization

# Create an archive
archive = GridArchive{Float64,Float64}(
    2,                              # solution dimension
    (10, 10),                       # measure space dimensions
    [(0.0, 1.0), (0.0, 1.0)]       # measure ranges
)

# Create multiple emitters
emitter1 = GaussianEmitter{Float64,Float64}(
    archive,
    σ=[0.1, 0.1],
    x0=[0.5, 0.5],
    bounds=(0.0, 1.0)
)

emitter2 = GaussianEmitter{Float64,Float64}(
    archive,
    σ=[0.2, 0.2],
    x0=[0.3, 0.3],
    bounds=(0.0, 1.0)
)

# Create scheduler with multiple emitters
scheduler = RoundRobinScheduler([emitter1, emitter2], batch_size=10)

# Define objective and measure functions
objective_fn(x) = (objective = sum(x), measure = x)

# Run optimization
run!(
    scheduler,
    objective_fn,
    1000,                # total evaluations
    parallel=true,       # enable parallel evaluation
    show_progress=true   # show progress updates
)
```

## Components

### Archives
- ```GridArchive```: Divides measure space into uniform cells
- (More archive types coming soon from pyribs)

### Emitters
- ```GaussianEmitter```: Generates solutions using Gaussian perturbation
- ```IsoLineEmitter```: Generates solutions along iso-lines in measure space
- ```CMA-ES```: Generates solutions using Covariance Matrix Adaptation Evolution Strategy
- Coming soon:
  - Other non-gradient based emitters from pyribs

### Schedulers
- ```RoundRobinScheduler```: Alternates between emitters in round-robin fashion
- ```BanditScheduler```: Chooses emitters based on performance bandit strategy using Thompson sampling

## Flexibility

A key feature of QDOptimization.jl is its ability to mix and match different components:

- Use different emitter types with the same archive
- Combine multiple emitters in a single scheduler
- Mix emitters with different parameters for diverse exploration strategies

## Differences from pyribs

While inspired by pyribs, QDOptimization.jl:
- Has its own unique implementation
- May have different interfaces and behaviors
- Focuses on Julia-specific optimizations and features
- Is not intended as a direct port or replacement

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
