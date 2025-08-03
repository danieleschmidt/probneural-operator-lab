# ADR-0004: Hierarchical Package Structure

## Status
Accepted

## Context
The framework needs to support multiple neural operator architectures, uncertainty methods, and application domains. A clear package structure is essential for maintainability, discoverability, and extensibility.

We need to balance:
- Logical grouping of related functionality
- Ease of imports for common use cases
- Flexibility for advanced users
- Clear separation between core and domain-specific code

## Decision
Adopt a hierarchical package structure organized by functionality:

```
probneural_operator/
├── models/              # Neural operator architectures
│   ├── base/           # Base classes and interfaces
│   ├── fno/            # Fourier Neural Operator
│   ├── deeponet/       # Deep Operator Network
│   ├── gno/            # Graph Neural Operator
│   └── pino/           # Physics-Informed Neural Operator
├── posteriors/         # Uncertainty quantification methods
│   ├── base/           # Base posterior classes
│   ├── laplace/        # Laplace approximations
│   ├── variational/    # Variational inference
│   └── ensemble/       # Deep ensembles
├── active/             # Active learning
│   ├── acquisition/    # Acquisition functions
│   ├── strategies/     # Selection strategies
│   └── optimization/   # Acquisition optimization
├── calibration/        # Uncertainty calibration
│   ├── temperature/    # Temperature scaling
│   ├── isotonic/       # Isotonic regression
│   └── metrics/        # Calibration metrics
├── applications/       # Domain-specific implementations
│   ├── fluids/         # Fluid dynamics
│   ├── materials/      # Material science
│   └── climate/        # Climate modeling
├── data/              # Data handling and datasets
├── utils/             # Utility functions
└── visualization/     # Plotting and visualization
```

## Consequences

### Positive
- Clear logical organization by functionality
- Easy to find relevant modules
- Supports both simple and complex use cases
- Facilitates testing and documentation
- Enables domain experts to focus on relevant subpackages

### Negative
- May lead to deeper import paths
- Potential for circular dependencies
- Need to maintain consistency across subpackages

### Mitigation
- Provide convenience imports at top level for common classes
- Use dependency injection to avoid circular imports
- Establish clear interface contracts between packages
- Document import patterns and best practices