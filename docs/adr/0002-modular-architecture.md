# ADR-0002: Modular Architecture for Neural Operators

## Status
Accepted

## Context
The field of neural operators is rapidly evolving with new architectures (FNO, DeepONet, GNO, PINO) and uncertainty methods. We need an architecture that allows easy experimentation and extension while maintaining clean separation of concerns.

Users may want to:
- Try different neural operator architectures
- Experiment with various uncertainty quantification methods
- Combine different acquisition functions for active learning
- Apply the framework to new application domains

## Decision
Implement a modular architecture with clear separation between:

1. **Neural Operator Models** (`models/`): Core architectures (FNO, DeepONet, etc.)
2. **Posterior Approximation** (`posteriors/`): Uncertainty methods (Laplace, VI, ensembles)
3. **Active Learning** (`active/`): Acquisition functions and strategies
4. **Calibration** (`calibration/`): Post-hoc uncertainty calibration
5. **Applications** (`applications/`): Domain-specific implementations

Each module provides common interfaces allowing mix-and-match composition.

## Consequences

### Positive
- Easy to add new neural operator architectures
- Uncertainty methods can be applied to any model
- Clear separation of concerns aids testing and maintenance
- Enables comparative studies across methods
- Facilitates collaboration by allowing independent development

### Negative
- Additional abstraction overhead
- May require more complex configuration
- Potential for interface mismatches between modules

### Mitigation
- Provide comprehensive base classes with clear contracts
- Include integration tests across module combinations
- Offer high-level convenience functions for common workflows
- Maintain detailed documentation of interfaces