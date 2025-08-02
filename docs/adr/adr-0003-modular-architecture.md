# ADR-0003: Modular Component Architecture

**Date**: 2025-08-02  
**Status**: Accepted  
**Deciders**: Core Development Team

## Context

ProbNeural-Operator-Lab needs an architecture that supports:
- Multiple neural operator types (FNO, DeepONet, GNO, etc.)
- Various uncertainty quantification methods (Laplace, variational, ensembles)
- Different active learning strategies and acquisition functions
- Extensibility for research and new method development
- Clear separation of concerns for maintainability

The framework must balance flexibility with usability, allowing researchers to easily experiment with new components while providing stable interfaces for production use.

## Decision

We will implement a **modular component architecture** with clear interfaces between neural operators, posterior approximations, active learning, and calibration components.

## Alternatives Considered

### Monolithic Design
**Pros:**
- Simpler initial implementation
- Tighter integration between components
- Easier to optimize end-to-end performance

**Cons:**
- Difficult to extend with new methods
- High coupling between components
- Hard to test individual components
- Limited reusability across different use cases

### Plugin-Based Architecture
**Pros:**
- Maximum flexibility and extensibility
- Clear component boundaries
- Easy to add new implementations

**Cons:**
- Higher complexity and abstraction overhead
- Potential performance penalties
- More difficult for users to understand
- Versioning and compatibility challenges

### Inheritance-Heavy Design
**Pros:**
- Familiar object-oriented patterns
- Code reuse through inheritance
- Polymorphic behavior

**Cons:**
- Deep inheritance hierarchies hard to maintain
- Tight coupling between base and derived classes
- Difficult to compose different capabilities
- Fragile base class problem

## Decision Rationale

### Modular Architecture Benefits
1. **Separation of Concerns**: Each module has a single, well-defined responsibility
2. **Testability**: Components can be tested in isolation
3. **Extensibility**: New implementations can be added without modifying existing code
4. **Composition**: Users can mix and match components as needed
5. **Maintainability**: Changes to one component don't affect others

### Design Principles
- **Interface-based Design**: Define clear contracts between components
- **Dependency Injection**: Components receive dependencies rather than creating them
- **Factory Patterns**: Centralized creation of configured components
- **Registry Patterns**: Dynamic discovery and instantiation of implementations

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Neural Operators   │    │   Posterior Approx  │    │   Active Learning   │
│                 │    │                 │    │                 │
│ - FNO           │    │ - Laplace       │    │ - Acquisition   │
│ - DeepONet      │    │ - Variational   │    │ - Strategies    │
│ - GNO           │◄───┤ - Ensemble      │◄───┤ - Optimization  │
│ - PINO          │    │ - Dropout       │    │ - Multi-fidelity│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────┐
                    │   Calibration   │
                    │                 │
                    │ - Temperature   │
                    │ - Isotonic      │
                    │ - Metrics       │
                    └─────────────────┘
```

## Implementation Details

### Core Interfaces

#### Neural Operator Interface
```python
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch

class NeuralOperator(ABC):
    """Base interface for neural operators."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the operator."""
        pass
    
    @abstractmethod
    def parameters(self) -> Iterator[torch.Tensor]:
        """Return model parameters."""
        pass
    
    def train(self, mode: bool = True) -> 'NeuralOperator':
        """Set training mode."""
        return self
```

#### Posterior Approximation Interface
```python
class PosteriorApproximation(ABC):
    """Base interface for posterior approximations."""
    
    @abstractmethod
    def fit(self, model: NeuralOperator, data_loader: DataLoader) -> None:
        """Fit the posterior approximation."""
        pass
    
    @abstractmethod
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with uncertainty (mean, variance)."""
        pass
    
    @abstractmethod
    def sample(self, x: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Sample from posterior predictive."""
        pass
```

#### Active Learning Interface
```python
class AcquisitionFunction(ABC):
    """Base interface for acquisition functions."""
    
    @abstractmethod
    def __call__(self, model: PosteriorApproximation, x: torch.Tensor) -> torch.Tensor:
        """Compute acquisition scores."""
        pass

class ActiveLearningStrategy(ABC):
    """Base interface for active learning strategies."""
    
    @abstractmethod
    def select_batch(self, 
                    pool: torch.Tensor, 
                    acquisition: AcquisitionFunction,
                    batch_size: int) -> torch.Tensor:
        """Select batch of points for labeling."""
        pass
```

### Factory Pattern Implementation

```python
class PosteriorFactory:
    """Factory for creating posterior approximations."""
    
    _registry = {
        'laplace': LinearizedLaplace,
        'variational': VariationalPosterior,
        'ensemble': DeepEnsemble,
        'dropout': MonteCarloDropout
    }
    
    @classmethod
    def create(cls, method: str, **kwargs) -> PosteriorApproximation:
        """Create posterior approximation instance."""
        if method not in cls._registry:
            raise ValueError(f"Unknown posterior method: {method}")
        return cls._registry[method](**kwargs)
    
    @classmethod
    def register(cls, name: str, posterior_class: type) -> None:
        """Register new posterior approximation."""
        cls._registry[name] = posterior_class
```

### Configuration System

```python
@dataclass
class ModelConfig:
    """Configuration for neural operator models."""
    operator_type: str = "fno"
    operator_kwargs: dict = field(default_factory=dict)
    posterior_type: str = "laplace"
    posterior_kwargs: dict = field(default_factory=dict)
    
@dataclass
class ActiveLearningConfig:
    """Configuration for active learning."""
    acquisition_function: str = "bald"
    strategy: str = "greedy"
    batch_size: int = 10
    budget: int = 100
```

## Consequences

### Positive
- **Flexibility**: Easy to experiment with different combinations of components
- **Testability**: Each component can be unit tested independently
- **Extensibility**: New methods can be added without modifying existing code
- **Maintainability**: Clear boundaries reduce coupling and complexity
- **Reusability**: Components can be reused across different applications
- **Documentation**: Clear interfaces make behavior explicit

### Negative
- **Initial Complexity**: More upfront design and implementation effort
- **Performance Overhead**: Abstraction layers may introduce small performance costs
- **Learning Curve**: Users need to understand the component model
- **Over-Engineering Risk**: May be more complex than needed for simple use cases

### Neutral
- **Configuration**: Need robust configuration management system
- **Backwards Compatibility**: Interface changes require careful versioning
- **Error Handling**: Need consistent error handling across components

## Implementation Strategy

### Phase 1: Core Interfaces
1. Define base interfaces for all major components
2. Implement basic concrete classes for each interface
3. Create factory classes for component instantiation
4. Add comprehensive unit tests for interfaces

### Phase 2: Integration Framework
1. Implement high-level orchestration classes
2. Add configuration management system
3. Create integration tests
4. Develop usage examples and tutorials

### Phase 3: Advanced Features
1. Add plugin discovery mechanisms
2. Implement component validation and compatibility checking
3. Add performance monitoring and profiling
4. Create developer documentation for extending the framework

## Testing Strategy

### Component Testing
- Unit tests for each interface implementation
- Mock objects for testing component interactions
- Property-based testing for mathematical correctness
- Performance benchmarks for each component

### Integration Testing
- End-to-end workflows with different component combinations
- Configuration validation testing
- Error handling and recovery testing
- Backwards compatibility testing

## Documentation Requirements

### User Documentation
- Component overview and selection guide
- Configuration examples and best practices
- Tutorial for common workflows
- Performance tuning guide

### Developer Documentation
- Interface specifications and contracts
- Extension development guide
- Testing conventions and tools
- Architecture decision rationale

## References

- [Clean Architecture by Robert Martin](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [Design Patterns: Elements of Reusable Object-Oriented Software](https://en.wikipedia.org/wiki/Design_Patterns)
- [Modular Design Principles](https://en.wikipedia.org/wiki/Modular_design)
- [Dependency Injection Patterns](https://martinfowler.com/articles/injection.html)