# ADR-0003: PyTorch as Primary Backend

## Status
Accepted

## Context
Neural operators require automatic differentiation for physics-informed constraints and efficient GPU computation for large-scale problems. The choice of deep learning framework affects ease of development, performance, and ecosystem compatibility.

Main options considered:
- PyTorch: Dynamic graphs, strong research ecosystem, good autodiff
- JAX: Functional programming, XLA compilation, good for scientific computing
- TensorFlow: Production-ready, but more complex for research

## Decision
Use PyTorch as the primary backend for neural operator implementations.

Key factors:
- Dominant framework in neural operator research community
- Excellent automatic differentiation for physics-informed losses
- Dynamic computation graphs suit variable-size PDE domains
- Strong ecosystem for uncertainty quantification (GPyTorch, Pyro)
- Good GPU performance with straightforward optimization

## Consequences

### Positive
- Leverages existing neural operator implementations
- Easy integration with uncertainty libraries (GPyTorch, Pyro)
- Dynamic graphs support flexible PDE formulations
- Strong community and extensive documentation
- Good debugging experience for research

### Negative
- Potentially slower than JAX for some scientific computing workloads
- Less functional programming paradigm compared to JAX
- Dynamic nature can make some optimizations harder

### Mitigation
- Implement key performance-critical components efficiently
- Consider JAX backend for specific high-performance applications
- Use PyTorch's JIT compilation where appropriate
- Provide clear interfaces to enable future backend alternatives