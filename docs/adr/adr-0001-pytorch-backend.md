# ADR-0001: PyTorch as Primary Backend

**Date**: 2025-08-02  
**Status**: Accepted  
**Deciders**: Core Development Team

## Context

ProbNeural-Operator-Lab requires a deep learning framework backend to implement neural operators and uncertainty quantification methods. The choice of backend significantly impacts:
- Developer experience and ecosystem compatibility
- Performance characteristics
- Community adoption and contributions
- Integration with existing scientific computing workflows

The project needs support for:
- Automatic differentiation for gradient-based optimization
- GPU acceleration for large-scale computations
- Advanced mathematical operations (eigenvalue decomposition, matrix factorizations)
- Probabilistic programming capabilities
- Research-friendly experimentation features

## Decision

We will use **PyTorch** as the primary deep learning backend for ProbNeural-Operator-Lab.

## Alternatives Considered

### JAX
**Pros:**
- Functional programming paradigm
- JIT compilation with XLA
- Excellent performance for scientific computing
- Clean mathematical abstractions
- Growing ecosystem (Flax, Optax, etc.)

**Cons:**
- Steeper learning curve for many developers
- Smaller ecosystem compared to PyTorch
- Less mature tooling for model deployment
- Functional paradigm may be unfamiliar

### TensorFlow/Keras
**Pros:**
- Mature ecosystem and tooling
- Strong production deployment support
- TensorFlow Probability for probabilistic programming
- Extensive documentation and tutorials

**Cons:**
- More complex API surface
- Graph-based execution model less intuitive
- Declining research community adoption
- Eager execution still feels secondary

### NumPy + SciPy (Pure Python)
**Pros:**
- Maximum control over implementations
- No deep learning framework overhead
- Clear mathematical implementations
- Universal compatibility

**Cons:**
- No automatic differentiation
- Manual GPU implementation required
- Limited scalability for large models
- Significant development overhead

## Consequences

### Positive
- **Research Community Alignment**: PyTorch is widely adopted in neural operator research
- **Automatic Differentiation**: Built-in autograd enables easy gradient computation for uncertainty methods
- **Flexible Development**: Dynamic computation graphs support experimental research
- **Rich Ecosystem**: Access to libraries like Botorch (for Bayesian optimization), GPyTorch (for Gaussian processes)
- **GPU Support**: Seamless CUDA integration with minimal code changes
- **Probabilistic Programming**: Integration with Pyro for advanced Bayesian methods

### Negative
- **Performance**: May be slower than JAX for some numerical computations
- **Memory Usage**: Dynamic graphs can have higher memory overhead
- **JIT Limitations**: TorchScript limitations compared to JAX's JIT compilation
- **Deployment Complexity**: Production deployment more complex than TensorFlow Serving

### Neutral
- **Learning Curve**: Most target users already familiar with PyTorch
- **Breaking Changes**: PyTorch has good backward compatibility practices
- **Maintenance**: Active development with regular releases

## Implementation Notes

### Core Dependencies
```python
torch >= 2.0.0  # For improved compilation and memory efficiency
torchvision      # For computer vision utilities (if needed)
```

### Optional Integrations
```python
pyro-ppl         # For advanced probabilistic programming
botorch          # For Bayesian optimization in active learning
gpytorch         # For Gaussian process components
torchdiffeq      # For neural ODEs (if needed for physics applications)
```

### Architecture Patterns
- Use `torch.nn.Module` for all neural operator implementations
- Leverage `torch.autograd` for uncertainty quantification gradients
- Implement custom autograd functions for efficient Hessian computations
- Use `torch.jit.script` for performance-critical components

### GPU Strategy
- Default to CPU with easy GPU migration via `.to(device)`
- Support mixed precision training with `torch.cuda.amp`
- Implement memory-efficient attention patterns for large operators

### Testing Strategy
- Test numerical correctness against analytical solutions
- Benchmark performance across different hardware configurations
- Ensure gradient correctness with `torch.autograd.gradcheck`

### Future Considerations
- Monitor JAX ecosystem development for potential future migration
- Maintain clean abstractions to enable backend swapping if needed
- Consider implementing JAX backend as secondary option in v2.0+

## References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Neural Operator Research Papers using PyTorch](https://github.com/neuraloperator/neuraloperator)
- [Uncertainty Quantification with PyTorch](https://pytorch.org/tutorials/intermediate/uncertainty_quantification.html)
- [Scientific Computing with PyTorch](https://pytorch.org/ecosystem/)