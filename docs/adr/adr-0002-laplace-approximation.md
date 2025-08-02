# ADR-0002: Linearized Laplace as Primary Uncertainty Method

**Date**: 2025-08-02  
**Status**: Accepted  
**Deciders**: Core Development Team, Research Advisors

## Context

ProbNeural-Operator-Lab requires a primary uncertainty quantification method for neural operators. The choice significantly impacts:
- Computational efficiency and scalability
- Quality of uncertainty estimates
- Implementation complexity
- Research alignment with recent advances

Key requirements:
- Efficient uncertainty quantification for large neural operators
- Well-calibrated confidence intervals
- Theoretical grounding and interpretability
- Compatibility with active learning workflows
- Minimal computational overhead compared to deterministic training

Recent research (ICML 2025) demonstrates that linearized Laplace approximations can provide high-quality uncertainty estimates for neural operators with significantly lower computational cost than alternatives.

## Decision

We will implement **Linearized Laplace Approximation** as the primary uncertainty quantification method, with other methods (variational inference, ensembles) as secondary options.

## Alternatives Considered

### Deep Ensembles
**Pros:**
- Simple to implement and understand
- Often provides high-quality uncertainty estimates
- No modification to training procedure required
- Well-established in the literature

**Cons:**
- Computationally expensive (5-10x training and inference cost)
- Memory intensive for large models
- No theoretical guarantees
- May underestimate uncertainty in some cases

### Monte Carlo Dropout
**Pros:**
- Minimal computational overhead
- Easy to implement
- Works with existing architectures

**Cons:**
- Poor theoretical justification
- Often provides poorly calibrated uncertainties
- Requires careful hyperparameter tuning
- May interfere with batch normalization

### Variational Inference (Mean-Field)
**Pros:**
- Principled Bayesian approach
- Single forward pass inference
- Theoretical guarantees

**Cons:**
- Mean-field assumption often too restrictive
- Complex implementation for neural operators
- Optimization can be unstable
- May underestimate uncertainty

### Full Bayesian Neural Networks (MCMC)
**Pros:**
- Most principled approach
- No approximation assumptions
- Gold standard for uncertainty

**Cons:**
- Computationally prohibitive for large models
- Convergence diagnostics challenging
- Implementation complexity
- Not practical for real-time applications

## Decision Rationale

### Linearized Laplace Advantages
1. **Efficiency**: Near-deterministic computational cost
2. **Quality**: Recent research shows competitive uncertainty quality
3. **Scalability**: Efficient approximations for large neural operators
4. **Theoretical Foundation**: Well-grounded in Bayesian statistics
5. **Implementation**: Moderate complexity with clear mathematical foundation

### Supporting Research
- ICML 2025 paper demonstrates superior performance on neural operator benchmarks
- Kronecker-factored approximations enable scaling to large models
- Linearization around mode provides good local approximation for neural operators

## Consequences

### Positive
- **Performance**: Minimal computational overhead compared to ensembles
- **Quality**: High-quality uncertainty estimates validated on neural operator tasks
- **Scalability**: Can handle large neural operators efficiently
- **Interpretability**: Clear mathematical interpretation of uncertainty sources
- **Active Learning**: Efficient gradient-based acquisition functions
- **Research Alignment**: Builds on cutting-edge uncertainty quantification research

### Negative
- **Implementation Complexity**: More complex than dropout or ensembles
- **Hessian Computation**: Requires efficient approximations for large models
- **Local Approximation**: May be poor for highly non-linear posteriors
- **Hyperparameter Sensitivity**: Requires careful prior selection

### Neutral
- **Novelty**: Relatively new method may require user education
- **Validation**: Need comprehensive benchmarking against established methods

## Implementation Notes

### Core Components

#### Hessian Approximations
```python
# Kronecker-factored Laplace approximation
class KroneckerLaplace:
    def fit(self, model, train_loader):
        # Compute Kronecker factors of Hessian
        # A ⊗ B approximation for efficiency
        
    def predict(self, x):
        # Linear approximation around mode
        # μ + Φ(x)^T Σ ε
```

#### Efficient Implementation
- Use Kronecker-factored approximations for convolutional layers
- Block-diagonal approximations for fully connected layers
- Lazy evaluation for memory efficiency
- Vectorized operations for batch predictions

### Integration Strategy

#### Model Interface
```python
class ProbabilisticNeuralOperator:
    def __init__(self, base_model, posterior_type="laplace"):
        self.posterior = self._create_posterior(posterior_type)
    
    def fit_posterior(self, train_data):
        self.posterior.fit(self.base_model, train_data)
    
    def predict_with_uncertainty(self, x):
        return self.posterior.predict(x)
```

#### Hyperparameter Selection
- Prior precision: Cross-validation or marginal likelihood
- Hessian structure: Based on model architecture
- Linearization point: MAP estimate from standard training

### Performance Considerations
- Memory-efficient Hessian computation using backpropagation
- Precompute and cache factors when possible
- Support for mixed precision to reduce memory usage
- Batch processing for prediction efficiency

### Validation Strategy
- Compare against analytical solutions on simple problems
- Benchmark against ensemble methods on standard datasets
- Verify calibration using reliability diagrams
- Test scalability on progressively larger models

## Future Extensions

### Advanced Approximations
- **Full-rank approximations**: For critical applications requiring highest quality
- **Hierarchical priors**: For structured uncertainty in multi-scale problems
- **Online updates**: For continual learning scenarios

### Specialized Implementations
- **Physics-informed priors**: Incorporating PDE structure
- **Multi-fidelity**: Combining different approximation levels
- **Adaptive linearization**: Re-linearizing for improved approximations

## References

- [ICML 2025: Linearization for Probabilistic Neural Operators](https://example.com/icml2025)
- [Laplace Redux: Effortless Bayesian Deep Learning](https://arxiv.org/abs/2106.14806)
- [Kronecker-factored approximations](https://arxiv.org/abs/1503.05671)
- [Neural Operator Uncertainty Quantification Survey](https://example.com/survey)