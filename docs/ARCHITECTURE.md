# Architecture Overview

## Core Design Principles

1. **Modular Design**: Independent components for models, posteriors, and active learning
2. **Extensibility**: Easy to add new neural operators and uncertainty methods
3. **Efficiency**: Optimized implementations with minimal computational overhead
4. **Flexibility**: Support for multiple backends and configuration options

## Component Architecture

### Neural Operator Models (`models/`)
- **Base Classes**: Common interfaces for all neural operators
- **Implementations**: FNO, DeepONet, GNO with probabilistic extensions
- **Training**: Unified training loops with uncertainty-aware losses

### Posterior Approximation (`posteriors/`)
- **Laplace**: Linearized Laplace approximation (primary method)
- **Variational**: Mean-field and full-rank variational inference
- **Ensemble**: Deep ensemble implementations

### Active Learning (`active/`)
- **Acquisition Functions**: BALD, variance-based, gradient-based
- **Strategies**: Batch selection and diversity enforcement
- **Optimization**: Efficient acquisition optimization

### Calibration (`calibration/`)
- **Temperature Scaling**: Post-hoc calibration
- **Metrics**: ECE, reliability diagrams, sharpness measures

## Data Flow

```
Input Data → Neural Operator → Posterior → Calibration → Predictions
     ↓              ↓            ↓            ↓
Active Learning ← Acquisition ← Uncertainty ← Metrics
```

## Key Interfaces

### Model Interface
```python
class ProbabilisticNeuralOperator:
    def fit(self, data, epochs, lr)
    def predict(self, x, return_std=True)
    def sample(self, x, n_samples)
    def log_marginal_likelihood()
```

### Posterior Interface
```python
class PosteriorApproximation:
    def fit(self, model, data)
    def predict(self, x)
    def sample(self, x, n_samples)
    def uncertainty(self, x)
```

## Performance Considerations

- **Memory**: Efficient Hessian approximations for large models
- **Computation**: Vectorized operations and GPU acceleration
- **Scalability**: Batch processing and distributed training support