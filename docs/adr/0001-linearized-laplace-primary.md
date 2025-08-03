# ADR-0001: Use Linearized Laplace for Primary Uncertainty Method

## Status
Accepted

## Context
Neural operators require uncertainty quantification for scientific computing applications, but traditional methods like deep ensembles are computationally expensive and variational inference can be difficult to tune properly.

The recent ICML 2025 work on linearized Laplace approximation provides a principled, efficient approach to uncertainty quantification that maintains theoretical guarantees while being computationally tractable for large neural operators.

## Decision
We will use linearized Laplace approximation as the primary uncertainty quantification method for probabilistic neural operators, with other methods (variational inference, ensembles) available as alternatives.

Key aspects:
- Linearized Laplace provides efficient posterior approximation
- Kronecker-factored Hessian approximation for scalability
- Post-hoc application to pre-trained models
- Well-calibrated uncertainty estimates

## Consequences

### Positive
- Efficient uncertainty quantification with minimal computational overhead
- Theoretical foundation from Bayesian deep learning
- Can be applied to existing pre-trained models
- Good calibration properties for scientific applications
- Enables marginal likelihood computation for model selection

### Negative
- Requires Hessian computation which can be memory intensive
- Gaussian approximation may not capture complex posterior shapes
- Limited to second-order approximations
- May require careful hyperparameter tuning for prior precision

### Mitigation
- Implement efficient Hessian approximation techniques
- Provide ensemble and variational alternatives for comparison
- Include calibration methods to validate uncertainty quality