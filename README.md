# ProbNeural-Operator-Lab

Framework for probabilistic neural operators with linearized Laplace approximation and active learning capabilities. Implements ICML 2025's "Linearization → Probabilistic NO" approach for uncertainty-aware PDE solving with neural operators.

## Overview

ProbNeural-Operator-Lab extends neural operators with principled uncertainty quantification through linearized Laplace approximations. This enables active learning, risk-aware decision making, and reliable extrapolation for scientific computing applications where understanding model confidence is crucial.

## Key Features

- **Probabilistic Neural Operators**: Uncertainty quantification for DeepONet, FNO, and GNO
- **Linearized Laplace**: Efficient posterior approximation with minimal overhead
- **Active Learning**: Optimal data acquisition for expensive simulations
- **Multi-Fidelity**: Combine high and low-fidelity data sources
- **Calibrated Uncertainty**: Well-calibrated confidence intervals
- **Physics-Informed**: Incorporate PDE constraints in uncertainty

## Installation

```bash
# Basic installation
pip install probneural-operator-lab

# With all backends
pip install probneural-operator-lab[full]

# With GPU support
pip install probneural-operator-lab[gpu]

# Development installation
git clone https://github.com/yourusername/probneural-operator-lab
cd probneural-operator-lab
pip install -e ".[dev]"
```

## Quick Start

### Basic Probabilistic FNO

```python
from probneural_operator import ProbabilisticFNO
from probneural_operator.datasets import NavierStokes

# Load dataset
dataset = NavierStokes(
    resolution=64,
    viscosity=1e-3,
    time_steps=50
)

# Create probabilistic FNO
prob_fno = ProbabilisticFNO(
    modes=12,
    width=32,
    depth=4,
    posterior="laplace",  # or "ensemble", "dropout"
    prior_precision=1.0
)

# Train with uncertainty
prob_fno.train(
    train_data=dataset.train,
    val_data=dataset.val,
    epochs=100,
    lr=1e-3
)

# Predict with uncertainty
mean, std = prob_fno.predict(
    initial_condition,
    return_std=True,
    num_samples=100
)

print(f"Prediction uncertainty: {std.mean():.3f}")
```

### Active Learning Loop

```python
from probneural_operator.active import ActiveLearner

# Initialize active learner
active_learner = ActiveLearner(
    model=prob_fno,
    acquisition="bald",  # Bayesian Active Learning by Disagreement
    budget=100
)

# Active learning loop
for iteration in range(10):
    # Select most informative points
    query_points = active_learner.query(
        candidate_pool=unlabeled_data,
        batch_size=10
    )
    
    # Run expensive simulation
    new_labels = run_simulation(query_points)
    
    # Update model
    active_learner.update(query_points, new_labels)
    
    print(f"Iteration {iteration}: Test error = {active_learner.test_error:.3f}")
```

## Architecture

```
probneural-operator-lab/
├── probneural_operator/
│   ├── models/
│   │   ├── deeponet/      # Probabilistic DeepONet
│   │   ├── fno/           # Probabilistic FNO
│   │   ├── gno/           # Probabilistic GNO
│   │   └── pino/          # Probabilistic PINO
│   ├── posteriors/
│   │   ├── laplace/       # Laplace approximations
│   │   ├── variational/   # Variational inference
│   │   └── ensemble/      # Deep ensembles
│   ├── priors/
│   │   ├── gaussian/      # Gaussian priors
│   │   ├── hierarchical/  # Hierarchical priors
│   │   └── physics/       # Physics-informed priors
│   ├── active/
│   │   ├── acquisition/   # Acquisition functions
│   │   ├── strategies/    # Selection strategies
│   │   └── optimization/  # Acquisition optimization
│   ├── calibration/
│   │   ├── temperature/   # Temperature scaling
│   │   ├── isotonic/      # Isotonic regression
│   │   └── metrics/       # Calibration metrics
│   └── applications/
│       ├── fluids/        # Fluid dynamics
│       ├── materials/     # Material science
│       └── climate/       # Climate modeling
├── benchmarks/            # Uncertainty benchmarks
├── experiments/          # Reproducible experiments
└── tutorials/            # Tutorial notebooks
```

## Uncertainty Quantification Methods

### Linearized Laplace Approximation

```python
from probneural_operator.posteriors import LinearizedLaplace

# Configure Laplace approximation
laplace = LinearizedLaplace(
    model=neural_operator,
    likelihood="regression",
    hessian_structure="kron",  # Kronecker-factored
    prior_precision=1.0
)

# Fit posterior
laplace.fit(train_loader)

# Posterior predictive
def predict_with_uncertainty(x):
    mean = laplace.mean(x)
    variance = laplace.variance(x)
    
    # Sample from posterior
    samples = laplace.sample(x, n_samples=1000)
    
    return mean, variance, samples

# Marginal likelihood for model selection
log_marginal_likelihood = laplace.log_marginal_likelihood()
```

### Variational Inference

```python
from probneural_operator.posteriors import VariationalPosterior

# Variational neural operator
var_model = VariationalPosterior(
    base_model=neural_operator,
    prior="gaussian",
    posterior_type="mean_field"  # or "full_rank", "low_rank"
)

# ELBO optimization
optimizer = torch.optim.Adam(var_model.parameters(), lr=1e-3)

for epoch in range(epochs):
    for x, y in train_loader:
        # Sample from variational posterior
        loss = var_model.elbo_loss(x, y, n_samples=5)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Deep Ensembles

```python
from probneural_operator.posteriors import DeepEnsemble

# Create ensemble
ensemble = DeepEnsemble(
    base_model_fn=lambda: FourierNeuralOperator(...),
    n_members=5,
    init_strategy="diverse"  # Different initializations
)

# Train ensemble members
ensemble.train(
    train_data=dataset,
    epochs=100,
    adversarial_training=True  # For better uncertainty
)

# Ensemble predictions
mean, epistemic_var, aleatoric_var = ensemble.predict(
    x,
    decompose_uncertainty=True
)
```

## Active Learning

### Acquisition Functions

```python
from probneural_operator.active import AcquisitionFunctions

# Various acquisition functions
acquisitions = {
    "variance": AcquisitionFunctions.max_variance,
    "bald": AcquisitionFunctions.bald,  # Mutual information
    "badge": AcquisitionFunctions.badge,  # Gradient diversity
    "lcmd": AcquisitionFunctions.lcmd,   # Loss prediction
}

# Custom physics-aware acquisition
def physics_aware_acquisition(model, x):
    # High uncertainty + PDE residual
    uncertainty = model.epistemic_uncertainty(x)
    residual = compute_pde_residual(model, x)
    return uncertainty + 0.1 * residual

# Batch acquisition with diversity
from probneural_operator.active import DiverseBatchAcquisition

batch_selector = DiverseBatchAcquisition(
    acquisition_fn=acquisitions["bald"],
    diversity_weight=0.5,
    similarity_metric="gradient"
)

selected_batch = batch_selector.select(
    model=prob_model,
    pool=candidate_pool,
    batch_size=20
)
```

### Multi-Fidelity Active Learning

```python
from probneural_operator.active import MultiFidelityActiveLearner

# Multiple fidelity levels
fidelities = {
    "low": {"cost": 1, "simulator": coarse_mesh_solver},
    "medium": {"cost": 10, "simulator": medium_mesh_solver},
    "high": {"cost": 100, "simulator": fine_mesh_solver}
}

# Multi-fidelity learner
mf_learner = MultiFidelityActiveLearner(
    model=prob_model,
    fidelities=fidelities,
    budget=1000,
    strategy="cost_aware_variance"
)

# Optimal fidelity selection
for iteration in range(max_iterations):
    # Select point and fidelity
    point, fidelity = mf_learner.query()
    
    # Run simulation at selected fidelity
    value = fidelities[fidelity]["simulator"](point)
    
    # Update multi-fidelity model
    mf_learner.update(point, value, fidelity)
```

## Calibration

### Temperature Scaling

```python
from probneural_operator.calibration import TemperatureScaling

# Calibrate uncertainties
calibrator = TemperatureScaling()
calibrator.fit(
    model=prob_model,
    val_loader=val_loader
)

# Apply calibration
calibrated_model = calibrator.calibrate(prob_model)

# Check calibration
from probneural_operator.calibration import CalibrationMetrics

metrics = CalibrationMetrics()
ece = metrics.expected_calibration_error(
    predictions=calibrated_predictions,
    confidence_levels=np.linspace(0, 1, 100)
)

print(f"Expected Calibration Error: {ece:.3f}")
```

### Reliability Diagrams

```python
from probneural_operator.calibration import ReliabilityDiagram

# Generate reliability diagram
diagram = ReliabilityDiagram()
fig = diagram.plot(
    model=calibrated_model,
    test_data=test_loader,
    n_bins=10,
    return_figure=True
)

# Confidence-accuracy plot
diagram.plot_confidence_accuracy(
    model=calibrated_model,
    test_data=test_loader,
    save_path="calibration_plot.pdf"
)
```

## Physics-Informed Uncertainty

### PDE-Constrained Posterior

```python
from probneural_operator.physics import PhysicsInformedPosterior

# Define PDE constraints
def n
