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
def navier_stokes_residual(u, x, t, nu=0.01):
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    
    return u_t + u * u_x - nu * u_xx

# Physics-informed posterior
pi_posterior = PhysicsInformedPosterior(
    neural_operator=prob_fno,
    pde_residual=navier_stokes_residual,
    boundary_conditions=bc_function,
    physics_weight=0.1
)

# Train with physics constraints
pi_posterior.train(
    data_loader=train_loader,
    collocation_points=collocation_sampler,
    epochs=200
)

# Uncertainty that respects physics
mean, std = pi_posterior.predict(test_input)
```

### Conservation-Aware Uncertainty

```python
from probneural_operator.physics import ConservationLaws

# Enforce conservation in predictions
conservation = ConservationLaws(
    quantities=["mass", "momentum", "energy"],
    tolerance=1e-6
)

# Project predictions to conservation manifold
constrained_mean = conservation.project(mean)
constrained_samples = conservation.project_samples(posterior_samples)

# Uncertainty on conservation manifold
constrained_std = conservation.manifold_uncertainty(
    model=prob_model,
    point=test_point
)
```

## Applications

### Fluid Dynamics

```python
from probneural_operator.applications import FluidDynamics

# Turbulent flow with uncertainty
turb_solver = FluidDynamics.TurbulentFlow(
    model=prob_fno,
    reynolds_number=10000,
    grid_size=256
)

# Predict with uncertainty propagation
flow_mean, flow_std = turb_solver.solve(
    initial_condition=ic,
    time_steps=100,
    return_uncertainty=True
)

# Risk metrics
risk_metrics = turb_solver.compute_risk_metrics(
    flow_mean, 
    flow_std,
    metrics=["max_velocity", "pressure_drop", "vorticity"]
)
```

### Material Science

```python
from probneural_operator.applications import MaterialScience

# Microstructure evolution with uncertainty
material_model = MaterialScience.PhaseField(
    model=prob_deeponet,
    material_properties=titanium_alloy
)

# Predict with epistemic uncertainty
microstructure, uncertainty = material_model.evolve(
    initial_microstructure=initial,
    processing_conditions=conditions,
    time=processing_time
)

# Identify high-uncertainty regions
uncertain_regions = material_model.identify_uncertain_regions(
    uncertainty,
    threshold=0.95  # 95th percentile
)
```

### Climate Modeling

```python
from probneural_operator.applications import ClimateModeling

# Weather prediction with uncertainty
weather_model = ClimateModeling.WeatherPredictor(
    model=prob_gno,
    resolution="0.25deg",
    variables=["temperature", "pressure", "humidity"]
)

# Ensemble forecast
ensemble_forecast = weather_model.ensemble_forecast(
    initial_state=current_weather,
    lead_time=7*24,  # 7 days
    num_members=50
)

# Extreme event probabilities
extreme_probs = weather_model.extreme_event_probability(
    ensemble_forecast,
    thresholds={
        "temperature": 40,  # °C
        "wind_speed": 100,  # km/h
        "precipitation": 100  # mm/day
    }
)
```

## Advanced Features

### Hierarchical Uncertainty

```python
from probneural_operator.hierarchical import HierarchicalUncertainty

# Multi-scale uncertainty quantification
hierarchical = HierarchicalUncertainty(
    scales=["global", "regional", "local"],
    base_model=prob_model
)

# Learn scale-dependent uncertainties
hierarchical.train(
    multi_scale_data=data,
    scale_coupling="multiplicative"
)

# Decompose uncertainty by scale
uncertainties = hierarchical.decompose_uncertainty(
    prediction_location=x,
    return_correlations=True
)

for scale, unc in uncertainties.items():
    print(f"{scale} uncertainty: {unc:.3f}")
```

### Out-of-Distribution Detection

```python
from probneural_operator.ood import OODDetector

# OOD detection for neural operators
ood_detector = OODDetector(
    model=prob_model,
    method="energy",  # or "mahalanobis", "gradient_norm"
    calibration_data=train_data
)

# Detect OOD inputs
ood_scores = ood_detector.score(test_inputs)
is_ood = ood_scores > ood_detector.threshold

# Adaptive uncertainty inflation for OOD
inflated_uncertainty = ood_detector.inflate_uncertainty(
    base_uncertainty=std,
    ood_scores=ood_scores
)
```

## Benchmarking

### Uncertainty Metrics

```python
from probneural_operator.benchmarks import UncertaintyBenchmark

benchmark = UncertaintyBenchmark(
    datasets=["burgers", "navier_stokes", "darcy_flow"],
    metrics=["nll", "crps", "interval_score", "calibration_error"]
)

# Run comprehensive evaluation
results = benchmark.evaluate(
    model=prob_model,
    baselines=["dropout", "ensemble", "vanilla"],
    n_trials=5
)

# Generate report
benchmark.generate_report(
    results,
    include_plots=True,
    save_path="uncertainty_benchmark.pdf"
)
```

### Computational Efficiency

```python
from probneural_operator.benchmarks import EfficiencyProfiler

profiler = EfficiencyProfiler()

# Profile uncertainty methods
efficiency_results = profiler.profile({
    "laplace": linearized_laplace_model,
    "variational": variational_model,
    "ensemble": ensemble_model,
    "dropout": dropout_model
})

# Compare compute/memory/accuracy tradeoffs
profiler.plot_pareto_frontier(
    efficiency_results,
    metrics=["inference_time", "memory", "uncertainty_quality"]
)
```

## Deployment

### Uncertainty-Aware API

```python
from probneural_operator.serving import UncertaintyAwareServer

# Create server with uncertainty
server = UncertaintyAwareServer(
    model=prob_model,
    confidence_levels=[0.68, 0.95, 0.99],
    include_samples=False
)

@server.endpoint("/predict")
async def predict(input_data):
    result = await server.predict_with_uncertainty(
        input_data,
        return_quantiles=True,
        return_risk_metrics=True
    )
    
    return {
        "mean": result.mean,
        "std": result.std,
        "quantiles": result.quantiles,
        "confidence_intervals": result.confidence_intervals,
        "risk_metrics": result.risk_metrics
    }

server.run(host="0.0.0.0", port=8000)
```

### Real-Time Uncertainty Visualization

```python
from probneural_operator.visualization import UncertaintyVisualizer

visualizer = UncertaintyVisualizer(
    model=prob_model,
    update_rate=10  # Hz
)

# Real-time uncertainty plot
visualizer.start_realtime(
    data_stream=sensor_stream,
    plot_types=["mean_field", "uncertainty_bands", "sample_paths"],
    window_size=100
)

# Interactive 3D uncertainty visualization
visualizer.interactive_3d(
    domain=simulation_domain,
    uncertainty_field=uncertainty,
    isosurfaces=[0.1, 0.5, 0.9],
    colormap="viridis"
)
```

## Best Practices

### Uncertainty Validation

```python
# Synthetic uncertainty validation
from probneural_operator.validation import UncertaintyValidator

validator = UncertaintyValidator()

# Generate synthetic data with known uncertainty
synthetic_data = validator.generate_synthetic_data(
    true_operator=analytical_solution,
    noise_model="heteroscedastic",
    n_samples=1000
)

# Validate uncertainty estimates
validation_results = validator.validate(
    model=prob_model,
    synthetic_data=synthetic_data,
    metrics=["coverage", "sharpness", "bias"]
)

# Uncertainty consistency check
consistency = validator.check_consistency(
    model=prob_model,
    transformations=["translation", "rotation", "scaling"]
)
```

### Hyperparameter Selection

```python
from probneural_operator.tuning import UncertaintyHyperparameterTuner

# Tune uncertainty-specific hyperparameters
tuner = UncertaintyHyperparameterTuner(
    model_class=ProbabilisticFNO,
    param_space={
        "prior_precision": [0.01, 0.1, 1.0, 10.0],
        "likelihood_noise": [1e-4, 1e-3, 1e-2],
        "posterior_samples": [10, 50, 100]
    }
)

# Optimize for uncertainty quality
best_params = tuner.optimize(
    train_data=train_data,
    val_data=val_data,
    objective="log_likelihood",
    n_trials=50
)
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

```bibtex
@software{probneural_operator_lab,
  title={ProbNeural-Operator-Lab: Probabilistic Neural Operators with Active Learning},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/probneural-operator-lab}
}

@inproceedings{linearized_probabilistic_no_2025,
  title={Linearization for Probabilistic Neural Operators},
  author={ICML Authors},
  booktitle={ICML},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- ICML 2025 authors for linearized Laplace methodology
- Neural operator community for foundational work
- Scientific computing community for application domains
