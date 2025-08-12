# ProbNeural Operator Lab - API Reference

## Core Models

### ProbabilisticFNO

The main Fourier Neural Operator with uncertainty quantification.

```python
from probneural_operator.models import ProbabilisticFNO

model = ProbabilisticFNO(
    input_dim=3,           # Input function dimension
    output_dim=1,          # Output function dimension  
    modes=16,              # Number of Fourier modes
    width=128,             # Hidden dimension
    depth=4,               # Number of layers
    spatial_dim=2,         # Spatial dimensions (1D, 2D, 3D)
    posterior_type="laplace",  # Posterior approximation method
    prior_precision=1.0    # Prior precision parameter
)
```

#### Methods

##### `forward(x: torch.Tensor) -> torch.Tensor`
Forward pass through the network.

**Parameters:**
- `x`: Input tensor of shape `(batch, input_dim, *spatial_dims)`

**Returns:**
- Output tensor of shape `(batch, output_dim, *spatial_dims)`

##### `predict_with_uncertainty(x: torch.Tensor, num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]`
Generate predictions with epistemic uncertainty estimates.

**Parameters:**
- `x`: Input tensor
- `num_samples`: Number of posterior samples

**Returns:**
- `predictions`: Mean predictions
- `uncertainties`: Epistemic uncertainty (standard deviation)

##### `fit_posterior(train_loader: DataLoader, val_loader: DataLoader = None)`
Fit the posterior approximation using training data.

**Parameters:**
- `train_loader`: Training data loader
- `val_loader`: Optional validation data loader

### MultiFidelityFNO

Multi-fidelity neural operator with hierarchical uncertainty propagation.

```python
from probneural_operator.models.multifidelity import MultiFidelityFNO

model = MultiFidelityFNO(
    input_dim=3,
    output_dim=1,
    num_fidelities=3,      # Number of fidelity levels
    modes=[8, 12, 16],     # Modes for each fidelity
    widths=[32, 64, 96],   # Hidden dims for each fidelity
    depths=[2, 3, 4],      # Depths for each fidelity
    fidelity_correlation=0.8  # Inter-fidelity correlation
)
```

#### Methods

##### `forward(x: torch.Tensor, target_fidelity: int = -1) -> torch.Tensor`
Forward pass at specified fidelity level.

**Parameters:**
- `x`: Input tensor
- `target_fidelity`: Target fidelity level (-1 for highest)

##### `predict_with_multifidelity_uncertainty(x: torch.Tensor, return_all_fidelities: bool = False, num_samples: int = 100) -> Dict[str, torch.Tensor]`
Generate predictions with multi-fidelity uncertainty estimates.

**Returns:**
- Dictionary with keys:
  - `highest_fidelity_prediction`: Prediction at highest fidelity
  - `total_uncertainty`: Combined uncertainty estimate
  - `fidelity_predictions`: Predictions at all fidelity levels (if requested)
  - `fidelity_uncertainties`: Uncertainties at all fidelity levels

##### `optimal_fidelity_selection(x: torch.Tensor, computational_budget: float, fidelity_costs: List[float]) -> int`
Select optimal fidelity level given computational constraints.

**Parameters:**
- `x`: Input tensor
- `computational_budget`: Available computational budget
- `fidelity_costs`: Cost for each fidelity level

**Returns:**
- Optimal fidelity level index

## Posterior Approximations

### LinearizedLaplaceApproximation

Full Jacobian Laplace approximation for neural operators.

```python
from probneural_operator.posteriors.laplace import LinearizedLaplaceApproximation

laplace = LinearizedLaplaceApproximation(
    model=model,
    prior_precision=1.0,
    temperature_scaling=True
)

# Fit posterior
laplace.fit(train_loader, val_loader)

# Generate samples
samples = laplace.sample_predictions(test_data, num_samples=100)
```

#### Methods

##### `fit(train_loader: DataLoader, val_loader: DataLoader = None, optimizer_kwargs: Dict = None)`
Fit the Laplace approximation.

##### `sample_predictions(x: torch.Tensor, num_samples: int = 100) -> torch.Tensor`
Sample from the posterior predictive distribution.

##### `marginal_likelihood() -> float`
Compute the marginal likelihood (model evidence).

## Active Learning

### PhysicsAwareAcquisition

Physics-informed acquisition function for active learning.

```python
from probneural_operator.active import PhysicsAwareAcquisition

acquisition = PhysicsAwareAcquisition(
    model=model,
    pde_type="navier_stokes",
    physics_weight=0.1,
    conservation_weight=0.05
)

# Select next training points
next_points = acquisition.select_batch(
    candidate_points=candidates,
    batch_size=10
)
```

#### PDE Types Supported

- `"burgers"`: Burgers equation
- `"navier_stokes"`: Navier-Stokes equations
- `"wave"`: Wave equation  
- `"heat"`: Heat/diffusion equation
- `"custom"`: User-defined PDE residual function

#### Methods

##### `select_batch(candidate_points: torch.Tensor, batch_size: int = 1) -> torch.Tensor`
Select next batch of training points.

##### `compute_acquisition(x: torch.Tensor) -> torch.Tensor`
Compute acquisition function values.

## Calibration

### MultiDimensionalTemperatureScaling

Advanced uncertainty calibration with spatial-temporal awareness.

```python
from probneural_operator.calibration import MultiDimensionalTemperatureScaling

calibrator = MultiDimensionalTemperatureScaling(
    spatial_dims=2,
    temporal_aware=True,
    learnable_spatial=True
)

# Fit calibration parameters
metrics = calibrator.fit(model, val_loader)

# Apply calibration
calibrated_predictions = calibrator.forward(predictions, coordinates)
```

### PhysicsConstrainedCalibration

Calibration that preserves physical constraints.

```python
from probneural_operator.calibration import PhysicsConstrainedCalibration

# Define conservation laws
def mass_conservation(predictions, coordinates):
    # Return mass conservation residual
    pass

def momentum_conservation(predictions, coordinates):
    # Return momentum conservation residual
    pass

calibrator = PhysicsConstrainedCalibration(
    base_calibrator=base_calibrator,
    conservation_laws=[mass_conservation, momentum_conservation],
    constraint_weight=0.1
)
```

## Benchmarking and Validation

### ResearchBenchmark

Comprehensive benchmarking framework with statistical validation.

```python
from probneural_operator.benchmarks import ResearchBenchmark

benchmark = ResearchBenchmark(
    output_dir="benchmark_results",
    n_runs=5,
    confidence_level=0.95
)

# Benchmark model
results = benchmark.benchmark_model(
    model_factory=lambda: ProbabilisticFNO(...),
    model_name="ProbFNO",
    train_loader=train_loader,
    val_loader=val_loader, 
    test_loader=test_loader,
    dataset_name="Burgers2D",
    training_config={'epochs': 100, 'lr': 1e-3}
)

# Compare methods
comparison = benchmark.compare_methods(
    baseline_results, 
    method_results, 
    metric="mse"
)

# Generate report
report_path = benchmark.generate_report()
```

#### Methods

##### `benchmark_model(model_factory: Callable, model_name: str, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, dataset_name: str, training_config: Dict) -> List[BenchmarkResult]`
Benchmark a model with multiple independent runs.

##### `compare_methods(baseline_results: List[BenchmarkResult], method_results: List[BenchmarkResult], metric: str) -> Dict`
Compare two methods with statistical significance testing.

##### `generate_report(include_plots: bool = True, include_tables: bool = True) -> str`
Generate comprehensive benchmark report.

### UncertaintyValidationSuite

Specialized validation for uncertainty quality.

```python
from probneural_operator.benchmarks import UncertaintyValidationSuite

validator = UncertaintyValidationSuite()

# Validate calibration
calibration_metrics = validator.validate_calibration(
    predictions, uncertainties, targets
)

# Validate uncertainty decomposition
decomposition_metrics = validator.validate_uncertainty_decomposition(
    model, test_loader
)
```

## Distributed Training

### DistributedBayesianTraining

Distributed training for probabilistic neural operators.

```python
from probneural_operator.scaling import DistributedBayesianTraining

config = DistributedConfig(
    backend="nccl",
    world_size=4,
    mixed_precision=True,
    gradient_clipping=1.0
)

trainer = DistributedBayesianTraining(model, config)

history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    lr=1e-3,
    checkpoint_dir="checkpoints/"
)
```

### HPCWorkflowManager

HPC integration with SLURM job management.

```python  
from probneural_operator.scaling import HPCWorkflowManager

hpc = HPCWorkflowManager(
    scheduler="slurm",
    partition="gpu",
    account="research-account"
)

# Submit training job
job_id = hpc.submit_training_job(
    script_path="train_script.py",
    job_name="probfno_training",
    nodes=2,
    gpus_per_node=4,
    time_limit="24:00:00"
)

# Monitor job
status = hpc.get_job_status(job_id)
```

## Utilities

### Data Loading and Preprocessing

```python
from probneural_operator.utils.data import PDEDataset, create_dataloaders

# Create PDE dataset
dataset = PDEDataset(
    data_path="data/burgers/",
    equation_type="burgers",
    spatial_resolution=128,
    temporal_steps=100
)

# Create data loaders
train_loader, val_loader, test_loader = create_dataloaders(
    dataset,
    batch_size=32,
    train_ratio=0.7,
    val_ratio=0.15
)
```

### Visualization

```python
from probneural_operator.utils.visualization import (
    plot_predictions_with_uncertainty,
    plot_uncertainty_calibration,
    plot_active_learning_selection
)

# Plot predictions with uncertainty bands
plot_predictions_with_uncertainty(
    predictions, uncertainties, targets,
    save_path="predictions.png"
)

# Plot calibration diagram
plot_uncertainty_calibration(
    predictions, uncertainties, targets,
    save_path="calibration.png"
)
```

### Validation and Error Handling

```python
from probneural_operator.utils.validation import (
    validate_tensor_shape,
    validate_tensor_finite,
    ValidationError
)

try:
    validate_tensor_shape(tensor, expected_shape=(batch_size, channels, height, width))
    validate_tensor_finite(tensor)
except ValidationError as e:
    print(f"Validation error: {e}")
```

## Configuration Management

### Loading Configuration

```python
from probneural_operator.config import load_config

# Load from YAML file
config = load_config("config/experiment.yaml")

# Access nested configuration
model_config = config.model
training_config = config.training
```

### Configuration Schema

```yaml
# config/experiment.yaml
model:
  type: "ProbabilisticFNO"
  input_dim: 3
  output_dim: 1
  modes: 16
  width: 128
  depth: 4
  spatial_dim: 2
  posterior_type: "laplace"

training:
  epochs: 100
  batch_size: 32
  learning_rate: 1e-3
  optimizer: "AdamW"
  scheduler: "CosineAnnealingLR"

active_learning:
  acquisition_function: "physics_aware"
  pde_type: "navier_stokes"
  batch_size: 10
  max_iterations: 50

calibration:
  method: "multidimensional_temperature"
  spatial_dims: 2
  temporal_aware: true
```

## Error Handling

### Custom Exceptions

```python
from probneural_operator.utils.exceptions import (
    PosteriorNotFittedError,
    InvalidPDETypeError, 
    DistributedTrainingError,
    CalibrationError
)

try:
    model.predict_with_uncertainty(x)
except PosteriorNotFittedError:
    print("Posterior approximation not fitted. Call fit_posterior() first.")
```

### Common Error Patterns

```python
# Check if posterior is fitted
if hasattr(model, '_posterior') and model._posterior is not None:
    predictions, uncertainties = model.predict_with_uncertainty(x)
else:
    print("Posterior not available, using point predictions")
    predictions = model(x)
```

## Performance Optimization

### Memory Management

```python
from probneural_operator.utils.memory import MemoryTracker

tracker = MemoryTracker()
tracker.start_monitoring()

# Your training code here

memory_usage = tracker.get_peak_usage()
print(f"Peak memory usage: {memory_usage:.2f} GB")
```

### Profiling

```python
from probneural_operator.utils.profiling import profile_model

# Profile model forward pass
profile_model(
    model,
    input_tensor,
    output_path="profile_results/"
)
```

This API reference provides comprehensive documentation for using the ProbNeural Operator Lab framework. For more detailed examples and tutorials, see the `examples/` directory in the repository.