# 📚 ProbNeural-Operator-Lab API Reference

Complete API documentation for the probabilistic neural operators framework.

## 🏗️ Core Architecture

### Models (`probneural_operator.models`)

#### ProbabilisticFNO
Fourier Neural Operator with uncertainty quantification.

```python
from probneural_operator.models import ProbabilisticFNO

model = ProbabilisticFNO(
    modes=12,           # Number of Fourier modes
    width=64,           # Hidden dimension size  
    depth=4,            # Number of layers
    posterior="laplace", # Posterior approximation method
    prior_precision=1.0  # Prior precision parameter
)

# Training
model.train_model(
    train_loader=train_data,
    val_loader=val_data, 
    epochs=100,
    lr=1e-3
)

# Prediction with uncertainty
mean, std = model.predict_with_uncertainty(
    x_input,
    return_std=True,
    num_samples=100
)
```

**Methods:**
- `forward(x)`: Forward pass through the network
- `train_model(train_loader, val_loader, epochs, lr)`: Train the model
- `predict_with_uncertainty(x, return_std, num_samples)`: Predict with uncertainty
- `get_epistemic_uncertainty(x)`: Get epistemic uncertainty
- `get_aleatoric_uncertainty(x)`: Get aleatoric uncertainty

#### ProbabilisticDeepONet
Deep Operator Network with uncertainty quantification.

```python
from probneural_operator.models import ProbabilisticDeepONet

model = ProbabilisticDeepONet(
    branch_net_sizes=[100, 100, 100],  # Branch network architecture
    trunk_net_sizes=[2, 100, 100],     # Trunk network architecture
    output_size=1,                     # Output dimension
    posterior="laplace"                # Posterior approximation
)
```

**Methods:**
- `forward(branch_input, trunk_input)`: Forward pass
- `train_model(train_loader, val_loader, epochs, lr)`: Training
- `predict_with_uncertainty(branch_input, trunk_input)`: Uncertain prediction

### Posteriors (`probneural_operator.posteriors`)

#### LinearizedLaplace
Linearized Laplace approximation for uncertainty quantification.

```python
from probneural_operator.posteriors import LinearizedLaplace

laplace = LinearizedLaplace(
    model=neural_operator,
    likelihood="regression",
    hessian_structure="kron",  # 'diag', 'kron', 'full'
    prior_precision=1.0
)

# Fit posterior
laplace.fit(train_loader)

# Sample from posterior  
samples = laplace.sample(x_input, n_samples=100)

# Compute marginal likelihood
log_ml = laplace.log_marginal_likelihood()
```

**Methods:**
- `fit(train_loader)`: Fit the Laplace approximation
- `sample(x, n_samples)`: Sample from posterior
- `predictive_mean(x)`: Posterior predictive mean
- `predictive_variance(x)`: Posterior predictive variance
- `log_marginal_likelihood()`: Compute marginal likelihood

### Active Learning (`probneural_operator.active`)

#### ActiveLearner
Intelligent data acquisition for expensive simulations.

```python
from probneural_operator.active import ActiveLearner

learner = ActiveLearner(
    model=prob_model,
    acquisition="bald",  # Acquisition function
    budget=1000         # Total acquisition budget
)

# Active learning loop
for iteration in range(10):
    # Query most informative points
    query_points = learner.query(
        candidate_pool=unlabeled_data,
        batch_size=20
    )
    
    # Get labels (expensive simulation)
    labels = expensive_simulation(query_points)
    
    # Update model
    learner.update(query_points, labels)
```

**Acquisition Functions:**
- `bald`: Bayesian Active Learning by Disagreement
- `max_variance`: Maximum predictive variance
- `max_entropy`: Maximum predictive entropy
- `random`: Random selection baseline
- `physics_aware`: Physics-informed acquisition

#### Calibration (`probneural_operator.calibration`)

#### TemperatureScaling
Post-hoc uncertainty calibration.

```python
from probneural_operator.calibration import TemperatureScaling

calibrator = TemperatureScaling()

# Fit temperature parameter
calibrator.fit(
    model=prob_model,
    val_loader=validation_data
)

# Apply calibration
calibrated_model = calibrator.calibrate(prob_model)

# Compute calibration metrics
from probneural_operator.calibration import CalibrationMetrics
metrics = CalibrationMetrics()
ece = metrics.expected_calibration_error(predictions, true_labels)
```

### Data (`probneural_operator.data`)

#### Datasets
Pre-implemented PDE datasets.

```python
from probneural_operator.data import NavierStokes, Burgers, DarcyFlow

# Navier-Stokes dataset
ns_dataset = NavierStokes(
    resolution=64,
    viscosity=1e-3,
    time_steps=50,
    n_samples=1000
)

# Burgers equation dataset  
burgers = Burgers(
    resolution=128,
    viscosity=0.01,
    time_horizon=1.0,
    n_samples=500
)

# Darcy flow dataset
darcy = DarcyFlow(
    resolution=64, 
    permeability_range=(0.1, 10.0),
    n_samples=800
)
```

#### Synthetic Generators
Generate PDE solutions programmatically.

```python
from probneural_operator.data import PDESolutionGenerator

generator = PDESolutionGenerator()

# Generate Navier-Stokes solutions
solutions = generator.generate_navier_stokes(
    n_samples=100,
    resolution=64,
    reynolds_number=1000
)

# Generate Burgers solutions
solutions = generator.generate_burgers(
    n_samples=50,
    resolution=128, 
    viscosity=0.01
)
```

## 🚀 Advanced Features

### Scaling (`probneural_operator.scaling`)

#### Distributed Training
Multi-GPU and multi-node training support.

```python
from probneural_operator.scaling import DistributedTrainer

trainer = DistributedTrainer(
    model=prob_model,
    world_size=4,  # Number of GPUs/nodes
    backend="nccl"
)

# Train across multiple GPUs
trainer.train(
    train_loader=distributed_train_loader,
    epochs=100,
    lr=1e-3
)
```

#### Auto-Scaling
Dynamic resource allocation.

```python
from probneural_operator.scaling import AutoScaler

scaler = AutoScaler(
    min_workers=1,
    max_workers=10,
    target_cpu_utilization=70,
    scale_up_threshold=0.8,
    scale_down_threshold=0.3
)

# Start auto-scaling
scaler.start_monitoring()
```

#### Caching System
Intelligent prediction caching.

```python
from probneural_operator.scaling import PredictionCache

cache = PredictionCache(
    max_size=1000,
    ttl=3600,  # 1 hour
    compression=True
)

# Cache predictions
@cache.cached_prediction
def expensive_prediction(input_data):
    return model.predict(input_data)
```

### Production Serving (`probneural_operator.scaling.serving`)

#### REST API Server
High-performance serving infrastructure.

```python
from probneural_operator.scaling.serving import UncertaintyAwareServer

server = UncertaintyAwareServer(
    model=prob_model,
    host="0.0.0.0",
    port=8000,
    workers=4
)

# Start server
server.run()
```

**API Endpoints:**
- `POST /predict`: Single prediction with uncertainty
- `POST /predict/batch`: Batch predictions  
- `GET /health`: Health check
- `GET /metrics`: Prometheus metrics
- `GET /model/info`: Model information

## 🛠️ Utilities

### Configuration (`probneural_operator.utils.config`)

```python
from probneural_operator.utils.config import Config

# Load configuration
config = Config.from_file("config.yaml")

# Access nested values
batch_size = config.training.batch_size
learning_rate = config.optimizer.lr

# Environment-specific overrides
config.override_from_env("PROD")
```

### Logging (`probneural_operator.utils.logging_config`)

```python
from probneural_operator.utils.logging_config import setup_logging

# Setup structured logging
logger = setup_logging(
    level="INFO",
    format="json",
    output_file="probneural.log"
)

# Use specialized loggers
training_logger = get_training_logger()
uncertainty_logger = get_uncertainty_logger()
```

### Validation (`probneural_operator.utils.validation`)

```python
from probneural_operator.utils.validation import validate_tensor, validate_config

# Validate tensor inputs
validate_tensor(
    tensor=input_data,
    expected_shape=(batch_size, channels, height, width),
    dtype=torch.float32,
    device="cuda"
)

# Validate configuration
validate_config(config, schema="training_config.json")
```

### Monitoring (`probneural_operator.utils.monitoring`)

```python
from probneural_operator.utils.monitoring import ResourceMonitor

monitor = ResourceMonitor()

# Start monitoring
monitor.start()

# Get current metrics
metrics = monitor.get_metrics()
print(f"GPU Usage: {metrics.gpu_utilization}%")
print(f"Memory Usage: {metrics.memory_usage_gb} GB")
```

## 📊 Examples

### Basic Training Example

```python
import torch
from probneural_operator import ProbabilisticFNO
from probneural_operator.data import NavierStokes
from probneural_operator.calibration import TemperatureScaling

# Load dataset
dataset = NavierStokes(resolution=64, n_samples=1000)
train_loader, val_loader = dataset.get_loaders(batch_size=32)

# Create model
model = ProbabilisticFNO(
    modes=12,
    width=64, 
    depth=4,
    posterior="laplace"
)

# Train model
model.train_model(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    lr=1e-3
)

# Calibrate uncertainties
calibrator = TemperatureScaling()
calibrator.fit(model, val_loader)
calibrated_model = calibrator.calibrate(model)

# Predict with calibrated uncertainty
test_input = dataset.get_test_sample()
mean, std = calibrated_model.predict_with_uncertainty(
    test_input,
    return_std=True
)

print(f"Prediction: {mean.item():.3f} ± {std.item():.3f}")
```

### Active Learning Example

```python
from probneural_operator.active import ActiveLearner
from probneural_operator.data import DarcyFlow

# Setup
dataset = DarcyFlow(resolution=64, n_samples=10000)
initial_data, pool_data = dataset.get_active_learning_split(initial_size=100)

model = ProbabilisticFNO(modes=8, width=32, depth=3)
learner = ActiveLearner(model=model, acquisition="bald", budget=500)

# Active learning loop
for iteration in range(10):
    # Query most informative samples
    query_indices = learner.query(
        candidate_pool=pool_data,
        batch_size=50
    )
    
    # Simulate expensive labeling
    new_labels = expensive_simulation(pool_data[query_indices])
    
    # Update model
    learner.update(pool_data[query_indices], new_labels)
    
    # Evaluate performance
    test_error = evaluate_model(model, dataset.test_loader)
    print(f"Iteration {iteration}: Test Error = {test_error:.3f}")
```

### Production Deployment

```python
# Deploy with Docker
docker build -t probneural-operator .
docker run -p 8000:8000 probneural-operator

# Deploy to Kubernetes
kubectl apply -f k8s-deployment.yaml

# Use deployment script
./scripts/deploy.sh -e production -p kubernetes -g -m
```

For complete examples, see the `examples/` directory in the repository.