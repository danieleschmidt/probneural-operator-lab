# Best Practices Guide for ProbNeural-Operator-Lab

This guide provides recommended practices for using the ProbNeural-Operator-Lab framework effectively and avoiding common pitfalls.

## Table of Contents

1. [Project Setup and Organization](#project-setup-and-organization)
2. [Configuration Management](#configuration-management)
3. [Data Handling and Preprocessing](#data-handling-and-preprocessing)
4. [Model Architecture Design](#model-architecture-design)
5. [Training Best Practices](#training-best-practices)
6. [Uncertainty Quantification Guidelines](#uncertainty-quantification-guidelines)
7. [Performance Optimization](#performance-optimization)
8. [Monitoring and Logging](#monitoring-and-logging)
9. [Testing and Validation](#testing-and-validation)
10. [Production Deployment](#production-deployment)

---

## Project Setup and Organization

### Recommended Directory Structure

```
your_project/
├── configs/
│   ├── base_config.yaml
│   ├── fno_darcy.yaml
│   └── deeponet_burgers.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
├── experiments/
│   ├── experiment_001/
│   │   ├── config.yaml
│   │   ├── model.pth
│   │   ├── results.json
│   │   └── plots/
│   └── experiment_002/
├── src/
│   ├── data_generation.py
│   ├── preprocessing.py
│   └── evaluation.py
├── logs/
├── outputs/
├── requirements.txt
└── main.py
```

### Environment Setup

1. **Use virtual environments:**
   ```bash
   python -m venv probneural_env
   source probneural_env/bin/activate  # Linux/Mac
   # or
   probneural_env\Scripts\activate  # Windows
   ```

2. **Pin dependencies:**
   ```python
   # requirements.txt
   torch>=2.0.0
   probneural-operator-lab>=0.1.0
   numpy>=1.21.0
   scipy>=1.7.0
   matplotlib>=3.5.0
   ```

3. **Set up logging early:**
   ```python
   from probneural_operator.utils import setup_logging
   
   # At the start of your main script
   setup_logging(log_level="INFO", log_dir="./logs")
   ```

---

## Configuration Management

### Use Configuration Files

**❌ Avoid hardcoding parameters:**
```python
# Bad: Parameters scattered throughout code
model = ProbabilisticFNO(modes=16, width=64, depth=4)
history = model.fit(train_loader, epochs=100, lr=0.001)
```

**✅ Use centralized configuration:**
```python
# Good: Configuration-driven approach
from probneural_operator.utils import load_config

config = load_config("configs/my_experiment.yaml")
config.validate()

model = ProbabilisticFNO(
    modes=config.model.modes,
    width=config.model.width,
    depth=config.model.depth,
    posterior_type=config.posterior.method
)

history = model.fit(
    train_loader, 
    epochs=config.training.epochs,
    lr=config.model.learning_rate
)
```

### Configuration Best Practices

1. **Use environment-specific configs:**
   ```python
   from probneural_operator.utils import set_environment
   
   # Development: faster, smaller models
   set_environment("development")
   config = load_config("configs/base.yaml")  # Auto-applies dev overrides
   ```

2. **Validate configurations:**
   ```python
   from probneural_operator.utils import validate_config_compatibility
   
   config = load_config("config.yaml")
   config.validate()  # Check parameter ranges
   
   warnings = validate_config_compatibility(config)  # Check interactions
   if warnings:
       logger.warning(f"Configuration warnings: {warnings}")
   ```

3. **Use meaningful experiment names:**
   ```yaml
   # Good: Descriptive names
   name: "fno_burgers_1d_modes16_laplace_lr001"
   
   # Bad: Generic names
   name: "experiment_1"
   ```

---

## Data Handling and Preprocessing

### Data Validation

**Always validate your data:**
```python
from probneural_operator.utils import validate_training_data

# Before training
for batch_data, batch_targets in train_loader:
    try:
        validate_training_data(batch_data, batch_targets)
    except ValidationError as e:
        logger.error(f"Invalid batch: {e}")
        # Handle or fix the problematic batch
```

### Data Preprocessing Pipeline

```python
class DataPreprocessor:
    """Robust data preprocessing pipeline."""
    
    def __init__(self, normalize=True, add_noise=False, noise_level=0.01):
        self.normalize = normalize
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.stats = None
    
    def fit(self, data):
        """Fit preprocessing parameters."""
        if self.normalize:
            self.stats = {
                'mean': data.mean(dim=0, keepdim=True),
                'std': data.std(dim=0, keepdim=True) + 1e-8  # Avoid division by zero
            }
        return self
    
    def transform(self, data):
        """Apply preprocessing."""
        # Validate input
        validate_tensor_finite(data, "input_data")
        
        # Normalize
        if self.normalize and self.stats:
            data = (data - self.stats['mean']) / self.stats['std']
        
        # Add noise for regularization (training only)
        if self.add_noise and self.training:
            noise = torch.randn_like(data) * self.noise_level
            data = data + noise
        
        return data
    
    def inverse_transform(self, data):
        """Reverse preprocessing."""
        if self.normalize and self.stats:
            data = data * self.stats['std'] + self.stats['mean']
        return data
```

### Data Loading Best Practices

1. **Use appropriate batch sizes:**
   ```python
   # Start with reasonable batch size, adjust based on memory
   def find_optimal_batch_size(model, dataset, max_batch_size=128):
       for batch_size in [4, 8, 16, 32, 64, 128]:
           if batch_size > max_batch_size:
               break
           try:
               test_loader = DataLoader(dataset, batch_size=batch_size)
               batch_data, _ = next(iter(test_loader))
               _ = model(batch_data)  # Test forward pass
               logger.info(f"Batch size {batch_size} works")
               optimal_size = batch_size
           except RuntimeError as e:
               if "out of memory" in str(e).lower():
                   break
               raise
       return optimal_size
   ```

2. **Handle data imbalance:**
   ```python
   from torch.utils.data import WeightedRandomSampler
   
   # For imbalanced datasets
   weights = compute_sample_weights(targets)
   sampler = WeightedRandomSampler(weights, len(weights))
   
   dataloader = DataLoader(
       dataset, 
       batch_size=32,
       sampler=sampler  # Instead of shuffle=True
   )
   ```

---

## Model Architecture Design

### Choosing Model Parameters

1. **Start small, scale up:**
   ```python
   # Phase 1: Proof of concept
   model_small = ProbabilisticFNO(modes=8, width=32, depth=2)
   
   # Phase 2: Performance optimization
   model_medium = ProbabilisticFNO(modes=16, width=64, depth=4)
   
   # Phase 3: Production model
   model_large = ProbabilisticFNO(modes=32, width=128, depth=6)
   ```

2. **Match spatial dimensions:**
   ```python
   # For 1D problems (e.g., Burgers equation)
   model = ProbabilisticFNO(spatial_dim=1, modes=16)
   
   # For 2D problems (e.g., Darcy flow)
   model = ProbabilisticFNO(spatial_dim=2, modes=12)  # Lower modes for 2D
   
   # For 3D problems (computationally expensive)
   model = ProbabilisticFNO(spatial_dim=3, modes=8)   # Even lower modes
   ```

3. **Fourier modes selection:**
   ```python
   # Rule of thumb: modes ≤ min(spatial_resolution) // 2
   spatial_size = 64
   max_modes = spatial_size // 2  # 32
   
   # Start with fewer modes
   recommended_modes = max_modes // 2  # 16
   ```

### Architecture Guidelines

```python
def create_robust_model(problem_type, spatial_resolution, complexity="medium"):
    """Create model with robust defaults."""
    
    # Base parameters by complexity
    params = {
        "simple": {"modes": 8, "width": 32, "depth": 2},
        "medium": {"modes": 16, "width": 64, "depth": 4},
        "complex": {"modes": 32, "width": 128, "depth": 6}
    }
    
    model_params = params[complexity].copy()
    
    # Problem-specific adjustments
    if problem_type == "burgers_1d":
        model_params.update(spatial_dim=1, activation="gelu")
    elif problem_type == "darcy_2d":
        model_params.update(spatial_dim=2, activation="gelu")
        # Reduce modes for 2D problems
        model_params["modes"] = min(model_params["modes"], 16)
    elif problem_type == "navier_stokes":
        model_params.update(spatial_dim=2, activation="gelu")
        # Higher width for complex dynamics
        model_params["width"] = model_params["width"] * 2
    
    # Adjust for resolution
    max_modes = min(spatial_resolution) // 2
    model_params["modes"] = min(model_params["modes"], max_modes)
    
    return ProbabilisticFNO(**model_params)
```

---

## Training Best Practices

### Robust Training Loop

```python
def robust_training(model, train_loader, val_loader, config):
    """Robust training with comprehensive monitoring."""
    
    # Set up monitoring
    from probneural_operator.utils import create_monitoring_suite
    monitoring = create_monitoring_suite(config.name)
    
    # Start monitoring
    monitoring['resource_monitor'].start()
    monitoring['training_monitor'].start_training()
    monitoring['health_monitor'].start()
    
    try:
        with secure_operation("training", max_memory_gb=config.max_memory):
            # Training with error handling
            history = model.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=config.training.epochs,
                lr=config.model.learning_rate,
                device=config.training.device
            )
            
            # Fit posterior for uncertainty
            model.fit_posterior(train_loader, val_loader)
            
            return history
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        # Save partial results
        torch.save(model.state_dict(), f"partial_model_{config.name}.pth")
        raise
    
    finally:
        # Stop monitoring
        monitoring['resource_monitor'].stop()
        monitoring['health_monitor'].stop()
```

### Training Hyperparameters

1. **Learning rate scheduling:**
   ```python
   # Use learning rate scheduling for better convergence
   optimizer = torch.optim.Adam(model.parameters(), lr=config.model.learning_rate)
   scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
       optimizer, mode='min', patience=10, factor=0.5
   )
   
   for epoch in range(epochs):
       # ... training loop ...
       scheduler.step(val_loss)
   ```

2. **Early stopping:**
   ```python
   from probneural_operator.utils import ConvergenceMonitor
   
   convergence = ConvergenceMonitor(patience=20, min_improvement=1e-6)
   
   for epoch in range(max_epochs):
       # ... training ...
       status = convergence.update(val_loss)
       
       if status['converged']:
           logger.info(f"Early stopping at epoch {epoch}")
           break
   ```

3. **Gradient monitoring:**
   ```python
   # Monitor gradients for stability
   def log_gradient_stats(model, epoch):
       total_norm = 0
       param_count = 0
       
       for name, param in model.named_parameters():
           if param.grad is not None:
               param_norm = param.grad.data.norm(2)
               total_norm += param_norm.item() ** 2
               param_count += 1
       
       total_norm = total_norm ** (1. / 2)
       logger.info(f"Epoch {epoch}: Gradient norm: {total_norm:.4f}")
       
       return total_norm
   ```

---

## Uncertainty Quantification Guidelines

### Posterior Method Selection

```python
def choose_posterior_method(dataset_size, model_complexity, computational_budget):
    """Choose appropriate posterior approximation method."""
    
    if dataset_size < 1000:
        # Small datasets: use full Laplace
        return PosteriorConfig(
            method="laplace",
            hessian_structure="full" if model_complexity < 10000 else "diagonal"
        )
    
    elif computational_budget == "low":
        # Fast approximation
        return PosteriorConfig(
            method="laplace",
            hessian_structure="diagonal"
        )
    
    elif computational_budget == "high":
        # Most accurate but expensive
        return PosteriorConfig(
            method="ensemble",
            ensemble_size=10
        )
    
    else:
        # Balanced approach
        return PosteriorConfig(
            method="variational",
            num_samples=50
        )
```

### Uncertainty Validation

```python
def validate_uncertainties(model, test_loader, num_samples=100):
    """Validate uncertainty estimates using calibration metrics."""
    
    predictions = []
    uncertainties = []
    targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            mean, std = model.predict_with_uncertainty(
                data, num_samples=num_samples, return_std=True
            )
            
            predictions.extend(mean.cpu().numpy())
            uncertainties.extend(std.cpu().numpy())
            targets.extend(target.cpu().numpy())
    
    # Calculate calibration metrics
    predictions = np.array(predictions)
    uncertainties = np.array(uncertainties)
    targets = np.array(targets)
    
    # Expected calibration error
    errors = np.abs(predictions - targets)
    ece = expected_calibration_error(errors, uncertainties)
    
    logger.info(f"Expected Calibration Error: {ece:.4f}")
    
    # Reliability diagram
    plot_reliability_diagram(errors, uncertainties, save_path="reliability.png")
    
    return {
        'calibration_error': ece,
        'mean_uncertainty': np.mean(uncertainties),
        'uncertainty_correlation': np.corrcoef(errors, uncertainties)[0, 1]
    }
```

---

## Performance Optimization

### Memory Optimization

1. **Use gradient checkpointing for large models:**
   ```python
   # Enable gradient checkpointing to trade compute for memory
   if hasattr(model, 'gradient_checkpointing_enable'):
       model.gradient_checkpointing_enable()
   ```

2. **Optimize data loading:**
   ```python
   # Efficient data loading
   dataloader = DataLoader(
       dataset,
       batch_size=optimal_batch_size,
       shuffle=True,
       num_workers=4,           # Parallel loading
       pin_memory=True,         # Faster GPU transfer
       persistent_workers=True   # Reuse workers
   )
   ```

3. **Mixed precision training:**
   ```python
   from torch.cuda.amp import autocast, GradScaler
   
   scaler = GradScaler()
   
   for data, target in dataloader:
       with autocast():
           output = model(data)
           loss = criterion(output, target)
       
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()
   ```

### Computational Optimization

```python
def optimize_model_for_inference(model):
    """Optimize model for inference."""
    
    # Set to evaluation mode
    model.eval()
    
    # Compile model (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
    
    # JIT scripting for deployment
    # model = torch.jit.script(model)
    
    return model
```

---

## Monitoring and Logging

### Comprehensive Monitoring Setup

```python
def setup_comprehensive_monitoring(experiment_name):
    """Set up comprehensive monitoring for experiments."""
    
    # Create monitoring suite
    monitoring = create_monitoring_suite(experiment_name)
    
    # Create specialized loggers
    training_logger = create_training_logger(experiment_name)
    uncertainty_logger = create_uncertainty_logger(experiment_name)
    performance_logger = create_performance_logger(experiment_name)
    
    # Start all monitoring
    monitoring['resource_monitor'].start()
    monitoring['health_monitor'].start()
    
    return {
        'monitoring': monitoring,
        'loggers': {
            'training': training_logger,
            'uncertainty': uncertainty_logger,
            'performance': performance_logger
        }
    }
```

### Logging Best Practices

1. **Structure your logs:**
   ```python
   # Good: Structured logging with context
   logger.info("Training started", extra={
       'event_type': 'training_start',
       'model_type': 'FNO',
       'batch_size': 32,
       'learning_rate': 0.001
   })
   
   # Bad: Unstructured strings
   logger.info("Starting training with FNO, batch=32, lr=0.001")
   ```

2. **Log at appropriate levels:**
   ```python
   logger.debug("Batch processed")        # Detailed info
   logger.info("Epoch completed")         # Important milestones
   logger.warning("High memory usage")    # Potential issues
   logger.error("Training failed")        # Serious problems
   ```

---

## Testing and Validation

### Unit Testing

```python
import pytest
from probneural_operator.models import ProbabilisticFNO

class TestFNOModel:
    """Test FNO model functionality."""
    
    @pytest.fixture
    def small_model(self):
        return ProbabilisticFNO(
            input_dim=1, output_dim=1, modes=4, width=16, depth=2
        )
    
    def test_forward_pass(self, small_model):
        """Test basic forward pass."""
        x = torch.randn(2, 1, 32)
        y = small_model(x)
        assert y.shape == x.shape
        assert torch.all(torch.isfinite(y))
    
    def test_uncertainty_quantification(self, small_model, sample_dataloader):
        """Test UQ functionality."""
        # Quick training
        history = small_model.fit(sample_dataloader, epochs=2)
        small_model.fit_posterior(sample_dataloader)
        
        # Test UQ
        x = torch.randn(1, 1, 32)
        mean, std = small_model.predict_with_uncertainty(x, num_samples=10)
        
        assert mean.shape == x.shape
        assert std.shape == x.shape
        assert torch.all(std > 0)  # Positive uncertainties
```

### Integration Testing

```python
def test_full_workflow():
    """Test complete training and inference workflow."""
    
    # Setup
    config = create_default_config("fno")
    config.training.epochs = 3  # Quick test
    
    # Create synthetic data
    inputs = torch.randn(20, 1, 32)
    targets = torch.randn(20, 1, 32)
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=4)
    
    # Create and train model
    model = ProbabilisticFNO(**config.model.to_dict())
    
    # Test training
    history = model.fit(dataloader, epochs=config.training.epochs)
    assert len(history['train_loss']) == config.training.epochs
    
    # Test posterior fitting
    model.fit_posterior(dataloader)
    assert model._is_fitted
    
    # Test uncertainty quantification
    test_input = torch.randn(2, 1, 32)
    mean, std = model.predict_with_uncertainty(test_input)
    
    assert mean.shape == test_input.shape
    assert std.shape == test_input.shape
    assert torch.all(std > 0)
```

---

## Production Deployment

### Model Serialization

```python
def save_model_for_production(model, config, metrics, save_path):
    """Save model with all necessary information for production."""
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': config.model.to_dict(),
        'posterior_config': config.posterior.to_dict(),
        'training_metrics': metrics,
        'framework_version': probneural_operator.__version__,
        'pytorch_version': torch.__version__,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"Model saved for production: {save_path}")

def load_model_for_production(checkpoint_path):
    """Load model in production environment."""
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Recreate model
    model = ProbabilisticFNO(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    return model, checkpoint
```

### Performance Monitoring in Production

```python
class ProductionMonitor:
    """Monitor model performance in production."""
    
    def __init__(self, model):
        self.model = model
        self.prediction_times = []
        self.uncertainty_stats = []
    
    def predict_with_monitoring(self, x, num_samples=50):
        """Make prediction with performance monitoring."""
        
        start_time = time.time()
        
        with torch.no_grad():
            mean, std = self.model.predict_with_uncertainty(
                x, num_samples=num_samples
            )
        
        prediction_time = time.time() - start_time
        self.prediction_times.append(prediction_time)
        
        # Track uncertainty statistics
        avg_uncertainty = torch.mean(std).item()
        self.uncertainty_stats.append(avg_uncertainty)
        
        # Alert on anomalies
        if avg_uncertainty > 0.5:  # Threshold
            logger.warning(f"High uncertainty detected: {avg_uncertainty:.4f}")
        
        return mean, std
    
    def get_performance_stats(self):
        """Get performance statistics."""
        return {
            'avg_prediction_time': np.mean(self.prediction_times),
            'p95_prediction_time': np.percentile(self.prediction_times, 95),
            'avg_uncertainty': np.mean(self.uncertainty_stats),
            'prediction_count': len(self.prediction_times)
        }
```

---

## Summary Checklist

### Before Training
- [ ] Configuration validated and warnings addressed
- [ ] Data validated and preprocessed
- [ ] Model architecture appropriate for problem
- [ ] Monitoring and logging set up
- [ ] Resource limits configured

### During Training
- [ ] Monitor training progress and resource usage
- [ ] Watch for gradient issues and numerical stability
- [ ] Check convergence and early stopping
- [ ] Save checkpoints regularly

### After Training
- [ ] Fit posterior for uncertainty quantification
- [ ] Validate uncertainty estimates
- [ ] Run comprehensive diagnostics
- [ ] Save model with metadata for production
- [ ] Document experiment results

### Production Deployment
- [ ] Model optimized for inference
- [ ] Performance monitoring in place
- [ ] Error handling and fallbacks implemented
- [ ] Documentation updated

Following these best practices will help you avoid common issues and achieve reliable, high-quality results with the ProbNeural-Operator-Lab framework.