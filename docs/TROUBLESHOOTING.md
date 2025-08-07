# Troubleshooting Guide for ProbNeural-Operator-Lab

This comprehensive troubleshooting guide covers common issues, their causes, and solutions when using the ProbNeural-Operator-Lab framework.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Model Training Problems](#model-training-problems)
3. [Uncertainty Quantification Issues](#uncertainty-quantification-issues)
4. [Performance Problems](#performance-problems)
5. [Memory Issues](#memory-issues)
6. [Numerical Stability Problems](#numerical-stability-problems)
7. [Configuration Errors](#configuration-errors)
8. [Data-Related Issues](#data-related-issues)
9. [Device and Hardware Issues](#device-and-hardware-issues)
10. [Common Error Messages](#common-error-messages)

---

## Installation Issues

### Problem: Import errors after installation
```python
ImportError: No module named 'probneural_operator'
```

**Causes:**
- Package not installed correctly
- Virtual environment not activated
- Python path issues

**Solutions:**
1. Verify installation:
   ```bash
   pip list | grep probneural-operator-lab
   ```

2. Reinstall in development mode:
   ```bash
   pip install -e .
   ```

3. Check Python path:
   ```python
   import sys
   print(sys.path)
   ```

### Problem: Dependency conflicts
```
ERROR: Could not find a version that satisfies the requirement torch>=2.0.0
```

**Solutions:**
1. Update pip:
   ```bash
   pip install --upgrade pip
   ```

2. Install with specific versions:
   ```bash
   pip install torch==2.0.1 torchvision==0.15.2
   pip install -e .
   ```

3. Use conda for better dependency resolution:
   ```bash
   conda install pytorch torchvision -c pytorch
   pip install -e .
   ```

---

## Model Training Problems

### Problem: Training loss not decreasing
```
Epoch 50: Train Loss: 1.234567, Val Loss: 1.234567
```

**Causes:**
- Learning rate too high or too low
- Model architecture issues
- Data preprocessing problems
- Gradient clipping too aggressive

**Solutions:**

1. **Adjust learning rate:**
   ```python
   # Try different learning rates
   learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]
   for lr in learning_rates:
       model.fit(train_loader, lr=lr, epochs=10)
   ```

2. **Check gradient norms:**
   ```python
   from probneural_operator.utils import ModelHealthChecker
   
   checker = ModelHealthChecker(model)
   diagnostics = checker.run_full_health_check()
   print(diagnostics['gradient_health'])
   ```

3. **Verify data:**
   ```python
   # Check data statistics
   for batch_data, batch_targets in train_loader:
       print(f"Input range: [{batch_data.min():.3f}, {batch_data.max():.3f}]")
       print(f"Target range: [{batch_targets.min():.3f}, {batch_targets.max():.3f}]")
       break
   ```

### Problem: Training crashes with NaN loss
```
RuntimeError: Loss became nan at epoch 15, batch 23
```

**Causes:**
- Gradient explosion
- Numerical instability
- Invalid input data
- Learning rate too high

**Solutions:**

1. **Enable gradient clipping:**
   ```python
   # Already enabled by default in enhanced fit method
   # But you can adjust the max_norm
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

2. **Check for problematic inputs:**
   ```python
   from probneural_operator.utils import validate_training_data
   
   for batch_data, batch_targets in train_loader:
       try:
           validate_training_data(batch_data, batch_targets)
       except Exception as e:
           print(f"Invalid batch found: {e}")
   ```

3. **Reduce learning rate:**
   ```python
   model.fit(train_loader, lr=1e-4, epochs=100)  # Smaller LR
   ```

### Problem: Model not learning spatial patterns
```
Predictions look like constant values across space
```

**Causes:**
- Insufficient Fourier modes
- Wrong spatial dimensions
- Inappropriate padding

**Solutions:**

1. **Increase Fourier modes:**
   ```python
   model = ProbabilisticFNO(
       modes=32,  # Increase from default 16
       width=128,  # Also consider increasing width
       # ... other params
   )
   ```

2. **Check spatial dimensions:**
   ```python
   # For 1D problems
   model = ProbabilisticFNO(spatial_dim=1, ...)
   # For 2D problems  
   model = ProbabilisticFNO(spatial_dim=2, ...)
   ```

3. **Verify input shape:**
   ```python
   print(f"Input shape: {input_tensor.shape}")
   # Should be (batch, channels, *spatial_dims)
   # e.g., (32, 1, 64) for 1D or (32, 1, 64, 64) for 2D
   ```

---

## Uncertainty Quantification Issues

### Problem: Posterior fitting fails
```
PosteriorNotFittedError: Posterior approximation not fitted
```

**Solution:**
```python
# Always fit posterior after training
model.fit(train_loader, epochs=100)
model.fit_posterior(train_loader)  # Required for UQ

# Then you can use uncertainty methods
mean, std = model.predict_with_uncertainty(test_input)
```

### Problem: Unrealistic uncertainty estimates
```
Standard deviation values are too large/small
```

**Causes:**
- Inappropriate prior precision
- Insufficient posterior samples
- Model hasn't converged

**Solutions:**

1. **Adjust prior precision:**
   ```python
   model = ProbabilisticFNO(
       prior_precision=0.1,  # Increase for less uncertainty
       # or
       prior_precision=10.0,  # Decrease for more uncertainty
   )
   ```

2. **Increase posterior samples:**
   ```python
   mean, std = model.predict_with_uncertainty(
       x, 
       num_samples=200  # Increase from default 100
   )
   ```

3. **Check model convergence:**
   ```python
   from probneural_operator.utils import ConvergenceMonitor
   
   monitor = ConvergenceMonitor(patience=20)
   for epoch in range(epochs):
       # ... training code ...
       status = monitor.update(train_loss)
       if status['converged']:
           print(f"Converged at epoch {epoch}")
           break
   ```

### Problem: Laplace approximation numerical issues
```
NumericalStabilityError: Matrix is ill-conditioned
```

**Solutions:**

1. **Use diagonal Laplace:**
   ```python
   from probneural_operator.utils import PosteriorConfig
   
   config = PosteriorConfig(
       method="laplace",
       hessian_structure="diagonal"  # More stable than "full"
   )
   ```

2. **Increase damping:**
   ```python
   config = PosteriorConfig(
       method="laplace",
       damping=1e-2  # Increase from default 1e-3
   )
   ```

3. **Switch to ensemble method:**
   ```python
   model = ProbabilisticFNO(
       posterior_type="ensemble",  # More robust than Laplace
       ensemble_size=5
   )
   ```

---

## Performance Problems

### Problem: Training is very slow
```
Each epoch takes several minutes
```

**Causes:**
- Model too large
- Inefficient data loading
- CPU/GPU mismatch
- Too many Fourier modes

**Solutions:**

1. **Optimize model size:**
   ```python
   # Reduce model complexity
   model = ProbabilisticFNO(
       modes=8,      # Reduce from 16
       width=32,     # Reduce from 64
       depth=2       # Reduce from 4
   )
   ```

2. **Optimize data loading:**
   ```python
   dataloader = DataLoader(
       dataset,
       batch_size=32,
       shuffle=True,
       num_workers=4,    # Parallel loading
       pin_memory=True   # For GPU training
   )
   ```

3. **Use GPU if available:**
   ```python
   model.fit(train_loader, device="cuda")
   ```

4. **Enable monitoring to identify bottlenecks:**
   ```python
   from probneural_operator.utils import create_monitoring_suite
   
   monitoring = create_monitoring_suite()
   monitoring['resource_monitor'].start()
   # ... training ...
   monitoring['resource_monitor'].stop()
   ```

### Problem: Poor GPU utilization
```
GPU memory usage is low during training
```

**Solutions:**

1. **Increase batch size:**
   ```python
   # Try larger batches if memory allows
   dataloader = DataLoader(dataset, batch_size=64)  # vs 32
   ```

2. **Mixed precision training:**
   ```python
   # Enable automatic mixed precision (if using PyTorch >= 1.6)
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

---

## Memory Issues

### Problem: Out of memory errors
```
RuntimeError: CUDA out of memory
```

**Solutions:**

1. **Reduce batch size:**
   ```python
   dataloader = DataLoader(dataset, batch_size=8)  # Reduce from 32
   ```

2. **Enable memory monitoring:**
   ```python
   from probneural_operator.utils import MemoryMonitor
   
   monitor = MemoryMonitor(max_memory_gb=8.0)
   with monitor.monitor_operation("training"):
       model.fit(train_loader)
   ```

3. **Use gradient checkpointing:**
   ```python
   # Enable gradient checkpointing for memory efficiency
   model.gradient_checkpointing_enable()  # If available
   ```

4. **Clear cache regularly:**
   ```python
   import gc
   import torch
   
   if torch.cuda.is_available():
       torch.cuda.empty_cache()
   gc.collect()
   ```

### Problem: Memory leaks during training
```
Memory usage keeps increasing across epochs
```

**Solutions:**

1. **Use memory monitoring:**
   ```python
   from probneural_operator.utils import create_monitoring_suite
   
   monitoring = create_monitoring_suite()
   health_monitor = monitoring['health_monitor']
   health_monitor.start()
   # Will automatically detect memory leaks
   ```

2. **Clear intermediate results:**
   ```python
   # Don't keep references to intermediate tensors
   for epoch in range(epochs):
       # ... training ...
       
       # Clear any stored results
       if 'stored_outputs' in locals():
           del stored_outputs
           
       torch.cuda.empty_cache()  # Clear GPU cache
   ```

---

## Numerical Stability Problems

### Problem: Gradient explosion
```
RuntimeError: Gradient explosion detected: 1234.56
```

**Solutions:**

1. **Gradient clipping (already enabled):**
   ```python
   # Adjust clipping threshold
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
   ```

2. **Reduce learning rate:**
   ```python
   model.fit(train_loader, lr=1e-4)  # Reduce from 1e-3
   ```

3. **Check input normalization:**
   ```python
   # Normalize inputs to [-1, 1] or [0, 1]
   data_normalized = (data - data.mean()) / data.std()
   ```

### Problem: Ill-conditioned matrices
```
NumericalStabilityError: Matrix has high condition number
```

**Solutions:**

1. **Increase regularization:**
   ```python
   model = ProbabilisticFNO(
       prior_precision=10.0  # Higher regularization
   )
   ```

2. **Use more stable algorithms:**
   ```python
   from probneural_operator.utils import safe_inversion
   
   # Use SVD-based inversion for stability
   inv_matrix = safe_inversion(matrix, method="svd")
   ```

---

## Configuration Errors

### Problem: Invalid configuration parameters
```
ConfigurationError: Invalid activation: invalid_activation
```

**Solutions:**

1. **Check valid options:**
   ```python
   from probneural_operator.utils import FNOConfig
   
   # See valid activation functions
   config = FNOConfig(activation="gelu")  # Valid options: gelu, relu, tanh, etc.
   ```

2. **Use configuration validation:**
   ```python
   from probneural_operator.utils import validate_config_compatibility
   
   config = create_default_config("fno")
   warnings = validate_config_compatibility(config)
   if warnings:
       print("Configuration warnings:", warnings)
   ```

3. **Load from examples:**
   ```python
   from probneural_operator.utils import load_config
   
   # Load a working configuration
   config = load_config("configs/fno_example.yaml")
   ```

---

## Data-Related Issues

### Problem: Shape mismatch errors
```
RuntimeError: Expected input shape (batch, 1, spatial), got (batch, spatial)
```

**Solutions:**

1. **Add channel dimension:**
   ```python
   # For FNO, ensure channel dimension exists
   if data.ndim == 2:  # (batch, spatial)
       data = data.unsqueeze(1)  # -> (batch, 1, spatial)
   ```

2. **Check data loader output:**
   ```python
   for batch_data, batch_targets in dataloader:
       print(f"Data shape: {batch_data.shape}")
       print(f"Target shape: {batch_targets.shape}")
       break
   ```

### Problem: Data type errors
```
RuntimeError: Expected dtype float32, got int64
```

**Solutions:**

1. **Convert data types:**
   ```python
   data = data.float()  # Convert to float32
   targets = targets.float()
   ```

2. **Set dtype in dataset creation:**
   ```python
   data_tensor = torch.tensor(data, dtype=torch.float32)
   ```

---

## Device and Hardware Issues

### Problem: CUDA errors
```
RuntimeError: CUDA error: device-side assert triggered
```

**Solutions:**

1. **Run system diagnostics:**
   ```python
   from probneural_operator.utils import run_comprehensive_diagnostics
   
   diagnostics = run_comprehensive_diagnostics(model)
   print(diagnostics['cuda_compatibility'])
   ```

2. **Check GPU memory:**
   ```python
   if torch.cuda.is_available():
       print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
       print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
   ```

3. **Fall back to CPU:**
   ```python
   model.fit(train_loader, device="cpu")
   ```

---

## Common Error Messages

### `PosteriorNotFittedError`
**Cause:** Trying to use uncertainty methods before fitting posterior.
**Solution:** Always call `model.fit_posterior()` after training.

### `ValidationError: Input batch size != target batch size`
**Cause:** Mismatched data dimensions.
**Solution:** Check your data loading and preprocessing.

### `NumericalStabilityError: Matrix is ill-conditioned`
**Cause:** Numerical instability in Laplace approximation.
**Solution:** Use diagonal Laplace or ensemble methods.

### `ConfigurationError: Invalid dataset_type`
**Cause:** Typo in configuration.
**Solution:** Check valid options in configuration documentation.

### `MemoryError: Critical memory usage`
**Cause:** Insufficient memory for operation.
**Solution:** Reduce batch size or model complexity.

---

## Getting Help

If you encounter issues not covered here:

1. **Check the logs:**
   ```python
   from probneural_operator.utils import setup_logging
   setup_logging(log_level="DEBUG")
   ```

2. **Run diagnostics:**
   ```python
   from probneural_operator.utils import run_comprehensive_diagnostics
   diagnostics = run_comprehensive_diagnostics(model, sample_input, sample_target)
   ```

3. **Enable monitoring:**
   ```python
   from probneural_operator.utils import create_monitoring_suite
   monitoring = create_monitoring_suite()
   # Monitor resource usage, health, etc.
   ```

4. **Check system compatibility:**
   ```python
   from probneural_operator.utils import SystemCompatibilityChecker
   checker = SystemCompatibilityChecker()
   compatibility = checker.check_full_compatibility()
   ```

---

## Best Practices for Avoiding Issues

1. **Always validate configurations:**
   ```python
   config.validate()
   warnings = validate_config_compatibility(config)
   ```

2. **Use comprehensive monitoring:**
   ```python
   monitoring = create_monitoring_suite()
   monitoring['health_monitor'].start()
   ```

3. **Start with small models:**
   ```python
   # Test with small models first
   model = ProbabilisticFNO(modes=4, width=16, depth=2)
   ```

4. **Use provided examples:**
   - Start with `examples/comprehensive_usage_guide.py`
   - Check `configs/` for working configurations

5. **Enable logging:**
   ```python
   setup_logging(log_level="INFO")
   ```

This troubleshooting guide covers the most common issues. For additional help, refer to the comprehensive usage examples and API documentation.