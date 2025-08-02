"""Testing utilities and helper functions."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import tempfile
import contextlib
import warnings


# Device and computation utilities
def get_available_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def assert_tensors_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    msg: str = ""
) -> None:
    """Assert that two tensors are close with informative error messages."""
    try:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    except AssertionError as e:
        error_msg = f"Tensors not close: {e}"
        if msg:
            error_msg = f"{msg}: {error_msg}"
        
        # Add additional debugging information
        error_msg += f"\nActual shape: {actual.shape}, Expected shape: {expected.shape}"
        error_msg += f"\nActual dtype: {actual.dtype}, Expected dtype: {expected.dtype}"
        error_msg += f"\nActual device: {actual.device}, Expected device: {expected.device}"
        error_msg += f"\nMax absolute difference: {(actual - expected).abs().max().item()}"
        error_msg += f"\nMean absolute difference: {(actual - expected).abs().mean().item()}"
        
        raise AssertionError(error_msg)


def assert_tensor_properties(
    tensor: torch.Tensor,
    shape: Optional[Tuple[int, ...]] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    finite: bool = True,
    requires_grad: Optional[bool] = None
) -> None:
    """Assert tensor has expected properties."""
    if shape is not None:
        assert tensor.shape == shape, f"Expected shape {shape}, got {tensor.shape}"
    
    if dtype is not None:
        assert tensor.dtype == dtype, f"Expected dtype {dtype}, got {tensor.dtype}"
    
    if device is not None:
        assert tensor.device == device, f"Expected device {device}, got {tensor.device}"
    
    if finite:
        assert torch.isfinite(tensor).all(), "Tensor contains non-finite values"
    
    if requires_grad is not None:
        assert tensor.requires_grad == requires_grad, \
            f"Expected requires_grad={requires_grad}, got {tensor.requires_grad}"


# Data generation utilities
def generate_random_pde_data(
    nx: int = 64,
    nt: int = 50,
    batch_size: int = 10,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32
) -> Dict[str, torch.Tensor]:
    """Generate random PDE data for testing."""
    x = torch.linspace(0, 1, nx, device=device, dtype=dtype)
    t = torch.linspace(0, 1, nt, device=device, dtype=dtype)
    
    # Random initial conditions
    u0 = torch.randn(batch_size, nx, device=device, dtype=dtype)
    
    # Random solutions (not physically meaningful, but good for testing)
    u = torch.randn(batch_size, nt, nx, device=device, dtype=dtype)
    
    return {
        "x": x,
        "t": t,
        "u0": u0,
        "u": u,
        "batch_size": batch_size
    }


def generate_synthetic_regression_data(
    n_samples: int = 1000,
    n_features: int = 10,
    noise_level: float = 0.1,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic regression data with known ground truth."""
    X = torch.randn(n_samples, n_features, device=device, dtype=dtype)
    
    # True coefficients
    true_coeff = torch.randn(n_features, device=device, dtype=dtype)
    
    # Generate targets with noise
    y = X @ true_coeff + noise_level * torch.randn(n_samples, device=device, dtype=dtype)
    
    return X, y


# Model testing utilities
def test_model_forward_pass(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    expected_output_shape: Tuple[int, ...],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Test model forward pass with given input shape."""
    model.eval()
    x = torch.randn(*input_shape, device=device, dtype=dtype)
    
    with torch.no_grad():
        output = model(x)
    
    assert_tensor_properties(output, shape=expected_output_shape, device=device, dtype=dtype)
    return output


def test_model_gradients(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32
) -> None:
    """Test that model can compute gradients."""
    model.train()
    x = torch.randn(*input_shape, device=device, dtype=dtype, requires_grad=True)
    
    output = model(x)
    loss = output.sum()
    loss.backward()
    
    # Check that gradients were computed
    assert x.grad is not None, "Input gradients not computed"
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} gradients not computed"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradients"


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Memory and performance utilities
@contextlib.contextmanager
def measure_memory(device: torch.device):
    """Context manager to measure memory usage."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        start_memory = torch.cuda.memory_allocated(device)
        
        yield
        
        peak_memory = torch.cuda.max_memory_allocated(device)
        end_memory = torch.cuda.memory_allocated(device)
        
        print(f"Memory used: {(end_memory - start_memory) / 1024**2:.2f} MB")
        print(f"Peak memory: {peak_memory / 1024**2:.2f} MB")
    else:
        # For CPU, we could use tracemalloc, but it's more complex
        yield


@contextlib.contextmanager
def measure_time():
    """Context manager to measure execution time."""
    import time
    start = time.time()
    yield
    end = time.time()
    print(f"Execution time: {end - start:.4f} seconds")


# Numerical stability utilities
def check_numerical_stability(
    tensor: torch.Tensor,
    max_value: float = 1e6,
    min_value: float = -1e6
) -> bool:
    """Check if tensor values are numerically stable."""
    if torch.isnan(tensor).any():
        return False
    if torch.isinf(tensor).any():
        return False
    if (tensor > max_value).any() or (tensor < min_value).any():
        return False
    return True


def safe_normalize(tensor: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """Safely normalize tensor to avoid division by zero."""
    norm = torch.norm(tensor, dim=dim, keepdim=True)
    return tensor / torch.clamp(norm, min=eps)


# Configuration and setup utilities
def setup_test_environment(seed: int = 42) -> None:
    """Setup test environment with deterministic behavior."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def suppress_warnings():
    """Suppress common warnings during testing."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")


# File and data utilities
@contextlib.contextmanager
def temporary_file(suffix: str = ".pt"):
    """Context manager for temporary files."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        temp_path = Path(f.name)
    
    try:
        yield temp_path
    finally:
        if temp_path.exists():
            temp_path.unlink()


def save_and_load_model(model: torch.nn.Module, temp_dir: Path) -> torch.nn.Module:
    """Test model serialization by saving and loading."""
    model_path = temp_dir / "test_model.pt"
    
    # Save model
    torch.save(model.state_dict(), model_path)
    
    # Create new model instance and load state
    new_model = type(model)()
    new_model.load_state_dict(torch.load(model_path))
    
    return new_model


# Statistical testing utilities
def statistical_test_predictions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    rtol: float = 0.1
) -> Dict[str, float]:
    """Perform statistical tests on predictions."""
    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()
    
    # Correlation coefficient
    pred_mean = predictions.mean()
    target_mean = targets.mean()
    
    numerator = ((predictions - pred_mean) * (targets - target_mean)).sum()
    denominator = torch.sqrt(
        ((predictions - pred_mean) ** 2).sum() * 
        ((targets - target_mean) ** 2).sum()
    )
    
    correlation = (numerator / denominator).item() if denominator > 0 else 0.0
    
    return {
        "mse": mse,
        "mae": mae,
        "correlation": correlation
    }


# Uncertainty testing utilities
def test_uncertainty_properties(
    mean: torch.Tensor,
    std: torch.Tensor,
    samples: Optional[torch.Tensor] = None
) -> None:
    """Test properties of uncertainty estimates."""
    # Standard deviation should be positive
    assert (std >= 0).all(), "Standard deviation must be non-negative"
    
    # Check shapes match
    assert mean.shape == std.shape, "Mean and std shapes must match"
    
    if samples is not None:
        # Check sample statistics match mean/std approximately
        sample_mean = samples.mean(dim=0)
        sample_std = samples.std(dim=0)
        
        # Allow some tolerance for statistical variation
        assert_tensors_close(sample_mean, mean, rtol=0.1, atol=0.1, 
                           msg="Sample mean doesn't match predicted mean")
        assert_tensors_close(sample_std, std, rtol=0.2, atol=0.1,
                           msg="Sample std doesn't match predicted std")


def test_calibration_properties(
    predictions: torch.Tensor,
    uncertainties: torch.Tensor,
    targets: torch.Tensor,
    confidence_levels: List[float] = [0.68, 0.95]
) -> Dict[str, float]:
    """Test calibration properties of uncertainty estimates."""
    results = {}
    
    for confidence in confidence_levels:
        # Compute confidence intervals
        z_score = torch.distributions.Normal(0, 1).icdf(torch.tensor((1 + confidence) / 2))
        lower = predictions - z_score * uncertainties
        upper = predictions + z_score * uncertainties
        
        # Check coverage
        in_interval = (targets >= lower) & (targets <= upper)
        actual_coverage = in_interval.float().mean().item()
        
        results[f"coverage_{confidence}"] = actual_coverage
        
        # Coverage should be close to nominal confidence level
        coverage_error = abs(actual_coverage - confidence)
        assert coverage_error < 0.1, \
            f"Coverage error {coverage_error:.3f} too large for confidence {confidence}"
    
    return results


# Active learning testing utilities
def test_acquisition_function_properties(
    acquisition_scores: torch.Tensor,
    check_positive: bool = True,
    check_finite: bool = True
) -> None:
    """Test properties of acquisition function scores."""
    if check_finite:
        assert torch.isfinite(acquisition_scores).all(), "Acquisition scores must be finite"
    
    if check_positive:
        assert (acquisition_scores >= 0).all(), "Acquisition scores must be non-negative"
    
    # Should have some variation (not all identical)
    assert acquisition_scores.std() > 0, "Acquisition scores should have some variation"


# Debugging utilities
def debug_tensor_info(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Print detailed tensor information for debugging."""
    print(f"\n=== {name} ===")
    print(f"Shape: {tensor.shape}")
    print(f"Dtype: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    print(f"Requires grad: {tensor.requires_grad}")
    print(f"Min: {tensor.min().item():.6f}")
    print(f"Max: {tensor.max().item():.6f}")
    print(f"Mean: {tensor.mean().item():.6f}")
    print(f"Std: {tensor.std().item():.6f}")
    print(f"Has NaN: {torch.isnan(tensor).any().item()}")
    print(f"Has Inf: {torch.isinf(tensor).any().item()}")
    print("=" * (len(name) + 8))


def compare_models(model1: torch.nn.Module, model2: torch.nn.Module, 
                  input_tensor: torch.Tensor) -> Dict[str, float]:
    """Compare outputs of two models."""
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        out1 = model1(input_tensor)
        out2 = model2(input_tensor)
    
    diff = (out1 - out2).abs()
    
    return {
        "max_diff": diff.max().item(),
        "mean_diff": diff.mean().item(),
        "relative_diff": (diff / (out1.abs() + 1e-8)).mean().item()
    }