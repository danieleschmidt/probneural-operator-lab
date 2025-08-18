"""Custom assertion helpers for testing."""

import torch
import numpy as np
import warnings
from typing import Union, Optional, Tuple, Any, Dict
from pathlib import Path


def assert_tensor_equal(
    actual: torch.Tensor, 
    expected: torch.Tensor, 
    rtol: float = 1e-5, 
    atol: float = 1e-8,
    msg: Optional[str] = None
) -> None:
    """Assert two tensors are equal within tolerance."""
    if msg is None:
        msg = f"Tensors not equal within tolerance (rtol={rtol}, atol={atol})"
    
    # Check shapes first
    assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} != {expected.shape}"
    
    # Check values
    assert torch.allclose(actual, expected, rtol=rtol, atol=atol), msg


def assert_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...], msg: Optional[str] = None) -> None:
    """Assert tensor has expected shape."""
    if msg is None:
        msg = f"Expected shape {expected_shape}, got {tensor.shape}"
    assert tensor.shape == expected_shape, msg


def assert_tensor_dtype(tensor: torch.Tensor, expected_dtype: torch.dtype, msg: Optional[str] = None) -> None:
    """Assert tensor has expected dtype."""
    if msg is None:
        msg = f"Expected dtype {expected_dtype}, got {tensor.dtype}"
    assert tensor.dtype == expected_dtype, msg


def assert_tensor_device(tensor: torch.Tensor, expected_device: Union[str, torch.device], msg: Optional[str] = None) -> None:
    """Assert tensor is on expected device."""
    if isinstance(expected_device, str):
        expected_device = torch.device(expected_device)
    
    if msg is None:
        msg = f"Expected device {expected_device}, got {tensor.device}"
    assert tensor.device == expected_device, msg


def assert_no_nan(tensor: torch.Tensor, msg: Optional[str] = None) -> None:
    """Assert tensor contains no NaN values."""
    if msg is None:
        msg = "Tensor contains NaN values"
    assert not torch.isnan(tensor).any(), msg


def assert_no_inf(tensor: torch.Tensor, msg: Optional[str] = None) -> None:
    """Assert tensor contains no infinite values."""
    if msg is None:
        msg = "Tensor contains infinite values"
    assert not torch.isinf(tensor).any(), msg


def assert_finite(tensor: torch.Tensor, msg: Optional[str] = None) -> None:
    """Assert all tensor values are finite."""
    if msg is None:
        msg = "Tensor contains non-finite values"
    assert torch.isfinite(tensor).all(), msg


def assert_positive(tensor: torch.Tensor, msg: Optional[str] = None) -> None:
    """Assert all tensor values are positive."""
    if msg is None:
        msg = "Tensor contains non-positive values"
    assert (tensor > 0).all(), msg


def assert_in_range(tensor: torch.Tensor, min_val: float, max_val: float, msg: Optional[str] = None) -> None:
    """Assert tensor values are in specified range."""
    if msg is None:
        msg = f"Tensor values not in range [{min_val}, {max_val}]"
    assert (tensor >= min_val).all() and (tensor <= max_val).all(), msg


def assert_normalized(tensor: torch.Tensor, dim: int = -1, rtol: float = 1e-5, msg: Optional[str] = None) -> None:
    """Assert tensor is normalized along specified dimension."""
    norms = torch.norm(tensor, dim=dim)
    expected_norms = torch.ones_like(norms)
    
    if msg is None:
        msg = f"Tensor not normalized along dim {dim}"
    
    assert torch.allclose(norms, expected_norms, rtol=rtol), msg


def assert_symmetric(tensor: torch.Tensor, rtol: float = 1e-5, msg: Optional[str] = None) -> None:
    """Assert tensor is symmetric (for 2D tensors)."""
    assert tensor.dim() == 2, "Tensor must be 2D for symmetry check"
    assert tensor.shape[0] == tensor.shape[1], "Tensor must be square for symmetry check"
    
    if msg is None:
        msg = "Tensor is not symmetric"
    
    assert torch.allclose(tensor, tensor.T, rtol=rtol), msg


def assert_positive_definite(tensor: torch.Tensor, msg: Optional[str] = None) -> None:
    """Assert tensor is positive definite."""
    assert tensor.dim() == 2, "Tensor must be 2D for positive definite check"
    assert tensor.shape[0] == tensor.shape[1], "Tensor must be square for positive definite check"
    
    eigenvals = torch.linalg.eigvals(tensor).real
    
    if msg is None:
        msg = "Tensor is not positive definite"
    
    assert (eigenvals > 0).all(), msg


def assert_gradient_exists(tensor: torch.Tensor, msg: Optional[str] = None) -> None:
    """Assert tensor has gradient."""
    if msg is None:
        msg = "Tensor has no gradient"
    assert tensor.grad is not None, msg


def assert_gradient_finite(tensor: torch.Tensor, msg: Optional[str] = None) -> None:
    """Assert tensor gradient is finite."""
    assert_gradient_exists(tensor)
    assert_finite(tensor.grad, msg)


def assert_model_output_shape(model: torch.nn.Module, input_tensor: torch.Tensor, expected_shape: Tuple[int, ...]) -> None:
    """Assert model output has expected shape."""
    with torch.no_grad():
        output = model(input_tensor)
    assert_tensor_shape(output, expected_shape, f"Model output shape mismatch")


def assert_model_deterministic(model: torch.nn.Module, input_tensor: torch.Tensor, num_runs: int = 3) -> None:
    """Assert model gives deterministic outputs."""
    model.eval()
    outputs = []
    
    for _ in range(num_runs):
        with torch.no_grad():
            output = model(input_tensor)
        outputs.append(output)
    
    for i in range(1, num_runs):
        assert_tensor_equal(outputs[0], outputs[i], msg="Model outputs are not deterministic")


def assert_loss_decreasing(losses: list, tolerance: int = 5, msg: Optional[str] = None) -> None:
    """Assert loss generally decreases (allowing for some fluctuation)."""
    if len(losses) < 10:
        warnings.warn("Loss list too short for reliable decreasing trend check")
        return
    
    # Check if overall trend is decreasing
    start_avg = np.mean(losses[:5])
    end_avg = np.mean(losses[-5:])
    
    if msg is None:
        msg = f"Loss not decreasing: start={start_avg:.4f}, end={end_avg:.4f}"
    
    assert end_avg < start_avg, msg


def assert_convergence(values: list, threshold: float = 1e-6, window: int = 10, msg: Optional[str] = None) -> None:
    """Assert values have converged."""
    if len(values) < window:
        warnings.warn(f"Value list too short for convergence check (need at least {window})")
        return
    
    recent_values = values[-window:]
    std = np.std(recent_values)
    
    if msg is None:
        msg = f"Values have not converged: std={std:.6f} > threshold={threshold:.6f}"
    
    assert std < threshold, msg


def assert_uncertainty_calibrated(predictions: torch.Tensor, uncertainties: torch.Tensor, true_values: torch.Tensor, 
                                confidence_level: float = 0.95, tolerance: float = 0.05, msg: Optional[str] = None) -> None:
    """Assert uncertainty estimates are well-calibrated."""
    z_scores = torch.abs(predictions - true_values) / (uncertainties + 1e-8)
    
    # For Gaussian uncertainty, z-scores should be within confidence interval
    from scipy.stats import norm
    critical_value = norm.ppf((1 + confidence_level) / 2)
    
    fraction_within = (z_scores <= critical_value).float().mean()
    expected_fraction = confidence_level
    
    if msg is None:
        msg = f"Uncertainty not calibrated: {fraction_within:.3f} != {expected_fraction:.3f} ± {tolerance:.3f}"
    
    assert abs(fraction_within - expected_fraction) <= tolerance, msg


def assert_file_exists(file_path: Union[str, Path], msg: Optional[str] = None) -> None:
    """Assert file exists."""
    path = Path(file_path)
    if msg is None:
        msg = f"File does not exist: {path}"
    assert path.exists(), msg


def assert_directory_exists(dir_path: Union[str, Path], msg: Optional[str] = None) -> None:
    """Assert directory exists."""
    path = Path(dir_path)
    if msg is None:
        msg = f"Directory does not exist: {path}"
    assert path.exists() and path.is_dir(), msg


def assert_config_valid(config: Dict[str, Any], required_keys: list, msg: Optional[str] = None) -> None:
    """Assert configuration contains required keys."""
    missing_keys = [key for key in required_keys if key not in config]
    
    if msg is None:
        msg = f"Configuration missing required keys: {missing_keys}"
    
    assert len(missing_keys) == 0, msg


def assert_memory_usage_reasonable(func, max_memory_mb: float = 1000, msg: Optional[str] = None):
    """Assert function doesn't use excessive memory."""
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    result = func()
    
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    mem_used = mem_after - mem_before
    
    if msg is None:
        msg = f"Function used {mem_used:.1f}MB > {max_memory_mb:.1f}MB limit"
    
    assert mem_used <= max_memory_mb, msg
    
    return result


class WarningAssertions:
    """Context manager for asserting warnings."""
    
    def __init__(self, expected_warning: type, match: Optional[str] = None):
        self.expected_warning = expected_warning
        self.match = match
        self.caught_warnings = []
    
    def __enter__(self):
        self.warning_context = warnings.catch_warnings(record=True)
        self.caught_warnings = self.warning_context.__enter__()
        warnings.simplefilter("always")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.warning_context.__exit__(exc_type, exc_val, exc_tb)
        
        matching_warnings = [
            w for w in self.caught_warnings 
            if issubclass(w.category, self.expected_warning)
        ]
        
        if self.match is not None:
            matching_warnings = [
                w for w in matching_warnings
                if self.match in str(w.message)
            ]
        
        assert len(matching_warnings) > 0, f"Expected warning {self.expected_warning.__name__} not raised"


def assert_warns(expected_warning: type, match: Optional[str] = None):
    """Assert that a warning is raised."""
    return WarningAssertions(expected_warning, match)