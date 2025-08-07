"""
Comprehensive validation and error handling utilities.

This module provides robust validation functions for ensuring data integrity,
parameter bounds, and numerical stability throughout the framework.
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


class NumericalStabilityError(ValidationError):
    """Exception for numerical stability issues."""
    pass


class ParameterBoundsError(ValidationError):
    """Exception for parameter boundary violations."""
    pass


class TensorValidationError(ValidationError):
    """Exception for tensor validation failures."""
    pass


def validate_tensor_shape(
    tensor: torch.Tensor,
    expected_shape: Optional[Tuple[int, ...]] = None,
    min_dims: Optional[int] = None,
    max_dims: Optional[int] = None,
    name: str = "tensor"
) -> None:
    """Validate tensor shape and dimensions.
    
    Args:
        tensor: Input tensor to validate
        expected_shape: Expected exact shape (None dimensions ignored)
        min_dims: Minimum number of dimensions
        max_dims: Maximum number of dimensions
        name: Name of tensor for error messages
        
    Raises:
        TensorValidationError: If validation fails
    """
    if not isinstance(tensor, torch.Tensor):
        raise TensorValidationError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    shape = tensor.shape
    ndim = len(shape)
    
    # Check dimension bounds
    if min_dims is not None and ndim < min_dims:
        raise TensorValidationError(
            f"{name} must have at least {min_dims} dimensions, got {ndim}"
        )
    
    if max_dims is not None and ndim > max_dims:
        raise TensorValidationError(
            f"{name} must have at most {max_dims} dimensions, got {ndim}"
        )
    
    # Check exact shape if provided
    if expected_shape is not None:
        if len(expected_shape) != ndim:
            raise TensorValidationError(
                f"{name} must have {len(expected_shape)} dimensions, got {ndim}"
            )
        
        for i, (actual, expected) in enumerate(zip(shape, expected_shape)):
            if expected is not None and actual != expected:
                raise TensorValidationError(
                    f"{name} dimension {i} must be {expected}, got {actual}"
                )


def validate_tensor_dtype(
    tensor: torch.Tensor,
    allowed_dtypes: Union[torch.dtype, List[torch.dtype]],
    name: str = "tensor"
) -> None:
    """Validate tensor data type.
    
    Args:
        tensor: Input tensor to validate
        allowed_dtypes: Allowed data type(s)
        name: Name of tensor for error messages
        
    Raises:
        TensorValidationError: If validation fails
    """
    if isinstance(allowed_dtypes, torch.dtype):
        allowed_dtypes = [allowed_dtypes]
    
    if tensor.dtype not in allowed_dtypes:
        raise TensorValidationError(
            f"{name} must have dtype in {allowed_dtypes}, got {tensor.dtype}"
        )


def validate_tensor_finite(
    tensor: torch.Tensor,
    name: str = "tensor",
    check_nan: bool = True,
    check_inf: bool = True
) -> None:
    """Validate that tensor contains only finite values.
    
    Args:
        tensor: Input tensor to validate
        name: Name of tensor for error messages
        check_nan: Whether to check for NaN values
        check_inf: Whether to check for infinite values
        
    Raises:
        NumericalStabilityError: If validation fails
    """
    if check_nan and torch.any(torch.isnan(tensor)):
        nan_count = torch.sum(torch.isnan(tensor)).item()
        raise NumericalStabilityError(
            f"{name} contains {nan_count} NaN values"
        )
    
    if check_inf and torch.any(torch.isinf(tensor)):
        inf_count = torch.sum(torch.isinf(tensor)).item()
        raise NumericalStabilityError(
            f"{name} contains {inf_count} infinite values"
        )


def validate_parameter_bounds(
    value: Union[int, float],
    name: str,
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    exclusive_min: bool = False,
    exclusive_max: bool = False
) -> None:
    """Validate parameter is within bounds.
    
    Args:
        value: Parameter value to validate
        name: Parameter name for error messages
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        exclusive_min: Whether minimum is exclusive
        exclusive_max: Whether maximum is exclusive
        
    Raises:
        ParameterBoundsError: If validation fails
    """
    if min_value is not None:
        if exclusive_min and value <= min_value:
            raise ParameterBoundsError(
                f"{name} must be > {min_value}, got {value}"
            )
        elif not exclusive_min and value < min_value:
            raise ParameterBoundsError(
                f"{name} must be >= {min_value}, got {value}"
            )
    
    if max_value is not None:
        if exclusive_max and value >= max_value:
            raise ParameterBoundsError(
                f"{name} must be < {max_value}, got {value}"
            )
        elif not exclusive_max and value > max_value:
            raise ParameterBoundsError(
                f"{name} must be <= {max_value}, got {value}"
            )


def validate_integer_parameter(
    value: Any,
    name: str,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None
) -> int:
    """Validate and convert to integer parameter.
    
    Args:
        value: Value to validate and convert
        name: Parameter name for error messages
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        
    Returns:
        Validated integer value
        
    Raises:
        ParameterBoundsError: If validation fails
    """
    if not isinstance(value, (int, np.integer)):
        try:
            value = int(value)
        except (ValueError, TypeError):
            raise ParameterBoundsError(f"{name} must be an integer, got {type(value)}")
    
    validate_parameter_bounds(value, name, min_value, max_value)
    return int(value)


def validate_float_parameter(
    value: Any,
    name: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    exclusive_min: bool = False,
    exclusive_max: bool = False
) -> float:
    """Validate and convert to float parameter.
    
    Args:
        value: Value to validate and convert
        name: Parameter name for error messages
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        exclusive_min: Whether minimum is exclusive
        exclusive_max: Whether maximum is exclusive
        
    Returns:
        Validated float value
        
    Raises:
        ParameterBoundsError: If validation fails
    """
    if not isinstance(value, (int, float, np.number)):
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise ParameterBoundsError(f"{name} must be a number, got {type(value)}")
    
    if not np.isfinite(value):
        raise ParameterBoundsError(f"{name} must be finite, got {value}")
    
    validate_parameter_bounds(value, name, min_value, max_value, exclusive_min, exclusive_max)
    return float(value)


def check_numerical_stability(
    tensors: Dict[str, torch.Tensor],
    max_condition_number: float = 1e12,
    min_eigenvalue: float = 1e-12
) -> Dict[str, Any]:
    """Check numerical stability of tensors (matrices).
    
    Args:
        tensors: Dictionary of tensors to check
        max_condition_number: Maximum allowed condition number
        min_eigenvalue: Minimum allowed eigenvalue
        
    Returns:
        Dictionary with stability diagnostics
        
    Raises:
        NumericalStabilityError: If critical stability issues found
    """
    diagnostics = {}
    
    for name, tensor in tensors.items():
        if tensor.ndim < 2:
            continue  # Skip non-matrix tensors
        
        # Reshape to 2D if needed
        original_shape = tensor.shape
        if tensor.ndim > 2:
            tensor = tensor.view(-1, tensor.size(-1))
        
        try:
            # Condition number check
            if tensor.size(0) == tensor.size(1):  # Square matrix
                cond = torch.linalg.cond(tensor).item()
                if cond > max_condition_number:
                    raise NumericalStabilityError(
                        f"Matrix {name} is ill-conditioned (cond={cond:.2e})"
                    )
                
                # Eigenvalue check for symmetric matrices
                if torch.allclose(tensor, tensor.T, rtol=1e-5):
                    eigenvals = torch.linalg.eigvals(tensor).real
                    min_eig = torch.min(eigenvals).item()
                    if min_eig < min_eigenvalue:
                        warnings.warn(
                            f"Matrix {name} has very small eigenvalue ({min_eig:.2e})",
                            UserWarning
                        )
                
                diagnostics[name] = {
                    "condition_number": cond,
                    "min_eigenvalue": min_eig if 'min_eig' in locals() else None,
                    "shape": original_shape
                }
            else:
                # For non-square matrices, check singular values
                svd = torch.linalg.svd(tensor, compute_uv=False)
                cond = (svd[0] / svd[-1]).item()
                if cond > max_condition_number:
                    warnings.warn(
                        f"Matrix {name} has high condition number ({cond:.2e})",
                        UserWarning
                    )
                
                diagnostics[name] = {
                    "condition_number": cond,
                    "singular_values": svd.cpu().numpy(),
                    "shape": original_shape
                }
                
        except Exception as e:
            logger.warning(f"Could not check stability of {name}: {e}")
            diagnostics[name] = {"error": str(e), "shape": original_shape}
    
    return diagnostics


def validate_training_data(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    min_samples: int = 1
) -> None:
    """Validate training data consistency.
    
    Args:
        inputs: Input tensor
        targets: Target tensor
        min_samples: Minimum number of samples required
        
    Raises:
        ValidationError: If validation fails
    """
    # Basic tensor validation
    validate_tensor_shape(inputs, min_dims=2, name="inputs")
    validate_tensor_shape(targets, min_dims=2, name="targets")
    
    # Check batch size consistency
    if inputs.size(0) != targets.size(0):
        raise ValidationError(
            f"Input batch size ({inputs.size(0)}) != target batch size ({targets.size(0)})"
        )
    
    # Check minimum samples
    batch_size = inputs.size(0)
    if batch_size < min_samples:
        raise ValidationError(
            f"Need at least {min_samples} samples, got {batch_size}"
        )
    
    # Check for finite values
    validate_tensor_finite(inputs, "inputs")
    validate_tensor_finite(targets, "targets")
    
    # Check for reasonable value ranges
    input_range = (inputs.min().item(), inputs.max().item())
    target_range = (targets.min().item(), targets.max().item())
    
    if abs(input_range[1] - input_range[0]) < 1e-10:
        warnings.warn("Input data has very small range", UserWarning)
    
    if abs(target_range[1] - target_range[0]) < 1e-10:
        warnings.warn("Target data has very small range", UserWarning)


def safe_inversion(
    matrix: torch.Tensor,
    regularization: float = 1e-6,
    method: str = "cholesky"
) -> torch.Tensor:
    """Safely compute matrix inverse with regularization.
    
    Args:
        matrix: Input matrix to invert
        regularization: Regularization parameter for numerical stability
        method: Inversion method ("cholesky", "lu", "svd")
        
    Returns:
        Inverted matrix
        
    Raises:
        NumericalStabilityError: If inversion fails
    """
    if matrix.ndim != 2 or matrix.size(0) != matrix.size(1):
        raise ValueError("Matrix must be square")
    
    # Add regularization to diagonal
    regularized = matrix + regularization * torch.eye(
        matrix.size(0), device=matrix.device, dtype=matrix.dtype
    )
    
    try:
        if method == "cholesky":
            # Try Cholesky decomposition (assumes positive definite)
            L = torch.linalg.cholesky(regularized)
            inv = torch.cholesky_inverse(L)
        elif method == "lu":
            # LU decomposition
            inv = torch.linalg.inv(regularized)
        elif method == "svd":
            # SVD-based pseudo-inverse
            U, S, Vh = torch.linalg.svd(regularized, full_matrices=False)
            # Filter out small singular values
            S_inv = torch.where(S > regularization, 1.0 / S, 0.0)
            inv = Vh.T @ torch.diag(S_inv) @ U.T
        else:
            raise ValueError(f"Unknown inversion method: {method}")
        
        # Validate result
        validate_tensor_finite(inv, f"inverted matrix ({method})")
        
        return inv
        
    except Exception as e:
        raise NumericalStabilityError(f"Matrix inversion failed ({method}): {e}")


def validate_device_compatibility(
    tensors: List[torch.Tensor],
    target_device: Optional[torch.device] = None
) -> torch.device:
    """Validate and ensure device compatibility.
    
    Args:
        tensors: List of tensors to check
        target_device: Target device (if None, use first tensor's device)
        
    Returns:
        Validated device
        
    Raises:
        ValidationError: If device compatibility issues found
    """
    if not tensors:
        return torch.device("cpu")
    
    if target_device is None:
        target_device = tensors[0].device
    
    # Check all tensors are on compatible devices
    for i, tensor in enumerate(tensors):
        if tensor.device != target_device:
            raise ValidationError(
                f"Tensor {i} is on device {tensor.device}, expected {target_device}"
            )
    
    # Check CUDA availability if needed
    if target_device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValidationError("CUDA device requested but CUDA is not available")
        
        if target_device.index is not None:
            if target_device.index >= torch.cuda.device_count():
                raise ValidationError(
                    f"CUDA device {target_device.index} requested but only "
                    f"{torch.cuda.device_count()} devices available"
                )
    
    return target_device


class ValidationContext:
    """Context manager for validation with automatic error collection."""
    
    def __init__(self, strict: bool = True):
        """Initialize validation context.
        
        Args:
            strict: If True, raise on first error; if False, collect all errors
        """
        self.strict = strict
        self.errors = []
        self.warnings = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.errors and self.strict:
            raise ValidationError(f"Validation failed with {len(self.errors)} errors")
        return False
    
    def validate(self, condition: bool, message: str, warning: bool = False):
        """Add validation check.
        
        Args:
            condition: Condition to check
            message: Error message if condition fails
            warning: If True, treat as warning instead of error
        """
        if not condition:
            if warning:
                self.warnings.append(message)
                warnings.warn(message, UserWarning)
            else:
                self.errors.append(message)
                if self.strict:
                    raise ValidationError(message)
    
    def get_summary(self) -> Dict[str, List[str]]:
        """Get validation summary."""
        return {"errors": self.errors, "warnings": self.warnings}