"""
Security and input sanitization utilities for the ProbNeural-Operator-Lab framework.

This module provides comprehensive security measures including tensor validation,
memory monitoring, safe file operations, and protection against adversarial inputs.
"""

import os
import hashlib
import tempfile
import contextlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import warnings
import logging

import torch
import numpy as np

from .validation import (
    validate_tensor_shape, validate_tensor_finite, validate_tensor_dtype,
    ValidationError
)
from .exceptions import SecurityError, MemoryError as FrameworkMemoryError

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Exception raised for security-related issues."""
    pass


class TensorSecurityValidator:
    """Validates tensors for security concerns."""
    
    def __init__(self, 
                 max_tensor_size_gb: float = 10.0,
                 max_dimensions: int = 10,
                 allowed_dtypes: Optional[List[torch.dtype]] = None):
        """Initialize tensor security validator.
        
        Args:
            max_tensor_size_gb: Maximum allowed tensor size in GB
            max_dimensions: Maximum number of tensor dimensions
            allowed_dtypes: List of allowed data types
        """
        self.max_tensor_size_gb = max_tensor_size_gb
        self.max_dimensions = max_dimensions
        self.allowed_dtypes = allowed_dtypes or [
            torch.float32, torch.float64, torch.int32, torch.int64,
            torch.complex64, torch.complex128
        ]
        
        # Security thresholds
        self.max_tensor_size_bytes = int(max_tensor_size_gb * 1024**3)
    
    def validate_tensor(self, tensor: torch.Tensor, name: str = "tensor") -> None:
        """Validate tensor for security concerns.
        
        Args:
            tensor: Tensor to validate
            name: Name of tensor for error messages
            
        Raises:
            SecurityError: If security validation fails
        """
        if not isinstance(tensor, torch.Tensor):
            raise SecurityError(f"{name} must be a torch.Tensor")
        
        # Check tensor size
        tensor_size_bytes = tensor.element_size() * tensor.numel()
        if tensor_size_bytes > self.max_tensor_size_bytes:
            raise SecurityError(
                f"{name} size ({tensor_size_bytes / 1024**3:.2f} GB) exceeds maximum "
                f"allowed size ({self.max_tensor_size_gb} GB)"
            )
        
        # Check dimensions
        if tensor.ndim > self.max_dimensions:
            raise SecurityError(
                f"{name} has {tensor.ndim} dimensions, maximum allowed is {self.max_dimensions}"
            )
        
        # Check data type
        if tensor.dtype not in self.allowed_dtypes:
            raise SecurityError(
                f"{name} has disallowed dtype {tensor.dtype}, allowed: {self.allowed_dtypes}"
            )
        
        # Check for suspicious values
        self._check_suspicious_values(tensor, name)
    
    def _check_suspicious_values(self, tensor: torch.Tensor, name: str) -> None:
        """Check for potentially malicious values in tensor.
        
        Args:
            tensor: Tensor to check
            name: Name of tensor for error messages
        """
        # Check for NaN/Inf values
        if torch.any(torch.isnan(tensor)):
            raise SecurityError(f"{name} contains NaN values")
        
        if torch.any(torch.isinf(tensor)):
            raise SecurityError(f"{name} contains infinite values")
        
        # Check for extremely large values that could cause overflow
        if tensor.dtype.is_floating_point:
            max_val = torch.max(torch.abs(tensor)).item()
            if max_val > 1e10:
                warnings.warn(
                    f"{name} contains very large values (max: {max_val:.2e})",
                    UserWarning
                )
        
        # Check for suspicious patterns (all zeros, all ones, etc.)
        if tensor.numel() > 100:  # Only check for larger tensors
            unique_values = torch.unique(tensor)
            if len(unique_values) == 1:
                warnings.warn(f"{name} contains only constant values", UserWarning)


class MemoryMonitor:
    """Monitors and controls memory usage."""
    
    def __init__(self, 
                 max_memory_gb: float = 32.0,
                 warning_threshold: float = 0.8,
                 critical_threshold: float = 0.95):
        """Initialize memory monitor.
        
        Args:
            max_memory_gb: Maximum allowed memory usage in GB
            warning_threshold: Warning threshold as fraction of max
            critical_threshold: Critical threshold as fraction of max
        """
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        
    def check_memory_usage(self, operation: str = "operation") -> Dict[str, Any]:
        """Check current memory usage.
        
        Args:
            operation: Name of operation for logging
            
        Returns:
            Dictionary with memory usage information
            
        Raises:
            FrameworkMemoryError: If memory usage is critical
        """
        import psutil
        import gc
        
        # System memory
        system_memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        memory_info = {
            'system_total_gb': system_memory.total / 1024**3,
            'system_used_gb': system_memory.used / 1024**3,
            'system_percent': system_memory.percent,
            'process_rss_gb': process_memory.rss / 1024**3,
            'process_vms_gb': process_memory.vms / 1024**3
        }
        
        # GPU memory if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                memory_info[f'gpu_{i}_allocated_gb'] = allocated
                memory_info[f'gpu_{i}_reserved_gb'] = reserved
        
        # Check thresholds
        current_usage = process_memory.rss
        usage_fraction = current_usage / self.max_memory_bytes
        
        if usage_fraction > self.critical_threshold:
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Re-check after cleanup
            process_memory = process.memory_info()
            current_usage = process_memory.rss
            usage_fraction = current_usage / self.max_memory_bytes
            
            if usage_fraction > self.critical_threshold:
                raise FrameworkMemoryError(
                    f"Critical memory usage during {operation}: "
                    f"{usage_fraction:.1%} of {self.max_memory_bytes / 1024**3:.1f} GB limit"
                )
        
        elif usage_fraction > self.warning_threshold:
            warnings.warn(
                f"High memory usage during {operation}: "
                f"{usage_fraction:.1%} of {self.max_memory_bytes / 1024**3:.1f} GB limit",
                UserWarning
            )
        
        memory_info['usage_fraction'] = usage_fraction
        return memory_info
    
    @contextlib.contextmanager
    def monitor_operation(self, operation_name: str):
        """Context manager to monitor memory during an operation.
        
        Args:
            operation_name: Name of the operation
        """
        initial_memory = self.check_memory_usage(f"{operation_name}_start")
        
        try:
            yield
        finally:
            final_memory = self.check_memory_usage(f"{operation_name}_end")
            
            # Log memory change
            memory_change = (final_memory['process_rss_gb'] - 
                           initial_memory['process_rss_gb'])
            
            if memory_change > 0.1:  # Log if more than 100MB change
                logger.info(
                    f"Memory change during {operation_name}: "
                    f"{memory_change:+.2f} GB"
                )


class SafeFileHandler:
    """Handles file operations safely with security checks."""
    
    def __init__(self, 
                 allowed_extensions: Optional[List[str]] = None,
                 max_file_size_mb: float = 1000.0,
                 safe_directories: Optional[List[str]] = None):
        """Initialize safe file handler.
        
        Args:
            allowed_extensions: List of allowed file extensions
            max_file_size_mb: Maximum file size in MB
            safe_directories: List of safe directory patterns
        """
        self.allowed_extensions = allowed_extensions or [
            '.pt', '.pth', '.h5', '.hdf5', '.npz', '.npy', '.pkl', '.json',
            '.yaml', '.yml', '.txt', '.log', '.csv'
        ]
        self.max_file_size_bytes = int(max_file_size_mb * 1024**2)
        self.safe_directories = safe_directories or []
        
    def validate_file_path(self, file_path: Union[str, Path]) -> Path:
        """Validate file path for security.
        
        Args:
            file_path: File path to validate
            
        Returns:
            Validated Path object
            
        Raises:
            SecurityError: If path validation fails
        """
        path = Path(file_path).resolve()
        
        # Check for directory traversal attempts
        if '..' in str(path):
            raise SecurityError(f"Directory traversal detected in path: {file_path}")
        
        # Check file extension
        if path.suffix.lower() not in self.allowed_extensions:
            raise SecurityError(
                f"File extension {path.suffix} not allowed. "
                f"Allowed: {self.allowed_extensions}"
            )
        
        # Check if in safe directory (if specified)
        if self.safe_directories:
            path_str = str(path)
            is_safe = any(safe_dir in path_str for safe_dir in self.safe_directories)
            if not is_safe:
                raise SecurityError(
                    f"File path {path} not in allowed directories: {self.safe_directories}"
                )
        
        return path
    
    def safe_load(self, file_path: Union[str, Path]) -> Any:
        """Safely load data from file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Loaded data
            
        Raises:
            SecurityError: If file loading fails security checks
        """
        path = self.validate_file_path(file_path)
        
        # Check if file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Check file size
        file_size = path.stat().st_size
        if file_size > self.max_file_size_bytes:
            raise SecurityError(
                f"File size ({file_size / 1024**2:.1f} MB) exceeds maximum "
                f"allowed size ({self.max_file_size_bytes / 1024**2:.1f} MB)"
            )
        
        # Load based on extension
        suffix = path.suffix.lower()
        
        try:
            if suffix in ['.pt', '.pth']:
                # PyTorch tensors - use safe loading
                return torch.load(path, map_location='cpu', weights_only=True)
            
            elif suffix in ['.h5', '.hdf5']:
                import h5py
                with h5py.File(path, 'r') as f:
                    # Read metadata first
                    if 'data' in f:
                        return torch.tensor(f['data'][:])
                    else:
                        raise SecurityError("HDF5 file structure not recognized")
            
            elif suffix in ['.npy', '.npz']:
                data = np.load(path, allow_pickle=False)  # Disable pickle for security
                if isinstance(data, np.ndarray):
                    return torch.from_numpy(data)
                else:
                    return {key: torch.from_numpy(array) for key, array in data.items()}
            
            elif suffix == '.json':
                import json
                with open(path, 'r') as f:
                    return json.load(f)
            
            elif suffix in ['.yaml', '.yml']:
                try:
                    import yaml
                    with open(path, 'r') as f:
                        return yaml.safe_load(f)  # Use safe_load
                except ImportError:
                    raise SecurityError("PyYAML not available for YAML file loading")
            
            elif suffix in ['.txt', '.log', '.csv']:
                with open(path, 'r') as f:
                    return f.read()
            
            else:
                raise SecurityError(f"No safe loader available for {suffix} files")
                
        except Exception as e:
            if isinstance(e, SecurityError):
                raise
            else:
                raise SecurityError(f"Failed to safely load file {path}: {e}")
    
    def safe_save(self, data: Any, file_path: Union[str, Path]) -> None:
        """Safely save data to file.
        
        Args:
            data: Data to save
            file_path: Path to save file
            
        Raises:
            SecurityError: If file saving fails security checks
        """
        path = self.validate_file_path(file_path)
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use temporary file for atomic writes
        with tempfile.NamedTemporaryFile(
            dir=path.parent, 
            prefix=f".{path.name}_tmp_",
            delete=False
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            
            try:
                suffix = path.suffix.lower()
                
                if suffix in ['.pt', '.pth']:
                    torch.save(data, tmp_path)
                
                elif suffix in ['.h5', '.hdf5']:
                    import h5py
                    with h5py.File(tmp_path, 'w') as f:
                        if isinstance(data, torch.Tensor):
                            f.create_dataset('data', data=data.numpy())
                        else:
                            raise SecurityError("Only tensors supported for HDF5 saving")
                
                elif suffix in ['.npy', '.npz']:
                    if isinstance(data, torch.Tensor):
                        np.save(tmp_path, data.numpy())
                    elif isinstance(data, dict):
                        np.savez(tmp_path, **{k: v.numpy() if isinstance(v, torch.Tensor) else v 
                                            for k, v in data.items()})
                    else:
                        raise SecurityError("Unsupported data type for numpy save")
                
                elif suffix == '.json':
                    import json
                    with open(tmp_path, 'w') as f:
                        json.dump(data, f, indent=2, default=str)
                
                elif suffix in ['.yaml', '.yml']:
                    try:
                        import yaml
                        with open(tmp_path, 'w') as f:
                            yaml.safe_dump(data, f)
                    except ImportError:
                        raise SecurityError("PyYAML not available for YAML file saving")
                
                elif suffix in ['.txt', '.log']:
                    with open(tmp_path, 'w') as f:
                        f.write(str(data))
                
                else:
                    raise SecurityError(f"No safe saver available for {suffix} files")
                
                # Atomic move
                tmp_path.replace(path)
                
            except Exception as e:
                # Clean up temporary file
                tmp_path.unlink(missing_ok=True)
                if isinstance(e, SecurityError):
                    raise
                else:
                    raise SecurityError(f"Failed to safely save file {path}: {e}")


class AdversarialInputDetector:
    """Detects potentially adversarial inputs."""
    
    def __init__(self, 
                 statistical_threshold: float = 3.0,
                 gradient_threshold: float = 10.0):
        """Initialize adversarial input detector.
        
        Args:
            statistical_threshold: Z-score threshold for statistical anomalies
            gradient_threshold: Threshold for gradient-based detection
        """
        self.statistical_threshold = statistical_threshold
        self.gradient_threshold = gradient_threshold
        
        # Reference statistics (updated during training)
        self.reference_stats: Dict[str, Dict[str, float]] = {}
    
    def update_reference_stats(self, data: torch.Tensor, name: str = "default") -> None:
        """Update reference statistics from clean data.
        
        Args:
            data: Clean training data
            name: Name of dataset/domain
        """
        with torch.no_grad():
            stats = {
                'mean': torch.mean(data).item(),
                'std': torch.std(data).item(),
                'min': torch.min(data).item(),
                'max': torch.max(data).item(),
                'median': torch.median(data).item()
            }
            
            self.reference_stats[name] = stats
            logger.info(f"Updated reference statistics for {name}: {stats}")
    
    def detect_statistical_anomalies(self, data: torch.Tensor, 
                                   reference_name: str = "default") -> Dict[str, Any]:
        """Detect statistical anomalies in input data.
        
        Args:
            data: Input data to check
            reference_name: Name of reference statistics to use
            
        Returns:
            Dictionary with anomaly detection results
        """
        if reference_name not in self.reference_stats:
            warnings.warn(f"No reference statistics for {reference_name}", UserWarning)
            return {'anomalies_detected': False, 'reason': 'no_reference'}
        
        ref_stats = self.reference_stats[reference_name]
        
        with torch.no_grad():
            current_mean = torch.mean(data).item()
            current_std = torch.std(data).item()
            
            # Z-score for mean
            mean_z_score = abs(current_mean - ref_stats['mean']) / (ref_stats['std'] + 1e-8)
            
            # Standard deviation ratio
            std_ratio = current_std / (ref_stats['std'] + 1e-8)
            
            # Range check
            data_min, data_max = torch.min(data).item(), torch.max(data).item()
            range_violation = (data_min < ref_stats['min'] - 3 * ref_stats['std'] or
                             data_max > ref_stats['max'] + 3 * ref_stats['std'])
            
            anomalies = []
            
            if mean_z_score > self.statistical_threshold:
                anomalies.append(f"Mean anomaly (z-score: {mean_z_score:.2f})")
            
            if std_ratio > 2.0 or std_ratio < 0.5:
                anomalies.append(f"Standard deviation anomaly (ratio: {std_ratio:.2f})")
            
            if range_violation:
                anomalies.append(f"Range violation (min: {data_min:.2f}, max: {data_max:.2f})")
            
            return {
                'anomalies_detected': len(anomalies) > 0,
                'anomalies': anomalies,
                'statistics': {
                    'mean_z_score': mean_z_score,
                    'std_ratio': std_ratio,
                    'data_min': data_min,
                    'data_max': data_max
                }
            }
    
    def detect_gradient_anomalies(self, model: torch.nn.Module, 
                                data: torch.Tensor,
                                target: torch.Tensor) -> Dict[str, Any]:
        """Detect gradient-based adversarial inputs.
        
        Args:
            model: Neural network model
            data: Input data
            target: Target values
            
        Returns:
            Dictionary with gradient anomaly detection results
        """
        data_with_grad = data.clone().detach().requires_grad_(True)
        
        try:
            # Forward pass
            output = model(data_with_grad)
            loss = torch.nn.functional.mse_loss(output, target)
            
            # Backward pass
            loss.backward()
            
            if data_with_grad.grad is not None:
                grad_norm = torch.norm(data_with_grad.grad).item()
                grad_max = torch.max(torch.abs(data_with_grad.grad)).item()
                
                anomalies = []
                
                if grad_norm > self.gradient_threshold:
                    anomalies.append(f"High gradient norm: {grad_norm:.2f}")
                
                if grad_max > self.gradient_threshold / 2:
                    anomalies.append(f"High gradient magnitude: {grad_max:.2f}")
                
                return {
                    'anomalies_detected': len(anomalies) > 0,
                    'anomalies': anomalies,
                    'gradient_norm': grad_norm,
                    'gradient_max': grad_max
                }
            else:
                return {'anomalies_detected': False, 'reason': 'no_gradients'}
                
        except Exception as e:
            logger.warning(f"Gradient anomaly detection failed: {e}")
            return {'anomalies_detected': False, 'reason': f'error: {e}'}


# Utility functions for common security operations
def sanitize_tensor(tensor: torch.Tensor, 
                   validator: Optional[TensorSecurityValidator] = None,
                   name: str = "tensor") -> torch.Tensor:
    """Sanitize a tensor by applying security validation and cleaning.
    
    Args:
        tensor: Tensor to sanitize
        validator: Security validator (creates default if None)
        name: Name for error messages
        
    Returns:
        Sanitized tensor
        
    Raises:
        SecurityError: If tensor fails security validation
    """
    if validator is None:
        validator = TensorSecurityValidator()
    
    # Validate security
    validator.validate_tensor(tensor, name)
    
    # Create a clean copy
    clean_tensor = tensor.clone().detach()
    
    # Ensure finite values
    if torch.any(torch.isnan(clean_tensor)) or torch.any(torch.isinf(clean_tensor)):
        logger.warning(f"Replacing non-finite values in {name}")
        clean_tensor = torch.where(
            torch.isfinite(clean_tensor), 
            clean_tensor, 
            torch.zeros_like(clean_tensor)
        )
    
    return clean_tensor


def compute_data_hash(data: torch.Tensor) -> str:
    """Compute SHA-256 hash of tensor data for integrity verification.
    
    Args:
        data: Tensor to hash
        
    Returns:
        SHA-256 hash string
    """
    # Convert to bytes
    data_bytes = data.detach().cpu().numpy().tobytes()
    
    # Compute hash
    hash_obj = hashlib.sha256(data_bytes)
    return hash_obj.hexdigest()


def verify_data_integrity(data: torch.Tensor, expected_hash: str) -> bool:
    """Verify data integrity using hash comparison.
    
    Args:
        data: Tensor to verify
        expected_hash: Expected SHA-256 hash
        
    Returns:
        True if hashes match, False otherwise
    """
    actual_hash = compute_data_hash(data)
    return actual_hash == expected_hash


# Context manager for secure operations
@contextlib.contextmanager
def secure_operation(operation_name: str = "operation",
                    max_memory_gb: float = 32.0,
                    tensor_validator: Optional[TensorSecurityValidator] = None):
    """Context manager for secure operations with resource monitoring.
    
    Args:
        operation_name: Name of the operation
        max_memory_gb: Maximum memory limit
        tensor_validator: Tensor security validator
    """
    memory_monitor = MemoryMonitor(max_memory_gb=max_memory_gb)
    
    with memory_monitor.monitor_operation(operation_name):
        yield {
            'memory_monitor': memory_monitor,
            'tensor_validator': tensor_validator or TensorSecurityValidator()
        }