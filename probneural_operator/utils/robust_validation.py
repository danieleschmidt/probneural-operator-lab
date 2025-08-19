"""
Enhanced robust validation system for probabilistic neural operators.

This module provides comprehensive validation with graceful fallbacks,
detailed error reporting, and automatic recovery mechanisms.
"""

import logging
import traceback
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from functools import wraps
import time
import os
import sys

from .exceptions import (
    ValidationError, 
    ModelInitializationError,
    DataLoadingError
)

logger = logging.getLogger(__name__)

class RobustValidator:
    """Enhanced validator with automatic fallbacks and recovery."""
    
    def __init__(self, 
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 enable_fallbacks: bool = True):
        """Initialize robust validator.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries (seconds)
            enable_fallbacks: Whether to enable automatic fallbacks
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_fallbacks = enable_fallbacks
        self.validation_history = []
    
    def validate_with_retry(self, 
                          validation_func: Callable,
                          data: Any,
                          fallback_func: Optional[Callable] = None,
                          **kwargs) -> Tuple[bool, Any, Optional[str]]:
        """Validate data with automatic retry and fallback.
        
        Args:
            validation_func: Function to validate data
            data: Data to validate
            fallback_func: Optional fallback function
            **kwargs: Additional arguments for validation
            
        Returns:
            Tuple of (success, result/data, error_message)
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = validation_func(data, **kwargs)
                
                # Log successful validation
                self.validation_history.append({
                    'timestamp': time.time(),
                    'status': 'success',
                    'attempt': attempt + 1,
                    'data_type': type(data).__name__
                })
                
                return True, result, None
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Validation attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                    continue
                else:
                    break
        
        # All retries failed, try fallback if available
        if self.enable_fallbacks and fallback_func:
            try:
                logger.info("Attempting fallback validation...")
                fallback_result = fallback_func(data, **kwargs)
                
                self.validation_history.append({
                    'timestamp': time.time(),
                    'status': 'fallback_success',
                    'attempt': self.max_retries + 1,
                    'data_type': type(data).__name__
                })
                
                return True, fallback_result, f"Fallback used after {self.max_retries} failed attempts"
                
            except Exception as fallback_error:
                logger.error(f"Fallback validation also failed: {fallback_error}")
        
        # Complete failure
        self.validation_history.append({
            'timestamp': time.time(),
            'status': 'failure',
            'attempt': self.max_retries + 1,
            'data_type': type(data).__name__,
            'error': last_error
        })
        
        return False, data, last_error

def robust_validation(max_retries: int = 3, 
                     retry_delay: float = 1.0,
                     fallback_func: Optional[Callable] = None):
    """Decorator for robust validation with retry and fallback."""
    
    def decorator(validation_func: Callable):
        @wraps(validation_func)
        def wrapper(*args, **kwargs):
            validator = RobustValidator(max_retries, retry_delay, True)
            
            def _validate(data):
                return validation_func(*args, **kwargs)
            
            success, result, error = validator.validate_with_retry(
                _validate, 
                args[0] if args else None,
                fallback_func
            )
            
            if not success:
                raise ValidationError(f"Validation failed after {max_retries} retries: {error}")
            
            return result
        
        return wrapper
    return decorator

def validate_tensor_with_fallback(tensor: Any, 
                                expected_shape: Optional[Tuple] = None,
                                expected_dtype: Optional[Any] = None,
                                allow_nan: bool = False,
                                allow_inf: bool = False) -> Tuple[bool, Any, Optional[str]]:
    """Validate tensor with comprehensive checks and fallbacks."""
    
    try:
        # Import torch only when needed
        import torch
        
        # Check if input is actually a tensor
        if not torch.is_tensor(tensor):
            try:
                tensor = torch.tensor(tensor)
            except Exception as e:
                return False, tensor, f"Cannot convert to tensor: {e}"
        
        # Check shape
        if expected_shape and tensor.shape != expected_shape:
            # Try to reshape if possible
            try:
                tensor = tensor.reshape(expected_shape)
            except Exception:
                return False, tensor, f"Shape mismatch: got {tensor.shape}, expected {expected_shape}"
        
        # Check dtype
        if expected_dtype and tensor.dtype != expected_dtype:
            try:
                tensor = tensor.to(expected_dtype)
            except Exception:
                return False, tensor, f"Cannot convert to dtype {expected_dtype}"
        
        # Check for NaN/Inf values
        if not allow_nan and torch.isnan(tensor).any():
            if hasattr(torch, 'nan_to_num'):
                tensor = torch.nan_to_num(tensor, nan=0.0)
            else:
                return False, tensor, "Tensor contains NaN values"
        
        if not allow_inf and torch.isinf(tensor).any():
            if hasattr(torch, 'nan_to_num'):
                tensor = torch.nan_to_num(tensor, posinf=1e6, neginf=-1e6)
            else:
                return False, tensor, "Tensor contains infinite values"
        
        return True, tensor, None
        
    except ImportError:
        # Fallback for when torch is not available
        try:
            import numpy as np
            
            if not isinstance(tensor, np.ndarray):
                tensor = np.array(tensor)
            
            if expected_shape and tensor.shape != expected_shape:
                tensor = tensor.reshape(expected_shape)
            
            if not allow_nan and np.isnan(tensor).any():
                tensor = np.nan_to_num(tensor, nan=0.0)
            
            if not allow_inf and np.isinf(tensor).any():
                tensor = np.nan_to_num(tensor, posinf=1e6, neginf=-1e6)
            
            return True, tensor, None
            
        except Exception as e:
            return False, tensor, f"Validation failed with numpy fallback: {e}"
    
    except Exception as e:
        return False, tensor, f"Tensor validation failed: {e}"

def validate_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize model configuration with robust fallbacks."""
    
    # Define required fields with fallback values
    required_fields = {
        'input_dim': 1,
        'output_dim': 1,
        'width': 64,
        'depth': 4,
        'modes': 16
    }
    
    # Define valid ranges
    valid_ranges = {
        'input_dim': (1, 1000),
        'output_dim': (1, 1000),
        'width': (8, 2048),
        'depth': (1, 20),
        'modes': (1, 512)
    }
    
    validated_config = config.copy()
    warnings = []
    
    # Check required fields
    for field, default_value in required_fields.items():
        if field not in validated_config:
            validated_config[field] = default_value
            warnings.append(f"Missing field '{field}', using default: {default_value}")
    
    # Validate ranges
    for field, (min_val, max_val) in valid_ranges.items():
        if field in validated_config:
            value = validated_config[field]
            
            if not isinstance(value, (int, float)):
                try:
                    value = float(value)
                    validated_config[field] = int(value) if field != 'learning_rate' else value
                except (ValueError, TypeError):
                    validated_config[field] = required_fields[field]
                    warnings.append(f"Invalid type for '{field}', using default: {required_fields[field]}")
                    continue
            
            if value < min_val:
                validated_config[field] = min_val
                warnings.append(f"'{field}' below minimum, clamped to {min_val}")
            elif value > max_val:
                validated_config[field] = max_val
                warnings.append(f"'{field}' above maximum, clamped to {max_val}")
    
    # Log warnings
    if warnings:
        logger.warning(f"Model config validation warnings: {warnings}")
    
    return validated_config

def validate_training_data(data: Any, 
                         min_samples: int = 10,
                         max_samples: int = 1000000) -> Tuple[bool, Any, List[str]]:
    """Validate training data with comprehensive checks."""
    
    issues = []
    
    try:
        # Check if we have torch available
        try:
            import torch
            torch_available = True
        except ImportError:
            torch_available = False
            import numpy as np
        
        # Convert to appropriate format
        if torch_available and not torch.is_tensor(data):
            try:
                data = torch.tensor(data)
            except Exception as e:
                issues.append(f"Cannot convert to tensor: {e}")
                return False, data, issues
        elif not torch_available and not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
            except Exception as e:
                issues.append(f"Cannot convert to numpy array: {e}")
                return False, data, issues
        
        # Check number of samples
        num_samples = data.shape[0] if len(data.shape) > 0 else 1
        
        if num_samples < min_samples:
            issues.append(f"Too few samples: {num_samples} < {min_samples}")
        
        if num_samples > max_samples:
            issues.append(f"Too many samples: {num_samples} > {max_samples}")
            # Optionally subsample
            if torch_available:
                data = data[:max_samples]
            else:
                data = data[:max_samples]
            issues.append(f"Data subsampled to {max_samples} samples")
        
        # Check for data quality issues
        if torch_available:
            if torch.isnan(data).any():
                issues.append("Data contains NaN values")
                data = torch.nan_to_num(data, nan=0.0)
            
            if torch.isinf(data).any():
                issues.append("Data contains infinite values")
                data = torch.nan_to_num(data, posinf=1e6, neginf=-1e6)
        else:
            if np.isnan(data).any():
                issues.append("Data contains NaN values")
                data = np.nan_to_num(data, nan=0.0)
            
            if np.isinf(data).any():
                issues.append("Data contains infinite values")
                data = np.nan_to_num(data, posinf=1e6, neginf=-1e6)
        
        # Check data range
        data_min = data.min()
        data_max = data.max()
        
        if abs(data_max - data_min) < 1e-10:
            issues.append("Data has very small dynamic range")
        
        if abs(data_max) > 1e6 or abs(data_min) > 1e6:
            issues.append("Data values are very large, consider normalization")
        
        return len(issues) == 0, data, issues
        
    except Exception as e:
        issues.append(f"Data validation failed: {e}")
        return False, data, issues

class SafeExecutor:
    """Execute operations safely with automatic error recovery."""
    
    def __init__(self, max_retries: int = 3, timeout: float = 30.0):
        self.max_retries = max_retries
        self.timeout = timeout
    
    def execute_safely(self, 
                      operation: Callable,
                      fallback_operation: Optional[Callable] = None,
                      *args, **kwargs) -> Tuple[bool, Any, Optional[str]]:
        """Execute operation safely with timeout and retry."""
        
        import signal
        import threading
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {self.timeout} seconds")
        
        for attempt in range(self.max_retries):
            try:
                # Set timeout if supported
                if hasattr(signal, 'SIGALRM'):
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(int(self.timeout))
                
                result = operation(*args, **kwargs)
                
                # Clear timeout
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)
                
                return True, result, None
                
            except Exception as e:
                # Clear timeout
                if hasattr(signal, 'SIGALRM'):
                    signal.alarm(0)
                
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt == self.max_retries - 1:
                    # Try fallback on final attempt
                    if fallback_operation:
                        try:
                            result = fallback_operation(*args, **kwargs)
                            return True, result, f"Used fallback after {self.max_retries} attempts"
                        except Exception as fallback_error:
                            return False, None, f"Original error: {e}, Fallback error: {fallback_error}"
                    else:
                        return False, None, str(e)
        
        return False, None, "All attempts failed"

# Global instances for easy access
global_validator = RobustValidator()
global_executor = SafeExecutor()

# Convenience functions
def validate_safely(validation_func: Callable, data: Any, **kwargs):
    """Convenience function for safe validation."""
    return global_validator.validate_with_retry(validation_func, data, **kwargs)

def execute_safely(operation: Callable, *args, **kwargs):
    """Convenience function for safe execution."""
    return global_executor.execute_safely(operation, *args, **kwargs)