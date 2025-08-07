"""
Custom exceptions for the ProbNeural-Operator-Lab framework.

This module defines all custom exceptions used throughout the framework,
providing clear error messages and context for different failure modes.
"""

import traceback
from typing import Any, Dict, List, Optional


class ProbNeuralOperatorError(Exception):
    """Base exception for all ProbNeural-Operator-Lab errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Initialize exception with message and optional context.
        
        Args:
            message: Error message
            context: Additional context information
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
    
    def __str__(self) -> str:
        """String representation with context."""
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class ModelError(ProbNeuralOperatorError):
    """Base exception for model-related errors."""
    pass


class ModelInitializationError(ModelError):
    """Exception raised during model initialization."""
    pass


class ModelTrainingError(ModelError):
    """Exception raised during model training."""
    pass


class PosteriorError(ProbNeuralOperatorError):
    """Base exception for posterior approximation errors."""
    pass


class PosteriorNotFittedError(PosteriorError):
    """Exception raised when posterior is not fitted."""
    
    def __init__(self, message: str = "Posterior approximation not fitted. Call fit_posterior() first."):
        super().__init__(message)


class LaplaceFitError(PosteriorError):
    """Exception raised during Laplace approximation fitting."""
    pass


class DataError(ProbNeuralOperatorError):
    """Base exception for data-related errors."""
    pass


class DataLoaderError(DataError):
    """Exception raised with data loading issues."""
    pass


class DataValidationError(DataError):
    """Exception raised when data validation fails."""
    pass


class ConfigurationError(ProbNeuralOperatorError):
    """Exception raised for configuration issues."""
    pass


class DeviceError(ProbNeuralOperatorError):
    """Exception raised for device/hardware issues."""
    pass


class MemoryError(ProbNeuralOperatorError):
    """Exception raised for memory-related issues."""
    pass


class NumericalError(ProbNeuralOperatorError):
    """Exception raised for numerical computation issues."""
    pass


class ConvergenceError(ProbNeuralOperatorError):
    """Exception raised when algorithms fail to converge."""
    pass


class ActiveLearningError(ProbNeuralOperatorError):
    """Exception raised during active learning processes."""
    pass


class CalibrationError(ProbNeuralOperatorError):
    """Exception raised during uncertainty calibration."""
    pass


class BenchmarkError(ProbNeuralOperatorError):
    """Exception raised during benchmarking."""
    pass


def handle_exception(
    func_name: str,
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    reraise_as: Optional[type] = None
) -> None:
    """Handle and optionally re-raise exceptions with additional context.
    
    Args:
        func_name: Name of function where exception occurred
        exception: Original exception
        context: Additional context to include
        reraise_as: Exception class to re-raise as
    
    Raises:
        Exception: Re-raised with additional context
    """
    full_context = {"function": func_name}
    if context:
        full_context.update(context)
    
    if reraise_as:
        if issubclass(reraise_as, ProbNeuralOperatorError):
            raise reraise_as(str(exception), full_context) from exception
        else:
            raise reraise_as(str(exception)) from exception
    else:
        # Re-raise with additional context if it's our custom exception
        if isinstance(exception, ProbNeuralOperatorError):
            exception.context.update(full_context)
        raise exception


class ErrorCollector:
    """Utility class to collect multiple errors and warnings."""
    
    def __init__(self):
        """Initialize error collector."""
        self.errors: List[Exception] = []
        self.warnings: List[str] = []
        self.context: Dict[str, Any] = {}
    
    def add_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """Add an error to the collection.
        
        Args:
            error: Exception to add
            context: Additional context
        """
        if context:
            if isinstance(error, ProbNeuralOperatorError):
                error.context.update(context)
        self.errors.append(error)
    
    def add_warning(self, message: str) -> None:
        """Add a warning message.
        
        Args:
            message: Warning message
        """
        self.warnings.append(message)
    
    def has_errors(self) -> bool:
        """Check if any errors have been collected."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if any warnings have been collected."""
        return len(self.warnings) > 0
    
    def raise_if_errors(self) -> None:
        """Raise a combined exception if any errors exist."""
        if not self.has_errors():
            return
        
        if len(self.errors) == 1:
            raise self.errors[0]
        
        # Create a combined error message
        error_messages = [str(error) for error in self.errors]
        combined_message = f"Multiple errors occurred:\n" + "\n".join(
            f"  {i+1}. {msg}" for i, msg in enumerate(error_messages)
        )
        
        raise ProbNeuralOperatorError(combined_message, {"error_count": len(self.errors)})
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of collected errors and warnings."""
        return {
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": [str(error) for error in self.errors],
            "warnings": self.warnings,
            "context": self.context
        }
    
    def clear(self) -> None:
        """Clear all collected errors and warnings."""
        self.errors.clear()
        self.warnings.clear()
        self.context.clear()


def create_detailed_traceback(exception: Exception) -> str:
    """Create a detailed traceback string for an exception.
    
    Args:
        exception: Exception to create traceback for
        
    Returns:
        Detailed traceback string
    """
    return "".join(traceback.format_exception(
        type(exception), exception, exception.__traceback__
    ))


def log_exception(
    logger,
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    level: str = "error"
) -> None:
    """Log an exception with context information.
    
    Args:
        logger: Logger instance
        exception: Exception to log
        context: Additional context
        level: Log level ("error", "warning", "info")
    """
    context_str = ""
    if context:
        context_str = f" Context: {context}"
    
    message = f"Exception occurred: {exception}{context_str}"
    
    if level == "error":
        logger.error(message, exc_info=True)
    elif level == "warning":
        logger.warning(message, exc_info=True)
    elif level == "info":
        logger.info(message, exc_info=True)
    else:
        logger.debug(message, exc_info=True)


def safe_execute(
    func,
    *args,
    default_return=None,
    exception_types=(Exception,),
    logger=None,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
):
    """Safely execute a function with exception handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments for function
        default_return: Default return value if exception occurs
        exception_types: Tuple of exception types to catch
        logger: Logger for exception logging
        context: Additional context for logging
        **kwargs: Keyword arguments for function
        
    Returns:
        Function result or default_return if exception occurs
    """
    try:
        return func(*args, **kwargs)
    except exception_types as e:
        if logger:
            log_exception(logger, e, context)
        return default_return


class RetryConfig:
    """Configuration for retry mechanisms."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        exceptions: tuple = (Exception,)
    ):
        """Initialize retry configuration.
        
        Args:
            max_attempts: Maximum number of retry attempts
            base_delay: Base delay between attempts (seconds)
            max_delay: Maximum delay between attempts (seconds)
            backoff_factor: Factor to multiply delay by each attempt
            exceptions: Tuple of exception types that trigger retry
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.exceptions = exceptions


def retry_on_exception(config: RetryConfig, logger=None):
    """Decorator for retrying functions on specific exceptions.
    
    Args:
        config: Retry configuration
        logger: Optional logger for retry messages
        
    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            delay = config.base_delay
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    return func(*args, **kwargs)
                except config.exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts - 1:
                        # Last attempt failed
                        if logger:
                            logger.error(
                                f"Function {func.__name__} failed after "
                                f"{config.max_attempts} attempts: {e}"
                            )
                        raise
                    
                    if logger:
                        logger.warning(
                            f"Attempt {attempt + 1} of {func.__name__} failed: {e}. "
                            f"Retrying in {delay} seconds..."
                        )
                    
                    import time
                    time.sleep(delay)
                    delay = min(delay * config.backoff_factor, config.max_delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator