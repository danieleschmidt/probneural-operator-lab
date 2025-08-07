"""Utility functions for ProbNeural-Operator-Lab."""

from .performance import PerformanceProfiler, MemoryTracker
from .optimization import ModelOptimizer, DataLoaderOptimizer
from .validation import (
    ValidationError, NumericalStabilityError, ParameterBoundsError,
    TensorValidationError, ValidationContext, validate_tensor_shape,
    validate_tensor_dtype, validate_tensor_finite, validate_parameter_bounds,
    validate_integer_parameter, validate_float_parameter, check_numerical_stability,
    validate_training_data, safe_inversion, validate_device_compatibility
)
from .exceptions import (
    ProbNeuralOperatorError, ModelError, ModelInitializationError, ModelTrainingError,
    PosteriorError, PosteriorNotFittedError, LaplaceFitError, DataError, DataLoaderError,
    DataValidationError, ConfigurationError, DeviceError, MemoryError, NumericalError,
    ConvergenceError, ActiveLearningError, CalibrationError, BenchmarkError,
    ErrorCollector, handle_exception, retry_on_exception, RetryConfig
)
from .logging_config import (
    setup_logging, get_logger, JSONFormatter, TrainingProgressLogger,
    UncertaintyTracker, PerformanceTracker, ContextLogger,
    create_training_logger, create_uncertainty_logger, create_performance_logger
)
from .monitoring import (
    MetricCollector, ResourceMonitor, TrainingMonitor, SystemHealthMonitor,
    create_monitoring_suite
)
from .security import (
    SecurityError, TensorSecurityValidator, MemoryMonitor, SafeFileHandler,
    AdversarialInputDetector, sanitize_tensor, compute_data_hash, verify_data_integrity,
    secure_operation
)
from .diagnostics import (
    HealthStatus, DiagnosticResult, ModelHealthChecker, ConvergenceMonitor,
    SystemCompatibilityChecker, run_comprehensive_diagnostics
)
from .config import (
    ConfigFormat, ModelConfig, FNOConfig, DeepONetConfig, PosteriorConfig,
    TrainingConfig, ActiveLearningConfig, ExperimentConfig, ConfigManager,
    load_config, save_config, create_default_config, set_environment,
    get_config_manager, validate_config_compatibility
)

__all__ = [
    # Performance utilities
    "PerformanceProfiler",
    "MemoryTracker", 
    "ModelOptimizer",
    "DataLoaderOptimizer",
    
    # Validation utilities
    "ValidationError", 
    "NumericalStabilityError", 
    "ParameterBoundsError",
    "TensorValidationError",
    "ValidationContext",
    "validate_tensor_shape",
    "validate_tensor_dtype",
    "validate_tensor_finite",
    "validate_parameter_bounds",
    "validate_integer_parameter",
    "validate_float_parameter",
    "check_numerical_stability",
    "validate_training_data",
    "safe_inversion",
    "validate_device_compatibility",
    
    # Exception handling
    "ProbNeuralOperatorError",
    "ModelError",
    "ModelInitializationError", 
    "ModelTrainingError",
    "PosteriorError",
    "PosteriorNotFittedError",
    "LaplaceFitError",
    "DataError",
    "DataLoaderError",
    "DataValidationError",
    "ConfigurationError",
    "DeviceError",
    "MemoryError",
    "NumericalError",
    "ConvergenceError",
    "ActiveLearningError",
    "CalibrationError",
    "BenchmarkError",
    "ErrorCollector",
    "handle_exception",
    "retry_on_exception",
    "RetryConfig",
    
    # Logging utilities
    "setup_logging",
    "get_logger",
    "JSONFormatter",
    "TrainingProgressLogger",
    "UncertaintyTracker",
    "PerformanceTracker",
    "ContextLogger",
    "create_training_logger",
    "create_uncertainty_logger",
    "create_performance_logger",
    
    # Monitoring utilities
    "MetricCollector",
    "ResourceMonitor",
    "TrainingMonitor",
    "SystemHealthMonitor",
    "create_monitoring_suite",
    
    # Security utilities
    "SecurityError",
    "TensorSecurityValidator",
    "MemoryMonitor",
    "SafeFileHandler", 
    "AdversarialInputDetector",
    "sanitize_tensor",
    "compute_data_hash",
    "verify_data_integrity",
    "secure_operation",
    
    # Diagnostics utilities
    "HealthStatus",
    "DiagnosticResult",
    "ModelHealthChecker",
    "ConvergenceMonitor",
    "SystemCompatibilityChecker",
    "run_comprehensive_diagnostics",
    
    # Configuration management
    "ConfigFormat",
    "ModelConfig",
    "FNOConfig", 
    "DeepONetConfig",
    "PosteriorConfig",
    "TrainingConfig",
    "ActiveLearningConfig",
    "ExperimentConfig",
    "ConfigManager",
    "load_config",
    "save_config",
    "create_default_config",
    "set_environment",
    "get_config_manager",
    "validate_config_compatibility"
]