"""
Comprehensive logging configuration for the ProbNeural-Operator-Lab framework.

This module provides structured logging with multiple levels, formatters,
and handlers for different components of the framework.
"""

import logging
import logging.config
import sys
import os
from typing import Dict, Any, Optional
from datetime import datetime
import json

# Default logging configuration
DEFAULT_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'simple': {
            'format': '%(levelname)s - %(name)s - %(message)s'
        },
        'json': {
            '()': 'probneural_operator.utils.logging_config.JSONFormatter'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'simple',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': 'probneural_operator.log',
            'mode': 'a'
        },
        'error_file': {
            'class': 'logging.FileHandler',
            'level': 'ERROR',
            'formatter': 'detailed',
            'filename': 'probneural_operator_errors.log',
            'mode': 'a'
        }
    },
    'loggers': {
        'probneural_operator': {
            'level': 'DEBUG',
            'handlers': ['console', 'file', 'error_file'],
            'propagate': False
        },
        'probneural_operator.models': {
            'level': 'DEBUG',
            'handlers': ['console', 'file'],
            'propagate': False
        },
        'probneural_operator.training': {
            'level': 'INFO',
            'handlers': ['console', 'file'],
            'propagate': False
        },
        'probneural_operator.active': {
            'level': 'INFO',
            'handlers': ['console', 'file'],
            'propagate': False
        }
    },
    'root': {
        'level': 'WARNING',
        'handlers': ['console']
    }
}


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            'timestamp': self.formatTime(record, datefmt='%Y-%m-%dT%H:%M:%S'),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage()
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'message', 'exc_info', 'exc_text',
                          'stack_info', 'getMessage']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class TrainingProgressLogger:
    """Specialized logger for training progress monitoring."""
    
    def __init__(self, name: str = "training_progress", log_dir: str = "logs"):
        """Initialize training progress logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
        """
        self.name = name
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # File handler for training logs
        log_file = os.path.join(log_dir, f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # JSON formatter for structured data
        formatter = JSONFormatter()
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
        # Training state
        self.training_start_time = None
        self.current_epoch = 0
        self.total_epochs = 0
        
    def start_training(self, total_epochs: int, model_config: Dict[str, Any]):
        """Log training start event.
        
        Args:
            total_epochs: Total number of epochs
            model_config: Model configuration
        """
        self.training_start_time = datetime.now()
        self.total_epochs = total_epochs
        
        self.logger.info(
            "Training started",
            extra={
                "event_type": "training_start",
                "total_epochs": total_epochs,
                "model_config": model_config,
                "timestamp": self.training_start_time.isoformat()
            }
        )
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None,
                  metrics: Optional[Dict[str, float]] = None, epoch_time: Optional[float] = None):
        """Log epoch results.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss (optional)
            metrics: Additional metrics dictionary
            epoch_time: Time taken for epoch in seconds
        """
        self.current_epoch = epoch
        
        log_data = {
            "event_type": "epoch_complete",
            "epoch": epoch,
            "train_loss": train_loss,
            "progress": epoch / self.total_epochs if self.total_epochs > 0 else 0
        }
        
        if val_loss is not None:
            log_data["val_loss"] = val_loss
        
        if metrics:
            log_data.update(metrics)
        
        if epoch_time is not None:
            log_data["epoch_time"] = epoch_time
        
        if self.training_start_time:
            elapsed = (datetime.now() - self.training_start_time).total_seconds()
            log_data["total_elapsed_time"] = elapsed
        
        self.logger.info("Epoch completed", extra=log_data)
    
    def log_batch(self, epoch: int, batch_idx: int, batch_loss: float, 
                  batch_size: int, total_batches: int):
        """Log batch-level information (used sparingly).
        
        Args:
            epoch: Current epoch
            batch_idx: Batch index
            batch_loss: Batch loss
            batch_size: Size of batch
            total_batches: Total number of batches
        """
        # Only log every 100 batches to avoid spam
        if batch_idx % 100 == 0:
            self.logger.debug(
                "Batch processed",
                extra={
                    "event_type": "batch_complete",
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "batch_loss": batch_loss,
                    "batch_size": batch_size,
                    "batch_progress": batch_idx / total_batches if total_batches > 0 else 0
                }
            )
    
    def log_validation(self, epoch: int, val_metrics: Dict[str, float]):
        """Log validation results.
        
        Args:
            epoch: Current epoch
            val_metrics: Validation metrics
        """
        log_data = {
            "event_type": "validation_complete",
            "epoch": epoch,
            **val_metrics
        }
        
        self.logger.info("Validation completed", extra=log_data)
    
    def training_complete(self, final_metrics: Dict[str, float]):
        """Log training completion.
        
        Args:
            final_metrics: Final training metrics
        """
        if self.training_start_time:
            total_time = (datetime.now() - self.training_start_time).total_seconds()
        else:
            total_time = None
        
        log_data = {
            "event_type": "training_complete",
            "total_epochs": self.current_epoch,
            **final_metrics
        }
        
        if total_time:
            log_data["total_training_time"] = total_time
        
        self.logger.info("Training completed", extra=log_data)


class UncertaintyTracker:
    """Logger for uncertainty quantification tracking."""
    
    def __init__(self, name: str = "uncertainty_tracker", log_dir: str = "logs"):
        """Initialize uncertainty tracker.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set up file handler
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"uncertainty_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(file_handler)
    
    def log_posterior_fit(self, method: str, fit_time: float, 
                         convergence_info: Optional[Dict[str, Any]] = None):
        """Log posterior fitting results.
        
        Args:
            method: Posterior approximation method
            fit_time: Time taken to fit posterior
            convergence_info: Convergence information
        """
        log_data = {
            "event_type": "posterior_fit",
            "method": method,
            "fit_time": fit_time
        }
        
        if convergence_info:
            log_data["convergence_info"] = convergence_info
        
        self.logger.info("Posterior fitted", extra=log_data)
    
    def log_uncertainty_estimation(self, method: str, num_samples: int,
                                 mean_uncertainty: float, max_uncertainty: float,
                                 estimation_time: float):
        """Log uncertainty estimation results.
        
        Args:
            method: Uncertainty estimation method
            num_samples: Number of samples used
            mean_uncertainty: Mean uncertainty value
            max_uncertainty: Maximum uncertainty value  
            estimation_time: Time for estimation
        """
        self.logger.info(
            "Uncertainty estimated",
            extra={
                "event_type": "uncertainty_estimation",
                "method": method,
                "num_samples": num_samples,
                "mean_uncertainty": mean_uncertainty,
                "max_uncertainty": max_uncertainty,
                "estimation_time": estimation_time
            }
        )
    
    def log_calibration_metrics(self, calibration_error: float, 
                               reliability_diagram_data: Dict[str, Any]):
        """Log uncertainty calibration metrics.
        
        Args:
            calibration_error: Expected calibration error
            reliability_diagram_data: Data for reliability diagram
        """
        self.logger.info(
            "Calibration metrics computed",
            extra={
                "event_type": "calibration_metrics",
                "calibration_error": calibration_error,
                "reliability_data": reliability_diagram_data
            }
        )


class PerformanceTracker:
    """Logger for performance metrics tracking."""
    
    def __init__(self, name: str = "performance_tracker", log_dir: str = "logs"):
        """Initialize performance tracker.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set up file handler
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(file_handler)
    
    def log_memory_usage(self, stage: str, memory_stats: Dict[str, float]):
        """Log memory usage statistics.
        
        Args:
            stage: Training stage (e.g., "training", "validation", "inference")
            memory_stats: Memory usage statistics
        """
        self.logger.info(
            "Memory usage recorded",
            extra={
                "event_type": "memory_usage",
                "stage": stage,
                **memory_stats
            }
        )
    
    def log_computation_time(self, operation: str, execution_time: float,
                           context: Optional[Dict[str, Any]] = None):
        """Log computation time for operations.
        
        Args:
            operation: Operation name
            execution_time: Execution time in seconds
            context: Additional context information
        """
        log_data = {
            "event_type": "computation_time",
            "operation": operation,
            "execution_time": execution_time
        }
        
        if context:
            log_data.update(context)
        
        self.logger.info("Operation timed", extra=log_data)


def setup_logging(config: Optional[Dict[str, Any]] = None, 
                 log_level: str = "INFO",
                 log_dir: str = "logs") -> None:
    """Set up logging configuration for the framework.
    
    Args:
        config: Custom logging configuration (uses default if None)
        log_level: Global log level
        log_dir: Directory for log files
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    # Update file paths to use log_dir
    if 'handlers' in config:
        for handler_config in config['handlers'].values():
            if 'filename' in handler_config:
                handler_config['filename'] = os.path.join(log_dir, handler_config['filename'])
    
    # Apply global log level
    if 'loggers' in config:
        for logger_config in config['loggers'].values():
            logger_config['level'] = log_level
    
    # Configure logging
    logging.config.dictConfig(config)
    
    # Log setup completion
    logger = logging.getLogger('probneural_operator')
    logger.info(f"Logging configured with level {log_level}, logs in {log_dir}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger
    """
    return logging.getLogger(name)


class ContextLogger:
    """Context manager for adding context to log messages."""
    
    def __init__(self, logger: logging.Logger, **context):
        """Initialize context logger.
        
        Args:
            logger: Base logger
            **context: Context key-value pairs
        """
        self.logger = logger
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        """Enter context."""
        self.old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context."""
        logging.setLogRecordFactory(self.old_factory)


# Convenience function to create specialized loggers
def create_training_logger(experiment_name: str = "training") -> TrainingProgressLogger:
    """Create a training progress logger.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Configured training logger
    """
    return TrainingProgressLogger(f"training_{experiment_name}")


def create_uncertainty_logger(experiment_name: str = "uncertainty") -> UncertaintyTracker:
    """Create an uncertainty tracker.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Configured uncertainty tracker
    """
    return UncertaintyTracker(f"uncertainty_{experiment_name}")


def create_performance_logger(experiment_name: str = "performance") -> PerformanceTracker:
    """Create a performance tracker.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Configured performance tracker
    """
    return PerformanceTracker(f"performance_{experiment_name}")