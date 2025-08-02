"""Enhanced logging configuration for ProbNeural Operator Lab."""

import logging
import logging.config
import os
import sys
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager

import torch
import numpy as np


class PerformanceLogger:
    """Logger for performance metrics and timing."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._timers = {}
        self._lock = threading.Lock()
    
    @contextmanager
    def timer(self, name: str, log_level: int = logging.INFO):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            self.log_timing(name, elapsed, log_level)
    
    def log_timing(self, operation: str, elapsed_time: float, log_level: int = logging.INFO):
        """Log timing information."""
        self.logger.log(log_level, f"⏱️  {operation}: {elapsed_time:.4f}s")
    
    def log_memory_usage(self, operation: str = "", device: Optional[torch.device] = None):
        """Log current memory usage."""
        if device and device.type == "cuda":
            allocated = torch.cuda.memory_allocated(device) / 1024**2  # MB
            cached = torch.cuda.memory_reserved(device) / 1024**2  # MB
            self.logger.info(f"🧠 GPU Memory {operation}: {allocated:.1f}MB allocated, {cached:.1f}MB cached")
        else:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024**2
            self.logger.info(f"🧠 CPU Memory {operation}: {memory_mb:.1f}MB")
    
    def log_tensor_stats(self, tensor: torch.Tensor, name: str = "tensor"):
        """Log tensor statistics."""
        stats = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "min": tensor.min().item() if tensor.numel() > 0 else None,
            "max": tensor.max().item() if tensor.numel() > 0 else None,
            "mean": tensor.mean().item() if tensor.numel() > 0 else None,
            "std": tensor.std().item() if tensor.numel() > 1 else None,
            "has_nan": torch.isnan(tensor).any().item(),
            "has_inf": torch.isinf(tensor).any().item(),
        }
        self.logger.debug(f"📊 Tensor '{name}': {json.dumps(stats, indent=2)}")


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "levelname", "levelno", "pathname", 
                          "filename", "module", "lineno", "funcName", "created", 
                          "msecs", "relativeCreated", "thread", "threadName", 
                          "processName", "process", "getMessage", "exc_info", 
                          "exc_text", "stack_info"]:
                log_entry[key] = value
        
        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Union[str, Path]] = None,
    console_output: bool = True,
    json_output: bool = False,
    file_output: bool = True,
    performance_logging: bool = True
) -> Dict[str, logging.Logger]:
    """Setup comprehensive logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: logs/)
        console_output: Enable console logging
        json_output: Use JSON format for file logs
        file_output: Enable file logging
        performance_logging: Enable performance logging
    
    Returns:
        Dictionary of configured loggers
    """
    # Create log directory
    if log_dir is None:
        log_dir = Path("logs")
    else:
        log_dir = Path(log_dir)
    
    log_dir.mkdir(exist_ok=True)
    
    # Get numeric log level
    numeric_level = getattr(logging, log_level.upper())
    
    # Clear existing handlers
    logging.getLogger().handlers.clear()
    
    # Logging configuration
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "simple": {
                "format": "%(levelname)s - %(name)s - %(message)s"
            },
            "colored": {
                "()": ColoredFormatter,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "datefmt": "%H:%M:%S"
            },
            "json": {
                "()": JSONFormatter
            }
        },
        "handlers": {},
        "loggers": {
            "probneural_operator": {
                "level": log_level.upper(),
                "handlers": [],
                "propagate": False
            },
            "performance": {
                "level": "DEBUG" if performance_logging else "WARNING",
                "handlers": [],
                "propagate": False
            },
            "training": {
                "level": log_level.upper(),
                "handlers": [],
                "propagate": False
            },
            "evaluation": {
                "level": log_level.upper(),
                "handlers": [],
                "propagate": False
            },
            "uncertainty": {
                "level": log_level.upper(),
                "handlers": [],
                "propagate": False
            }
        },
        "root": {
            "level": log_level.upper(),
            "handlers": []
        }
    }
    
    handler_names = []
    
    # Console handler
    if console_output:
        config["handlers"]["console"] = {
            "class": "logging.StreamHandler",
            "level": log_level.upper(),
            "formatter": "colored",
            "stream": "ext://sys.stdout"
        }
        handler_names.append("console")
    
    # File handlers
    if file_output:
        # Main log file
        config["handlers"]["file_main"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level.upper(),
            "formatter": "json" if json_output else "detailed",
            "filename": str(log_dir / "probneural_operator.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5
        }
        handler_names.append("file_main")
        
        # Error log file
        config["handlers"]["file_error"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "json" if json_output else "detailed",
            "filename": str(log_dir / "errors.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 3
        }
        handler_names.append("file_error")
        
        # Performance log file
        if performance_logging:
            config["handlers"]["file_performance"] = {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "json" if json_output else "detailed",
                "filename": str(log_dir / "performance.log"),
                "maxBytes": 10485760,  # 10MB
                "backupCount": 3
            }
    
    # Assign handlers to loggers
    for logger_name in config["loggers"]:
        if logger_name == "performance" and performance_logging and file_output:
            config["loggers"][logger_name]["handlers"] = ["file_performance"] + (["console"] if console_output else [])
        else:
            config["loggers"][logger_name]["handlers"] = handler_names
    
    config["root"]["handlers"] = handler_names
    
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Create logger instances
    loggers = {
        "main": logging.getLogger("probneural_operator"),
        "performance": logging.getLogger("performance"),
        "training": logging.getLogger("training"),
        "evaluation": logging.getLogger("evaluation"),
        "uncertainty": logging.getLogger("uncertainty"),
    }
    
    # Add performance logging capabilities
    for name, logger in loggers.items():
        logger.perf = PerformanceLogger(logger)
    
    # Log startup information
    main_logger = loggers["main"]
    main_logger.info("🚀 Logging system initialized")
    main_logger.info(f"📝 Log level: {log_level}")
    main_logger.info(f"📁 Log directory: {log_dir.absolute()}")
    main_logger.info(f"🎨 Console output: {console_output}")
    main_logger.info(f"📄 File output: {file_output}")
    main_logger.info(f"🔧 JSON format: {json_output}")
    main_logger.info(f"⚡ Performance logging: {performance_logging}")
    
    # Log system information
    main_logger.info(f"🐍 Python version: {sys.version}")
    main_logger.info(f"🔥 PyTorch version: {torch.__version__}")
    main_logger.info(f"🖥️  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        main_logger.info(f"🎮 GPU count: {torch.cuda.device_count()}")
        main_logger.info(f"🎮 Current device: {torch.cuda.current_device()}")
    
    return loggers


def get_logger(name: str = "probneural_operator") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


# Convenience functions
def log_model_info(model: torch.nn.Module, logger: Optional[logging.Logger] = None):
    """Log model information."""
    if logger is None:
        logger = get_logger()
    
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"🏗️  Model: {model.__class__.__name__}")
    logger.info(f"📊 Total parameters: {param_count:,}")
    logger.info(f"🎯 Trainable parameters: {trainable_params:,}")
    logger.info(f"🔒 Frozen parameters: {param_count - trainable_params:,}")


def log_config(config: Dict[str, Any], logger: Optional[logging.Logger] = None):
    """Log configuration information."""
    if logger is None:
        logger = get_logger()
    
    logger.info("⚙️  Configuration:")
    logger.info(json.dumps(config, indent=2, default=str))


def log_experiment_start(experiment_name: str, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
    """Log experiment start with configuration."""
    if logger is None:
        logger = get_logger()
    
    logger.info(f"🧪 Starting experiment: {experiment_name}")
    logger.info(f"⏰ Start time: {datetime.now().isoformat()}")
    log_config(config, logger)


def log_experiment_end(experiment_name: str, results: Dict[str, Any], logger: Optional[logging.Logger] = None):
    """Log experiment end with results."""
    if logger is None:
        logger = get_logger()
    
    logger.info(f"🏁 Experiment completed: {experiment_name}")
    logger.info(f"⏰ End time: {datetime.now().isoformat()}")
    logger.info("📊 Results:")
    logger.info(json.dumps(results, indent=2, default=str))


# Initialize default logging if not already configured
def init_default_logging():
    """Initialize default logging configuration."""
    if not logging.getLogger().handlers:
        log_level = os.getenv("LOG_LEVEL", "INFO")
        setup_logging(log_level=log_level)


# Auto-initialize on import
init_default_logging()