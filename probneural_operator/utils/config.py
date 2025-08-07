"""
Comprehensive configuration management system for ProbNeural-Operator-Lab.

This module provides a robust configuration system with validation, serialization,
environment-specific settings, and hierarchical configuration management.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Type, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from copy import deepcopy

from .validation import validate_float_parameter, validate_integer_parameter
from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class ConfigFormat(Enum):
    """Configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    PYTHON = "python"


@dataclass
class ModelConfig:
    """Configuration for neural operator models."""
    # Core architecture parameters
    input_dim: int = 1
    output_dim: int = 1
    
    # Model-specific parameters (will be extended by subclasses)
    activation: str = "gelu"
    dropout: float = 0.0
    batch_norm: bool = False
    
    # Training parameters
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"
    scheduler: Optional[str] = None
    
    # Regularization
    l1_reg: float = 0.0
    l2_reg: float = 0.0
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        self.input_dim = validate_integer_parameter(self.input_dim, "input_dim", min_value=1)
        self.output_dim = validate_integer_parameter(self.output_dim, "output_dim", min_value=1)
        self.dropout = validate_float_parameter(self.dropout, "dropout", min_value=0.0, max_value=1.0)
        self.learning_rate = validate_float_parameter(
            self.learning_rate, "learning_rate", min_value=0.0, exclusive_min=True
        )
        self.weight_decay = validate_float_parameter(
            self.weight_decay, "weight_decay", min_value=0.0
        )
        self.l1_reg = validate_float_parameter(self.l1_reg, "l1_reg", min_value=0.0)
        self.l2_reg = validate_float_parameter(self.l2_reg, "l2_reg", min_value=0.0)
        
        # Validate string parameters
        valid_activations = ["relu", "gelu", "tanh", "sigmoid", "leaky_relu", "elu"]
        if self.activation not in valid_activations:
            raise ConfigurationError(f"Invalid activation: {self.activation}. Valid: {valid_activations}")
        
        valid_optimizers = ["adam", "adamw", "sgd", "rmsprop"]
        if self.optimizer not in valid_optimizers:
            raise ConfigurationError(f"Invalid optimizer: {self.optimizer}. Valid: {valid_optimizers}")


@dataclass
class FNOConfig(ModelConfig):
    """Configuration for Fourier Neural Operators."""
    # FNO-specific parameters
    modes: int = 16
    width: int = 64
    depth: int = 4
    spatial_dim: int = 1
    
    # Spectral parameters
    factor: int = 1
    ff_weight_norm: bool = True
    n_ff_layers: int = 2
    
    def validate(self) -> None:
        """Validate FNO-specific parameters."""
        super().validate()
        self.modes = validate_integer_parameter(self.modes, "modes", min_value=1, max_value=128)
        self.width = validate_integer_parameter(self.width, "width", min_value=8, max_value=1024)
        self.depth = validate_integer_parameter(self.depth, "depth", min_value=1, max_value=20)
        self.spatial_dim = validate_integer_parameter(self.spatial_dim, "spatial_dim", min_value=1, max_value=3)
        self.factor = validate_integer_parameter(self.factor, "factor", min_value=1, max_value=4)
        self.n_ff_layers = validate_integer_parameter(self.n_ff_layers, "n_ff_layers", min_value=1, max_value=5)


@dataclass
class DeepONetConfig(ModelConfig):
    """Configuration for DeepONet models."""
    # DeepONet-specific parameters
    branch_layers: List[int] = field(default_factory=lambda: [128, 128, 128])
    trunk_layers: List[int] = field(default_factory=lambda: [128, 128, 128])
    basis_functions: int = 100
    
    # Architecture options
    use_bias: bool = True
    branch_activation: str = "relu"
    trunk_activation: str = "relu"
    
    def validate(self) -> None:
        """Validate DeepONet-specific parameters."""
        super().validate()
        self.basis_functions = validate_integer_parameter(
            self.basis_functions, "basis_functions", min_value=10, max_value=1000
        )
        
        # Validate layer configurations
        for i, layer_size in enumerate(self.branch_layers):
            self.branch_layers[i] = validate_integer_parameter(
                layer_size, f"branch_layer_{i}", min_value=8, max_value=2048
            )
        
        for i, layer_size in enumerate(self.trunk_layers):
            self.trunk_layers[i] = validate_integer_parameter(
                layer_size, f"trunk_layer_{i}", min_value=8, max_value=2048
            )
        
        valid_activations = ["relu", "gelu", "tanh", "sigmoid", "leaky_relu", "elu"]
        if self.branch_activation not in valid_activations:
            raise ConfigurationError(f"Invalid branch_activation: {self.branch_activation}")
        if self.trunk_activation not in valid_activations:
            raise ConfigurationError(f"Invalid trunk_activation: {self.trunk_activation}")


@dataclass
class PosteriorConfig:
    """Configuration for posterior approximation methods."""
    method: str = "laplace"
    prior_precision: float = 1.0
    
    # Laplace-specific
    hessian_structure: str = "diagonal"
    damping: float = 1e-3
    
    # Variational-specific
    num_samples: int = 10
    kl_weight: float = 1.0
    
    # Ensemble-specific
    ensemble_size: int = 5
    
    def validate(self) -> None:
        """Validate posterior configuration."""
        valid_methods = ["laplace", "variational", "ensemble"]
        if self.method not in valid_methods:
            raise ConfigurationError(f"Invalid posterior method: {self.method}. Valid: {valid_methods}")
        
        self.prior_precision = validate_float_parameter(
            self.prior_precision, "prior_precision", min_value=0.0, exclusive_min=True
        )
        self.damping = validate_float_parameter(
            self.damping, "damping", min_value=0.0, exclusive_min=True
        )
        self.num_samples = validate_integer_parameter(
            self.num_samples, "num_samples", min_value=1, max_value=1000
        )
        self.kl_weight = validate_float_parameter(
            self.kl_weight, "kl_weight", min_value=0.0
        )
        self.ensemble_size = validate_integer_parameter(
            self.ensemble_size, "ensemble_size", min_value=2, max_value=20
        )
        
        valid_hessian_structures = ["diagonal", "block_diagonal", "full"]
        if self.hessian_structure not in valid_hessian_structures:
            raise ConfigurationError(
                f"Invalid hessian_structure: {self.hessian_structure}. Valid: {valid_hessian_structures}"
            )


@dataclass
class TrainingConfig:
    """Configuration for training procedures."""
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.2
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-6
    
    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_freq: int = 10
    save_best_only: bool = True
    
    # Logging
    log_freq: int = 10
    verbose: bool = True
    
    # Device and performance
    device: str = "auto"
    num_workers: int = 4
    pin_memory: bool = True
    
    def validate(self) -> None:
        """Validate training configuration."""
        self.epochs = validate_integer_parameter(self.epochs, "epochs", min_value=1)
        self.batch_size = validate_integer_parameter(self.batch_size, "batch_size", min_value=1)
        self.validation_split = validate_float_parameter(
            self.validation_split, "validation_split", min_value=0.0, max_value=0.5
        )
        self.patience = validate_integer_parameter(self.patience, "patience", min_value=1)
        self.min_delta = validate_float_parameter(self.min_delta, "min_delta", min_value=0.0)
        self.checkpoint_freq = validate_integer_parameter(self.checkpoint_freq, "checkpoint_freq", min_value=1)
        self.log_freq = validate_integer_parameter(self.log_freq, "log_freq", min_value=1)
        self.num_workers = validate_integer_parameter(self.num_workers, "num_workers", min_value=0)


@dataclass
class ActiveLearningConfig:
    """Configuration for active learning."""
    acquisition_function: str = "bald"
    budget: int = 1000
    batch_size: int = 10
    initial_size: int = 50
    
    # Acquisition-specific parameters
    beta: float = 1.0  # For UCB
    temperature: float = 1.0  # For softmax sampling
    
    def validate(self) -> None:
        """Validate active learning configuration."""
        valid_acquisitions = ["bald", "max_entropy", "max_variance", "ucb", "random"]
        if self.acquisition_function not in valid_acquisitions:
            raise ConfigurationError(
                f"Invalid acquisition_function: {self.acquisition_function}. Valid: {valid_acquisitions}"
            )
        
        self.budget = validate_integer_parameter(self.budget, "budget", min_value=10)
        self.batch_size = validate_integer_parameter(self.batch_size, "batch_size", min_value=1)
        self.initial_size = validate_integer_parameter(self.initial_size, "initial_size", min_value=5)
        self.beta = validate_float_parameter(self.beta, "beta", min_value=0.0)
        self.temperature = validate_float_parameter(
            self.temperature, "temperature", min_value=0.0, exclusive_min=True
        )


@dataclass
class ExperimentConfig:
    """Top-level experiment configuration."""
    # Experiment metadata
    name: str = "experiment"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Component configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    posterior: PosteriorConfig = field(default_factory=PosteriorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    active_learning: Optional[ActiveLearningConfig] = None
    
    # Data configuration
    data_path: Optional[str] = None
    dataset_type: str = "synthetic"
    
    # Output configuration
    output_dir: str = "./outputs"
    experiment_id: Optional[str] = None
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    
    def validate(self) -> None:
        """Validate all configuration components."""
        # Validate individual components
        self.model.validate()
        self.posterior.validate()
        self.training.validate()
        
        if self.active_learning is not None:
            self.active_learning.validate()
        
        # Validate experiment-level parameters
        self.seed = validate_integer_parameter(self.seed, "seed", min_value=0)
        
        valid_dataset_types = ["synthetic", "burgers", "darcy", "navier_stokes", "custom"]
        if self.dataset_type not in valid_dataset_types:
            raise ConfigurationError(f"Invalid dataset_type: {self.dataset_type}. Valid: {valid_dataset_types}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)


class ConfigManager:
    """Manages configuration loading, saving, and validation."""
    
    def __init__(self, config_dir: str = "./configs"):
        """Initialize configuration manager.
        
        Args:
            config_dir: Directory for configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Environment-specific settings
        self.environment = os.getenv("PROBNEURAL_ENV", "development")
        
        # Configuration cache
        self._config_cache: Dict[str, ExperimentConfig] = {}
    
    def load_config(self, 
                   config_path: Union[str, Path],
                   config_format: Optional[ConfigFormat] = None) -> ExperimentConfig:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            config_format: Format of configuration file (auto-detected if None)
            
        Returns:
            Loaded and validated configuration
            
        Raises:
            ConfigurationError: If loading or validation fails
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        
        # Auto-detect format if not specified
        if config_format is None:
            suffix = config_path.suffix.lower()
            if suffix == '.json':
                config_format = ConfigFormat.JSON
            elif suffix in ['.yaml', '.yml']:
                config_format = ConfigFormat.YAML
            elif suffix == '.py':
                config_format = ConfigFormat.PYTHON
            else:
                raise ConfigurationError(f"Cannot auto-detect format for file: {config_path}")
        
        try:
            # Load raw configuration data
            if config_format == ConfigFormat.JSON:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
            
            elif config_format == ConfigFormat.YAML:
                with open(config_path, 'r') as f:
                    config_data = yaml.safe_load(f)
            
            elif config_format == ConfigFormat.PYTHON:
                # Execute Python file and extract config
                import importlib.util
                spec = importlib.util.spec_from_file_location("config", config_path)
                config_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config_module)
                
                if hasattr(config_module, 'config'):
                    config_data = config_module.config
                else:
                    raise ConfigurationError("Python config file must define 'config' variable")
            
            else:
                raise ConfigurationError(f"Unsupported config format: {config_format}")
            
            # Convert to ExperimentConfig
            config = self._dict_to_config(config_data)
            
            # Apply environment-specific overrides
            config = self._apply_environment_overrides(config)
            
            # Validate configuration
            config.validate()
            
            # Cache configuration
            cache_key = str(config_path)
            self._config_cache[cache_key] = config
            
            logger.info(f"Loaded configuration from {config_path}")
            return config
            
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration from {config_path}: {e}")
    
    def save_config(self, 
                   config: ExperimentConfig,
                   config_path: Union[str, Path],
                   config_format: ConfigFormat = ConfigFormat.YAML) -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration to save
            config_path: Path to save configuration
            config_format: Format to save configuration in
            
        Raises:
            ConfigurationError: If saving fails
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_dict = config.to_dict()
            
            if config_format == ConfigFormat.JSON:
                with open(config_path, 'w') as f:
                    json.dump(config_dict, f, indent=2, default=str)
            
            elif config_format == ConfigFormat.YAML:
                with open(config_path, 'w') as f:
                    yaml.safe_dump(config_dict, f, indent=2, default_flow_style=False)
            
            else:
                raise ConfigurationError(f"Saving not supported for format: {config_format}")
            
            logger.info(f"Saved configuration to {config_path}")
            
        except Exception as e:
            raise ConfigurationError(f"Error saving configuration to {config_path}: {e}")
    
    def create_default_config(self, model_type: str = "fno") -> ExperimentConfig:
        """Create a default configuration for a model type.
        
        Args:
            model_type: Type of model ("fno", "deeponet", "gno")
            
        Returns:
            Default configuration
            
        Raises:
            ConfigurationError: If model type is invalid
        """
        if model_type == "fno":
            model_config = FNOConfig()
        elif model_type == "deeponet":
            model_config = DeepONetConfig()
        else:
            model_config = ModelConfig()
        
        return ExperimentConfig(
            name=f"default_{model_type}_experiment",
            model=model_config,
            posterior=PosteriorConfig(),
            training=TrainingConfig(),
        )
    
    def _dict_to_config(self, config_data: Dict[str, Any]) -> ExperimentConfig:
        """Convert dictionary to ExperimentConfig."""
        # Handle nested configurations
        config_kwargs = {}
        
        for key, value in config_data.items():
            if key == "model":
                # Determine model type and create appropriate config
                model_type = value.get("type", "base")
                if model_type == "fno":
                    config_kwargs["model"] = self._dict_to_dataclass(value, FNOConfig)
                elif model_type == "deeponet":
                    config_kwargs["model"] = self._dict_to_dataclass(value, DeepONetConfig)
                else:
                    config_kwargs["model"] = self._dict_to_dataclass(value, ModelConfig)
            
            elif key == "posterior":
                config_kwargs["posterior"] = self._dict_to_dataclass(value, PosteriorConfig)
            
            elif key == "training":
                config_kwargs["training"] = self._dict_to_dataclass(value, TrainingConfig)
            
            elif key == "active_learning":
                config_kwargs["active_learning"] = self._dict_to_dataclass(value, ActiveLearningConfig)
            
            else:
                config_kwargs[key] = value
        
        return ExperimentConfig(**config_kwargs)
    
    def _dict_to_dataclass(self, data: Dict[str, Any], dataclass_type: Type) -> Any:
        """Convert dictionary to dataclass instance."""
        # Filter out keys that are not in the dataclass
        import inspect
        valid_keys = set(inspect.signature(dataclass_type.__init__).parameters.keys()) - {"self"}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        
        return dataclass_type(**filtered_data)
    
    def _apply_environment_overrides(self, config: ExperimentConfig) -> ExperimentConfig:
        """Apply environment-specific configuration overrides."""
        # Create a copy to avoid mutating the original
        config = deepcopy(config)
        
        # Environment-specific overrides
        env_overrides = {
            "development": {
                "training.epochs": 10,
                "training.verbose": True,
                "output_dir": "./dev_outputs"
            },
            "testing": {
                "training.epochs": 2,
                "training.batch_size": 4,
                "output_dir": "./test_outputs"
            },
            "production": {
                "training.save_checkpoints": True,
                "training.save_best_only": True,
                "output_dir": "./prod_outputs"
            }
        }
        
        if self.environment in env_overrides:
            overrides = env_overrides[self.environment]
            for key, value in overrides.items():
                self._set_nested_attribute(config, key, value)
        
        # Environment variable overrides
        env_prefix = "PROBNEURAL_"
        for key, value in os.environ.items():
            if key.startswith(env_prefix):
                config_key = key[len(env_prefix):].lower().replace("_", ".")
                try:
                    # Try to convert to appropriate type
                    if value.lower() in ["true", "false"]:
                        value = value.lower() == "true"
                    elif value.isdigit():
                        value = int(value)
                    elif "." in value and value.replace(".", "").isdigit():
                        value = float(value)
                    
                    self._set_nested_attribute(config, config_key, value)
                except Exception as e:
                    logger.warning(f"Could not apply environment override {key}={value}: {e}")
        
        return config
    
    def _set_nested_attribute(self, obj: Any, attr_path: str, value: Any) -> None:
        """Set nested attribute using dot notation."""
        parts = attr_path.split(".")
        current = obj
        
        for part in parts[:-1]:
            current = getattr(current, part)
        
        setattr(current, parts[-1], value)
    
    def get_cached_config(self, config_path: Union[str, Path]) -> Optional[ExperimentConfig]:
        """Get cached configuration if available."""
        cache_key = str(config_path)
        return self._config_cache.get(cache_key)
    
    def clear_cache(self) -> None:
        """Clear configuration cache."""
        self._config_cache.clear()
        logger.info("Configuration cache cleared")


# Global configuration manager instance
_config_manager = ConfigManager()


# Convenience functions
def load_config(config_path: Union[str, Path]) -> ExperimentConfig:
    """Load configuration from file using global manager."""
    return _config_manager.load_config(config_path)


def save_config(config: ExperimentConfig, config_path: Union[str, Path]) -> None:
    """Save configuration to file using global manager."""
    _config_manager.save_config(config, config_path)


def create_default_config(model_type: str = "fno") -> ExperimentConfig:
    """Create default configuration using global manager."""
    return _config_manager.create_default_config(model_type)


def set_environment(environment: str) -> None:
    """Set the current environment."""
    _config_manager.environment = environment
    logger.info(f"Environment set to: {environment}")


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager."""
    return _config_manager


# Configuration validation utilities
def validate_config_compatibility(config: ExperimentConfig) -> List[str]:
    """Validate configuration component compatibility.
    
    Args:
        config: Configuration to validate
        
    Returns:
        List of compatibility warnings
    """
    warnings = []
    
    # Check model-posterior compatibility
    if hasattr(config.model, 'spatial_dim'):
        if config.model.spatial_dim > 2 and config.posterior.method == "variational":
            warnings.append("Variational inference may be slow with high-dimensional spatial data")
    
    # Check training-model compatibility
    if config.training.batch_size < 8 and hasattr(config.model, 'batch_norm'):
        if config.model.batch_norm:
            warnings.append("Batch normalization may be unstable with small batch sizes")
    
    # Check active learning compatibility
    if config.active_learning is not None:
        if config.active_learning.budget < config.active_learning.initial_size:
            warnings.append("Active learning budget smaller than initial size")
        
        if config.posterior.method == "ensemble" and config.active_learning.acquisition_function == "bald":
            warnings.append("BALD acquisition may not work well with ensemble methods")
    
    return warnings