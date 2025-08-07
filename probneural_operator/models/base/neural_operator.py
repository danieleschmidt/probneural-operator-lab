"""Base neural operator classes with common interfaces."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
import warnings
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ...utils.validation import (
    validate_tensor_shape, validate_tensor_finite, validate_training_data,
    validate_integer_parameter, validate_float_parameter, validate_device_compatibility,
    ValidationContext
)
from ...utils.exceptions import (
    ModelInitializationError, ModelTrainingError, PosteriorNotFittedError,
    handle_exception, safe_execute
)

logger = logging.getLogger(__name__)


class NeuralOperator(nn.Module, ABC):
    """Abstract base class for neural operators.
    
    This defines the common interface that all neural operators must implement.
    Neural operators map between function spaces, learning operators that can
    generalize across different discretizations and domains.
    """
    
    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        """Initialize the neural operator.
        
        Args:
            input_dim: Dimension of input functions
            output_dim: Dimension of output functions
            **kwargs: Additional model-specific parameters
            
        Raises:
            ModelInitializationError: If parameters are invalid
        """
        super().__init__()
        
        try:
            # Validate core parameters
            self.input_dim = validate_integer_parameter(
                input_dim, "input_dim", min_value=1, max_value=1000
            )
            self.output_dim = validate_integer_parameter(
                output_dim, "output_dim", min_value=1, max_value=1000
            )
            
            # Store configuration with validation
            self._config = self._validate_config(kwargs)
            
            logger.info(
                f"Initialized {self.__class__.__name__} with input_dim={self.input_dim}, "
                f"output_dim={self.output_dim}"
            )
            
        except Exception as e:
            handle_exception("__init__", e, 
                           {"input_dim": input_dim, "output_dim": output_dim},
                           ModelInitializationError)
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration parameters.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validated configuration
            
        Raises:
            ModelInitializationError: If configuration is invalid
        """
        validated_config = {}
        
        with ValidationContext(strict=True) as ctx:
            # Common parameter validation patterns
            float_params = ['lr', 'weight_decay', 'dropout', 'prior_precision']
            int_params = ['width', 'depth', 'modes', 'epochs', 'batch_size']
            
            for key, value in config.items():
                if key in float_params and value is not None:
                    try:
                        validated_config[key] = validate_float_parameter(
                            value, key, min_value=0.0, exclusive_min=(key != 'weight_decay')
                        )
                    except Exception as e:
                        ctx.validate(False, f"Invalid {key}: {e}")
                
                elif key in int_params and value is not None:
                    try:
                        validated_config[key] = validate_integer_parameter(
                            value, key, min_value=1
                        )
                    except Exception as e:
                        ctx.validate(False, f"Invalid {key}: {e}")
                
                else:
                    # Store as-is for unknown parameters (subclasses can handle them)
                    validated_config[key] = value
        
        return validated_config
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural operator.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim, ...)
            
        Returns:
            Output tensor of shape (batch_size, output_dim, ...)
        """
        pass
    
    def fit(self, 
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            epochs: int = 100,
            lr: float = 1e-3,
            device: str = "auto") -> Dict[str, Any]:
        """Train the neural operator.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            epochs: Number of training epochs
            lr: Learning rate
            device: Device to train on ("auto", "cpu", "cuda")
            
        Returns:
            Training history dictionary
            
        Raises:
            ModelTrainingError: If training fails
        """
        try:
            # Validate parameters
            epochs = validate_integer_parameter(epochs, "epochs", min_value=1)
            lr = validate_float_parameter(lr, "lr", min_value=0.0, exclusive_min=True)
            
            # Validate data loaders
            if len(train_loader) == 0:
                raise ModelTrainingError("Training data loader is empty")
            
            # Set up device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            device = torch.device(device)
            
            # Validate device compatibility
            sample_batch = next(iter(train_loader))
            if isinstance(sample_batch, (list, tuple)):
                sample_tensors = [t for t in sample_batch if isinstance(t, torch.Tensor)]
            else:
                sample_tensors = [sample_batch]
            
            # Move model to device first
            self.to(device)
            
            logger.info(f"Starting training for {epochs} epochs on device {device}")
            
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            history = {"train_loss": [], "val_loss": [], "epoch_times": []}
            
            import time
            start_time = time.time()
            
            for epoch in range(epochs):
                epoch_start = time.time()
                
                # Training phase
                self.train()
                train_loss = 0.0
                num_batches = 0
                
                try:
                    for batch_idx, (data, target) in enumerate(train_loader):
                        # Validate batch data
                        validate_tensor_finite(data, f"training_data_batch_{batch_idx}")
                        validate_tensor_finite(target, f"training_target_batch_{batch_idx}")
                        
                        data, target = data.to(device), target.to(device)
                        
                        optimizer.zero_grad()
                        output = self(data)
                        
                        # Validate model output
                        validate_tensor_finite(output, f"model_output_epoch_{epoch}_batch_{batch_idx}")
                        
                        loss = criterion(output, target)
                        
                        # Check for numerical issues
                        if torch.isnan(loss) or torch.isinf(loss):
                            raise ModelTrainingError(
                                f"Loss became {loss.item()} at epoch {epoch}, batch {batch_idx}"
                            )
                        
                        loss.backward()
                        
                        # Gradient clipping for numerical stability
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                        
                        optimizer.step()
                        
                        train_loss += loss.item()
                        num_batches += 1
                        
                        # Check for memory issues periodically
                        if batch_idx % 100 == 0 and device.type == "cuda":
                            if torch.cuda.memory_allocated(device) > 0.9 * torch.cuda.max_memory_allocated(device):
                                warnings.warn("High GPU memory usage detected", UserWarning)
                
                except Exception as e:
                    handle_exception("fit_training_loop", e, 
                                   {"epoch": epoch, "batch": batch_idx},
                                   ModelTrainingError)
                
                if num_batches > 0:
                    train_loss /= num_batches
                history["train_loss"].append(train_loss)
                
                # Validation phase
                val_loss = None
                if val_loader is not None:
                    try:
                        self.eval()
                        val_loss = 0.0
                        num_val_batches = 0
                        
                        with torch.no_grad():
                            for data, target in val_loader:
                                validate_tensor_finite(data, "validation_data")
                                validate_tensor_finite(target, "validation_target")
                                
                                data, target = data.to(device), target.to(device)
                                output = self(data)
                                
                                validate_tensor_finite(output, "validation_output")
                                val_loss += criterion(output, target).item()
                                num_val_batches += 1
                        
                        if num_val_batches > 0:
                            val_loss /= num_val_batches
                        history["val_loss"].append(val_loss)
                        
                    except Exception as e:
                        logger.warning(f"Validation failed at epoch {epoch}: {e}")
                        history["val_loss"].append(float('inf'))
                        val_loss = float('inf')
                
                epoch_time = time.time() - epoch_start
                history["epoch_times"].append(epoch_time)
                
                # Logging
                if epoch % 10 == 0 or epoch == epochs - 1:
                    msg = f"Epoch {epoch}: Train Loss: {train_loss:.6f}"
                    if val_loss is not None:
                        msg += f", Val Loss: {val_loss:.6f}"
                    msg += f", Time: {epoch_time:.2f}s"
                    logger.info(msg)
            
            total_time = time.time() - start_time
            logger.info(f"Training completed in {total_time:.2f}s")
            
            history["total_time"] = total_time
            return history
            
        except Exception as e:
            handle_exception("fit", e, 
                           {"epochs": epochs, "lr": lr, "device": device},
                           ModelTrainingError)
    
    def predict(self, x: torch.Tensor, device: str = "auto") -> torch.Tensor:
        """Make predictions with the neural operator.
        
        Args:
            x: Input tensor
            device: Device for computation
            
        Returns:
            Predictions
            
        Raises:
            ValidationError: If input validation fails
            ModelError: If prediction fails
        """
        try:
            # Validate input tensor
            validate_tensor_shape(x, min_dims=2, name="input")
            validate_tensor_finite(x, "input")
            
            # Set device
            if device == "auto":
                device = next(self.parameters()).device
            
            device = torch.device(device) if isinstance(device, str) else device
            
            self.eval()
            with torch.no_grad():
                x = x.to(device)
                output = self(x)
                
                # Validate output
                validate_tensor_finite(output, "prediction_output")
                
                return output
                
        except Exception as e:
            handle_exception("predict", e, 
                           {"input_shape": x.shape if 'x' in locals() else None},
                           reraise_as=type(e))
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        config = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            **self._config
        }
        
        # Add FNO-specific attributes if they exist
        if hasattr(self, 'modes'):
            config['modes'] = self.modes
        if hasattr(self, 'width'):
            config['width'] = self.width
        if hasattr(self, 'depth'):
            config['depth'] = self.depth
        if hasattr(self, 'spatial_dim'):
            config['spatial_dim'] = self.spatial_dim
            
        return config


class ProbabilisticNeuralOperator(NeuralOperator):
    """Base class for probabilistic neural operators with uncertainty quantification.
    
    This extends the base neural operator to support uncertainty quantification
    through various posterior approximation methods.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 output_dim: int,
                 posterior_type: str = "laplace",
                 prior_precision: float = 1.0,
                 **kwargs):
        """Initialize the probabilistic neural operator.
        
        Args:
            input_dim: Dimension of input functions
            output_dim: Dimension of output functions
            posterior_type: Type of posterior approximation ("laplace", "variational", "ensemble")
            prior_precision: Prior precision for Bayesian inference
            **kwargs: Additional model-specific parameters
        """
        super().__init__(input_dim, output_dim, **kwargs)
        self.posterior_type = posterior_type
        self.prior_precision = prior_precision
        self._posterior = None
        self._is_fitted = False
    
    def fit_posterior(self, 
                     train_loader: DataLoader,
                     val_loader: Optional[DataLoader] = None) -> None:
        """Fit the posterior approximation after training.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
        """
        from probneural_operator.posteriors import get_posterior
        
        self._posterior = get_posterior(
            self.posterior_type,
            model=self,
            prior_precision=self.prior_precision
        )
        
        self._posterior.fit(train_loader, val_loader)
        self._is_fitted = True
    
    def predict_with_uncertainty(self, 
                               x: torch.Tensor,
                               return_std: bool = True,
                               num_samples: int = 100,
                               device: str = "auto") -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Make predictions with uncertainty estimates.
        
        Args:
            x: Input tensor
            return_std: Whether to return standard deviation
            num_samples: Number of posterior samples for uncertainty estimation
            device: Device for computation
            
        Returns:
            If return_std=True: (mean, std) tuple
            If return_std=False: mean predictions only
            
        Raises:
            PosteriorNotFittedError: If posterior not fitted
            ValidationError: If input validation fails
        """
        try:
            if not self._is_fitted:
                raise PosteriorNotFittedError()
            
            # Validate inputs
            validate_tensor_shape(x, min_dims=2, name="input")
            validate_tensor_finite(x, "input")
            num_samples = validate_integer_parameter(num_samples, "num_samples", min_value=1)
            
            if device == "auto":
                device = next(self.parameters()).device
            
            device = torch.device(device) if isinstance(device, str) else device
            x = x.to(device)
            
            # Get posterior predictive distribution
            mean, variance = self._posterior.predict(x, num_samples=num_samples)
            
            # Validate outputs
            validate_tensor_finite(mean, "prediction_mean")
            validate_tensor_finite(variance, "prediction_variance")
            
            if return_std:
                # Ensure variance is non-negative
                variance = torch.clamp(variance, min=1e-10)
                std = torch.sqrt(variance)
                return mean, std
            else:
                return mean
                
        except Exception as e:
            handle_exception("predict_with_uncertainty", e,
                           {"input_shape": x.shape if 'x' in locals() else None,
                            "num_samples": num_samples if 'num_samples' in locals() else None},
                           reraise_as=type(e))
    
    def sample_predictions(self, 
                         x: torch.Tensor,
                         num_samples: int = 100,
                         device: str = "auto") -> torch.Tensor:
        """Sample predictions from the posterior.
        
        Args:
            x: Input tensor
            num_samples: Number of samples to draw
            device: Device for computation
            
        Returns:
            Samples of shape (num_samples, batch_size, output_dim, ...)
        """
        if not self._is_fitted:
            raise RuntimeError("Posterior not fitted. Call fit_posterior() first.")
        
        if device == "auto":
            device = next(self.parameters()).device
        
        x = x.to(device)
        return self._posterior.sample(x, num_samples=num_samples)
    
    def epistemic_uncertainty(self, x: torch.Tensor, num_samples: int = 100) -> torch.Tensor:
        """Compute epistemic (model) uncertainty.
        
        Args:
            x: Input tensor
            num_samples: Number of posterior samples
            
        Returns:
            Epistemic uncertainty estimates
        """
        samples = self.sample_predictions(x, num_samples=num_samples)
        return torch.var(samples, dim=0)
    
    def aleatoric_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """Compute aleatoric (data) uncertainty.
        
        Args:
            x: Input tensor
            
        Returns:
            Aleatoric uncertainty estimates
        """
        # For regression, this would typically be learned noise parameter
        # For now, return zeros as placeholder
        mean = self.predict(x)
        return torch.zeros_like(mean)
    
    def log_marginal_likelihood(self, train_loader: DataLoader) -> float:
        """Compute log marginal likelihood for model selection.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Log marginal likelihood
        """
        if not self._is_fitted:
            raise RuntimeError("Posterior not fitted. Call fit_posterior() first.")
        
        return self._posterior.log_marginal_likelihood(train_loader)