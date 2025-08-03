"""Base neural operator classes with common interfaces."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self._config = kwargs
        
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
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        history = {"train_loss": [], "val_loss": []}
        
        for epoch in range(epochs):
            # Training
            self.train()
            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = self(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)
            
            # Validation
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(device), target.to(device)
                        output = self(data)
                        val_loss += criterion(output, target).item()
                
                val_loss /= len(val_loader)
                history["val_loss"].append(val_loss)
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}")
        
        return history
    
    def predict(self, x: torch.Tensor, device: str = "auto") -> torch.Tensor:
        """Make predictions with the neural operator.
        
        Args:
            x: Input tensor
            device: Device for computation
            
        Returns:
            Predictions
        """
        if device == "auto":
            device = next(self.parameters()).device
        
        self.eval()
        with torch.no_grad():
            x = x.to(device)
            return self(x)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            **self._config
        }


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
        """
        if not self._is_fitted:
            raise RuntimeError("Posterior not fitted. Call fit_posterior() first.")
        
        if device == "auto":
            device = next(self.parameters()).device
        
        x = x.to(device)
        
        # Get posterior predictive distribution
        mean, variance = self._posterior.predict(x, num_samples=num_samples)
        
        if return_std:
            std = torch.sqrt(variance)
            return mean, std
        else:
            return mean
    
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