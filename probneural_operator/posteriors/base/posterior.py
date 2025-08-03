"""Base posterior approximation interface."""

from abc import ABC, abstractmethod
from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class PosteriorApproximation(ABC):
    """Abstract base class for posterior approximation methods.
    
    This defines the interface that all posterior approximation methods
    (Laplace, variational, ensemble) must implement.
    """
    
    def __init__(self, model: nn.Module, prior_precision: float = 1.0):
        """Initialize posterior approximation.
        
        Args:
            model: The neural network model
            prior_precision: Prior precision (inverse variance)
        """
        self.model = model
        self.prior_precision = prior_precision
        self._is_fitted = False
    
    @abstractmethod
    def fit(self, 
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None) -> None:
        """Fit the posterior approximation.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
        """
        pass
    
    @abstractmethod
    def predict(self, 
                x: torch.Tensor,
                num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with uncertainty.
        
        Args:
            x: Input tensor
            num_samples: Number of samples for uncertainty estimation
            
        Returns:
            Tuple of (mean, variance) predictions
        """
        pass
    
    @abstractmethod
    def sample(self, 
               x: torch.Tensor,
               num_samples: int = 100) -> torch.Tensor:
        """Sample predictions from posterior.
        
        Args:
            x: Input tensor
            num_samples: Number of samples to draw
            
        Returns:
            Tensor of samples with shape (num_samples, batch_size, ...)
        """
        pass
    
    def log_marginal_likelihood(self, train_loader: DataLoader) -> float:
        """Compute log marginal likelihood.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Log marginal likelihood estimate
        """
        # Default implementation returns 0 (implement in subclasses)
        return 0.0
    
    def reset(self) -> None:
        """Reset the posterior approximation."""
        self._is_fitted = False