"""Acquisition functions for active learning."""

import math
from abc import ABC, abstractmethod
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AcquisitionFunction(ABC):
    """Abstract base class for acquisition functions."""
    
    @abstractmethod
    def __call__(self, 
                 model: nn.Module,
                 x: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Compute acquisition scores.
        
        Args:
            model: Probabilistic model
            x: Input tensor
            **kwargs: Additional arguments
            
        Returns:
            Acquisition scores
        """
        pass


class BALD(AcquisitionFunction):
    """Bayesian Active Learning by Disagreement.
    
    BALD measures the mutual information between model parameters and predictions.
    It selects points where the model is most uncertain about what it should predict.
    
    References:
        Houlsby et al. "Bayesian Active Learning for Classification and Preference Learning"
        arXiv:1112.5745 (2011)
    """
    
    def __init__(self, num_samples: int = 100):
        """Initialize BALD acquisition function.
        
        Args:
            num_samples: Number of posterior samples for estimation
        """
        self.num_samples = num_samples
    
    def __call__(self, 
                 model: nn.Module,
                 x: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Compute BALD scores.
        
        Args:
            model: Probabilistic model with sample_predictions method
            x: Input tensor of shape (batch_size, ...)
            
        Returns:
            BALD scores of shape (batch_size,)
        """
        if not hasattr(model, 'sample_predictions'):
            raise ValueError("Model must have sample_predictions method for BALD")
        
        # Sample predictions from posterior
        samples = model.sample_predictions(x, num_samples=self.num_samples)
        # samples shape: (num_samples, batch_size, output_dim, ...)
        
        # Compute predictive entropy (epistemic uncertainty)
        mean_pred = samples.mean(dim=0)  # (batch_size, output_dim, ...)
        predictive_entropy = self._compute_entropy(mean_pred)
        
        # Compute expected entropy (aleatoric uncertainty)
        expected_entropy = 0.0
        for i in range(self.num_samples):
            expected_entropy += self._compute_entropy(samples[i])
        expected_entropy /= self.num_samples
        
        # BALD = Predictive Entropy - Expected Entropy
        bald_scores = predictive_entropy - expected_entropy
        
        # Aggregate over spatial dimensions if necessary
        while bald_scores.ndim > 1:
            bald_scores = bald_scores.mean(dim=-1)
        
        return bald_scores
    
    def _compute_entropy(self, pred: torch.Tensor) -> torch.Tensor:
        """Compute entropy for regression (using Gaussian assumption).
        
        Args:
            pred: Predictions tensor
            
        Returns:
            Entropy estimates
        """
        # For regression, entropy is related to prediction variance
        # H = 0.5 * log(2π * var) for Gaussian
        # We approximate this using prediction magnitude as proxy for uncertainty
        return 0.5 * torch.log(2 * math.pi * torch.var(pred, dim=-1, keepdim=True) + 1e-8)


class MaxVariance(AcquisitionFunction):
    """Maximum variance acquisition function.
    
    Selects points with highest predictive variance (epistemic uncertainty).
    """
    
    def __init__(self, num_samples: int = 100):
        """Initialize MaxVariance acquisition function.
        
        Args:
            num_samples: Number of posterior samples
        """
        self.num_samples = num_samples
    
    def __call__(self, 
                 model: nn.Module,
                 x: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Compute variance-based acquisition scores.
        
        Args:
            model: Probabilistic model
            x: Input tensor
            
        Returns:
            Variance scores
        """
        if hasattr(model, 'predict_with_uncertainty'):
            mean, variance = model.predict_with_uncertainty(x, return_std=False)
            scores = variance
        elif hasattr(model, 'sample_predictions'):
            samples = model.sample_predictions(x, num_samples=self.num_samples)
            scores = torch.var(samples, dim=0)
        else:
            raise ValueError("Model must support uncertainty estimation")
        
        # Aggregate over spatial/output dimensions
        while scores.ndim > 1:
            scores = scores.mean(dim=-1)
        
        return scores


class MaxEntropy(AcquisitionFunction):
    """Maximum entropy acquisition function.
    
    Selects points with highest predictive entropy.
    """
    
    def __init__(self, num_samples: int = 100):
        """Initialize MaxEntropy acquisition function.
        
        Args:
            num_samples: Number of posterior samples
        """
        self.num_samples = num_samples
    
    def __call__(self, 
                 model: nn.Module,
                 x: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Compute entropy-based acquisition scores.
        
        Args:
            model: Probabilistic model
            x: Input tensor
            
        Returns:
            Entropy scores
        """
        if hasattr(model, 'sample_predictions'):
            samples = model.sample_predictions(x, num_samples=self.num_samples)
            mean_pred = samples.mean(dim=0)
            entropy = self._compute_entropy(mean_pred)
        else:
            # Fallback to variance-based approximation
            if hasattr(model, 'predict_with_uncertainty'):
                mean, variance = model.predict_with_uncertainty(x, return_std=False)
                entropy = 0.5 * torch.log(2 * math.pi * variance + 1e-8)
            else:
                raise ValueError("Model must support uncertainty estimation")
        
        # Aggregate over spatial/output dimensions
        while entropy.ndim > 1:
            entropy = entropy.mean(dim=-1)
        
        return entropy
    
    def _compute_entropy(self, pred: torch.Tensor) -> torch.Tensor:
        """Compute entropy for predictions."""
        # Gaussian entropy approximation
        variance = torch.var(pred, dim=-1, keepdim=True)
        return 0.5 * torch.log(2 * math.pi * variance + 1e-8)


class Random(AcquisitionFunction):
    """Random acquisition (baseline)."""
    
    def __call__(self, 
                 model: nn.Module,
                 x: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Return random scores.
        
        Args:
            model: Model (unused)
            x: Input tensor
            
        Returns:
            Random scores
        """
        return torch.rand(x.shape[0], device=x.device)


class PhysicsAware(AcquisitionFunction):
    """Physics-aware acquisition function.
    
    Combines uncertainty with physics constraints (e.g., PDE residuals).
    """
    
    def __init__(self, 
                 base_acquisition: AcquisitionFunction,
                 physics_residual_fn: Callable,
                 physics_weight: float = 0.1):
        """Initialize physics-aware acquisition.
        
        Args:
            base_acquisition: Base acquisition function
            physics_residual_fn: Function computing physics residuals
            physics_weight: Weight for physics term
        """
        self.base_acquisition = base_acquisition
        self.physics_residual_fn = physics_residual_fn
        self.physics_weight = physics_weight
    
    def __call__(self, 
                 model: nn.Module,
                 x: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Compute physics-aware acquisition scores.
        
        Args:
            model: Probabilistic model
            x: Input tensor
            
        Returns:
            Combined acquisition scores
        """
        # Base uncertainty
        uncertainty_scores = self.base_acquisition(model, x, **kwargs)
        
        # Physics residuals
        with torch.enable_grad():
            x_physics = x.detach().requires_grad_(True)
            pred = model(x_physics)
            residuals = self.physics_residual_fn(pred, x_physics)
            physics_scores = torch.abs(residuals).mean(dim=tuple(range(1, residuals.ndim)))
        
        # Combine scores
        combined_scores = uncertainty_scores + self.physics_weight * physics_scores
        
        return combined_scores


# Factory for acquisition functions
class AcquisitionFunctions:
    """Factory class for acquisition functions."""
    
    @staticmethod
    def bald(num_samples: int = 100) -> BALD:
        """Create BALD acquisition function."""
        return BALD(num_samples)
    
    @staticmethod
    def max_variance(num_samples: int = 100) -> MaxVariance:
        """Create MaxVariance acquisition function."""
        return MaxVariance(num_samples)
    
    @staticmethod
    def max_entropy(num_samples: int = 100) -> MaxEntropy:
        """Create MaxEntropy acquisition function."""
        return MaxEntropy(num_samples)
    
    @staticmethod
    def random() -> Random:
        """Create Random acquisition function."""
        return Random()
    
    @staticmethod
    def physics_aware(base_type: str,
                     physics_residual_fn: Callable,
                     physics_weight: float = 0.1,
                     **kwargs) -> PhysicsAware:
        """Create physics-aware acquisition function."""
        if base_type == "bald":
            base = AcquisitionFunctions.bald(**kwargs)
        elif base_type == "variance":
            base = AcquisitionFunctions.max_variance(**kwargs)
        elif base_type == "entropy":
            base = AcquisitionFunctions.max_entropy(**kwargs)
        else:
            raise ValueError(f"Unknown base acquisition type: {base_type}")
        
        return PhysicsAware(base, physics_residual_fn, physics_weight)