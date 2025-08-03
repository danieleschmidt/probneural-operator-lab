"""Temperature scaling for uncertainty calibration."""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class TemperatureScaling(nn.Module):
    """Temperature scaling for calibrating uncertainty estimates.
    
    Temperature scaling is a simple post-hoc calibration method that scales
    the logits/predictions by a learned temperature parameter to improve
    calibration of uncertainty estimates.
    
    References:
        Guo et al. "On Calibration of Modern Neural Networks" ICML 2017
    """
    
    def __init__(self, temperature: float = 1.0):
        """Initialize temperature scaling.
        
        Args:
            temperature: Initial temperature value
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * temperature)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits.
        
        Args:
            logits: Input logits/predictions
            
        Returns:
            Temperature-scaled predictions
        """
        return logits / self.temperature
    
    def fit(self, 
            model: nn.Module,
            val_loader: DataLoader,
            max_iter: int = 50,
            lr: float = 0.01) -> None:
        """Fit temperature parameter on validation data.
        
        Args:
            model: Trained model to calibrate
            val_loader: Validation data loader
            max_iter: Maximum optimization iterations
            lr: Learning rate for temperature optimization
        """
        # Collect validation predictions and targets
        model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                predictions.append(output)
                targets.append(target)
        
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
        
        # Optimize temperature using LBFGS
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            scaled_predictions = self.forward(predictions)
            loss = F.mse_loss(scaled_predictions, targets)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        print(f"Optimized temperature: {self.temperature.item():.4f}")
    
    def calibrate(self, model: nn.Module) -> 'CalibratedModel':
        """Create calibrated version of model.
        
        Args:
            model: Model to calibrate
            
        Returns:
            Calibrated model wrapper
        """
        return CalibratedModel(model, self.temperature.item())


class CalibratedModel(nn.Module):
    """Wrapper for temperature-calibrated model."""
    
    def __init__(self, model: nn.Module, temperature: float):
        """Initialize calibrated model.
        
        Args:
            model: Base model
            temperature: Temperature parameter
        """
        super().__init__()
        self.model = model
        self.temperature = temperature
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with temperature scaling."""
        output = self.model(x)
        return output / self.temperature
    
    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Calibrated predictions."""
        return self.forward(x)
    
    def predict_with_uncertainty(self, 
                                x: torch.Tensor,
                                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calibrated predictions with uncertainty."""
        if hasattr(self.model, 'predict_with_uncertainty'):
            mean, var = self.model.predict_with_uncertainty(x, **kwargs)
            calibrated_mean = mean / self.temperature
            calibrated_var = var / (self.temperature ** 2)
            return calibrated_mean, calibrated_var
        else:
            raise ValueError("Base model does not support uncertainty prediction")
    
    def sample_predictions(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Calibrated prediction samples."""
        if hasattr(self.model, 'sample_predictions'):
            samples = self.model.sample_predictions(x, **kwargs)
            return samples / self.temperature
        else:
            raise ValueError("Base model does not support sampling")


class CalibrationMetrics:
    """Metrics for evaluating uncertainty calibration."""
    
    @staticmethod
    def expected_calibration_error(predictions: torch.Tensor,
                                 targets: torch.Tensor,
                                 confidences: torch.Tensor,
                                 n_bins: int = 10) -> float:
        """Compute Expected Calibration Error (ECE).
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets  
            confidences: Confidence estimates
            n_bins: Number of calibration bins
            
        Returns:
            ECE value
        """
        # Create bins
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        total_samples = len(predictions)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                # Accuracy in this bin
                bin_predictions = predictions[in_bin]
                bin_targets = targets[in_bin]
                bin_accuracy = (bin_predictions - bin_targets).abs().mean()
                
                # Confidence in this bin  
                bin_confidence = confidences[in_bin].mean()
                
                # Add to ECE
                ece += torch.abs(bin_accuracy - bin_confidence) * prop_in_bin
        
        return ece.item()
    
    @staticmethod
    def reliability_diagram_data(predictions: torch.Tensor,
                               targets: torch.Tensor,
                               confidences: torch.Tensor,
                               n_bins: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate data for reliability diagram.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            confidences: Confidence estimates  
            n_bins: Number of bins
            
        Returns:
            Tuple of (bin_centers, accuracies, counts)
        """
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        accuracies = torch.zeros(n_bins)
        counts = torch.zeros(n_bins)
        
        for i, (bin_lower, bin_upper) in enumerate(zip(bin_boundaries[:-1], bin_boundaries[1:])):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_predictions = predictions[in_bin]
                bin_targets = targets[in_bin]
                accuracies[i] = 1.0 - (bin_predictions - bin_targets).abs().mean()
                counts[i] = in_bin.sum()
        
        return bin_centers, accuracies, counts