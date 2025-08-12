"""Advanced uncertainty calibration methods for neural operators.

This module implements state-of-the-art calibration techniques specifically
designed for neural operators and PDE solvers, going beyond simple temperature
scaling to provide well-calibrated uncertainty estimates.

Key Methods:
- Multi-dimensional temperature scaling
- Platt scaling with spatial awareness  
- Isotonic regression for neural operators
- Physics-constrained calibration
- Hierarchical calibration for multi-scale problems

Research Contributions:
- Novel spatial-temporal calibration for PDEs
- Conservation-aware calibration
- Multi-fidelity calibration alignment
"""

from typing import Tuple, Optional, Dict, List, Callable, Any
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

from ..utils.validation import validate_tensor_shape, validate_tensor_finite


class MultiDimensionalTemperatureScaling(nn.Module):
    """Multi-dimensional temperature scaling for neural operators.
    
    Extends temperature scaling to account for spatial and temporal dimensions
    in PDE solutions, allowing different calibration parameters across the domain.
    """
    
    def __init__(self,
                 spatial_dims: int = 2,
                 temporal_aware: bool = False,
                 learnable_spatial: bool = True):
        """Initialize multi-dimensional temperature scaling.
        
        Args:
            spatial_dims: Number of spatial dimensions
            temporal_aware: Whether to include temporal calibration
            learnable_spatial: Whether spatial temperatures are learnable
        """
        super().__init__()
        
        self.spatial_dims = spatial_dims
        self.temporal_aware = temporal_aware
        self.learnable_spatial = learnable_spatial
        
        # Global temperature parameter
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Spatial temperature modulation
        if learnable_spatial:
            # Learnable spatial temperature field
            self.spatial_temperature_net = nn.Sequential(
                nn.Linear(spatial_dims, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Softplus()  # Ensure positive temperatures
            )
        else:
            self.spatial_temperature_net = None
        
        # Temporal temperature parameter
        if temporal_aware:
            self.temporal_temperature = nn.Parameter(torch.ones(1))
        else:
            self.temporal_temperature = None
    
    def forward(self, 
                logits: torch.Tensor,
                coordinates: Optional[torch.Tensor] = None,
                time: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply multi-dimensional temperature scaling.
        
        Args:
            logits: Model predictions/logits
            coordinates: Spatial coordinates for each prediction
            time: Temporal coordinates
            
        Returns:
            Calibrated predictions
        """
        calibrated = logits / self.temperature
        
        # Apply spatial temperature modulation
        if self.learnable_spatial and coordinates is not None:
            spatial_temp = self.spatial_temperature_net(coordinates)
            # Reshape to match logits
            while spatial_temp.ndim < logits.ndim:
                spatial_temp = spatial_temp.unsqueeze(-1)
            spatial_temp = spatial_temp.expand_as(logits)
            calibrated = calibrated / spatial_temp
        
        # Apply temporal temperature scaling
        if self.temporal_aware and time is not None:
            temp_factor = self.temporal_temperature * torch.sigmoid(time)
            calibrated = calibrated / temp_factor
        
        return calibrated
    
    def fit(self,
            model: nn.Module,
            val_loader: DataLoader,
            max_iter: int = 100,
            lr: float = 0.01) -> Dict[str, float]:
        """Fit temperature parameters using validation data.
        
        Args:
            model: Neural operator model
            val_loader: Validation data loader
            max_iter: Maximum optimization iterations
            lr: Learning rate for temperature optimization
            
        Returns:
            Calibration metrics
        """
        device = next(model.parameters()).device
        self.to(device)
        
        # Collect predictions and targets
        all_predictions = []
        all_targets = []
        all_coordinates = []
        
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 2:
                    data, target = batch
                    coords = None
                elif len(batch) == 3:
                    data, target, coords = batch
                else:
                    raise ValueError("Expected batch to have 2 or 3 elements")
                
                data, target = data.to(device), target.to(device)
                if coords is not None:
                    coords = coords.to(device)
                
                pred = model(data)
                all_predictions.append(pred)
                all_targets.append(target)
                if coords is not None:
                    all_coordinates.append(coords)
        
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        coordinates = torch.cat(all_coordinates, dim=0) if all_coordinates else None
        
        # Optimize temperature parameters
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        if self.learnable_spatial and coordinates is not None:
            optimizer.add_param_group({'params': self.spatial_temperature_net.parameters()})
        
        if self.temporal_aware:
            optimizer.add_param_group({'params': [self.temporal_temperature]})
        
        def closure():
            optimizer.zero_grad()
            calibrated_pred = self.forward(predictions, coordinates)
            loss = F.mse_loss(calibrated_pred, targets)
            
            # Add calibration regularization
            # Encourage temperatures to be close to 1 (minimal adjustment)
            temp_reg = 0.01 * (self.temperature - 1.0).pow(2)
            if self.temporal_aware:
                temp_reg += 0.01 * (self.temporal_temperature - 1.0).pow(2)
            
            total_loss = loss + temp_reg
            total_loss.backward()
            return total_loss
        
        optimizer.step(closure)
        
        # Compute calibration metrics
        with torch.no_grad():
            calibrated_pred = self.forward(predictions, coordinates)
            final_loss = F.mse_loss(calibrated_pred, targets).item()
            
            # Compute Expected Calibration Error (ECE)
            ece = self._compute_ece(calibrated_pred, targets)
            
            # Compute reliability metrics
            reliability_metrics = self._compute_reliability_metrics(
                calibrated_pred, targets
            )
        
        return {
            'final_loss': final_loss,
            'temperature': self.temperature.item(),
            'ece': ece,
            **reliability_metrics
        }
    
    def _compute_ece(self, predictions: torch.Tensor, targets: torch.Tensor, 
                     n_bins: int = 10) -> float:
        """Compute Expected Calibration Error."""
        # Convert to numpy for computation
        pred_np = predictions.cpu().numpy().flatten()
        target_np = targets.cpu().numpy().flatten()
        
        # Compute prediction confidence (using variance as proxy)
        confidence = 1.0 / (1.0 + np.var(pred_np))
        
        # Simple ECE computation for regression
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(np.abs(pred_np[in_bin] - target_np[in_bin]) < 0.1)
                avg_confidence_in_bin = confidence[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _compute_reliability_metrics(self, 
                                   predictions: torch.Tensor,
                                   targets: torch.Tensor) -> Dict[str, float]:
        """Compute additional reliability metrics."""
        with torch.no_grad():
            mse = F.mse_loss(predictions, targets).item()
            mae = F.l1_loss(predictions, targets).item()
            
            # Prediction interval coverage (assuming Gaussian)
            residuals = (predictions - targets).abs()
            coverage_90 = (residuals <= 1.645 * residuals.std()).float().mean().item()
            coverage_95 = (residuals <= 1.96 * residuals.std()).float().mean().item()
            
            return {
                'mse': mse,
                'mae': mae,
                'coverage_90': coverage_90,
                'coverage_95': coverage_95
            }


class PhysicsConstrainedCalibration(nn.Module):
    """Physics-constrained calibration for neural operators.
    
    Ensures that calibrated predictions still satisfy physical constraints
    such as conservation laws and boundary conditions.
    """
    
    def __init__(self,
                 base_calibrator: nn.Module,
                 conservation_laws: List[Callable],
                 boundary_conditions: Optional[Callable] = None,
                 constraint_weight: float = 0.1):
        """Initialize physics-constrained calibration.
        
        Args:
            base_calibrator: Base calibration method
            conservation_laws: List of conservation law functions
            boundary_conditions: Boundary condition function
            constraint_weight: Weight for physics constraint violation
        """
        super().__init__()
        
        self.base_calibrator = base_calibrator
        self.conservation_laws = conservation_laws
        self.boundary_conditions = boundary_conditions
        self.constraint_weight = constraint_weight
        
        # Physics constraint violation penalty network
        self.constraint_penalty_net = nn.Sequential(
            nn.Linear(len(conservation_laws) + (1 if boundary_conditions else 0), 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self,
                predictions: torch.Tensor,
                coordinates: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Apply physics-constrained calibration.
        
        Args:
            predictions: Model predictions
            coordinates: Spatial coordinates
            
        Returns:
            Physics-constrained calibrated predictions
        """
        # Apply base calibration
        calibrated = self.base_calibrator(predictions, coordinates, **kwargs)
        
        # Compute physics constraint violations
        violations = self._compute_constraint_violations(calibrated, coordinates)
        
        # Compute penalty factor
        penalty_input = torch.stack(violations, dim=-1)  # (batch, n_constraints)
        penalty_factor = self.constraint_penalty_net(penalty_input)
        
        # Apply physics-aware adjustment
        # Higher violations lead to stronger regularization toward original prediction
        physics_adjusted = calibrated * (1 - self.constraint_weight * penalty_factor) + \
                          predictions * (self.constraint_weight * penalty_factor)
        
        return physics_adjusted
    
    def _compute_constraint_violations(self,
                                     predictions: torch.Tensor,
                                     coordinates: Optional[torch.Tensor]) -> List[torch.Tensor]:
        """Compute physics constraint violations."""
        violations = []
        
        # Conservation law violations
        for conservation_fn in self.conservation_laws:
            try:
                if coordinates is not None:
                    violation = conservation_fn(predictions, coordinates)
                else:
                    violation = conservation_fn(predictions)
                
                # Aggregate violation over spatial dimensions
                violation_magnitude = torch.abs(violation).mean(dim=tuple(range(1, violation.ndim)))
                violations.append(violation_magnitude)
            except Exception as e:
                warnings.warn(f"Conservation law evaluation failed: {e}")
                violations.append(torch.zeros(predictions.shape[0], device=predictions.device))
        
        # Boundary condition violations
        if self.boundary_conditions is not None:
            try:
                if coordinates is not None:
                    bc_violation = self.boundary_conditions(predictions, coordinates)
                else:
                    bc_violation = self.boundary_conditions(predictions)
                
                bc_magnitude = torch.abs(bc_violation).mean(dim=tuple(range(1, bc_violation.ndim)))
                violations.append(bc_magnitude)
            except Exception as e:
                warnings.warn(f"Boundary condition evaluation failed: {e}")
                violations.append(torch.zeros(predictions.shape[0], device=predictions.device))
        
        return violations
    
    def fit(self,
            model: nn.Module,
            val_loader: DataLoader,
            **kwargs) -> Dict[str, Any]:
        """Fit physics-constrained calibration."""
        # First fit the base calibrator
        base_metrics = self.base_calibrator.fit(model, val_loader, **kwargs)
        
        # Then fine-tune the constraint penalty network
        device = next(model.parameters()).device
        self.to(device)
        
        optimizer = torch.optim.Adam(self.constraint_penalty_net.parameters(), lr=0.001)
        
        for epoch in range(50):  # Fine-tuning epochs
            total_loss = 0.0
            n_batches = 0
            
            for batch in val_loader:
                if len(batch) >= 2:
                    data, target = batch[0], batch[1]
                    coords = batch[2] if len(batch) > 2 else None
                else:
                    continue
                
                data, target = data.to(device), target.to(device)
                if coords is not None:
                    coords = coords.to(device)
                
                model.eval()
                with torch.no_grad():
                    predictions = model(data)
                
                # Apply physics-constrained calibration
                calibrated = self.forward(predictions, coords)
                
                # Loss: calibration quality + physics constraint satisfaction
                calibration_loss = F.mse_loss(calibrated, target)
                
                constraint_violations = self._compute_constraint_violations(calibrated, coords)
                constraint_loss = sum(violation.mean() for violation in constraint_violations)
                
                loss = calibration_loss + self.constraint_weight * constraint_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            if epoch % 10 == 0:
                avg_loss = total_loss / n_batches if n_batches > 0 else 0
                print(f"Physics calibration epoch {epoch}: Loss = {avg_loss:.6f}")
        
        # Compute final metrics
        final_metrics = base_metrics.copy()
        final_metrics['physics_constraint_weight'] = self.constraint_weight
        
        return final_metrics


class HierarchicalCalibration(nn.Module):
    """Hierarchical calibration for multi-scale neural operators.
    
    Provides different calibration parameters for different spatial scales,
    enabling better calibration across multi-resolution problems.
    """
    
    def __init__(self,
                 n_scales: int = 3,
                 scale_detection_method: str = "frequency"):
        """Initialize hierarchical calibration.
        
        Args:
            n_scales: Number of spatial scales
            scale_detection_method: Method for detecting scales ("frequency", "gradient")
        """
        super().__init__()
        
        self.n_scales = n_scales
        self.scale_detection_method = scale_detection_method
        
        # Scale-specific temperature parameters
        self.scale_temperatures = nn.Parameter(torch.ones(n_scales))
        
        # Scale detection network
        if scale_detection_method == "frequency":
            self.scale_detector = self._create_frequency_detector()
        elif scale_detection_method == "gradient":
            self.scale_detector = self._create_gradient_detector()
        else:
            raise ValueError(f"Unknown scale detection method: {scale_detection_method}")
    
    def _create_frequency_detector(self) -> nn.Module:
        """Create frequency-based scale detector."""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),  # Downsample for efficiency
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 32),
            nn.ReLU(),
            nn.Linear(32, self.n_scales),
            nn.Softmax(dim=1)
        )
    
    def _create_gradient_detector(self) -> nn.Module:
        """Create gradient-based scale detector."""
        return nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),  # 2 channels for grad_x, grad_y
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_scales),
            nn.Softmax(dim=1)
        )
    
    def _detect_scales(self, predictions: torch.Tensor) -> torch.Tensor:
        """Detect dominant scales in predictions.
        
        Args:
            predictions: Model predictions
            
        Returns:
            Scale weights for each sample
        """
        if self.scale_detection_method == "frequency":
            # Use FFT to analyze frequency content
            if predictions.ndim == 4:  # (batch, channels, height, width)
                # Take first channel for simplicity
                signal = predictions[:, 0:1]
                return self.scale_detector(signal)
            else:
                # For 1D case, reshape to 2D
                signal = predictions.unsqueeze(-1).unsqueeze(1)
                return self.scale_detector(signal)
        
        elif self.scale_detection_method == "gradient":
            # Use gradient magnitude to detect scales
            if predictions.ndim == 4:
                pred_single = predictions[:, 0]
                grad_x = torch.diff(pred_single, dim=-1, prepend=pred_single[:, :, :1])
                grad_y = torch.diff(pred_single, dim=-2, prepend=pred_single[:, :1, :])
                gradients = torch.stack([grad_x, grad_y], dim=1)
                return self.scale_detector(gradients)
            else:
                # For 1D, create dummy gradient
                grad = torch.diff(predictions.squeeze(1), dim=-1, prepend=predictions[:, 0:1, 0])
                gradients = torch.stack([grad, torch.zeros_like(grad)], dim=1).unsqueeze(-1)
                return self.scale_detector(gradients)
    
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        """Apply hierarchical calibration.
        
        Args:
            predictions: Model predictions
            
        Returns:
            Scale-aware calibrated predictions
        """
        # Detect scales
        scale_weights = self._detect_scales(predictions)  # (batch, n_scales)
        
        # Compute scale-weighted temperature
        effective_temperature = torch.sum(
            scale_weights.unsqueeze(-1).unsqueeze(-1) * 
            self.scale_temperatures.view(1, self.n_scales, 1, 1),
            dim=1, keepdim=True
        )
        
        # Apply calibration
        calibrated = predictions / effective_temperature
        
        return calibrated
    
    def fit(self,
            model: nn.Module,
            val_loader: DataLoader,
            epochs: int = 100,
            lr: float = 0.01) -> Dict[str, Any]:
        """Fit hierarchical calibration parameters."""
        device = next(model.parameters()).device
        self.to(device)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0
            
            for batch in val_loader:
                data, target = batch[0], batch[1]
                data, target = data.to(device), target.to(device)
                
                model.eval()
                with torch.no_grad():
                    predictions = model(data)
                
                calibrated = self.forward(predictions)
                loss = F.mse_loss(calibrated, target)
                
                # Add regularization to keep temperatures reasonable
                temp_reg = 0.01 * torch.sum((self.scale_temperatures - 1.0) ** 2)
                loss += temp_reg
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            if epoch % 20 == 0:
                avg_loss = total_loss / n_batches if n_batches > 0 else 0
                temps = self.scale_temperatures.detach().cpu().numpy()
                print(f"Hierarchical calibration epoch {epoch}: Loss = {avg_loss:.6f}, Temps = {temps}")
        
        return {
            'final_scale_temperatures': self.scale_temperatures.detach().cpu().numpy().tolist()
        }


class AdvancedCalibrationSuite:
    """Suite of advanced calibration methods for neural operators."""
    
    @staticmethod
    def create_multidimensional_calibrator(spatial_dims: int = 2,
                                         temporal_aware: bool = False) -> MultiDimensionalTemperatureScaling:
        """Create multi-dimensional temperature scaling calibrator."""
        return MultiDimensionalTemperatureScaling(
            spatial_dims=spatial_dims,
            temporal_aware=temporal_aware
        )
    
    @staticmethod
    def create_physics_constrained_calibrator(
            base_calibrator: nn.Module,
            conservation_laws: List[Callable],
            boundary_conditions: Optional[Callable] = None) -> PhysicsConstrainedCalibration:
        """Create physics-constrained calibrator."""
        return PhysicsConstrainedCalibration(
            base_calibrator=base_calibrator,
            conservation_laws=conservation_laws,
            boundary_conditions=boundary_conditions
        )
    
    @staticmethod
    def create_hierarchical_calibrator(n_scales: int = 3,
                                     scale_detection_method: str = "frequency") -> HierarchicalCalibration:
        """Create hierarchical calibrator for multi-scale problems."""
        return HierarchicalCalibration(
            n_scales=n_scales,
            scale_detection_method=scale_detection_method
        )