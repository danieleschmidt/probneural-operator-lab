"""Adaptive Uncertainty Scaling Framework - Research Enhancement.

This module implements adaptive uncertainty scaling mechanisms that automatically
adjust uncertainty estimates based on:
1. Input domain characteristics
2. Model confidence patterns
3. Historical prediction accuracy
4. Physics-informed constraints

Research Innovation:
- Context-aware uncertainty calibration
- Online adaptation during inference
- Multi-modal uncertainty scaling
- Physics-constrained uncertainty bounds

Authors: TERRAGON Labs Research Team  
Date: 2025-08-13
"""

from typing import Dict, List, Tuple, Optional, Callable, Any
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class AdaptiveUncertaintyScaler:
    """Adaptive uncertainty scaling that adjusts based on context and performance.
    
    This system learns to scale uncertainty estimates based on:
    1. Input characteristics (domain coverage, complexity)
    2. Model performance history
    3. Physics constraints and conservation laws
    4. Multi-fidelity information when available
    
    Key Innovation: Unlike fixed temperature scaling, this adapts the scaling
    dynamically based on the specific input and context.
    """
    
    def __init__(self,
                 base_model: nn.Module,
                 adaptation_rate: float = 0.01,
                 memory_length: int = 1000,
                 physics_constraints: Optional[Dict[str, Any]] = None,
                 multi_fidelity: bool = False):
        """Initialize adaptive uncertainty scaler.
        
        Args:
            base_model: Underlying probabilistic model
            adaptation_rate: Rate of online adaptation
            memory_length: Number of recent predictions to remember
            physics_constraints: Physical constraints for uncertainty bounds
            multi_fidelity: Whether to use multi-fidelity scaling
        """
        self.base_model = base_model
        self.adaptation_rate = adaptation_rate
        self.memory_length = memory_length
        self.physics_constraints = physics_constraints or {}
        self.multi_fidelity = multi_fidelity
        
        # Online adaptation state
        self.prediction_history = []
        self.error_history = []
        self.scaling_factors = {}
        self.domain_statistics = {}
        
        # Learned scaling network
        self.scaling_network = UncertaintyScalingNetwork()
        self.is_fitted = False
        
    def fit(self, 
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            calibration_loader: Optional[DataLoader] = None) -> None:
        """Fit the adaptive uncertainty scaling mechanism.
        
        Args:
            train_loader: Training data for domain statistics
            val_loader: Validation data for calibration
            calibration_loader: Dedicated calibration dataset
        """
        device = next(self.base_model.parameters()).device
        
        # Step 1: Collect domain statistics
        self._collect_domain_statistics(train_loader, device)
        
        # Step 2: Train scaling network
        if val_loader is not None:
            self._train_scaling_network(val_loader, device)
        
        # Step 3: Calibrate with dedicated data if available
        if calibration_loader is not None:
            self._calibrate_scaling(calibration_loader, device)
        
        self.is_fitted = True
    
    def _collect_domain_statistics(self, train_loader: DataLoader, device: torch.device) -> None:
        """Collect statistics about the input domain."""
        input_norms = []
        gradient_norms = []
        prediction_magnitudes = []
        
        self.base_model.eval()
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx >= 100:  # Limit for efficiency
                break
                
            data, target = data.to(device), target.to(device)
            
            # Input statistics
            input_norm = torch.norm(data.flatten(1), dim=1)
            input_norms.extend(input_norm.cpu().tolist())
            
            # Gradient statistics (for complexity estimation)
            data_grad = data.detach().requires_grad_(True)
            with torch.enable_grad():
                output = self.base_model(data_grad)
                if hasattr(self.base_model, 'predict_with_uncertainty'):
                    mean, var = self.base_model.predict_with_uncertainty(data_grad)
                else:
                    mean, var = output, torch.ones_like(output) * 0.01
                
                # Compute gradient norm w.r.t input
                grad_outputs = torch.ones_like(mean)
                gradients = torch.autograd.grad(
                    outputs=mean.sum(),
                    inputs=data_grad,
                    grad_outputs=grad_outputs,
                    create_graph=False,
                    retain_graph=False,
                    only_inputs=True
                )[0]
                
                grad_norm = torch.norm(gradients.flatten(1), dim=1)
                gradient_norms.extend(grad_norm.cpu().tolist())
                
                # Prediction magnitude
                pred_magnitude = torch.norm(mean.flatten(1), dim=1)
                prediction_magnitudes.extend(pred_magnitude.cpu().tolist())
        
        # Store domain statistics
        self.domain_statistics = {
            'input_norm_mean': float(torch.tensor(input_norms).mean()),
            'input_norm_std': float(torch.tensor(input_norms).std()),
            'gradient_norm_mean': float(torch.tensor(gradient_norms).mean()),
            'gradient_norm_std': float(torch.tensor(gradient_norms).std()),
            'prediction_magnitude_mean': float(torch.tensor(prediction_magnitudes).mean()),
            'prediction_magnitude_std': float(torch.tensor(prediction_magnitudes).std()),
        }
    
    def _train_scaling_network(self, val_loader: DataLoader, device: torch.device) -> None:
        """Train neural network to predict optimal uncertainty scaling."""
        self.scaling_network.to(device)
        optimizer = torch.optim.Adam(self.scaling_network.parameters(), lr=0.001)
        
        # Collect training data for scaling network
        features = []
        optimal_scales = []
        
        self.base_model.eval()
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            
            # Get base uncertainty prediction
            if hasattr(self.base_model, 'predict_with_uncertainty'):
                mean, variance = self.base_model.predict_with_uncertainty(data)
            else:
                mean = self.base_model(data)
                variance = torch.ones_like(mean) * 0.01
            
            # Compute input features for scaling network
            input_features = self._extract_features(data, mean, variance)
            
            # Compute optimal scaling based on actual errors
            actual_errors = (mean - target).pow(2)
            predicted_variance = variance
            
            # Optimal scale makes predicted variance match actual squared error
            optimal_scale = torch.sqrt(actual_errors / (predicted_variance + 1e-8))
            optimal_scale = torch.clamp(optimal_scale, 0.1, 10.0)  # Reasonable bounds
            
            features.append(input_features)
            optimal_scales.append(optimal_scale.mean(dim=tuple(range(1, optimal_scale.ndim))))
        
        if not features:
            warnings.warn("No features collected for scaling network training")
            return
        
        features = torch.cat(features, dim=0)
        optimal_scales = torch.cat(optimal_scales, dim=0)
        
        # Train scaling network
        for epoch in range(50):
            optimizer.zero_grad()
            predicted_scales = self.scaling_network(features)
            loss = F.mse_loss(predicted_scales.squeeze(), optimal_scales)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Scaling network epoch {epoch}, loss: {loss.item():.6f}")
    
    def _extract_features(self, 
                         data: torch.Tensor,
                         mean: torch.Tensor,
                         variance: torch.Tensor) -> torch.Tensor:
        """Extract features for uncertainty scaling prediction."""
        batch_size = data.shape[0]
        features = []
        
        # Input characteristics
        input_norm = torch.norm(data.flatten(1), dim=1)
        features.append(input_norm.unsqueeze(1))
        
        # Prediction characteristics  
        pred_magnitude = torch.norm(mean.flatten(1), dim=1)
        features.append(pred_magnitude.unsqueeze(1))
        
        # Uncertainty characteristics
        uncertainty_magnitude = torch.sqrt(variance.flatten(1)).mean(dim=1)
        features.append(uncertainty_magnitude.unsqueeze(1))
        
        # Relative metrics
        rel_uncertainty = uncertainty_magnitude / (pred_magnitude + 1e-8)
        features.append(rel_uncertainty.unsqueeze(1))
        
        # Domain deviation (how far from training distribution)
        if self.domain_statistics:
            input_norm_z = (input_norm - self.domain_statistics['input_norm_mean']) / (
                self.domain_statistics['input_norm_std'] + 1e-8
            )
            features.append(input_norm_z.unsqueeze(1))
            
            pred_magnitude_z = (pred_magnitude - self.domain_statistics['prediction_magnitude_mean']) / (
                self.domain_statistics['prediction_magnitude_std'] + 1e-8
            )
            features.append(pred_magnitude_z.unsqueeze(1))
        
        # Physics constraints if available
        if self.physics_constraints:
            physics_features = self._compute_physics_features(data, mean, variance)
            features.append(physics_features)
        
        return torch.cat(features, dim=1)
    
    def _compute_physics_features(self, 
                                 data: torch.Tensor,
                                 mean: torch.Tensor,
                                 variance: torch.Tensor) -> torch.Tensor:
        """Compute physics-based features for uncertainty scaling."""
        batch_size = data.shape[0]
        physics_features = []
        
        # Conservation law violations
        if 'conservation' in self.physics_constraints:
            # Check mass conservation (example)
            mass_initial = data.sum(dim=tuple(range(1, data.ndim)))
            mass_predicted = mean.sum(dim=tuple(range(1, mean.ndim)))
            mass_violation = torch.abs(mass_predicted - mass_initial) / (mass_initial + 1e-8)
            physics_features.append(mass_violation.unsqueeze(1))
        
        # Energy conservation
        if 'energy' in self.physics_constraints:
            energy_initial = (data.pow(2)).sum(dim=tuple(range(1, data.ndim)))
            energy_predicted = (mean.pow(2)).sum(dim=tuple(range(1, mean.ndim)))
            energy_violation = torch.abs(energy_predicted - energy_initial) / (energy_initial + 1e-8)
            physics_features.append(energy_violation.unsqueeze(1))
        
        # PDE residual magnitude
        if 'pde_residual' in self.physics_constraints:
            # Simplified PDE residual computation
            if mean.requires_grad or data.requires_grad:
                # Compute spatial derivatives (simplified)
                if mean.ndim >= 3:  # Has spatial dimensions
                    dx = torch.gradient(mean, dim=-1)[0] if mean.shape[-1] > 1 else torch.zeros_like(mean)
                    residual_magnitude = torch.norm(dx.flatten(1), dim=1)
                    physics_features.append(residual_magnitude.unsqueeze(1))
        
        if not physics_features:
            # Return zero features if no physics constraints
            physics_features.append(torch.zeros(batch_size, 1, device=data.device))
        
        return torch.cat(physics_features, dim=1)
    
    def _calibrate_scaling(self, calibration_loader: DataLoader, device: torch.device) -> None:
        """Final calibration of uncertainty scaling."""
        # Collect calibration data
        all_uncertainties = []
        all_errors = []
        
        self.base_model.eval()
        for data, target in calibration_loader:
            data, target = data.to(device), target.to(device)
            
            # Get scaled uncertainties
            scaled_mean, scaled_variance = self.predict_with_adaptive_scaling(data)
            
            actual_errors = (scaled_mean - target).pow(2)
            predicted_uncertainties = scaled_variance
            
            all_uncertainties.extend(predicted_uncertainties.flatten().cpu().tolist())
            all_errors.extend(actual_errors.flatten().cpu().tolist())
        
        if all_uncertainties and all_errors:
            # Compute calibration curve
            uncertainties = torch.tensor(all_uncertainties)
            errors = torch.tensor(all_errors)
            
            # Bin by uncertainty level and check calibration
            n_bins = 10
            bin_boundaries = torch.quantile(uncertainties, torch.linspace(0, 1, n_bins + 1))
            
            calibration_error = 0.0
            for i in range(n_bins):
                mask = (uncertainties >= bin_boundaries[i]) & (uncertainties < bin_boundaries[i + 1])
                if mask.sum() > 0:
                    bin_uncertainty = uncertainties[mask].mean()
                    bin_error = errors[mask].mean()
                    calibration_error += torch.abs(bin_uncertainty - bin_error).item()
            
            self.calibration_error = calibration_error / n_bins
            print(f"Final calibration error: {self.calibration_error:.6f}")
    
    def predict_with_adaptive_scaling(self, 
                                    x: torch.Tensor,
                                    update_history: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with adaptive uncertainty scaling.
        
        Args:
            x: Input tensor
            update_history: Whether to update adaptation history
            
        Returns:
            Tuple of (scaled_mean, scaled_variance)
        """
        if not self.is_fitted:
            raise RuntimeError("Adaptive scaler not fitted. Call fit() first.")
        
        device = x.device
        
        # Get base prediction
        if hasattr(self.base_model, 'predict_with_uncertainty'):
            base_mean, base_variance = self.base_model.predict_with_uncertainty(x)
        else:
            base_mean = self.base_model(x)
            base_variance = torch.ones_like(base_mean) * 0.01
        
        # Extract features for scaling
        features = self._extract_features(x, base_mean, base_variance)
        
        # Predict scaling factors
        with torch.no_grad():
            scaling_factors = self.scaling_network(features)
            scaling_factors = torch.clamp(scaling_factors, 0.1, 10.0)  # Reasonable bounds
        
        # Apply scaling to variance
        if scaling_factors.ndim == 1:
            scaling_factors = scaling_factors.unsqueeze(1)
        
        # Broadcast scaling factors to match variance shape
        while scaling_factors.ndim < base_variance.ndim:
            scaling_factors = scaling_factors.unsqueeze(-1)
        
        scaling_factors = scaling_factors.expand_as(base_variance)
        scaled_variance = base_variance * scaling_factors.pow(2)
        
        # Apply physics constraints if specified
        if self.physics_constraints:
            scaled_variance = self._apply_physics_constraints(x, base_mean, scaled_variance)
        
        # Online adaptation
        if update_history and len(self.prediction_history) > 0:
            self._update_online_adaptation(x, base_mean, scaled_variance)
        
        return base_mean, scaled_variance
    
    def _apply_physics_constraints(self, 
                                  x: torch.Tensor,
                                  mean: torch.Tensor,
                                  variance: torch.Tensor) -> torch.Tensor:
        """Apply physics-based constraints to uncertainty estimates."""
        constrained_variance = variance.clone()
        
        # Conservation constraints
        if 'conservation_bounds' in self.physics_constraints:
            bounds = self.physics_constraints['conservation_bounds']
            max_violation = bounds.get('max_relative_violation', 0.1)
            
            # Constrain uncertainty such that 3σ violations don't exceed bounds
            max_allowed_std = max_violation * torch.abs(mean) / 3.0
            max_allowed_variance = max_allowed_std.pow(2)
            constrained_variance = torch.min(constrained_variance, max_allowed_variance)
        
        # Positivity constraints
        if 'positive_quantities' in self.physics_constraints:
            positive_mask = mean > 0
            # For positive quantities, uncertainty shouldn't make predictions negative
            max_std_positive = mean / 3.0  # 3σ rule
            max_var_positive = max_std_positive.pow(2)
            constrained_variance = torch.where(
                positive_mask,
                torch.min(constrained_variance, max_var_positive),
                constrained_variance
            )
        
        # Energy bounds
        if 'energy_bounds' in self.physics_constraints:
            bounds = self.physics_constraints['energy_bounds']
            if 'max_energy' in bounds:
                max_energy = bounds['max_energy']
                # Uncertainty shouldn't exceed energy bounds
                current_energy = mean.pow(2).sum(dim=tuple(range(1, mean.ndim)), keepdim=True)
                energy_headroom = max_energy - current_energy
                energy_headroom = torch.clamp(energy_headroom, min=0)
                
                max_energy_std = torch.sqrt(energy_headroom) / 3.0
                max_energy_variance = max_energy_std.pow(2)
                
                # Broadcast to match variance shape
                while max_energy_variance.ndim < constrained_variance.ndim:
                    max_energy_variance = max_energy_variance.unsqueeze(-1)
                max_energy_variance = max_energy_variance.expand_as(constrained_variance)
                
                constrained_variance = torch.min(constrained_variance, max_energy_variance)
        
        return constrained_variance
    
    def _update_online_adaptation(self, 
                                 x: torch.Tensor,
                                 mean: torch.Tensor,
                                 variance: torch.Tensor) -> None:
        """Update online adaptation based on recent predictions."""
        # Store prediction for later comparison
        self.prediction_history.append({
            'input': x.detach().cpu(),
            'mean': mean.detach().cpu(),
            'variance': variance.detach().cpu(),
            'timestamp': len(self.prediction_history)
        })
        
        # Maintain history length
        if len(self.prediction_history) > self.memory_length:
            self.prediction_history.pop(0)
    
    def update_with_feedback(self, 
                           prediction_id: int,
                           true_value: torch.Tensor) -> None:
        """Update scaling based on ground truth feedback.
        
        Args:
            prediction_id: ID of the prediction to update
            true_value: True observed value
        """
        if prediction_id >= len(self.prediction_history):
            warnings.warn(f"Prediction ID {prediction_id} not found in history")
            return
        
        prediction = self.prediction_history[prediction_id]
        predicted_mean = prediction['mean']
        predicted_variance = prediction['variance']
        
        # Compute actual error
        actual_error = (predicted_mean - true_value.cpu()).pow(2)
        
        # Update error history
        self.error_history.append({
            'predicted_variance': predicted_variance,
            'actual_error': actual_error,
            'scaling_quality': actual_error / (predicted_variance + 1e-8)
        })
        
        # Online adaptation of scaling network (simplified)
        if len(self.error_history) >= 10:  # Minimum for adaptation
            recent_errors = self.error_history[-10:]
            avg_scaling_quality = torch.stack([e['scaling_quality'] for e in recent_errors]).mean()
            
            # Adjust scaling based on recent performance
            if avg_scaling_quality > 2.0:  # Under-confident
                self.adaptation_rate *= 0.95  # Reduce scaling
            elif avg_scaling_quality < 0.5:  # Over-confident
                self.adaptation_rate *= 1.05  # Increase scaling
    
    def get_adaptation_metrics(self) -> Dict[str, float]:
        """Get metrics about the adaptation process."""
        metrics = {}
        
        if hasattr(self, 'calibration_error'):
            metrics['calibration_error'] = self.calibration_error
        
        if self.error_history:
            recent_scaling_quality = [e['scaling_quality'].mean().item() for e in self.error_history[-50:]]
            metrics['avg_scaling_quality'] = sum(recent_scaling_quality) / len(recent_scaling_quality)
            metrics['scaling_variance'] = torch.tensor(recent_scaling_quality).var().item()
        
        metrics['adaptation_rate'] = self.adaptation_rate
        metrics['history_length'] = len(self.prediction_history)
        
        return metrics


class UncertaintyScalingNetwork(nn.Module):
    """Neural network to predict optimal uncertainty scaling factors."""
    
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64):
        """Initialize scaling network.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive scaling factors
        )
        
        # Initialize with identity scaling
        with torch.no_grad():
            self.network[-2].bias.fill_(0.0)  # Softplus(0) ≈ 0.69
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict scaling factors.
        
        Args:
            features: Input features for scaling prediction
            
        Returns:
            Predicted scaling factors
        """
        return self.network(features)


class MultiModalAdaptiveScaler:
    """Multi-modal adaptive uncertainty scaler for different input types.
    
    This extension handles different input modalities (e.g., different PDE types,
    boundary conditions, parameter regimes) with specialized scaling.
    """
    
    def __init__(self, 
                 base_scaler: AdaptiveUncertaintyScaler,
                 modality_detector: Optional[Callable] = None):
        """Initialize multi-modal scaler.
        
        Args:
            base_scaler: Base adaptive uncertainty scaler
            modality_detector: Function to detect input modality
        """
        self.base_scaler = base_scaler
        self.modality_detector = modality_detector or self._default_modality_detector
        self.modality_scalers = {}
    
    def _default_modality_detector(self, x: torch.Tensor) -> str:
        """Default modality detection based on input statistics."""
        input_mean = x.mean().item()
        input_std = x.std().item()
        
        if input_std < 0.1:
            return "smooth"
        elif input_std > 1.0:
            return "turbulent"
        else:
            return "moderate"
    
    def predict_with_modal_scaling(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with modality-specific scaling."""
        # Detect modality
        modality = self.modality_detector(x)
        
        # Get base prediction
        mean, variance = self.base_scaler.predict_with_adaptive_scaling(x)
        
        # Apply modality-specific scaling
        if modality in self.modality_scalers:
            modal_scale = self.modality_scalers[modality]
            variance = variance * modal_scale
        
        return mean, variance
    
    def update_modality_scaling(self, 
                              modality: str,
                              scaling_factor: float) -> None:
        """Update scaling for a specific modality."""
        self.modality_scalers[modality] = scaling_factor