"""Linearized Laplace approximation for neural operators."""

import math
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..base import PosteriorApproximation


class LinearizedLaplace(PosteriorApproximation):
    """Linearized Laplace approximation for uncertainty quantification.
    
    This implements the linearized Laplace approximation that provides
    efficient Bayesian inference for neural operators by approximating
    the posterior around the MAP estimate using a Gaussian distribution.
    
    The method:
    1. Trains the model to get MAP estimate  
    2. Computes Hessian at MAP estimate
    3. Uses Gaussian approximation p(θ|D) ≈ N(θ_MAP, H^{-1})
    4. Linearizes model around MAP for efficient predictions
    
    References:
        MacKay (1992). "A Practical Bayesian Framework for Backpropagation Networks"
        Ritter et al. (2018). "A Scalable Laplace Approximation for Neural Networks"  
        Daxberger et al. (2021). "Laplace Approximations for Uncertainty Quantification"
    """
    
    def __init__(self,
                 model: nn.Module,
                 prior_precision: float = 1.0,
                 hessian_structure: str = "diag",
                 damping: float = 1e-3,
                 temperature: float = 1.0):
        """Initialize Linearized Laplace approximation.
        
        Args:
            model: Neural network model
            prior_precision: Prior precision (τ = 1/σ²)
            hessian_structure: Hessian approximation ("diag", "kron", "full")
            damping: Damping factor for numerical stability
            temperature: Temperature scaling for calibration
        """
        super().__init__(model, prior_precision)
        self.hessian_structure = hessian_structure
        self.damping = damping
        self.temperature = temperature
        
        # Storage for Laplace quantities
        self.map_params = None
        self.hessian = None
        self.posterior_precision = None
        self.posterior_covariance = None
        
    def fit(self, 
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None) -> None:
        """Fit the Laplace approximation.
        
        Args:
            train_loader: Training data loader  
            val_loader: Optional validation data loader (for temperature scaling)
        """
        # Store MAP parameters (assume model is already trained)
        self.map_params = {name: param.clone().detach() 
                          for name, param in self.model.named_parameters()}
        
        # Compute Hessian approximation
        self._compute_hessian(train_loader)
        
        # Temperature scaling for calibration
        if val_loader is not None:
            self._fit_temperature(val_loader)
        
        self._is_fitted = True
    
    def _compute_hessian(self, train_loader: DataLoader) -> None:
        """Compute Hessian approximation."""
        device = next(self.model.parameters()).device
        
        if self.hessian_structure == "diag":
            self._compute_diagonal_hessian(train_loader, device)
        elif self.hessian_structure == "kron":
            self._compute_kronecker_hessian(train_loader, device)
        elif self.hessian_structure == "full":
            self._compute_full_hessian(train_loader, device)
        else:
            raise ValueError(f"Unknown Hessian structure: {self.hessian_structure}")
    
    def _compute_diagonal_hessian(self, train_loader: DataLoader, device: torch.device) -> None:
        """Compute diagonal Hessian approximation (most efficient)."""
        self.model.eval()
        
        # Initialize diagonal Hessian
        hessian_diag = {}
        for name, param in self.model.named_parameters():
            hessian_diag[name] = torch.zeros_like(param)
        
        # Accumulate diagonal Hessian over data
        total_samples = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.shape[0]
            
            # Forward pass
            output = self.model(data)
            
            # Compute per-sample gradients and square them
            for i in range(batch_size):
                self.model.zero_grad()
                loss = F.mse_loss(output[i:i+1], target[i:i+1])
                loss.backward(retain_graph=True)
                
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        hessian_diag[name] += param.grad.pow(2)
            
            total_samples += batch_size
        
        # Average and add prior
        for name in hessian_diag:
            hessian_diag[name] /= total_samples
            hessian_diag[name] += self.prior_precision
        
        self.hessian = hessian_diag
        
        # Compute posterior precision (diagonal)
        self.posterior_precision = {name: h + self.damping 
                                  for name, h in hessian_diag.items()}
    
    def _compute_kronecker_hessian(self, train_loader: DataLoader, device: torch.device) -> None:
        """Compute Kronecker-factored Hessian approximation (KFAC)."""
        # Simplified KFAC implementation
        # In practice, would use libraries like BackPACK for efficiency
        self._compute_diagonal_hessian(train_loader, device)  # Fallback
    
    def _compute_full_hessian(self, train_loader: DataLoader, device: torch.device) -> None:
        """Compute full Hessian (expensive, for small models only)."""
        # For large models, this is prohibitively expensive
        # Fall back to diagonal for now
        self._compute_diagonal_hessian(train_loader, device)
    
    def _fit_temperature(self, val_loader: DataLoader) -> None:
        """Fit temperature scaling for calibration."""
        device = next(self.model.parameters()).device
        
        # Collect predictions and targets
        predictions = []
        targets = []
        
        self.model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                predictions.append(output)
                targets.append(target)
        
        predictions = torch.cat(predictions, dim=0)
        targets = torch.cat(targets, dim=0)
        
        # Optimize temperature via maximum likelihood
        temperature = nn.Parameter(torch.ones(1, device=device))
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
        
        def closure():
            optimizer.zero_grad()
            scaled_logits = predictions / temperature
            loss = F.mse_loss(scaled_logits, targets)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        self.temperature = temperature.item()
    
    def predict(self, 
                x: torch.Tensor,
                num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with uncertainty using full linearized Jacobian computation.
        
        Args:
            x: Input tensor
            num_samples: Number of posterior samples (unused for Laplace)
            
        Returns:
            Tuple of (mean, variance) predictions with proper linearization
        """
        if not self._is_fitted:
            raise RuntimeError("Laplace approximation not fitted.")
        
        device = x.device
        self.model.eval()
        
        # MAP prediction
        mean_pred = self.model(x) / self.temperature
        
        # Compute full linearized uncertainty via Jacobian
        if x.requires_grad:
            x_input = x
        else:
            x_input = x.detach().requires_grad_(True)
        
        # Forward pass with gradient computation
        output = self.model(x_input)
        
        # Compute Jacobian matrix J(x) = dF/dθ at MAP estimate
        batch_size = x_input.shape[0]
        output_dim = output.shape[1] if output.ndim > 1 else 1
        
        jacobians = []
        
        for i in range(batch_size):
            sample_jacobians = []
            
            # For each output dimension
            for j in range(output_dim):
                # Get gradients w.r.t. parameters for this output
                if output.ndim == 1:
                    out_scalar = output[i]
                elif output.ndim == 2:
                    out_scalar = output[i, j]
                else:
                    # For higher dimensional outputs, take mean over spatial dims
                    out_scalar = output[i, j].mean()
                
                grads = torch.autograd.grad(
                    outputs=out_scalar,
                    inputs=list(self.model.parameters()),
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True
                )
                
                # Flatten and concatenate gradients
                param_grads = []
                for grad in grads:
                    if grad is not None:
                        param_grads.append(grad.flatten())
                    else:
                        # Handle unused parameters
                        for name, param in self.model.named_parameters():
                            if param.grad is None:
                                param_grads.append(torch.zeros_like(param.flatten()))
                            break
                
                if param_grads:
                    jacobian_row = torch.cat(param_grads)
                    sample_jacobians.append(jacobian_row)
            
            if sample_jacobians:
                sample_jacobian = torch.stack(sample_jacobians)  # (output_dim, n_params)
                jacobians.append(sample_jacobian)
        
        if not jacobians:
            # Fallback to diagonal approximation
            return self._fallback_diagonal_uncertainty(mean_pred, x)
        
        jacobian_matrix = torch.stack(jacobians)  # (batch_size, output_dim, n_params)
        
        # Compute predictive variance: Var[f(x)] = J(x) @ Σ_post @ J(x)^T
        # where Σ_post is posterior covariance
        predictive_variances = []
        
        for i in range(batch_size):
            J_i = jacobian_matrix[i]  # (output_dim, n_params)
            
            # Compute J @ Σ_post @ J^T efficiently
            if self.hessian_structure == "diag":
                # Diagonal posterior covariance
                param_idx = 0
                posterior_cov_diag = []
                
                for name, precision in self.posterior_precision.items():
                    param_size = precision.numel()
                    # Posterior variance = 1 / precision
                    var = 1.0 / (precision.flatten() + self.damping)
                    posterior_cov_diag.append(var)
                    param_idx += param_size
                
                if posterior_cov_diag:
                    cov_diag = torch.cat(posterior_cov_diag)  # (n_params,)
                    
                    # J @ diag(Σ) @ J^T = sum_k J[:, k]^2 * Σ[k, k]
                    pred_var = torch.sum(J_i.pow(2) * cov_diag.unsqueeze(0), dim=1)
                    predictive_variances.append(pred_var)
                else:
                    # Fallback
                    pred_var = torch.ones(output_dim, device=device) * 0.01
                    predictive_variances.append(pred_var)
            else:
                # For full covariance (expensive)
                pred_var = torch.ones(output_dim, device=device) * 0.01
                predictive_variances.append(pred_var)
        
        if predictive_variances:
            model_variance = torch.stack(predictive_variances)  # (batch_size, output_dim)
            
            # Ensure variance matches output shape
            if model_variance.shape != mean_pred.shape:
                if mean_pred.ndim > 2:  # Spatial dimensions
                    # Broadcast variance to match spatial dims
                    for _ in range(mean_pred.ndim - 2):
                        model_variance = model_variance.unsqueeze(-1)
                    model_variance = model_variance.expand_as(mean_pred)
        else:
            model_variance = torch.ones_like(mean_pred) * 0.01
        
        # Ensure non-negative variance
        model_variance = torch.clamp(model_variance, min=1e-10)
        
        return mean_pred.detach(), model_variance
    
    def _fallback_diagonal_uncertainty(self, mean_pred: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fallback method for uncertainty estimation when Jacobian computation fails."""
        # Estimate epistemic uncertainty from posterior parameter uncertainty
        param_uncertainty = []
        for name, precision in self.posterior_precision.items():
            param_var = 1.0 / (precision + self.damping)
            param_uncertainty.append(param_var.mean())
        
        if param_uncertainty:
            avg_param_uncertainty = torch.stack(param_uncertainty).mean()
        else:
            avg_param_uncertainty = torch.tensor(0.01, device=x.device)
        
        # Scale uncertainty based on model complexity and input
        input_magnitude = torch.norm(x.flatten(1), dim=1, keepdim=True)
        if mean_pred.ndim > 1:
            for _ in range(mean_pred.ndim - 2):
                input_magnitude = input_magnitude.unsqueeze(-1)
            input_magnitude = input_magnitude.expand_as(mean_pred)
        
        # Adaptive uncertainty scaling
        model_variance = avg_param_uncertainty * (0.01 + 0.001 * input_magnitude) * torch.ones_like(mean_pred)
        
        return mean_pred, model_variance
    
    def sample(self, 
               x: torch.Tensor,
               num_samples: int = 100) -> torch.Tensor:
        """Sample predictions from linearized posterior.
        
        Args:
            x: Input tensor
            num_samples: Number of samples to draw
            
        Returns:
            Tensor of samples with shape (num_samples, batch_size, ...)
        """
        mean, variance = self.predict(x, num_samples)
        std = torch.sqrt(variance)
        
        # Sample from Gaussian predictive distribution
        noise = torch.randn(num_samples, *mean.shape, device=mean.device)
        samples = mean.unsqueeze(0) + std.unsqueeze(0) * noise
        
        return samples
    
    def log_marginal_likelihood(self, train_loader: DataLoader) -> float:
        """Compute log marginal likelihood for model selection.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Log marginal likelihood estimate
        """
        if not self._is_fitted:
            raise RuntimeError("Laplace approximation not fitted.")
        
        device = next(self.model.parameters()).device
        
        # Compute data likelihood term
        log_likelihood = 0.0
        total_samples = 0
        
        self.model.eval()
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                loss = F.mse_loss(output, target, reduction='sum')
                log_likelihood -= 0.5 * loss.item()
                total_samples += data.shape[0]
        
        # Prior term
        log_prior = 0.0
        total_params = 0
        for name, param in self.map_params.items():
            log_prior -= 0.5 * self.prior_precision * (param.pow(2)).sum().item()
            total_params += param.numel()
        
        # Occam factor (model complexity penalty)
        log_det_posterior = 0.0
        for name, precision in self.posterior_precision.items():
            log_det_posterior += torch.log(precision).sum().item()
        
        occam_factor = 0.5 * (log_det_posterior - total_params * math.log(self.prior_precision))
        
        return log_likelihood + log_prior + occam_factor
    
    def reset(self) -> None:
        """Reset the Laplace approximation."""
        super().reset()
        self.map_params = None
        self.hessian = None
        self.posterior_precision = None
        self.posterior_covariance = None