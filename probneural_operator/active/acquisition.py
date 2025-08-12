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
    """Advanced physics-aware acquisition function with multi-PDE support.
    
    Combines uncertainty with physics constraints using sophisticated PDE residual
    computation, conservation law enforcement, and adaptive weighting.
    """
    
    def __init__(self, 
                 base_acquisition: AcquisitionFunction,
                 physics_residual_fn: Optional[Callable] = None,
                 conservation_laws: Optional[list] = None,
                 boundary_conditions: Optional[Callable] = None,
                 physics_weight: float = 0.1,
                 adaptive_weighting: bool = True,
                 pde_type: str = "general"):
        """Initialize advanced physics-aware acquisition.
        
        Args:
            base_acquisition: Base acquisition function
            physics_residual_fn: Custom physics residual function
            conservation_laws: List of conservation law functions
            boundary_conditions: Boundary condition function
            physics_weight: Base weight for physics term
            adaptive_weighting: Whether to use adaptive weighting
            pde_type: Type of PDE ("navier_stokes", "burgers", "darcy", "wave", "heat")
        """
        self.base_acquisition = base_acquisition
        self.physics_residual_fn = physics_residual_fn
        self.conservation_laws = conservation_laws or []
        self.boundary_conditions = boundary_conditions
        self.physics_weight = physics_weight
        self.adaptive_weighting = adaptive_weighting
        self.pde_type = pde_type
        
        # Initialize PDE-specific residual functions
        if physics_residual_fn is None:
            self.physics_residual_fn = self._get_pde_residual_fn(pde_type)
    
    def _get_pde_residual_fn(self, pde_type: str) -> Callable:
        """Get PDE-specific residual function."""
        
        def navier_stokes_residual(u: torch.Tensor, x: torch.Tensor, 
                                 nu: float = 0.01, rho: float = 1.0) -> torch.Tensor:
            """Navier-Stokes residual computation."""
            # Assumes x has shape (batch, spatial_dims) and u has shape (batch, velocity_components, *spatial)
            if x.shape[-1] == 2:  # 2D case
                return self._compute_navier_stokes_2d(u, x, nu, rho)
            elif x.shape[-1] == 3:  # 3D case
                return self._compute_navier_stokes_3d(u, x, nu, rho)
            else:
                return torch.zeros_like(u[:, 0])
        
        def burgers_residual(u: torch.Tensor, x: torch.Tensor, 
                           nu: float = 0.01) -> torch.Tensor:
            """Burgers equation residual."""
            return self._compute_burgers_residual(u, x, nu)
        
        def wave_residual(u: torch.Tensor, x: torch.Tensor, 
                         c: float = 1.0) -> torch.Tensor:
            """Wave equation residual."""
            return self._compute_wave_residual(u, x, c)
        
        def heat_residual(u: torch.Tensor, x: torch.Tensor, 
                         alpha: float = 1.0) -> torch.Tensor:
            """Heat equation residual."""
            return self._compute_heat_residual(u, x, alpha)
        
        residual_functions = {
            "navier_stokes": navier_stokes_residual,
            "burgers": burgers_residual,
            "wave": wave_residual,
            "heat": heat_residual,
            "general": lambda u, x: torch.zeros_like(u)
        }
        
        return residual_functions.get(pde_type, residual_functions["general"])
    
    def _compute_navier_stokes_2d(self, u: torch.Tensor, x: torch.Tensor, 
                                nu: float, rho: float) -> torch.Tensor:
        """Compute 2D Navier-Stokes residual with proper derivatives."""
        # u should have shape (batch, 2, height, width) for velocity components
        # x should have spatial coordinates
        
        # Enable gradient computation
        x_grid = self._create_spatial_grid(x, u.shape)
        if not x_grid.requires_grad:
            x_grid = x_grid.requires_grad_(True)
        
        # Forward pass to get velocity field with gradients
        u_pred = u  # Assume u is already the velocity prediction
        
        # Compute spatial derivatives using finite differences (more stable than autograd)
        u_x, u_y = self._compute_spatial_derivatives_2d(u_pred, x_grid)
        u_xx, u_yy = self._compute_second_derivatives_2d(u_pred, x_grid)
        
        # Velocity components
        u_vel = u_pred[:, 0]  # u-velocity
        v_vel = u_pred[:, 1] if u_pred.shape[1] > 1 else torch.zeros_like(u_vel)  # v-velocity
        
        # Navier-Stokes equations:
        # ∂u/∂t + u ∂u/∂x + v ∂u/∂y = -1/ρ ∂p/∂x + ν(∂²u/∂x² + ∂²u/∂y²)
        # ∂v/∂t + u ∂v/∂x + v ∂v/∂y = -1/ρ ∂p/∂y + ν(∂²v/∂x² + ∂²v/∂y²)
        
        # For steady-state, ignore time derivatives
        # Convection terms
        convection_u = u_vel * u_x[:, 0] + v_vel * u_y[:, 0]
        convection_v = u_vel * u_x[:, 1] + v_vel * u_y[:, 1] if u_pred.shape[1] > 1 else torch.zeros_like(convection_u)
        
        # Viscous terms
        viscous_u = nu * (u_xx[:, 0] + u_yy[:, 0])
        viscous_v = nu * (u_xx[:, 1] + u_yy[:, 1]) if u_pred.shape[1] > 1 else torch.zeros_like(viscous_u)
        
        # Residuals (ignoring pressure gradient for now)
        residual_u = convection_u - viscous_u
        residual_v = convection_v - viscous_v
        
        # Combine residuals
        residual = torch.stack([residual_u, residual_v], dim=1)
        return residual.mean(dim=1)  # Average over velocity components
    
    def _compute_burgers_residual(self, u: torch.Tensor, x: torch.Tensor, nu: float) -> torch.Tensor:
        """Compute Burgers equation residual."""
        # Burgers: ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²
        
        x_grid = self._create_spatial_grid(x, u.shape)
        if not x_grid.requires_grad:
            x_grid = x_grid.requires_grad_(True)
        
        # Compute derivatives
        u_x, _ = self._compute_spatial_derivatives_1d(u, x_grid)
        u_xx, _ = self._compute_second_derivatives_1d(u, x_grid)
        
        # Burgers equation (steady-state)
        if u.ndim > 2:
            u_scalar = u.squeeze(1) if u.shape[1] == 1 else u.mean(dim=1)
        else:
            u_scalar = u
        
        convection = u_scalar * u_x
        diffusion = nu * u_xx
        
        residual = convection - diffusion
        return residual
    
    def _compute_spatial_derivatives_2d(self, u: torch.Tensor, x_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial derivatives using central differences."""
        # u: (batch, channels, height, width)
        # Returns u_x, u_y
        
        # Central difference approximation
        u_x = torch.zeros_like(u)
        u_y = torch.zeros_like(u)
        
        if u.shape[-1] > 2:  # width > 2
            u_x[:, :, :, 1:-1] = (u[:, :, :, 2:] - u[:, :, :, :-2]) / 2.0
            # Forward/backward differences at boundaries
            u_x[:, :, :, 0] = u[:, :, :, 1] - u[:, :, :, 0]
            u_x[:, :, :, -1] = u[:, :, :, -1] - u[:, :, :, -2]
        
        if u.shape[-2] > 2:  # height > 2
            u_y[:, :, 1:-1, :] = (u[:, :, 2:, :] - u[:, :, :-2, :]) / 2.0
            # Forward/backward differences at boundaries
            u_y[:, :, 0, :] = u[:, :, 1, :] - u[:, :, 0, :]
            u_y[:, :, -1, :] = u[:, :, -1, :] - u[:, :, -2, :]
        
        return u_x, u_y
    
    def _compute_second_derivatives_2d(self, u: torch.Tensor, x_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute second spatial derivatives."""
        # u: (batch, channels, height, width)
        
        u_xx = torch.zeros_like(u)
        u_yy = torch.zeros_like(u)
        
        if u.shape[-1] > 2:  # width > 2
            u_xx[:, :, :, 1:-1] = u[:, :, :, 2:] - 2*u[:, :, :, 1:-1] + u[:, :, :, :-2]
        
        if u.shape[-2] > 2:  # height > 2
            u_yy[:, :, 1:-1, :] = u[:, :, 2:, :] - 2*u[:, :, 1:-1, :] + u[:, :, :-2, :]
        
        return u_xx, u_yy
    
    def _compute_spatial_derivatives_1d(self, u: torch.Tensor, x_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 1D spatial derivatives."""
        u_x = torch.zeros_like(u)
        if u.shape[-1] > 2:
            u_x[..., 1:-1] = (u[..., 2:] - u[..., :-2]) / 2.0
            u_x[..., 0] = u[..., 1] - u[..., 0]
            u_x[..., -1] = u[..., -1] - u[..., -2]
        return u_x, torch.zeros_like(u_x)
    
    def _compute_second_derivatives_1d(self, u: torch.Tensor, x_grid: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 1D second derivatives."""
        u_xx = torch.zeros_like(u)
        if u.shape[-1] > 2:
            u_xx[..., 1:-1] = u[..., 2:] - 2*u[..., 1:-1] + u[..., :-2]
        return u_xx, torch.zeros_like(u_xx)
    
    def _create_spatial_grid(self, x: torch.Tensor, u_shape: torch.Size) -> torch.Tensor:
        """Create spatial grid for derivative computation."""
        # Simple grid creation - in practice would use actual spatial coordinates
        if len(u_shape) == 4:  # 2D spatial
            h, w = u_shape[-2:]
            x_coords = torch.linspace(0, 1, w, device=x.device)
            y_coords = torch.linspace(0, 1, h, device=x.device)
            grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing='xy')
            return torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        else:  # 1D spatial
            w = u_shape[-1]
            x_coords = torch.linspace(0, 1, w, device=x.device)
            return x_coords.unsqueeze(0)
    
    def __call__(self, 
                 model: nn.Module,
                 x: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Compute advanced physics-aware acquisition scores.
        
        Args:
            model: Probabilistic model
            x: Input tensor
            
        Returns:
            Combined acquisition scores with physics constraints
        """
        # Base uncertainty
        uncertainty_scores = self.base_acquisition(model, x, **kwargs)
        
        # Physics residuals with gradient computation
        with torch.enable_grad():
            x_physics = x.detach().requires_grad_(True)
            pred = model(x_physics)
            
            # Compute primary PDE residual
            residuals = self.physics_residual_fn(pred, x_physics)
            physics_scores = torch.abs(residuals).mean(dim=tuple(range(1, residuals.ndim)))
            
            # Add conservation law violations
            conservation_scores = torch.zeros_like(physics_scores)
            for conservation_fn in self.conservation_laws:
                conservation_residual = conservation_fn(pred, x_physics)
                conservation_scores += torch.abs(conservation_residual).mean(
                    dim=tuple(range(1, conservation_residual.ndim))
                )
            
            # Add boundary condition violations
            boundary_scores = torch.zeros_like(physics_scores)
            if self.boundary_conditions is not None:
                boundary_residual = self.boundary_conditions(pred, x_physics)
                boundary_scores = torch.abs(boundary_residual).mean(
                    dim=tuple(range(1, boundary_residual.ndim))
                )
        
        # Adaptive weighting based on local physics violation severity
        if self.adaptive_weighting:
            total_physics = physics_scores + conservation_scores + boundary_scores
            # Higher physics weight where violations are more severe
            adaptive_weight = self.physics_weight * (1.0 + torch.tanh(total_physics))
        else:
            adaptive_weight = self.physics_weight
            total_physics = physics_scores + conservation_scores + boundary_scores
        
        # Combine scores with adaptive weighting
        combined_scores = uncertainty_scores + adaptive_weight * total_physics
        
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
                     physics_residual_fn: Optional[Callable] = None,
                     conservation_laws: Optional[list] = None,
                     boundary_conditions: Optional[Callable] = None,
                     physics_weight: float = 0.1,
                     adaptive_weighting: bool = True,
                     pde_type: str = "general",
                     **kwargs) -> PhysicsAware:
        """Create advanced physics-aware acquisition function."""
        if base_type == "bald":
            base = AcquisitionFunctions.bald(**kwargs)
        elif base_type == "variance":
            base = AcquisitionFunctions.max_variance(**kwargs)
        elif base_type == "entropy":
            base = AcquisitionFunctions.max_entropy(**kwargs)
        else:
            raise ValueError(f"Unknown base acquisition type: {base_type}")
        
        return PhysicsAware(
            base=base,
            physics_residual_fn=physics_residual_fn,
            conservation_laws=conservation_laws,
            boundary_conditions=boundary_conditions,
            physics_weight=physics_weight,
            adaptive_weighting=adaptive_weighting,
            pde_type=pde_type
        )