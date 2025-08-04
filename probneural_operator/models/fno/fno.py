"""Fourier Neural Operator implementation with uncertainty quantification."""

from typing import Optional, List

import torch
import torch.nn as nn

from ..base import NeuralOperator, ProbabilisticNeuralOperator
from ..base.layers import SpectralLayer, FeedForwardLayer, LiftProjectLayer


class FourierNeuralOperator(NeuralOperator):
    """Fourier Neural Operator for learning operators on function spaces.
    
    FNO uses spectral convolutions in Fourier space to learn mappings between
    infinite-dimensional function spaces. It's particularly effective for PDEs
    and can generalize across different discretizations.
    
    References:
        Li et al. "Fourier Neural Operator for Parametric Partial Differential Equations"
        ICLR 2021.
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 modes: int = 12,
                 width: int = 64,
                 depth: int = 4,
                 activation: str = "gelu",
                 spatial_dim: int = 2,
                 **kwargs):
        """Initialize FNO.
        
        Args:
            input_dim: Input function dimension
            output_dim: Output function dimension  
            modes: Number of Fourier modes to keep
            width: Hidden dimension
            depth: Number of spectral layers
            activation: Activation function
            spatial_dim: Spatial dimension (1D, 2D, or 3D)
        """
        super().__init__(input_dim, output_dim, **kwargs)
        
        self.modes = modes
        self.width = width
        self.depth = depth
        self.spatial_dim = spatial_dim
        
        # Lifting layer: map input to higher dimension
        self.lift = nn.Linear(input_dim, width)
        
        # Spectral layers
        self.spectral_layers = nn.ModuleList([
            SpectralLayer(width, width, modes, spatial_dim)
            for _ in range(depth)
        ])
        
        # Local (pointwise) layers
        self.local_layers = nn.ModuleList([
            FeedForwardLayer(width, width, activation)
            for _ in range(depth)
        ])
        
        # Projection layer: map back to output dimension
        self.project = nn.Sequential(
            nn.Linear(width, width // 2),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Linear(width // 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FNO.
        
        Args:
            x: Input tensor of shape (batch, input_dim, *spatial_dims)
            
        Returns:
            Output tensor of shape (batch, output_dim, *spatial_dims)
        """
        # Move channel dimension to last for linear layers
        x = x.permute(0, *range(2, x.ndim), 1)  # (batch, *spatial, channels)
        
        # Lifting
        x = self.lift(x)
        
        # Spectral + local layers
        for spectral, local in zip(self.spectral_layers, self.local_layers):
            # Move channels back to second position for spectral layer
            x_spectral = x.permute(0, -1, *range(1, x.ndim-1))  # (batch, channels, *spatial)
            x_spectral = spectral(x_spectral)
            
            # Move channels back to last position
            x_spectral = x_spectral.permute(0, *range(2, x_spectral.ndim), 1)
            
            # Local transformation
            x_local = local(x)
            
            # Residual connection
            x = x_spectral + x_local
        
        # Projection
        x = self.project(x)
        
        # Move channels back to second position
        return x.permute(0, -1, *range(1, x.ndim-1))


class ProbabilisticFNO(ProbabilisticNeuralOperator):
    """Probabilistic Fourier Neural Operator with uncertainty quantification.
    
    Extends FNO with uncertainty quantification through posterior approximation
    methods like linearized Laplace approximation.
    """
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 modes: int = 12,
                 width: int = 64,
                 depth: int = 4,
                 activation: str = "gelu",
                 spatial_dim: int = 2,
                 posterior_type: str = "laplace",
                 prior_precision: float = 1.0,
                 **kwargs):
        """Initialize Probabilistic FNO.
        
        Args:
            input_dim: Input function dimension
            output_dim: Output function dimension
            modes: Number of Fourier modes to keep
            width: Hidden dimension
            depth: Number of spectral layers
            activation: Activation function
            spatial_dim: Spatial dimension (1D, 2D, or 3D)
            posterior_type: Type of posterior approximation
            prior_precision: Prior precision for Bayesian inference
        """
        super().__init__(
            input_dim, output_dim, 
            posterior_type=posterior_type,
            prior_precision=prior_precision,
            **kwargs
        )
        
        self.modes = modes
        self.width = width
        self.depth = depth
        self.spatial_dim = spatial_dim
        
        # Create the underlying FNO architecture
        self._build_architecture(activation)
    
    def _build_architecture(self, activation: str):
        """Build the FNO architecture."""
        # Lifting layer
        self.lift = nn.Linear(self.input_dim, self.width)
        
        # Spectral layers
        self.spectral_layers = nn.ModuleList([
            SpectralLayer(self.width, self.width, self.modes, self.spatial_dim)
            for _ in range(self.depth)
        ])
        
        # Local (pointwise) layers
        self.local_layers = nn.ModuleList([
            FeedForwardLayer(self.width, self.width, activation)
            for _ in range(self.depth)
        ])
        
        # Projection layer
        self.project = nn.Sequential(
            nn.Linear(self.width, self.width // 2),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Linear(self.width // 2, self.output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through Probabilistic FNO."""
        # Same forward pass as regular FNO
        x = x.permute(0, *range(2, x.ndim), 1)
        x = self.lift(x)
        
        for spectral, local in zip(self.spectral_layers, self.local_layers):
            x_spectral = x.permute(0, -1, *range(1, x.ndim-1))
            x_spectral = spectral(x_spectral)
            x_spectral = x_spectral.permute(0, *range(2, x_spectral.ndim), 1)
            x_local = local(x)
            x = x_spectral + x_local
        
        x = self.project(x)
        return x.permute(0, -1, *range(1, x.ndim-1))
    
    def fit(self, 
            train_loader,
            val_loader = None,
            epochs: int = 100,
            lr: float = 1e-3,
            device: str = "auto"):
        """Override fit method to handle tensor reshaping for FNO."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        
        history = {"train_loss": [], "val_loss": []}
        
        for epoch in range(epochs):
            # Training
            self.train()
            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                # Reshape data for FNO: add channel dimension
                if data.ndim == 2:  # 1D spatial case: (batch, spatial) -> (batch, 1, spatial)
                    data = data.unsqueeze(1)
                elif data.ndim == 3:  # 2D spatial case: (batch, height, width) -> (batch, 1, height, width)
                    data = data.unsqueeze(1)
                
                # Use last time step as target for time-series data
                if target.ndim > data.ndim - 1:
                    target = target[:, -1]  # Take final time step
                
                optimizer.zero_grad()
                output = self(data)
                
                # Remove channel dimension from output for loss computation
                if output.ndim > target.ndim:
                    output = output.squeeze(1)
                
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            if len(train_loader) > 0:
                train_loss /= len(train_loader)
            history["train_loss"].append(train_loss)
            
            # Validation
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(device), target.to(device)
                        
                        # Reshape data for FNO
                        if data.ndim == 2:
                            data = data.unsqueeze(1)
                        elif data.ndim == 3:
                            data = data.unsqueeze(1)
                        
                        # Use last time step as target
                        if target.ndim > data.ndim - 1:
                            target = target[:, -1]
                        
                        output = self(data)
                        
                        # Remove channel dimension from output
                        if output.ndim > target.ndim:
                            output = output.squeeze(1)
                        
                        val_loss += criterion(output, target).item()
                
                if len(val_loader) > 0:
                    val_loss /= len(val_loader)
                history["val_loss"].append(val_loss)
                
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}: Train Loss: {train_loss:.6f}")
        
        return history
    
    @classmethod
    def from_config(cls, config: dict) -> "ProbabilisticFNO":
        """Create ProbabilisticFNO from configuration dictionary.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configured ProbabilisticFNO instance
        """
        return cls(**config)


# Factory function for easy instantiation
def create_fno(config: dict) -> FourierNeuralOperator:
    """Create FNO from configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Configured FNO model
    """
    if config.get("probabilistic", False):
        return ProbabilisticFNO.from_config(config)
    else:
        return FourierNeuralOperator(**config)