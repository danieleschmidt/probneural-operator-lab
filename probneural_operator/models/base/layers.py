"""Common neural operator layers and building blocks."""

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralLayer(nn.Module):
    """Spectral convolution layer for Fourier Neural Operators.
    
    This layer performs convolution in the Fourier domain by:
    1. FFT of the input
    2. Pointwise multiplication with learnable spectral weights
    3. Inverse FFT to get output
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int, 
                 modes: int,
                 spatial_dim: int = 2):
        """Initialize spectral layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels  
            modes: Number of Fourier modes to keep
            spatial_dim: Spatial dimension (1D, 2D, or 3D)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.spatial_dim = spatial_dim
        
        # Initialize spectral weights
        scale = 1 / (in_channels * out_channels)
        if spatial_dim == 1:
            self.weights = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, self.modes, dtype=torch.cfloat)
            )
        elif spatial_dim == 2:
            self.weights = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, self.modes, self.modes, dtype=torch.cfloat)
            )
        elif spatial_dim == 3:
            self.weights = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, self.modes, self.modes, self.modes, dtype=torch.cfloat)
            )
        else:
            raise ValueError(f"Unsupported spatial dimension: {spatial_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spectral layer.
        
        Args:
            x: Input tensor of shape (batch, channels, *spatial_dims)
            
        Returns:
            Output tensor of same spatial shape but potentially different channels
        """
        batch_size = x.shape[0]
        
        if self.spatial_dim == 1:
            return self._forward_1d(x)
        elif self.spatial_dim == 2:
            return self._forward_2d(x)
        elif self.spatial_dim == 3:
            return self._forward_3d(x)
    
    def _forward_1d(self, x: torch.Tensor) -> torch.Tensor:
        """1D spectral convolution."""
        # FFT
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # Extract relevant modes
        x_ft = x_ft[:, :, :self.modes]
        
        # Spectral multiplication
        out_ft = torch.einsum("bix,iox->box", x_ft, self.weights)
        
        # Pad back to original size
        out_ft = F.pad(out_ft, (0, x.shape[-1]//2 + 1 - self.modes))
        
        # Inverse FFT
        return torch.fft.irfft(out_ft, n=x.shape[-1], dim=-1)
    
    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        """2D spectral convolution."""
        # FFT
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))
        
        # Extract relevant modes
        x_ft = x_ft[:, :, :self.modes, :self.modes]
        
        # Spectral multiplication
        out_ft = torch.einsum("bixy,ioxy->boxy", x_ft, self.weights)
        
        # Pad back to original size
        out_ft = F.pad(out_ft, (0, x.shape[-1]//2 + 1 - self.modes, 0, x.shape[-2] - self.modes))
        
        # Inverse FFT
        return torch.fft.irfft2(out_ft, s=x.shape[-2:], dim=(-2, -1))
    
    def _forward_3d(self, x: torch.Tensor) -> torch.Tensor:
        """3D spectral convolution."""
        # FFT
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        
        # Extract relevant modes
        x_ft = x_ft[:, :, :self.modes, :self.modes, :self.modes]
        
        # Spectral multiplication  
        out_ft = torch.einsum("bixyz,ioxyz->boxyz", x_ft, self.weights)
        
        # Pad back to original size
        out_ft = F.pad(out_ft, (
            0, x.shape[-1]//2 + 1 - self.modes,
            0, x.shape[-2] - self.modes,
            0, x.shape[-3] - self.modes
        ))
        
        # Inverse FFT
        return torch.fft.irfftn(out_ft, s=x.shape[-3:], dim=(-3, -2, -1))


class FeedForwardLayer(nn.Module):
    """Feed-forward layer with activation and optional dropout."""
    
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: str = "gelu",
                 dropout: float = 0.0):
        """Initialize feed-forward layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            activation: Activation function name
            dropout: Dropout probability
        """
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.dropout(self.activation(self.linear(x)))


class LiftProjectLayer(nn.Module):
    """Lifting and projection layers for neural operators.
    
    Lifting: Maps input functions to higher-dimensional representation
    Projection: Maps back to output space
    """
    
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int = 256):
        """Initialize lift/project layer.
        
        Args:
            in_dim: Input dimension
            out_dim: Output dimension  
            hidden_dim: Hidden dimension for intermediate representation
        """
        super().__init__()
        self.lift = nn.Linear(in_dim, hidden_dim)
        self.project = nn.Linear(hidden_dim, out_dim)
    
    def lift_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lift input to higher dimension."""
        return self.lift(x)
    
    def project_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project back to output dimension."""
        return self.project(x)


class PositionalEncoding(nn.Module):
    """Positional encoding for coordinate inputs."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:x.size(-2)]


class ResidualBlock(nn.Module):
    """Residual block for neural operators."""
    
    def __init__(self,
                 channels: int,
                 activation: str = "gelu",
                 norm: bool = True):
        """Initialize residual block.
        
        Args:
            channels: Number of channels
            activation: Activation function
            norm: Whether to use layer normalization
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(channels) if norm else nn.Identity()
        self.linear1 = nn.Linear(channels, channels)
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()
        
        self.norm2 = nn.LayerNorm(channels) if norm else nn.Identity()
        self.linear2 = nn.Linear(channels, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = x
        x = self.linear1(self.activation(self.norm1(x)))
        x = self.linear2(self.activation(self.norm2(x)))
        return x + residual