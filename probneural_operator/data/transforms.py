"""Data transformation utilities for neural operators."""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import numpy as np


class DataTransform(ABC):
    """Abstract base class for data transformations."""
    
    @abstractmethod
    def fit(self, data: torch.Tensor) -> None:
        """Fit transformation parameters to data.
        
        Args:
            data: Input data tensor
        """
        pass
    
    @abstractmethod
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply transformation to data.
        
        Args:
            data: Input data tensor
            
        Returns:
            Transformed data tensor
        """
        pass
    
    @abstractmethod
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply inverse transformation to data.
        
        Args:
            data: Transformed data tensor
            
        Returns:
            Original scale data tensor
        """
        pass
    
    def fit_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Fit transformation and apply to data.
        
        Args:
            data: Input data tensor
            
        Returns:
            Transformed data tensor
        """
        self.fit(data)
        return self.transform(data)


class StandardScaler(DataTransform):
    """Standard scaler for zero mean and unit variance normalization.
    
    Transforms data to have zero mean and unit variance:
    x_scaled = (x - mean) / std
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """Initialize scaler.
        
        Args:
            epsilon: Small constant to prevent division by zero
        """
        self.epsilon = epsilon
        self.mean = None
        self.std = None
        self.fitted = False
    
    def fit(self, data: torch.Tensor) -> None:
        """Fit scaler to data.
        
        Args:
            data: Input data of shape (batch_size, ...)
        """
        # Compute statistics over batch dimension
        self.mean = data.mean(dim=0, keepdim=True)
        self.std = data.std(dim=0, keepdim=True) + self.epsilon
        self.fitted = True
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply standardization.
        
        Args:
            data: Input data tensor
            
        Returns:
            Standardized data tensor
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply inverse standardization.
        
        Args:
            data: Standardized data tensor
            
        Returns:
            Original scale data tensor
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        return data * self.std + self.mean
    
    def get_params(self) -> Dict[str, torch.Tensor]:
        """Get scaler parameters."""
        return {
            "mean": self.mean,
            "std": self.std,
            "fitted": self.fitted
        }
    
    def set_params(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Set scaler parameters manually.
        
        Args:
            mean: Mean tensor
            std: Standard deviation tensor
        """
        self.mean = mean
        self.std = std
        self.fitted = True


class MinMaxScaler(DataTransform):
    """Min-max scaler for normalization to [0, 1] range.
    
    Transforms data to [0, 1] range:
    x_scaled = (x - min) / (max - min)
    """
    
    def __init__(self, feature_range: Tuple[float, float] = (0.0, 1.0), epsilon: float = 1e-8):
        """Initialize scaler.
        
        Args:
            feature_range: Target range for scaling
            epsilon: Small constant to prevent division by zero
        """
        self.feature_range = feature_range
        self.epsilon = epsilon
        self.min_val = None
        self.max_val = None
        self.fitted = False
    
    def fit(self, data: torch.Tensor) -> None:
        """Fit scaler to data.
        
        Args:
            data: Input data of shape (batch_size, ...)
        """
        # Compute min/max over batch dimension
        self.min_val = data.min(dim=0, keepdim=True)[0]
        self.max_val = data.max(dim=0, keepdim=True)[0]
        
        # Ensure range is not zero
        range_val = self.max_val - self.min_val
        range_val = torch.where(range_val < self.epsilon, torch.ones_like(range_val), range_val)
        self.max_val = self.min_val + range_val
        
        self.fitted = True
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply min-max scaling.
        
        Args:
            data: Input data tensor
            
        Returns:
            Scaled data tensor
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        # Scale to [0, 1]
        scaled = (data - self.min_val) / (self.max_val - self.min_val)
        
        # Scale to feature range
        min_range, max_range = self.feature_range
        scaled = scaled * (max_range - min_range) + min_range
        
        return scaled
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply inverse min-max scaling.
        
        Args:
            data: Scaled data tensor
            
        Returns:
            Original scale data tensor
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        # Inverse scale from feature range
        min_range, max_range = self.feature_range
        scaled = (data - min_range) / (max_range - min_range)
        
        # Inverse scale to original range
        return scaled * (self.max_val - self.min_val) + self.min_val
    
    def get_params(self) -> Dict[str, torch.Tensor]:
        """Get scaler parameters."""
        return {
            "min_val": self.min_val,
            "max_val": self.max_val,
            "feature_range": self.feature_range,
            "fitted": self.fitted
        }


class RobustScaler(DataTransform):
    """Robust scaler using median and IQR for outlier-resistant normalization.
    
    Transforms data using median and interquartile range:
    x_scaled = (x - median) / IQR
    """
    
    def __init__(self, quantile_range: Tuple[float, float] = (25.0, 75.0), epsilon: float = 1e-8):
        """Initialize scaler.
        
        Args:
            quantile_range: Quantile range for scaling (default: IQR)
            epsilon: Small constant to prevent division by zero
        """
        self.quantile_range = quantile_range
        self.epsilon = epsilon
        self.median = None
        self.scale = None
        self.fitted = False
    
    def fit(self, data: torch.Tensor) -> None:
        """Fit scaler to data.
        
        Args:
            data: Input data of shape (batch_size, ...)
        """
        # Flatten spatial dimensions for quantile computation
        original_shape = data.shape
        data_flat = data.view(original_shape[0], -1)
        
        # Compute median
        self.median = data_flat.median(dim=0, keepdim=True)[0]
        self.median = self.median.view(1, *original_shape[1:])
        
        # Compute scale (IQR)
        q_low = torch.quantile(data_flat, self.quantile_range[0] / 100.0, dim=0, keepdim=True)
        q_high = torch.quantile(data_flat, self.quantile_range[1] / 100.0, dim=0, keepdim=True)
        
        self.scale = q_high - q_low + self.epsilon
        self.scale = self.scale.view(1, *original_shape[1:])
        
        self.fitted = True
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply robust scaling.
        
        Args:
            data: Input data tensor
            
        Returns:
            Scaled data tensor
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        return (data - self.median) / self.scale
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply inverse robust scaling.
        
        Args:
            data: Scaled data tensor
            
        Returns:
            Original scale data tensor
        """
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        return data * self.scale + self.median


class LogTransform(DataTransform):
    """Logarithmic transformation for positive data.
    
    Applies log(x + offset) transformation.
    """
    
    def __init__(self, offset: float = 1e-8):
        """Initialize transform.
        
        Args:
            offset: Small offset to ensure positivity
        """
        self.offset = offset
        self.fitted = True  # No fitting required
    
    def fit(self, data: torch.Tensor) -> None:
        """Fit transform (no-op for log transform)."""
        pass
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply log transformation.
        
        Args:
            data: Input data tensor
            
        Returns:
            Log-transformed data tensor
        """
        return torch.log(data + self.offset)
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply inverse log transformation.
        
        Args:
            data: Log-transformed data tensor
            
        Returns:
            Original scale data tensor
        """
        return torch.exp(data) - self.offset


class CompositeTransform(DataTransform):
    """Composite transformation applying multiple transforms in sequence."""
    
    def __init__(self, transforms: list):
        """Initialize composite transform.
        
        Args:
            transforms: List of transforms to apply in order
        """
        self.transforms = transforms
        self.fitted = False
    
    def fit(self, data: torch.Tensor) -> None:
        """Fit all transforms sequentially.
        
        Args:
            data: Input data tensor
        """
        current_data = data
        
        for transform in self.transforms:
            transform.fit(current_data)
            current_data = transform.transform(current_data)
        
        self.fitted = True
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply all transforms in sequence.
        
        Args:
            data: Input data tensor
            
        Returns:
            Transformed data tensor
        """
        current_data = data
        
        for transform in self.transforms:
            current_data = transform.transform(current_data)
        
        return current_data
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply inverse transforms in reverse order.
        
        Args:
            data: Transformed data tensor
            
        Returns:
            Original scale data tensor
        """
        current_data = data
        
        # Apply inverse transforms in reverse order
        for transform in reversed(self.transforms):
            current_data = transform.inverse_transform(current_data)
        
        return current_data


class SpatialStandardScaler(DataTransform):
    """Spatial-aware standardization that preserves spatial structure.
    
    Computes statistics spatially rather than across all dimensions.
    """
    
    def __init__(self, spatial_dims: Tuple[int, ...] = (-2, -1), epsilon: float = 1e-8):
        """Initialize scaler.
        
        Args:
            spatial_dims: Dimensions to treat as spatial
            epsilon: Small constant to prevent division by zero
        """
        self.spatial_dims = spatial_dims
        self.epsilon = epsilon
        self.mean = None
        self.std = None
        self.fitted = False
    
    def fit(self, data: torch.Tensor) -> None:
        """Fit scaler preserving spatial structure.
        
        Args:
            data: Input data of shape (batch_size, channels, *spatial_dims)
        """
        # Compute statistics over batch and spatial dimensions
        reduce_dims = (0,) + tuple(range(len(data.shape)))[2:]  # batch + spatial
        
        self.mean = data.mean(dim=reduce_dims, keepdim=True)
        self.std = data.std(dim=reduce_dims, keepdim=True) + self.epsilon
        self.fitted = True
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply spatial standardization."""
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Apply inverse spatial standardization."""
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")
        
        return data * self.std + self.mean