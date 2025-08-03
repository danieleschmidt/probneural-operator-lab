"""Dataset classes for neural operator training."""

import os
import h5py
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any, List, Union
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import Dataset


class PDEDataset(Dataset, ABC):
    """Abstract base class for PDE datasets.
    
    This defines the common interface for all PDE datasets used in neural
    operator training. Each dataset should provide function pairs (input, output)
    that represent solutions to parametric PDEs.
    """
    
    def __init__(self,
                 data_path: Union[str, Path],
                 split: str = "train",
                 normalize: bool = True,
                 cache_data: bool = True):
        """Initialize PDE dataset.
        
        Args:
            data_path: Path to dataset file or directory
            split: Dataset split ("train", "val", "test")
            normalize: Whether to normalize the data
            cache_data: Whether to cache data in memory
        """
        self.data_path = Path(data_path)
        self.split = split
        self.normalize = normalize
        self.cache_data = cache_data
        
        # Data storage
        self.inputs = None
        self.outputs = None
        self.metadata = {}
        
        # Normalization statistics
        self.input_mean = None
        self.input_std = None
        self.output_mean = None
        self.output_std = None
        
        # Load data
        self._load_data()
        
        if self.normalize:
            self._compute_normalization_stats()
    
    @abstractmethod
    def _load_data(self) -> None:
        """Load dataset from storage. Must be implemented by subclasses."""
        pass
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.inputs) if self.inputs is not None else 0
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single data sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (input, output) tensors
        """
        if isinstance(idx, slice):
            # Handle slice indexing
            inputs = self.inputs[idx]
            outputs = self.outputs[idx]
        else:
            # Handle single index
            inputs = self.inputs[idx:idx+1]
            outputs = self.outputs[idx:idx+1]
        
        # Convert to tensors
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.from_numpy(inputs).float()
        if not isinstance(outputs, torch.Tensor):
            outputs = torch.from_numpy(outputs).float()
        
        # Apply normalization
        if self.normalize:
            inputs = self._normalize_input(inputs)
            outputs = self._normalize_output(outputs)
        
        return inputs.squeeze(0), outputs.squeeze(0)
    
    def _compute_normalization_stats(self) -> None:
        """Compute normalization statistics."""
        if self.inputs is not None and self.outputs is not None:
            # Convert to tensors if needed
            inputs = self.inputs if isinstance(self.inputs, torch.Tensor) else torch.from_numpy(self.inputs)
            outputs = self.outputs if isinstance(self.outputs, torch.Tensor) else torch.from_numpy(self.outputs)
            
            # Compute statistics over batch dimension
            self.input_mean = inputs.mean(dim=0, keepdim=True)
            self.input_std = inputs.std(dim=0, keepdim=True) + 1e-8
            self.output_mean = outputs.mean(dim=0, keepdim=True)
            self.output_std = outputs.std(dim=0, keepdim=True) + 1e-8
    
    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input tensor."""
        if self.input_mean is not None and self.input_std is not None:
            return (x - self.input_mean) / self.input_std
        return x
    
    def _normalize_output(self, y: torch.Tensor) -> torch.Tensor:
        """Normalize output tensor."""
        if self.output_mean is not None and self.output_std is not None:
            return (y - self.output_mean) / self.output_std
        return y
    
    def denormalize_output(self, y: torch.Tensor) -> torch.Tensor:
        """Denormalize output tensor."""
        if self.output_mean is not None and self.output_std is not None:
            return y * self.output_std + self.output_mean
        return y
    
    def get_grid(self) -> Optional[torch.Tensor]:
        """Get spatial grid for the dataset."""
        return self.metadata.get('grid', None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            'size': len(self),
            'input_shape': self.inputs.shape[1:] if self.inputs is not None else None,
            'output_shape': self.outputs.shape[1:] if self.outputs is not None else None,
            'input_mean': self.input_mean,
            'input_std': self.input_std,
            'output_mean': self.output_mean,
            'output_std': self.output_std,
            'metadata': self.metadata
        }


class NavierStokesDataset(PDEDataset):
    """Dataset for 2D Navier-Stokes equation.
    
    This dataset contains solutions to the 2D Navier-Stokes equation:
    ∂u/∂t + (u·∇)u = -∇p + ν∇²u + f
    
    where u is velocity, p is pressure, ν is viscosity, and f is forcing.
    """
    
    def __init__(self,
                 data_path: Union[str, Path],
                 split: str = "train",
                 resolution: int = 64,
                 time_steps: int = 50,
                 viscosity: float = 1e-3,
                 **kwargs):
        """Initialize Navier-Stokes dataset.
        
        Args:
            data_path: Path to dataset file
            split: Dataset split
            resolution: Spatial resolution  
            time_steps: Number of time steps
            viscosity: Fluid viscosity
        """
        self.resolution = resolution
        self.time_steps = time_steps
        self.viscosity = viscosity
        
        super().__init__(data_path, split, **kwargs)
    
    def _load_data(self) -> None:
        """Load Navier-Stokes data from HDF5 file."""
        if not self.data_path.exists():
            print(f"Data file not found: {self.data_path}")
            print("Generating synthetic Navier-Stokes data...")
            self._generate_synthetic_data()
            return
        
        try:
            with h5py.File(self.data_path, 'r') as f:
                # Load inputs (initial conditions)
                self.inputs = np.array(f[f'{self.split}/inputs'])
                
                # Load outputs (time evolution) 
                self.outputs = np.array(f[f'{self.split}/outputs'])
                
                # Load metadata
                if 'metadata' in f:
                    self.metadata = dict(f['metadata'].attrs)
                    
                # Create spatial grid
                x = np.linspace(0, 1, self.resolution)
                y = np.linspace(0, 1, self.resolution)
                xx, yy = np.meshgrid(x, y)
                self.metadata['grid'] = torch.from_numpy(np.stack([xx, yy], axis=0)).float()
                
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Generating synthetic data instead...")
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> None:
        """Generate synthetic Navier-Stokes data."""
        from .generators import SyntheticPDEGenerator
        
        generator = SyntheticPDEGenerator()
        
        # Generate different amounts for different splits
        n_samples = {"train": 1000, "val": 200, "test": 200}[self.split]
        
        print(f"Generating {n_samples} synthetic Navier-Stokes samples...")
        
        inputs, outputs = generator.generate_navier_stokes(
            n_samples=n_samples,
            resolution=self.resolution,
            time_steps=self.time_steps,
            viscosity=self.viscosity
        )
        
        self.inputs = inputs
        self.outputs = outputs
        
        # Create spatial grid
        x = np.linspace(0, 1, self.resolution)
        y = np.linspace(0, 1, self.resolution)
        xx, yy = np.meshgrid(x, y)
        self.metadata['grid'] = torch.from_numpy(np.stack([xx, yy], axis=0)).float()
        self.metadata['resolution'] = self.resolution
        self.metadata['time_steps'] = self.time_steps
        self.metadata['viscosity'] = self.viscosity


class BurgersDataset(PDEDataset):
    """Dataset for 1D Burgers' equation.
    
    This dataset contains solutions to the viscous Burgers' equation:
    ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
    """
    
    def __init__(self,
                 data_path: Union[str, Path],
                 split: str = "train", 
                 resolution: int = 256,
                 time_steps: int = 100,
                 viscosity: float = 0.01,
                 **kwargs):
        """Initialize Burgers dataset."""
        self.resolution = resolution
        self.time_steps = time_steps
        self.viscosity = viscosity
        
        super().__init__(data_path, split, **kwargs)
    
    def _load_data(self) -> None:
        """Load or generate Burgers data."""
        if not self.data_path.exists():
            self._generate_synthetic_data()
            return
        
        try:
            with h5py.File(self.data_path, 'r') as f:
                self.inputs = np.array(f[f'{self.split}/inputs'])
                self.outputs = np.array(f[f'{self.split}/outputs'])
                
                if 'metadata' in f:
                    self.metadata = dict(f['metadata'].attrs)
                    
        except Exception as e:
            print(f"Error loading data: {e}")
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> None:
        """Generate synthetic Burgers data."""
        from .generators import SyntheticPDEGenerator
        
        generator = SyntheticPDEGenerator()
        n_samples = {"train": 1000, "val": 200, "test": 200}[self.split]
        
        print(f"Generating {n_samples} synthetic Burgers samples...")
        
        inputs, outputs = generator.generate_burgers(
            n_samples=n_samples,
            resolution=self.resolution,
            time_steps=self.time_steps,
            viscosity=self.viscosity
        )
        
        self.inputs = inputs
        self.outputs = outputs
        
        # Create spatial grid
        x = np.linspace(0, 1, self.resolution)
        self.metadata['grid'] = torch.from_numpy(x).float()
        self.metadata['resolution'] = self.resolution
        self.metadata['time_steps'] = self.time_steps
        self.metadata['viscosity'] = self.viscosity


class DarcyFlowDataset(PDEDataset):
    """Dataset for Darcy flow equation.
    
    This dataset contains solutions to the Darcy flow equation:
    -∇·(κ(x)∇u) = f(x)
    
    where κ is permeability, u is pressure, and f is source term.
    """
    
    def __init__(self,
                 data_path: Union[str, Path],
                 split: str = "train",
                 resolution: int = 64,
                 **kwargs):
        """Initialize Darcy flow dataset."""
        self.resolution = resolution
        
        super().__init__(data_path, split, **kwargs)
    
    def _load_data(self) -> None:
        """Load or generate Darcy flow data."""
        if not self.data_path.exists():
            self._generate_synthetic_data()
            return
        
        try:
            with h5py.File(self.data_path, 'r') as f:
                # Inputs are permeability fields
                self.inputs = np.array(f[f'{self.split}/inputs'])
                
                # Outputs are pressure solutions
                self.outputs = np.array(f[f'{self.split}/outputs'])
                
                if 'metadata' in f:
                    self.metadata = dict(f['metadata'].attrs)
                    
        except Exception as e:
            print(f"Error loading data: {e}")
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> None:
        """Generate synthetic Darcy flow data."""
        from .generators import SyntheticPDEGenerator
        
        generator = SyntheticPDEGenerator()
        n_samples = {"train": 1000, "val": 200, "test": 200}[self.split]
        
        print(f"Generating {n_samples} synthetic Darcy flow samples...")
        
        inputs, outputs = generator.generate_darcy_flow(
            n_samples=n_samples,
            resolution=self.resolution
        )
        
        self.inputs = inputs
        self.outputs = outputs
        
        # Create spatial grid
        x = np.linspace(0, 1, self.resolution)
        y = np.linspace(0, 1, self.resolution)
        xx, yy = np.meshgrid(x, y)
        self.metadata['grid'] = torch.from_numpy(np.stack([xx, yy], axis=0)).float()
        self.metadata['resolution'] = self.resolution


class HeatEquationDataset(PDEDataset):
    """Dataset for heat equation.
    
    This dataset contains solutions to the heat equation:
    ∂u/∂t = α∇²u + f
    
    where α is thermal diffusivity and f is heat source.
    """
    
    def __init__(self,
                 data_path: Union[str, Path],
                 split: str = "train",
                 resolution: int = 64,
                 time_steps: int = 50,
                 diffusivity: float = 0.01,
                 **kwargs):
        """Initialize heat equation dataset."""
        self.resolution = resolution
        self.time_steps = time_steps
        self.diffusivity = diffusivity
        
        super().__init__(data_path, split, **kwargs)
    
    def _load_data(self) -> None:
        """Load or generate heat equation data."""
        if not self.data_path.exists():
            self._generate_synthetic_data()
            return
        
        try:
            with h5py.File(self.data_path, 'r') as f:
                self.inputs = np.array(f[f'{self.split}/inputs'])
                self.outputs = np.array(f[f'{self.split}/outputs'])
                
                if 'metadata' in f:
                    self.metadata = dict(f['metadata'].attrs)
                    
        except Exception as e:
            print(f"Error loading data: {e}")
            self._generate_synthetic_data()
    
    def _generate_synthetic_data(self) -> None:
        """Generate synthetic heat equation data."""
        from .generators import SyntheticPDEGenerator
        
        generator = SyntheticPDEGenerator()
        n_samples = {"train": 1000, "val": 200, "test": 200}[self.split]
        
        print(f"Generating {n_samples} synthetic heat equation samples...")
        
        inputs, outputs = generator.generate_heat_equation(
            n_samples=n_samples,
            resolution=self.resolution,
            time_steps=self.time_steps,
            diffusivity=self.diffusivity
        )
        
        self.inputs = inputs
        self.outputs = outputs
        
        # Create spatial grid
        x = np.linspace(0, 1, self.resolution)
        y = np.linspace(0, 1, self.resolution)
        xx, yy = np.meshgrid(x, y)
        self.metadata['grid'] = torch.from_numpy(np.stack([xx, yy], axis=0)).float()
        self.metadata['resolution'] = self.resolution
        self.metadata['time_steps'] = self.time_steps
        self.metadata['diffusivity'] = self.diffusivity


# Convenient aliases
NavierStokes = NavierStokesDataset
Burgers = BurgersDataset
DarcyFlow = DarcyFlowDataset
HeatEquation = HeatEquationDataset