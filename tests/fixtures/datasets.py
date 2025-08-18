"""Test fixtures for dataset generation and validation."""

import torch
import numpy as np
import pytest
from typing import Tuple, Dict, Any
from pathlib import Path


class SyntheticPDEData:
    """Generate synthetic PDE data for testing."""
    
    @staticmethod
    def burgers_equation(nx: int = 64, nt: int = 50, viscosity: float = 0.01) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic Burgers equation data."""
        x = np.linspace(0, 2*np.pi, nx, endpoint=False)
        t = np.linspace(0, 1, nt)
        
        # Initial condition: smooth random field
        u0 = np.sin(x) + 0.5 * np.sin(2*x) + 0.1 * np.random.randn(nx)
        
        # Simple finite difference solution (for testing only)
        dx = x[1] - x[0]
        dt = t[1] - t[0]
        
        u = np.zeros((nt, nx))
        u[0] = u0
        
        for i in range(1, nt):
            u_prev = u[i-1]
            # Upwind scheme
            dudt = np.zeros_like(u_prev)
            for j in range(nx):
                jp1 = (j + 1) % nx
                jm1 = (j - 1) % nx
                
                # Convection term (backward difference)
                dudt[j] = -u_prev[j] * (u_prev[j] - u_prev[jm1]) / dx
                
                # Viscosity term (central difference)  
                dudt[j] += viscosity * (u_prev[jp1] - 2*u_prev[j] + u_prev[jm1]) / (dx**2)
            
            u[i] = u_prev + dt * dudt
        
        # Return initial conditions and solutions
        initial = torch.tensor(u0, dtype=torch.float32).unsqueeze(0)  # (1, nx)
        solution = torch.tensor(u, dtype=torch.float32)  # (nt, nx)
        
        return initial, solution

    @staticmethod
    def darcy_flow(nx: int = 64, ny: int = 64, n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic Darcy flow data."""
        np.random.seed(42)
        
        # Generate random permeability fields
        permeability = []
        pressure = []
        
        for _ in range(n_samples):
            # Random permeability field
            k = np.random.lognormal(0, 1, (nx, ny))
            k = np.exp(k - np.mean(k))  # Normalize
            
            # Simple pressure solver (Laplace equation)
            p = np.random.randn(nx, ny)  # Placeholder solution
            
            permeability.append(k)
            pressure.append(p)
        
        permeability = torch.tensor(np.array(permeability), dtype=torch.float32)
        pressure = torch.tensor(np.array(pressure), dtype=torch.float32)
        
        return permeability, pressure

    @staticmethod
    def heat_equation(nx: int = 32, nt: int = 20, diffusivity: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic heat equation data."""
        x = np.linspace(0, 1, nx)
        t = np.linspace(0, 0.1, nt)
        
        # Initial condition: Gaussian
        u0 = np.exp(-(x - 0.5)**2 / 0.01)
        
        # Analytical solution for heat equation
        u = np.zeros((nt, nx))
        u[0] = u0
        
        for i, t_val in enumerate(t[1:], 1):
            # Analytical solution (simplified)
            u[i] = np.exp(-(x - 0.5)**2 / (0.01 + 2*diffusivity*t_val))
            u[i] = u[i] / np.sqrt(1 + 2*diffusivity*t_val/0.01)
        
        initial = torch.tensor(u0, dtype=torch.float32).unsqueeze(0)
        solution = torch.tensor(u, dtype=torch.float32)
        
        return initial, solution


@pytest.fixture
def synthetic_pde_data():
    """Fixture providing synthetic PDE datasets."""
    return SyntheticPDEData()


@pytest.fixture
def burgers_data(synthetic_pde_data):
    """Generate Burgers equation test data."""
    return synthetic_pde_data.burgers_equation()


@pytest.fixture
def darcy_data(synthetic_pde_data):
    """Generate Darcy flow test data."""
    return synthetic_pde_data.darcy_flow()


@pytest.fixture
def heat_data(synthetic_pde_data):
    """Generate heat equation test data."""
    return synthetic_pde_data.heat_equation()


@pytest.fixture
def multiscale_data():
    """Generate multi-scale test data."""
    # Different resolutions
    resolutions = [16, 32, 64]
    data = {}
    
    for res in resolutions:
        x = np.linspace(0, 1, res)
        y = np.sin(2*np.pi*x) + 0.5*np.sin(4*np.pi*x)
        data[f"res_{res}"] = torch.tensor(y, dtype=torch.float32)
    
    return data


@pytest.fixture  
def noisy_data():
    """Generate data with different noise levels."""
    x = np.linspace(0, 1, 100)
    clean_signal = np.sin(2*np.pi*x)
    
    noise_levels = [0.0, 0.1, 0.5, 1.0]
    data = {}
    
    for noise_level in noise_levels:
        noise = noise_level * np.random.randn(100)
        noisy_signal = clean_signal + noise
        data[f"noise_{noise_level}"] = {
            "clean": torch.tensor(clean_signal, dtype=torch.float32),
            "noisy": torch.tensor(noisy_signal, dtype=torch.float32)
        }
    
    return data


@pytest.fixture
def temporal_data():
    """Generate time series data for testing."""
    time_steps = 100
    n_features = 5
    
    # Generate sinusoidal time series with different frequencies
    t = np.linspace(0, 4*np.pi, time_steps)
    data = np.zeros((time_steps, n_features))
    
    for i in range(n_features):
        freq = (i + 1) * 0.5
        phase = i * np.pi / 4
        data[:, i] = np.sin(freq * t + phase) + 0.1 * np.random.randn(time_steps)
    
    return torch.tensor(data, dtype=torch.float32)


@pytest.fixture
def validation_metrics():
    """Common validation metrics for testing."""
    def mse(pred, target):
        return torch.mean((pred - target) ** 2)
    
    def mae(pred, target):
        return torch.mean(torch.abs(pred - target))
    
    def relative_error(pred, target):
        return torch.mean(torch.abs(pred - target) / (torch.abs(target) + 1e-8))
    
    return {
        "mse": mse,
        "mae": mae,
        "relative_error": relative_error
    }