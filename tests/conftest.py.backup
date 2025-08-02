"""Pytest configuration and fixtures."""

import pytest
import torch
import numpy as np


@pytest.fixture
def device():
    """Get computation device (CPU/GPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_data_1d():
    """Generate sample 1D data for testing."""
    x = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(100)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


@pytest.fixture
def sample_data_2d():
    """Generate sample 2D data for testing."""
    x = np.random.randn(100, 2)
    y = np.sum(x**2, axis=1) + 0.1 * np.random.randn(100)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


@pytest.fixture
def small_fno_config():
    """Small FNO configuration for testing."""
    return {
        "modes": 4,
        "width": 8,
        "depth": 2,
        "input_dim": 1,
        "output_dim": 1
    }