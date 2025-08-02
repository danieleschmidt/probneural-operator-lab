"""Pytest configuration and fixtures for ProbNeural Operator Lab."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


# Test configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "benchmark: marks tests as benchmarks")


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests based on their location and requirements."""
    for item in items:
        # Mark slow tests
        if "slow" in item.keywords:
            item.add_marker(pytest.mark.slow)
        
        # Mark GPU tests
        if "gpu" in str(item.fspath) or "cuda" in str(item.name).lower():
            item.add_marker(pytest.mark.gpu)
            
        # Mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)


# Device and computation fixtures
@pytest.fixture(scope="session")
def device():
    """Get computation device (CPU/GPU)."""
    if torch.cuda.is_available() and not os.getenv("SKIP_GPU_TESTS"):
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def cpu_device():
    """Force CPU device for tests that need to run on CPU."""
    return torch.device("cpu")


@pytest.fixture(scope="session")
def dtype():
    """Default floating point dtype for tests."""
    return torch.float32


# Random seed fixtures
@pytest.fixture(autouse=True)
def random_seed():
    """Set random seeds for reproducible tests."""
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Temporary directory fixtures
@pytest.fixture
def test_data_dir(tmp_path):
    """Create temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


@pytest.fixture
def model_checkpoint_dir(tmp_path):
    """Create temporary directory for model checkpoints."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    return checkpoint_dir


# Data generation fixtures
@pytest.fixture
def sample_data_1d(dtype, device):
    """Generate sample 1D data for testing."""
    np.random.seed(42)  # Ensure reproducibility
    x = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * x) + 0.01 * np.random.randn(100)  # Reduced noise for stable tests
    return (
        torch.tensor(x, dtype=dtype, device=device),
        torch.tensor(y, dtype=dtype, device=device)
    )


@pytest.fixture
def sample_data_2d(dtype, device):
    """Generate sample 2D data for testing."""
    np.random.seed(42)
    x = np.random.randn(100, 2)
    y = np.sum(x**2, axis=1) + 0.01 * np.random.randn(100)
    return (
        torch.tensor(x, dtype=dtype, device=device),
        torch.tensor(y, dtype=dtype, device=device)
    )


@pytest.fixture
def burgers_data(dtype, device):
    """Generate synthetic Burgers equation data."""
    # Simplified Burgers equation data for testing
    nx, nt = 64, 50
    x = torch.linspace(0, 1, nx, dtype=dtype, device=device)
    t = torch.linspace(0, 1, nt, dtype=dtype, device=device)
    
    # Initial condition: Gaussian pulse
    u0 = torch.exp(-100 * (x - 0.5)**2)
    
    # Simple time evolution (not exact Burgers, but good for testing)
    u = torch.zeros(nt, nx, dtype=dtype, device=device)
    u[0] = u0
    for i in range(1, nt):
        u[i] = u[i-1] * 0.99  # Simple decay
    
    return x, t, u


@pytest.fixture
def sample_pde_data(dtype, device):
    """Generate sample PDE data with known solution."""
    # Heat equation: u_t = u_xx with initial condition u(x,0) = sin(pi*x)
    nx, nt = 32, 20
    L = 1.0
    T = 0.1
    
    x = torch.linspace(0, L, nx, dtype=dtype, device=device)
    t = torch.linspace(0, T, nt, dtype=dtype, device=device)
    
    # Analytical solution: u(x,t) = exp(-pi^2 * t) * sin(pi * x)
    X, T_grid = torch.meshgrid(x, t, indexing='ij')
    u_exact = torch.exp(-np.pi**2 * T_grid) * torch.sin(np.pi * X)
    
    # Input: initial condition
    u0 = torch.sin(np.pi * x)
    
    return {
        'x': x,
        't': t,
        'u0': u0,
        'u_exact': u_exact.T,  # Shape: (nt, nx)
        'pde_params': {'diffusion': 1.0}
    }


# Model configuration fixtures
@pytest.fixture
def small_fno_config():
    """Small FNO configuration for testing."""
    return {
        "modes": 4,
        "width": 8,
        "depth": 2,
        "input_dim": 1,
        "output_dim": 1,
        "activation": "gelu"
    }


@pytest.fixture
def small_deeponet_config():
    """Small DeepONet configuration for testing."""
    return {
        "branch_net": {"layers": [100, 32, 32]},
        "trunk_net": {"layers": [1, 32, 32]},
        "output_dim": 1,
        "activation": "tanh"
    }


@pytest.fixture
def small_gno_config():
    """Small GNO configuration for testing."""
    return {
        "node_features": 8,
        "edge_features": 4,
        "hidden_dim": 16,
        "num_layers": 2,
        "output_dim": 1
    }


# Training configuration fixtures
@pytest.fixture
def training_config():
    """Standard training configuration for tests."""
    return {
        "batch_size": 4,
        "learning_rate": 1e-3,
        "epochs": 2,  # Very short for tests
        "optimizer": "adam",
        "scheduler": None,
        "early_stopping": False
    }


@pytest.fixture
def posterior_config():
    """Configuration for posterior approximation tests."""
    return {
        "laplace": {
            "hessian_structure": "diag",
            "prior_precision": 1.0,
            "temperature": 1.0
        },
        "variational": {
            "posterior_type": "mean_field",
            "kl_weight": 1.0,
            "num_samples": 5
        },
        "ensemble": {
            "num_members": 3,
            "init_strategy": "random"
        }
    }


@pytest.fixture
def active_learning_config():
    """Configuration for active learning tests."""
    return {
        "acquisition": "bald",
        "batch_size": 5,
        "budget": 20,
        "init_size": 10,
        "strategy": "greedy"
    }


# Data loader fixtures
@pytest.fixture
def sample_dataloader(sample_data_2d, training_config):
    """Create a DataLoader with sample data."""
    x, y = sample_data_2d
    dataset = TensorDataset(x, y)
    return DataLoader(
        dataset, 
        batch_size=training_config["batch_size"], 
        shuffle=False
    )


@pytest.fixture
def pde_dataloader(sample_pde_data, training_config):
    """Create a DataLoader with PDE data."""
    data = sample_pde_data
    # Create input-output pairs for supervised learning
    inputs = data['u0'].unsqueeze(0).repeat(data['u_exact'].shape[0], 1)
    outputs = data['u_exact']
    
    dataset = TensorDataset(inputs, outputs)
    return DataLoader(
        dataset,
        batch_size=training_config["batch_size"],
        shuffle=False
    )


# Mock fixtures for testing without real implementations
@pytest.fixture
def mock_neural_operator():
    """Mock neural operator for testing interfaces."""
    class MockNeuralOperator:
        def __init__(self, input_dim=1, output_dim=1):
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.training = True
            
        def forward(self, x):
            # Simple linear transformation for testing
            return torch.randn_like(x)
            
        def parameters(self):
            return [torch.randn(10, requires_grad=True)]
            
        def train(self, mode=True):
            self.training = mode
            return self
            
        def eval(self):
            return self.train(False)
    
    return MockNeuralOperator


@pytest.fixture
def mock_posterior():
    """Mock posterior approximation for testing."""
    class MockPosterior:
        def __init__(self):
            self.fitted = False
            
        def fit(self, model, data_loader):
            self.fitted = True
            
        def predict(self, x):
            mean = torch.randn_like(x)
            std = torch.ones_like(x) * 0.1
            return mean, std
            
        def sample(self, x, n_samples=10):
            return torch.randn(n_samples, *x.shape)
            
        def log_marginal_likelihood(self):
            return torch.tensor(-1.0)
    
    return MockPosterior


# Utility fixtures
@pytest.fixture
def tolerance():
    """Numerical tolerance for test assertions."""
    return {
        "rtol": 1e-4,
        "atol": 1e-6
    }


@pytest.fixture
def test_config():
    """Global test configuration."""
    return {
        "run_slow_tests": not os.getenv("SKIP_SLOW_TESTS", False),
        "run_gpu_tests": torch.cuda.is_available() and not os.getenv("SKIP_GPU_TESTS", False),
        "test_data_size": "small",  # small, medium, large
        "verbose": os.getenv("PYTEST_VERBOSE", False)
    }


# Benchmark fixtures
@pytest.fixture(scope="session")
def benchmark_data(device, dtype):
    """Large dataset for benchmarking."""
    if os.getenv("SKIP_BENCHMARK_TESTS"):
        pytest.skip("Benchmark tests disabled")
    
    np.random.seed(42)
    n_samples = 1000
    x = torch.randn(n_samples, 100, dtype=dtype, device=device)
    y = torch.randn(n_samples, 1, dtype=dtype, device=device)
    return x, y


# Error injection fixtures for robustness testing
@pytest.fixture
def corrupted_data(sample_data_2d):
    """Data with some corrupted values for robustness testing."""
    x, y = sample_data_2d
    # Inject some NaN and Inf values
    x_corrupt = x.clone()
    y_corrupt = y.clone()
    x_corrupt[0] = float('nan')
    y_corrupt[-1] = float('inf')
    return x_corrupt, y_corrupt


@pytest.fixture
def noisy_data(sample_data_2d):
    """Add significant noise to data for robustness testing."""
    x, y = sample_data_2d
    noise_level = 0.5
    y_noisy = y + noise_level * torch.randn_like(y)
    return x, y_noisy