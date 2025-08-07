"""Pytest configuration and fixtures."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducible tests
torch.manual_seed(42)
np.random.seed(42)


@pytest.fixture
def device():
    """Get computation device (CPU/GPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_data_1d():
    """Generate sample 1D data for testing."""
    torch.manual_seed(42)
    np.random.seed(42)
    x = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(100)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


@pytest.fixture
def sample_data_2d():
    """Generate sample 2D data for testing."""
    torch.manual_seed(42)
    np.random.seed(42)
    x = np.random.randn(100, 2)
    y = np.sum(x**2, axis=1) + 0.1 * np.random.randn(100)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


@pytest.fixture
def sample_tensor_data():
    """Generate sample tensor data for neural operators."""
    torch.manual_seed(42)
    batch_size, spatial_size = 8, 32
    
    # 1D data
    input_1d = torch.randn(batch_size, 1, spatial_size)
    output_1d = torch.randn(batch_size, 1, spatial_size)
    
    # 2D data
    input_2d = torch.randn(batch_size, 1, spatial_size, spatial_size)
    output_2d = torch.randn(batch_size, 1, spatial_size, spatial_size)
    
    return {
        "1d": (input_1d, output_1d),
        "2d": (input_2d, output_2d)
    }


@pytest.fixture
def sample_dataloader():
    """Create sample DataLoader for testing."""
    torch.manual_seed(42)
    inputs = torch.randn(32, 1, 64)
    targets = torch.randn(32, 1, 64)
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=4, shuffle=False)


@pytest.fixture
def small_fno_config():
    """Small FNO configuration for testing."""
    return {
        "modes": 4,
        "width": 8,
        "depth": 2,
        "input_dim": 1,
        "output_dim": 1,
        "spatial_dim": 1
    }


@pytest.fixture
def small_deeponet_config():
    """Small DeepONet configuration for testing."""
    return {
        "input_dim": 1,
        "output_dim": 1,
        "branch_layers": [16, 16],
        "trunk_layers": [16, 16],
        "basis_functions": 20
    }


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config():
    """Sample experiment configuration for testing."""
    from probneural_operator.utils.config import ExperimentConfig, FNOConfig, TrainingConfig
    
    return ExperimentConfig(
        name="test_experiment",
        model=FNOConfig(
            input_dim=1,
            output_dim=1,
            modes=4,
            width=16,
            depth=2
        ),
        training=TrainingConfig(
            epochs=5,
            batch_size=4,
            validation_split=0.2
        )
    )


@pytest.fixture
def mock_model():
    """Create a simple mock model for testing."""
    import torch.nn as nn
    
    class MockModel(nn.Module):
        def __init__(self, input_dim=10, output_dim=1):
            super().__init__()
            self.linear = nn.Linear(input_dim, output_dim)
        
        def forward(self, x):
            return self.linear(x)
    
    return MockModel()


@pytest.fixture
def problematic_tensors():
    """Create tensors with various numerical issues for testing."""
    torch.manual_seed(42)
    
    # Normal tensor
    normal = torch.randn(5, 5)
    
    # Tensor with NaN
    nan_tensor = torch.randn(5, 5)
    nan_tensor[2, 3] = float('nan')
    
    # Tensor with Inf
    inf_tensor = torch.randn(5, 5)
    inf_tensor[1, 4] = float('inf')
    
    # Very large values
    large_tensor = torch.randn(5, 5) * 1e10
    
    # Very small values
    small_tensor = torch.randn(5, 5) * 1e-20
    
    return {
        "normal": normal,
        "nan": nan_tensor,
        "inf": inf_tensor,
        "large": large_tensor,
        "small": small_tensor
    }


@pytest.fixture(scope="session")
def performance_baseline():
    """Performance baseline for regression testing."""
    return {
        "fno_forward_time_ms": 50.0,  # milliseconds
        "fno_backward_time_ms": 100.0,
        "memory_mb_per_sample": 10.0,  # MB
        "convergence_epochs": 100
    }


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "benchmark: marks tests as benchmark tests")
    config.addinivalue_line("markers", "property: marks property-based tests")


@pytest.fixture(autouse=True)
def setup_logging():
    """Set up logging for tests."""
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Suppress some verbose loggers during testing
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


@pytest.fixture
def capture_warnings():
    """Capture warnings during tests."""
    import warnings
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")
        yield warning_list


# Property-based testing fixtures
try:
    from hypothesis import given, strategies as st
    
    @pytest.fixture
    def tensor_strategy():
        """Hypothesis strategy for generating tensors."""
        return st.builds(
            torch.randn,
            st.integers(min_value=1, max_value=10),  # batch size
            st.integers(min_value=1, max_value=5),   # channels
            st.integers(min_value=8, max_value=64),  # spatial size
        )
    
    @pytest.fixture
    def config_strategy():
        """Hypothesis strategy for generating valid configurations."""
        from probneural_operator.utils.config import FNOConfig
        
        return st.builds(
            FNOConfig,
            input_dim=st.integers(min_value=1, max_value=3),
            output_dim=st.integers(min_value=1, max_value=3),
            modes=st.integers(min_value=2, max_value=16),
            width=st.integers(min_value=8, max_value=64),
            depth=st.integers(min_value=1, max_value=6),
        )

except ImportError:
    # Hypothesis not available, provide dummy fixtures
    @pytest.fixture
    def tensor_strategy():
        pytest.skip("Hypothesis not available")
    
    @pytest.fixture  
    def config_strategy():
        pytest.skip("Hypothesis not available")