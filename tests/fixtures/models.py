"""Test fixtures for model instantiation and testing."""

import torch
import torch.nn as nn
import pytest
from typing import Dict, Any, Optional, Callable
from unittest.mock import Mock, MagicMock


class DummyNeuralOperator(nn.Module):
    """Dummy neural operator for testing purposes."""
    
    def __init__(self, input_dim: int = 1, output_dim: int = 1, hidden_dim: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # Simple MLP as placeholder
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._uncertainty_mode = False
    
    def forward(self, x):
        # Flatten spatial dimensions
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1, self.input_dim)
        
        # Apply network
        output = self.layers(x_flat)
        
        # Reshape back
        output = output.view(batch_size, self.output_dim, -1)
        
        if self._uncertainty_mode:
            # Return mean and log variance
            mean = output
            log_var = torch.ones_like(mean) * -2.0  # Fixed uncertainty
            return mean, log_var
        
        return output
    
    def enable_uncertainty(self):
        """Enable uncertainty estimation mode."""
        self._uncertainty_mode = True
    
    def disable_uncertainty(self):
        """Disable uncertainty estimation mode."""
        self._uncertainty_mode = False


class MockPosterior:
    """Mock posterior approximation for testing."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.fitted = False
    
    def fit(self, dataloader, **kwargs):
        """Mock fitting process."""
        self.fitted = True
        return self
    
    def predict(self, x, return_std=True, num_samples=10):
        """Mock prediction with uncertainty."""
        with torch.no_grad():
            mean = self.model(x)
            
            if return_std:
                # Mock uncertainty
                std = 0.1 * torch.ones_like(mean)
                return mean, std
            return mean
    
    def sample(self, x, num_samples=10):
        """Mock posterior sampling."""
        mean = self.model(x)
        samples = []
        
        for _ in range(num_samples):
            noise = 0.1 * torch.randn_like(mean)
            samples.append(mean + noise)
        
        return torch.stack(samples, dim=0)
    
    def log_marginal_likelihood(self):
        """Mock log marginal likelihood."""
        return torch.tensor(-100.0)


class MockAcquisitionFunction:
    """Mock acquisition function for testing."""
    
    def __init__(self, name: str = "test_acquisition"):
        self.name = name
        self.call_count = 0
    
    def __call__(self, model, x_pool, **kwargs):
        """Mock acquisition function evaluation."""
        self.call_count += 1
        batch_size = x_pool.shape[0]
        # Return random scores
        return torch.rand(batch_size)


@pytest.fixture
def dummy_neural_operator():
    """Create a dummy neural operator for testing."""
    return DummyNeuralOperator()


@pytest.fixture
def dummy_neural_operator_2d():
    """Create a dummy 2D neural operator."""
    return DummyNeuralOperator(input_dim=2, output_dim=1, hidden_dim=64)


@pytest.fixture
def mock_posterior(dummy_neural_operator):
    """Create a mock posterior approximation."""
    return MockPosterior(dummy_neural_operator)


@pytest.fixture
def mock_acquisition():
    """Create a mock acquisition function."""
    return MockAcquisitionFunction()


@pytest.fixture
def model_factory():
    """Factory for creating different model types."""
    def _create_model(model_type: str, **kwargs):
        if model_type == "dummy":
            return DummyNeuralOperator(**kwargs)
        elif model_type == "linear":
            input_dim = kwargs.get("input_dim", 10)
            output_dim = kwargs.get("output_dim", 1)
            return nn.Linear(input_dim, output_dim)
        elif model_type == "mlp":
            input_dim = kwargs.get("input_dim", 10)
            hidden_dim = kwargs.get("hidden_dim", 32)
            output_dim = kwargs.get("output_dim", 1)
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    return _create_model


@pytest.fixture
def trained_model(dummy_neural_operator, sample_dataloader):
    """Create a partially trained model for testing."""
    model = dummy_neural_operator
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Quick training loop
    model.train()
    for batch_idx, (inputs, targets) in enumerate(sample_dataloader):
        if batch_idx >= 2:  # Only train for 2 batches
            break
            
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    model.eval()
    return model


@pytest.fixture
def model_with_checkpoints(trained_model, temp_dir):
    """Create model with saved checkpoints."""
    model = trained_model
    checkpoint_dir = temp_dir / "checkpoints"
    checkpoint_dir.mkdir()
    
    # Save multiple checkpoints
    checkpoints = []
    for epoch in [10, 20, 30]:
        checkpoint_path = checkpoint_dir / f"epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': {},
            'loss': 0.5 - epoch * 0.01,
        }, checkpoint_path)
        checkpoints.append(checkpoint_path)
    
    return model, checkpoints


@pytest.fixture
def problematic_model():
    """Create a model that can exhibit common training problems."""
    class ProblematicModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
            self._explode_gradients = False
            self._return_nan = False
        
        def forward(self, x):
            if self._return_nan:
                return torch.full_like(x[:, :1], float('nan'))
            
            output = self.linear(x)
            
            if self._explode_gradients:
                # Multiply by large number to cause gradient explosion
                output = output * 1e6
            
            return output
        
        def enable_gradient_explosion(self):
            self._explode_gradients = True
        
        def enable_nan_output(self):
            self._return_nan = True
    
    return ProblematicModel()


@pytest.fixture
def model_ensemble(model_factory):
    """Create an ensemble of models for testing."""
    models = []
    for i in range(5):
        model = model_factory("dummy", hidden_dim=32 + i*16)
        models.append(model)
    return models


@pytest.fixture
def mock_training_history():
    """Mock training history for testing."""
    epochs = 100
    history = {
        'train_loss': [1.0 - 0.008*i + 0.01*torch.rand(1).item() for i in range(epochs)],
        'val_loss': [1.0 - 0.006*i + 0.02*torch.rand(1).item() for i in range(epochs)],
        'train_acc': [0.3 + 0.005*i + 0.01*torch.rand(1).item() for i in range(epochs)],
        'val_acc': [0.3 + 0.004*i + 0.02*torch.rand(1).item() for i in range(epochs)],
    }
    
    # Ensure losses don't go below zero
    history['train_loss'] = [max(0.01, loss) for loss in history['train_loss']]
    history['val_loss'] = [max(0.01, loss) for loss in history['val_loss']]
    
    return history


@pytest.fixture
def calibration_data():
    """Generate data for calibration testing."""
    n_samples = 1000
    
    # Generate predictions with varying confidence
    predictions = torch.randn(n_samples)
    uncertainties = torch.abs(torch.randn(n_samples)) * 0.5 + 0.1
    
    # Generate true values with correlation to predictions
    noise = torch.randn(n_samples) * uncertainties
    true_values = predictions + noise
    
    return {
        'predictions': predictions,
        'uncertainties': uncertainties,
        'true_values': true_values
    }


@pytest.fixture
def benchmark_models():
    """Create models for benchmarking."""
    models = {
        'small': DummyNeuralOperator(hidden_dim=16),
        'medium': DummyNeuralOperator(hidden_dim=64),
        'large': DummyNeuralOperator(hidden_dim=256)
    }
    return models