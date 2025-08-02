"""Test that all fixtures work correctly."""

import pytest
import torch
import numpy as np


class TestBasicFixtures:
    """Test basic fixtures and configuration."""
    
    def test_device_fixture(self, device):
        """Test device fixture returns valid device."""
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda"]
    
    def test_dtype_fixture(self, dtype):
        """Test dtype fixture returns valid dtype."""
        assert dtype in [torch.float32, torch.float64, torch.float16]
    
    def test_random_seed_fixture(self):
        """Test that random seed fixture provides reproducible results."""
        # This test relies on the autouse=True random_seed fixture
        x1 = torch.randn(10)
        torch.manual_seed(42)  # Reset to same seed
        x2 = torch.randn(10)
        assert torch.allclose(x1, x2)


class TestDataFixtures:
    """Test data generation fixtures."""
    
    def test_sample_data_1d(self, sample_data_1d, dtype, device):
        """Test 1D sample data fixture."""
        x, y = sample_data_1d
        
        assert x.dtype == dtype
        assert y.dtype == dtype
        assert x.device == device
        assert y.device == device
        assert x.shape == (100,)
        assert y.shape == (100,)
        assert torch.isfinite(x).all()
        assert torch.isfinite(y).all()
    
    def test_sample_data_2d(self, sample_data_2d, dtype, device):
        """Test 2D sample data fixture."""
        x, y = sample_data_2d
        
        assert x.dtype == dtype
        assert y.dtype == dtype
        assert x.device == device
        assert y.device == device
        assert x.shape == (100, 2)
        assert y.shape == (100,)
        assert torch.isfinite(x).all()
        assert torch.isfinite(y).all()
    
    def test_burgers_data(self, burgers_data, dtype, device):
        """Test Burgers equation data fixture."""
        x, t, u = burgers_data
        
        assert x.dtype == dtype
        assert t.dtype == dtype
        assert u.dtype == dtype
        assert x.device == device
        assert t.device == device
        assert u.device == device
        
        assert x.shape == (64,)
        assert t.shape == (50,)
        assert u.shape == (50, 64)
        assert torch.isfinite(x).all()
        assert torch.isfinite(t).all()
        assert torch.isfinite(u).all()
    
    def test_sample_pde_data(self, sample_pde_data, dtype, device):
        """Test PDE data fixture."""
        data = sample_pde_data
        
        assert 'x' in data
        assert 't' in data
        assert 'u0' in data
        assert 'u_exact' in data
        assert 'pde_params' in data
        
        x, t, u0, u_exact = data['x'], data['t'], data['u0'], data['u_exact']
        
        assert x.dtype == dtype
        assert t.dtype == dtype
        assert u0.dtype == dtype
        assert u_exact.dtype == dtype
        
        assert x.shape == (32,)
        assert t.shape == (20,)
        assert u0.shape == (32,)
        assert u_exact.shape == (20, 32)


class TestConfigurationFixtures:
    """Test configuration fixtures."""
    
    def test_small_fno_config(self, small_fno_config):
        """Test FNO configuration fixture."""
        required_keys = ["modes", "width", "depth", "input_dim", "output_dim"]
        for key in required_keys:
            assert key in small_fno_config
        
        assert isinstance(small_fno_config["modes"], int)
        assert isinstance(small_fno_config["width"], int)
        assert isinstance(small_fno_config["depth"], int)
        assert small_fno_config["modes"] > 0
        assert small_fno_config["width"] > 0
        assert small_fno_config["depth"] > 0
    
    def test_small_deeponet_config(self, small_deeponet_config):
        """Test DeepONet configuration fixture."""
        assert "branch_net" in small_deeponet_config
        assert "trunk_net" in small_deeponet_config
        assert "output_dim" in small_deeponet_config
        
        assert "layers" in small_deeponet_config["branch_net"]
        assert "layers" in small_deeponet_config["trunk_net"]
        assert isinstance(small_deeponet_config["branch_net"]["layers"], list)
        assert isinstance(small_deeponet_config["trunk_net"]["layers"], list)
    
    def test_training_config(self, training_config):
        """Test training configuration fixture."""
        required_keys = ["batch_size", "learning_rate", "epochs"]
        for key in required_keys:
            assert key in training_config
        
        assert training_config["batch_size"] > 0
        assert training_config["learning_rate"] > 0
        assert training_config["epochs"] > 0
    
    def test_posterior_config(self, posterior_config):
        """Test posterior configuration fixture."""
        assert "laplace" in posterior_config
        assert "variational" in posterior_config
        assert "ensemble" in posterior_config
        
        # Check Laplace config
        laplace_config = posterior_config["laplace"]
        assert "hessian_structure" in laplace_config
        assert "prior_precision" in laplace_config
        
        # Check variational config
        variational_config = posterior_config["variational"]
        assert "posterior_type" in variational_config
        assert "kl_weight" in variational_config
        
        # Check ensemble config
        ensemble_config = posterior_config["ensemble"]
        assert "num_members" in ensemble_config
        assert "init_strategy" in ensemble_config


class TestMockFixtures:
    """Test mock fixtures for interface testing."""
    
    def test_mock_neural_operator(self, mock_neural_operator):
        """Test mock neural operator fixture."""
        mock_op = mock_neural_operator()
        
        assert hasattr(mock_op, 'forward')
        assert hasattr(mock_op, 'parameters')
        assert hasattr(mock_op, 'train')
        assert hasattr(mock_op, 'eval')
        
        # Test forward pass
        x = torch.randn(10, 1)
        output = mock_op.forward(x)
        assert output.shape == x.shape
        
        # Test parameters
        params = list(mock_op.parameters())
        assert len(params) > 0
        assert all(p.requires_grad for p in params)
        
        # Test train/eval mode
        mock_op.train()
        assert mock_op.training
        mock_op.eval()
        assert not mock_op.training
    
    def test_mock_posterior(self, mock_posterior):
        """Test mock posterior fixture."""
        mock_post = mock_posterior()
        
        assert hasattr(mock_post, 'fit')
        assert hasattr(mock_post, 'predict')
        assert hasattr(mock_post, 'sample')
        assert hasattr(mock_post, 'log_marginal_likelihood')
        
        # Test fitting
        assert not mock_post.fitted
        mock_post.fit(None, None)
        assert mock_post.fitted
        
        # Test prediction
        x = torch.randn(10, 1)
        mean, std = mock_post.predict(x)
        assert mean.shape == x.shape
        assert std.shape == x.shape
        assert torch.all(std > 0)
        
        # Test sampling
        samples = mock_post.sample(x, n_samples=5)
        assert samples.shape == (5, *x.shape)
        
        # Test log marginal likelihood
        lml = mock_post.log_marginal_likelihood()
        assert isinstance(lml, torch.Tensor)
        assert lml.shape == ()


class TestDataLoaderFixtures:
    """Test DataLoader fixtures."""
    
    def test_sample_dataloader(self, sample_dataloader, training_config):
        """Test sample DataLoader fixture."""
        assert len(sample_dataloader) > 0
        
        # Test batch
        batch = next(iter(sample_dataloader))
        assert len(batch) == 2  # x, y
        x, y = batch
        
        assert x.shape[0] == training_config["batch_size"]
        assert y.shape[0] == training_config["batch_size"]
        assert x.ndim == 2  # (batch_size, input_dim)
        assert y.ndim == 1  # (batch_size,)
    
    def test_pde_dataloader(self, pde_dataloader, training_config):
        """Test PDE DataLoader fixture."""
        assert len(pde_dataloader) > 0
        
        # Test batch
        batch = next(iter(pde_dataloader))
        assert len(batch) == 2  # inputs, outputs
        inputs, outputs = batch
        
        assert inputs.shape[0] == training_config["batch_size"]
        assert outputs.shape[0] == training_config["batch_size"]


class TestUtilityFixtures:
    """Test utility fixtures."""
    
    def test_tolerance(self, tolerance):
        """Test tolerance fixture."""
        assert "rtol" in tolerance
        assert "atol" in tolerance
        assert tolerance["rtol"] > 0
        assert tolerance["atol"] > 0
    
    def test_test_config(self, test_config):
        """Test global test configuration."""
        required_keys = ["run_slow_tests", "run_gpu_tests", "test_data_size"]
        for key in required_keys:
            assert key in test_config
        
        assert isinstance(test_config["run_slow_tests"], bool)
        assert isinstance(test_config["run_gpu_tests"], bool)
        assert test_config["test_data_size"] in ["small", "medium", "large"]


class TestErrorInjectionFixtures:
    """Test error injection fixtures for robustness testing."""
    
    def test_corrupted_data(self, corrupted_data):
        """Test corrupted data fixture."""
        x_corrupt, y_corrupt = corrupted_data
        
        # Should have NaN and Inf values
        assert torch.isnan(x_corrupt).any()
        assert torch.isinf(y_corrupt).any()
    
    def test_noisy_data(self, noisy_data):
        """Test noisy data fixture."""
        x, y_noisy = noisy_data
        
        # Should be finite but noisy
        assert torch.isfinite(x).all()
        assert torch.isfinite(y_noisy).all()
        
        # Check that noise was added by comparing variance
        # (This is a heuristic check)
        assert y_noisy.var() > 0.1  # Should have significant variance due to noise


class TestTemporaryDirectories:
    """Test temporary directory fixtures."""
    
    def test_test_data_dir(self, test_data_dir):
        """Test test data directory fixture."""
        assert test_data_dir.exists()
        assert test_data_dir.is_dir()
        assert test_data_dir.name == "test_data"
    
    def test_model_checkpoint_dir(self, model_checkpoint_dir):
        """Test model checkpoint directory fixture."""
        assert model_checkpoint_dir.exists()
        assert model_checkpoint_dir.is_dir()
        assert model_checkpoint_dir.name == "checkpoints"