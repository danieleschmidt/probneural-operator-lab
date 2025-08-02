"""Test utility functions and helpers."""

import pytest
import torch
import numpy as np
from pathlib import Path


class TestNumericalUtils:
    """Test numerical utility functions."""
    
    def test_torch_device_consistency(self, device, dtype):
        """Test PyTorch device and dtype consistency."""
        x = torch.randn(10, dtype=dtype, device=device)
        y = torch.randn(10, dtype=dtype, device=device)
        
        # Basic operations should work
        z = x + y
        assert z.device == device
        assert z.dtype == dtype
        
        # Matrix operations
        A = torch.randn(10, 10, dtype=dtype, device=device)
        w = A @ x
        assert w.device == device
        assert w.dtype == dtype
    
    def test_numerical_stability(self, tolerance):
        """Test numerical stability helpers."""
        rtol, atol = tolerance["rtol"], tolerance["atol"]
        
        # Test close values
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.0001, 2.0001, 3.0001])
        
        assert torch.allclose(x, y, rtol=rtol, atol=atol)
        
        # Test not-so-close values
        z = torch.tensor([1.1, 2.1, 3.1])
        assert not torch.allclose(x, z, rtol=rtol, atol=atol)
    
    def test_gradient_computation(self, device, dtype):
        """Test gradient computation utilities."""
        x = torch.randn(10, requires_grad=True, dtype=dtype, device=device)
        y = (x**2).sum()
        
        y.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert torch.allclose(x.grad, 2*x)
    
    def test_tensor_properties(self, sample_data_2d, tolerance):
        """Test tensor property checking utilities."""
        x, y = sample_data_2d
        
        # Test finite checks
        assert torch.isfinite(x).all()
        assert torch.isfinite(y).all()
        
        # Test shape consistency
        assert x.shape[0] == y.shape[0]  # Same batch size
        
        # Test value ranges (for generated data)
        assert x.abs().max() < 10  # Reasonable range
        assert y.abs().max() < 100  # Reasonable range


class TestDataValidation:
    """Test data validation utilities."""
    
    def test_data_shape_validation(self, sample_data_2d):
        """Test data shape validation."""
        x, y = sample_data_2d
        
        # Valid shapes
        assert len(x.shape) == 2  # 2D input
        assert len(y.shape) == 1  # 1D output
        assert x.shape[0] == y.shape[0]  # Consistent batch size
    
    def test_data_type_validation(self, sample_data_2d, dtype, device):
        """Test data type validation."""
        x, y = sample_data_2d
        
        assert x.dtype == dtype
        assert y.dtype == dtype
        assert x.device == device
        assert y.device == device
    
    def test_data_range_validation(self, sample_pde_data):
        """Test data range validation for PDE data."""
        data = sample_pde_data
        x, t = data['x'], data['t']
        
        # Spatial domain [0, 1]
        assert x.min() >= 0
        assert x.max() <= 1
        
        # Time domain [0, T]
        assert t.min() >= 0
        assert t.max() > 0
    
    def test_corrupted_data_detection(self, corrupted_data):
        """Test detection of corrupted data."""
        x_corrupt, y_corrupt = corrupted_data
        
        # Should detect NaN values
        assert torch.isnan(x_corrupt).any()
        assert torch.isinf(y_corrupt).any()
        
        # Helper functions would detect this
        def has_invalid_values(tensor):
            return torch.isnan(tensor).any() or torch.isinf(tensor).any()
        
        assert has_invalid_values(x_corrupt)
        assert has_invalid_values(y_corrupt)


class TestConfigValidation:
    """Test configuration validation utilities."""
    
    def test_fno_config_validation(self, small_fno_config):
        """Test FNO configuration validation."""
        config = small_fno_config
        
        # Required fields
        required_fields = ["modes", "width", "depth", "input_dim", "output_dim"]
        for field in required_fields:
            assert field in config
        
        # Value ranges
        assert config["modes"] > 0
        assert config["width"] > 0
        assert config["depth"] > 0
        assert config["input_dim"] > 0
        assert config["output_dim"] > 0
        
        # Type checking
        assert isinstance(config["modes"], int)
        assert isinstance(config["width"], int)
        assert isinstance(config["depth"], int)
    
    def test_training_config_validation(self, training_config):
        """Test training configuration validation."""
        config = training_config
        
        # Required fields
        required_fields = ["batch_size", "learning_rate", "epochs"]
        for field in required_fields:
            assert field in config
        
        # Value ranges
        assert config["batch_size"] > 0
        assert config["learning_rate"] > 0
        assert config["epochs"] > 0
        
        # Type checking
        assert isinstance(config["batch_size"], int)
        assert isinstance(config["learning_rate"], (int, float))
        assert isinstance(config["epochs"], int)
    
    def test_posterior_config_validation(self, posterior_config):
        """Test posterior configuration validation."""
        config = posterior_config
        
        # Check structure
        assert "laplace" in config
        assert "variational" in config
        assert "ensemble" in config
        
        # Validate Laplace config
        laplace = config["laplace"]
        assert "hessian_structure" in laplace
        assert "prior_precision" in laplace
        assert laplace["prior_precision"] > 0
        
        # Validate variational config
        variational = config["variational"]
        assert "kl_weight" in variational
        assert variational["kl_weight"] >= 0
        
        # Validate ensemble config
        ensemble = config["ensemble"]
        assert "num_members" in ensemble
        assert ensemble["num_members"] > 0


class TestFileOperations:
    """Test file operation utilities."""
    
    def test_temporary_directory_creation(self, test_data_dir, model_checkpoint_dir):
        """Test temporary directory creation."""
        assert test_data_dir.exists()
        assert test_data_dir.is_dir()
        assert model_checkpoint_dir.exists()
        assert model_checkpoint_dir.is_dir()
    
    def test_file_path_handling(self, test_data_dir):
        """Test file path handling utilities."""
        # Create a test file
        test_file = test_data_dir / "test.txt"
        test_file.write_text("test content")
        
        assert test_file.exists()
        assert test_file.is_file()
        assert test_file.read_text() == "test content"
    
    def test_model_checkpoint_paths(self, model_checkpoint_dir):
        """Test model checkpoint path utilities."""
        # Test checkpoint naming conventions
        checkpoint_name = "model_epoch_10.pt"
        checkpoint_path = model_checkpoint_dir / checkpoint_name
        
        # Save dummy checkpoint
        dummy_state = {"epoch": 10, "loss": 0.5}
        torch.save(dummy_state, checkpoint_path)
        
        assert checkpoint_path.exists()
        
        # Load and verify
        loaded_state = torch.load(checkpoint_path)
        assert loaded_state["epoch"] == 10
        assert loaded_state["loss"] == 0.5


class TestMockUtilities:
    """Test mock object utilities."""
    
    def test_mock_operator_interface(self, mock_neural_operator):
        """Test mock neural operator interface."""
        mock_op = mock_neural_operator()
        
        # Test interface compliance
        x = torch.randn(5, 10)
        
        # Forward pass
        output = mock_op.forward(x)
        assert output.shape == x.shape
        
        # Parameters
        params = list(mock_op.parameters())
        assert len(params) > 0
        
        # Training mode
        mock_op.train()
        assert mock_op.training
        mock_op.eval()
        assert not mock_op.training
    
    def test_mock_posterior_interface(self, mock_posterior):
        """Test mock posterior interface."""
        mock_post = mock_posterior()
        
        x = torch.randn(5, 10)
        
        # Fit interface
        mock_post.fit(None, None)
        assert mock_post.fitted
        
        # Prediction interface
        mean, std = mock_post.predict(x)
        assert mean.shape == x.shape
        assert std.shape == x.shape
        assert (std > 0).all()
        
        # Sampling interface
        samples = mock_post.sample(x, n_samples=3)
        assert samples.shape == (3, *x.shape)
        
        # Log marginal likelihood
        lml = mock_post.log_marginal_likelihood()
        assert isinstance(lml, torch.Tensor)


class TestBenchmarkUtilities:
    """Test benchmark and performance utilities."""
    
    @pytest.mark.benchmark
    def test_benchmark_data_generation(self, benchmark_data):
        """Test benchmark data generation."""
        x, y = benchmark_data
        
        assert x.shape[0] == 1000  # Large dataset
        assert y.shape[0] == 1000
        assert x.shape[1] == 100   # High-dimensional input
        
    def test_performance_measurement(self, sample_data_2d):
        """Test performance measurement utilities."""
        x, y = sample_data_2d
        
        import time
        
        # Simple timing utility
        start_time = time.time()
        result = torch.matmul(x, x.T)
        end_time = time.time()
        
        elapsed = end_time - start_time
        assert elapsed >= 0
        assert result.shape == (x.shape[0], x.shape[0])
    
    def test_memory_usage_tracking(self, sample_data_2d):
        """Test memory usage tracking utilities."""
        x, y = sample_data_2d
        
        if torch.cuda.is_available():
            # Test CUDA memory tracking
            initial_memory = torch.cuda.memory_allocated()
            
            # Allocate some memory
            large_tensor = torch.randn(1000, 1000, device=x.device)
            
            after_alloc_memory = torch.cuda.memory_allocated()
            assert after_alloc_memory > initial_memory
            
            # Free memory
            del large_tensor
            torch.cuda.empty_cache()


class TestErrorHandling:
    """Test error handling utilities."""
    
    def test_nan_detection(self):
        """Test NaN detection utilities."""
        # Valid tensor
        valid_tensor = torch.randn(10, 10)
        assert not torch.isnan(valid_tensor).any()
        
        # Tensor with NaN
        nan_tensor = torch.tensor([1.0, float('nan'), 3.0])
        assert torch.isnan(nan_tensor).any()
    
    def test_inf_detection(self):
        """Test infinity detection utilities."""
        # Valid tensor
        valid_tensor = torch.randn(10, 10)
        assert not torch.isinf(valid_tensor).any()
        
        # Tensor with infinity
        inf_tensor = torch.tensor([1.0, float('inf'), 3.0])
        assert torch.isinf(inf_tensor).any()
    
    def test_gradient_explosion_detection(self):
        """Test gradient explosion detection."""
        x = torch.randn(10, requires_grad=True)
        
        # Normal gradients
        y = (x**2).sum()
        y.backward()
        assert torch.isfinite(x.grad).all()
        assert x.grad.abs().max() < 100  # Reasonable gradient magnitude
        
        # Potential gradient explosion would be detected here
        # (In real implementation, we'd have monitoring utilities)
    
    def test_numerical_stability_checks(self, tolerance):
        """Test numerical stability checking utilities."""
        rtol, atol = tolerance["rtol"], tolerance["atol"]
        
        # Test matrix conditioning
        well_conditioned = torch.eye(5) + 0.1 * torch.randn(5, 5)
        assert torch.linalg.cond(well_conditioned) < 1000  # Well-conditioned
        
        # Test for numerical precision issues
        small_number = torch.tensor(1e-10)
        large_number = torch.tensor(1e10)
        ratio = large_number / small_number
        assert torch.isfinite(ratio)