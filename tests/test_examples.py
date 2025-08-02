"""Example tests demonstrating testing patterns and best practices."""

import pytest
import torch
import numpy as np
from typing import Tuple, List, Dict, Any


class TestBasicPatterns:
    """Demonstrate basic testing patterns."""
    
    def test_simple_assertion(self):
        """Most basic test pattern."""
        assert 2 + 2 == 4
        assert "hello" in "hello world"
        assert [1, 2, 3] == [1, 2, 3]
    
    def test_floating_point_comparison(self, tolerance):
        """Test floating point comparisons with tolerance."""
        a = 0.1 + 0.2
        b = 0.3
        
        # This would fail due to floating point precision
        # assert a == b
        
        # Use tolerance instead
        assert abs(a - b) < tolerance["atol"]
        
        # Or with numpy/torch
        assert np.isclose(a, b, atol=tolerance["atol"])
    
    def test_tensor_operations(self, device, dtype, tolerance):
        """Test tensor operations with proper comparisons."""
        x = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device=device)
        y = torch.tensor([1.0, 2.0, 3.0], dtype=dtype, device=device)
        
        # Element-wise comparison
        assert torch.allclose(x, y, rtol=tolerance["rtol"], atol=tolerance["atol"])
        
        # Shape comparison
        assert x.shape == y.shape
        
        # Device and dtype comparison
        assert x.device == device
        assert x.dtype == dtype
    
    def test_exception_handling(self):
        """Test exception handling patterns."""
        with pytest.raises(ValueError):
            raise ValueError("Expected error")
        
        with pytest.raises(RuntimeError, match="specific message"):
            raise RuntimeError("specific message in error")
        
        # Test that no exception is raised
        try:
            result = 10 / 2
            assert result == 5
        except ZeroDivisionError:
            pytest.fail("Unexpected ZeroDivisionError")


class TestParametrizedTests:
    """Demonstrate parametrized testing patterns."""
    
    @pytest.mark.parametrize("input_val,expected", [
        (0, 0),
        (1, 1),
        (2, 4),
        (3, 9),
        (-2, 4),
    ])
    def test_square_function(self, input_val, expected):
        """Test square function with multiple inputs."""
        def square(x):
            return x * x
        
        assert square(input_val) == expected
    
    @pytest.mark.parametrize("device_type", ["cpu", "cuda"])
    def test_device_compatibility(self, device_type):
        """Test compatibility across devices."""
        if device_type == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device(device_type)
        x = torch.randn(10, device=device)
        y = x * 2
        
        assert y.device == device
        assert torch.allclose(y, x * 2)
    
    @pytest.mark.parametrize("batch_size,input_dim,output_dim", [
        (1, 10, 1),
        (8, 20, 5),
        (32, 100, 10),
    ])
    def test_model_shapes(self, batch_size, input_dim, output_dim, device):
        """Test model with different input/output dimensions."""
        model = torch.nn.Linear(input_dim, output_dim).to(device)
        x = torch.randn(batch_size, input_dim, device=device)
        
        output = model(x)
        
        assert output.shape == (batch_size, output_dim)
        assert output.device == device


class TestFixtureUsage:
    """Demonstrate fixture usage patterns."""
    
    def test_data_fixtures(self, sample_data_1d, sample_data_2d):
        """Test using data fixtures."""
        x1, y1 = sample_data_1d
        x2, y2 = sample_data_2d
        
        # 1D data checks
        assert x1.ndim == 1
        assert y1.ndim == 1
        assert x1.shape[0] == y1.shape[0]
        
        # 2D data checks
        assert x2.ndim == 2
        assert y2.ndim == 1
        assert x2.shape[0] == y2.shape[0]
    
    def test_config_fixtures(self, small_fno_config, training_config):
        """Test using configuration fixtures."""
        # FNO config
        assert "modes" in small_fno_config
        assert "width" in small_fno_config
        assert small_fno_config["modes"] > 0
        
        # Training config
        assert "batch_size" in training_config
        assert "learning_rate" in training_config
        assert training_config["batch_size"] > 0
    
    def test_mock_fixtures(self, mock_neural_operator, mock_posterior):
        """Test using mock fixtures."""
        # Mock operator
        op = mock_neural_operator()
        x = torch.randn(5, 10)
        output = op.forward(x)
        assert output.shape == x.shape
        
        # Mock posterior
        post = mock_posterior()
        mean, std = post.predict(x)
        assert mean.shape == x.shape
        assert std.shape == x.shape


class TestAsyncAndSlow:
    """Demonstrate async and slow test patterns."""
    
    @pytest.mark.slow
    def test_computationally_expensive(self, benchmark_data):
        """Test marked as slow for optional execution."""
        x, y = benchmark_data
        
        # Simulate expensive computation
        result = torch.matmul(x, x.T)
        eigenvals = torch.linalg.eigvals(result)
        
        assert eigenvals.shape[0] == x.shape[0]
        assert torch.isreal(eigenvals).all()
    
    @pytest.mark.gpu
    def test_gpu_specific(self, device):
        """Test that requires GPU."""
        if device.type != "cuda":
            pytest.skip("GPU test requires CUDA")
        
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        # GPU-specific operations
        result = torch.matmul(x, y)
        assert result.device.type == "cuda"
    
    def test_with_timeout(self):
        """Test with timeout (conceptual - pytest-timeout plugin)."""
        import time
        
        # This should complete quickly
        start = time.time()
        result = sum(range(1000))
        end = time.time()
        
        assert result == 499500
        assert (end - start) < 1.0  # Should complete in less than 1 second


class TestErrorConditions:
    """Demonstrate testing error conditions and edge cases."""
    
    def test_empty_input(self, device, dtype):
        """Test handling of empty inputs."""
        empty_tensor = torch.empty(0, 10, dtype=dtype, device=device)
        
        # Many operations should handle empty tensors gracefully
        result = empty_tensor.sum()
        assert result == 0
        
        # But some operations might fail - test for expected failures
        with pytest.raises(RuntimeError):
            torch.linalg.inv(empty_tensor.reshape(0, 0))
    
    def test_invalid_shapes(self, device):
        """Test handling of invalid input shapes."""
        x = torch.randn(10, 5, device=device)
        y = torch.randn(3, 8, device=device)  # Incompatible shape
        
        # Matrix multiplication should fail
        with pytest.raises(RuntimeError):
            torch.matmul(x, y)
    
    def test_numerical_edge_cases(self, device, dtype):
        """Test numerical edge cases."""
        # Very large numbers
        large = torch.tensor(1e10, dtype=dtype, device=device)
        assert torch.isfinite(large)
        
        # Very small numbers
        small = torch.tensor(1e-10, dtype=dtype, device=device)
        assert torch.isfinite(small)
        assert small > 0
        
        # Operations that might cause overflow
        result = large * large
        # Check if overflow occurred (might become inf)
        if torch.isinf(result):
            pytest.skip("Expected overflow occurred")
    
    def test_corrupted_data_handling(self, corrupted_data):
        """Test handling of corrupted data."""
        x_corrupt, y_corrupt = corrupted_data
        
        # Detect corruption
        assert torch.isnan(x_corrupt).any()
        assert torch.isinf(y_corrupt).any()
        
        # Operations with corrupted data
        result_x = torch.nansum(x_corrupt)  # Should ignore NaN
        assert torch.isfinite(result_x)
        
        # Infinity handling
        y_clipped = torch.clamp(y_corrupt, -1e6, 1e6)
        assert torch.isfinite(y_clipped).all()


class TestRandomnessAndReproducibility:
    """Demonstrate testing with randomness and reproducibility."""
    
    def test_reproducible_randomness(self):
        """Test that randomness is reproducible."""
        # The random_seed fixture should make this reproducible
        x1 = torch.randn(10)
        
        # Reset seed manually (in real tests, fixture handles this)
        torch.manual_seed(42)
        x2 = torch.randn(10)
        
        assert torch.allclose(x1, x2)
    
    def test_statistical_properties(self, sample_data_1d):
        """Test statistical properties of random data."""
        x, y = sample_data_1d
        
        # Test that data has expected statistical properties
        # (Note: these might occasionally fail due to randomness)
        mean_x = x.mean()
        std_x = x.std()
        
        # For uniform data on [0,1], mean should be around 0.5
        assert 0.4 < mean_x < 0.6, f"Mean {mean_x} outside expected range"
        
        # Standard deviation should be reasonable
        assert 0.1 < std_x < 0.5, f"Std {std_x} outside expected range"
    
    def test_monte_carlo_estimation(self):
        """Test Monte Carlo estimation convergence."""
        n_samples = 10000
        
        # Estimate pi using Monte Carlo
        points = torch.rand(n_samples, 2) * 2 - 1  # Random points in [-1,1]^2
        distances = torch.norm(points, dim=1)
        inside_circle = (distances <= 1).float()
        pi_estimate = 4 * inside_circle.mean()
        
        # Should be close to pi (with some tolerance for randomness)
        assert abs(pi_estimate - np.pi) < 0.1, f"Pi estimate {pi_estimate} too far from {np.pi}"


class TestPerformanceAndMemory:
    """Demonstrate performance and memory testing."""
    
    def test_memory_cleanup(self, device):
        """Test that memory is properly cleaned up."""
        if device.type != "cuda":
            pytest.skip("Memory test for GPU only")
        
        initial_memory = torch.cuda.memory_allocated(device)
        
        # Allocate large tensor
        large_tensor = torch.randn(1000, 1000, device=device)
        allocated_memory = torch.cuda.memory_allocated(device)
        
        assert allocated_memory > initial_memory
        
        # Delete tensor
        del large_tensor
        torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated(device)
        assert final_memory <= initial_memory + 1024  # Allow small overhead
    
    def test_performance_regression(self, sample_data_2d, device):
        """Test for performance regressions."""
        x, y = sample_data_2d
        
        # Simple performance test
        model = torch.nn.Linear(2, 1).to(device)
        x = x.to(device)
        
        import time
        start_time = time.time()
        
        # Run multiple forward passes
        with torch.no_grad():
            for _ in range(1000):
                output = model(x)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert elapsed < 5.0, f"Performance regression: took {elapsed:.2f}s"
    
    def test_memory_efficient_operations(self, device):
        """Test memory-efficient operations."""
        if device.type != "cuda":
            pytest.skip("Memory test for GPU only")
        
        initial_memory = torch.cuda.memory_allocated(device)
        
        # Use in-place operations to save memory
        x = torch.randn(500, 500, device=device)
        
        # In-place operations
        x.add_(1.0)  # Instead of x = x + 1.0
        x.mul_(2.0)  # Instead of x = x * 2.0
        
        final_memory = torch.cuda.memory_allocated(device)
        
        # Should not have allocated much additional memory
        memory_increase = final_memory - initial_memory
        assert memory_increase < 1000000  # Less than 1MB increase


class TestIntegrationExamples:
    """Demonstrate integration testing patterns."""
    
    def test_end_to_end_workflow(self, sample_dataloader, device):
        """Test complete workflow integration."""
        # Create model
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.MSELoss()
        
        # Training phase
        model.train()
        for epoch in range(2):  # Short training
            for x, y in sample_dataloader:
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output.squeeze(), y)
                loss.backward()
                optimizer.step()
        
        # Evaluation phase
        model.eval()
        total_loss = 0
        count = 0
        
        with torch.no_grad():
            for x, y in sample_dataloader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output.squeeze(), y)
                total_loss += loss.item()
                count += 1
        
        avg_loss = total_loss / count
        
        # Verify workflow completed successfully
        assert avg_loss >= 0
        assert count > 0
        
        # Verify model learned something (not guaranteed with random data)
        # In real tests, you'd use data with known patterns
    
    def test_configuration_integration(self, small_fno_config, training_config, device):
        """Test integration of configuration systems."""
        # Use configurations to create components
        fno_config = small_fno_config
        train_config = training_config
        
        # Create model based on config
        input_dim = fno_config["input_dim"] 
        output_dim = fno_config["output_dim"]
        width = fno_config["width"]
        
        model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, width),
            torch.nn.ReLU(),
            torch.nn.Linear(width, output_dim)
        ).to(device)
        
        # Create optimizer based on config
        lr = train_config["learning_rate"]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        # Test that components work together
        x = torch.randn(train_config["batch_size"], input_dim, device=device)
        output = model(x)
        
        assert output.shape == (train_config["batch_size"], output_dim)
        
        # Test optimization step
        loss = output.sum()
        loss.backward()
        optimizer.step()
        
        # Verify gradients were computed and applied
        has_gradients = any(p.grad is not None for p in model.parameters())
        assert has_gradients