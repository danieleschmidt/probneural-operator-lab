"""Performance benchmarks for neural operators and uncertainty quantification."""

import pytest
import torch
import time
import psutil
import gc
from typing import Dict, Any


@pytest.mark.benchmark
class TestNeuralOperatorPerformance:
    """Benchmark neural operator performance."""
    
    def test_forward_pass_performance(self, benchmark_data, device):
        """Benchmark forward pass performance."""
        if device.type == "cpu":
            pytest.skip("GPU benchmarks only run on GPU")
            
        x, y = benchmark_data
        
        # Simple neural operator model
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        ).to(device)
        
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(x[:100])
        
        # Benchmark
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(100):
                output = model(x)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        throughput = (100 * x.shape[0]) / elapsed_time  # samples per second
        
        # Log results
        print(f"Forward pass throughput: {throughput:.2f} samples/sec")
        print(f"Batch size: {x.shape[0]}, Input dim: {x.shape[1]}")
        
        assert throughput > 0
        assert elapsed_time > 0
    
    def test_backward_pass_performance(self, benchmark_data, device):
        """Benchmark backward pass performance."""
        if device.type == "cpu":
            pytest.skip("GPU benchmarks only run on GPU")
            
        x, y = benchmark_data
        
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.MSELoss()
        
        # Warmup
        for _ in range(10):
            optimizer.zero_grad()
            output = model(x[:100])
            loss = criterion(output.squeeze(), y[:100])
            loss.backward()
            optimizer.step()
        
        # Benchmark
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.time()
        
        for _ in range(50):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.squeeze(), y)
            loss.backward()
            optimizer.step()
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        throughput = (50 * x.shape[0]) / elapsed_time
        
        print(f"Training throughput: {throughput:.2f} samples/sec")
        
        assert throughput > 0
        assert elapsed_time > 0


@pytest.mark.benchmark
class TestUncertaintyPerformance:
    """Benchmark uncertainty quantification performance."""
    
    def test_ensemble_prediction_performance(self, benchmark_data, device):
        """Benchmark ensemble prediction performance."""
        x, y = benchmark_data
        
        # Create ensemble of models
        ensemble_size = 5
        models = []
        
        for i in range(ensemble_size):
            model = torch.nn.Sequential(
                torch.nn.Linear(100, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1)
            ).to(device)
            models.append(model)
        
        # Benchmark ensemble prediction
        for model in models:
            model.eval()
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        start_time = time.time()
        
        predictions = []
        with torch.no_grad():
            for model in models:
                pred = model(x)
                predictions.append(pred)
        
        # Compute ensemble statistics
        predictions = torch.stack(predictions, dim=0)
        ensemble_mean = predictions.mean(dim=0)
        ensemble_std = predictions.std(dim=0)
        
        torch.cuda.synchronize() if device.type == "cuda" else None
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        throughput = x.shape[0] / elapsed_time
        
        print(f"Ensemble prediction throughput: {throughput:.2f} samples/sec")
        print(f"Ensemble size: {ensemble_size}")
        
        assert throughput > 0
        assert ensemble_mean.shape == (x.shape[0], 1)
        assert ensemble_std.shape == (x.shape[0], 1)
    
    def test_monte_carlo_sampling_performance(self, benchmark_data, mock_posterior):
        """Benchmark Monte Carlo sampling performance."""
        x, y = benchmark_data
        posterior = mock_posterior()
        
        # Benchmark sampling
        start_time = time.time()
        
        n_samples = 100
        samples = posterior.sample(x[:500], n_samples=n_samples)  # Smaller batch for sampling
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        samples_per_second = (500 * n_samples) / elapsed_time
        
        print(f"MC sampling rate: {samples_per_second:.2f} samples/sec")
        print(f"Number of posterior samples: {n_samples}")
        
        assert samples_per_second > 0
        assert samples.shape == (n_samples, 500, *x.shape[1:])


@pytest.mark.benchmark
class TestMemoryPerformance:
    """Benchmark memory usage and efficiency."""
    
    def test_memory_usage_scaling(self, device):
        """Test memory usage scaling with batch size."""
        if device.type == "cpu":
            pytest.skip("Memory benchmarks focus on GPU")
        
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1)
        ).to(device)
        
        batch_sizes = [32, 64, 128, 256, 512]
        memory_usage = []
        
        for batch_size in batch_sizes:
            torch.cuda.empty_cache()
            gc.collect()
            
            initial_memory = torch.cuda.memory_allocated(device)
            
            x = torch.randn(batch_size, 100, device=device)
            output = model(x)
            
            peak_memory = torch.cuda.max_memory_allocated(device)
            memory_used = (peak_memory - initial_memory) / 1024**2  # MB
            memory_usage.append(memory_used)
            
            print(f"Batch size {batch_size}: {memory_used:.2f} MB")
            
            # Cleanup
            del x, output
            torch.cuda.reset_peak_memory_stats(device)
        
        # Check that memory scales reasonably with batch size
        assert len(memory_usage) == len(batch_sizes)
        assert all(mem > 0 for mem in memory_usage)
    
    def test_gradient_memory_efficiency(self, device):
        """Test gradient computation memory efficiency."""
        if device.type == "cpu":
            pytest.skip("Memory benchmarks focus on GPU")
        
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        ).to(device)
        
        x = torch.randn(512, 100, device=device, requires_grad=True)
        
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device)
        
        # Forward pass
        output = model(x)
        after_forward = torch.cuda.memory_allocated(device)
        
        # Backward pass
        loss = output.sum()
        loss.backward()
        after_backward = torch.cuda.memory_allocated(device)
        
        forward_memory = (after_forward - initial_memory) / 1024**2
        backward_memory = (after_backward - after_forward) / 1024**2
        
        print(f"Forward pass memory: {forward_memory:.2f} MB")
        print(f"Backward pass memory: {backward_memory:.2f} MB")
        
        assert forward_memory > 0
        assert backward_memory > 0


@pytest.mark.benchmark 
class TestScalabilityBenchmarks:
    """Test scalability with problem size."""
    
    def test_input_dimension_scaling(self, device):
        """Test performance scaling with input dimension."""
        input_dims = [10, 50, 100, 200, 500]
        times = []
        
        for input_dim in input_dims:
            model = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1)
            ).to(device)
            
            x = torch.randn(256, input_dim, device=device)
            
            # Warmup
            for _ in range(10):
                _ = model(x)
            
            # Benchmark
            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = time.time()
            
            for _ in range(100):
                output = model(x)
            
            torch.cuda.synchronize() if device.type == "cuda" else None
            end_time = time.time()
            
            elapsed = end_time - start_time
            times.append(elapsed)
            
            print(f"Input dim {input_dim}: {elapsed:.4f} sec")
        
        # Check that times scale reasonably
        assert len(times) == len(input_dims)
        assert all(t > 0 for t in times)
    
    def test_model_depth_scaling(self, device):
        """Test performance scaling with model depth."""
        depths = [2, 4, 6, 8, 10]
        times = []
        
        for depth in depths:
            layers = []
            layers.append(torch.nn.Linear(100, 128))
            
            for _ in range(depth - 2):
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Linear(128, 128))
            
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Linear(128, 1))
            
            model = torch.nn.Sequential(*layers).to(device)
            x = torch.randn(256, 100, device=device)
            
            # Warmup
            for _ in range(10):
                _ = model(x)
            
            # Benchmark
            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = time.time()
            
            for _ in range(50):
                output = model(x)
            
            torch.cuda.synchronize() if device.type == "cuda" else None
            end_time = time.time()
            
            elapsed = end_time - start_time
            times.append(elapsed)
            
            print(f"Depth {depth}: {elapsed:.4f} sec")
        
        assert len(times) == len(depths)
        assert all(t > 0 for t in times)


@pytest.mark.benchmark
class TestNumericalPrecisionBenchmarks:
    """Test numerical precision and stability."""
    
    def test_mixed_precision_performance(self, device):
        """Test mixed precision training performance."""
        if device.type == "cpu":
            pytest.skip("Mixed precision requires GPU")
        
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1)
        ).to(device)
        
        x = torch.randn(512, 100, device=device)
        y = torch.randn(512, 1, device=device)
        
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler()
        
        # Benchmark with mixed precision
        start_time = time.time()
        
        for _ in range(50):
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                output = model(x)
                loss = criterion(output, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        end_time = time.time()
        mixed_precision_time = end_time - start_time
        
        # Benchmark without mixed precision
        start_time = time.time()
        
        for _ in range(50):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
        
        end_time = time.time()
        full_precision_time = end_time - start_time
        
        speedup = full_precision_time / mixed_precision_time
        
        print(f"Mixed precision time: {mixed_precision_time:.4f} sec")
        print(f"Full precision time: {full_precision_time:.4f} sec")
        print(f"Speedup: {speedup:.2f}x")
        
        assert mixed_precision_time > 0
        assert full_precision_time > 0
    
    def test_numerical_stability_under_load(self, device):
        """Test numerical stability under computational load."""
        model = torch.nn.Sequential(
            torch.nn.Linear(50, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 1)
        ).to(device)
        
        # Test with various input scales
        scales = [1e-3, 1e-1, 1.0, 1e1, 1e3]
        
        for scale in scales:
            x = torch.randn(100, 50, device=device) * scale
            
            with torch.no_grad():
                output = model(x)
                
                # Check for numerical issues
                assert torch.isfinite(output).all(), f"Non-finite outputs at scale {scale}"
                assert not torch.isnan(output).any(), f"NaN outputs at scale {scale}"
                assert not torch.isinf(output).any(), f"Inf outputs at scale {scale}"
                
                print(f"Scale {scale}: output range [{output.min():.2e}, {output.max():.2e}]")