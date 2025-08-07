"""Performance benchmark tests."""

import pytest
import torch
import time
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import psutil
import os

from probneural_operator.models import ProbabilisticFNO
from probneural_operator.utils import create_monitoring_suite


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Benchmark tests for performance regression detection."""
    
    def test_fno_forward_pass_performance(self, performance_baseline):
        """Benchmark FNO forward pass performance."""
        # Create models of different sizes
        configs = [
            {"modes": 8, "width": 32, "depth": 2},
            {"modes": 16, "width": 64, "depth": 4},
            {"modes": 32, "width": 128, "depth": 6},
        ]
        
        batch_sizes = [1, 8, 32]
        spatial_sizes = [64, 128, 256]
        
        results = {}
        
        for config in configs:
            model_key = f"modes_{config['modes']}_width_{config['width']}_depth_{config['depth']}"
            results[model_key] = {}
            
            model = ProbabilisticFNO(
                input_dim=1, output_dim=1, spatial_dim=1, **config
            )
            model.eval()
            
            for batch_size in batch_sizes:
                for spatial_size in spatial_sizes:
                    test_key = f"batch_{batch_size}_spatial_{spatial_size}"
                    
                    # Create test input
                    test_input = torch.randn(batch_size, 1, spatial_size)
                    
                    # Warmup
                    with torch.no_grad():
                        for _ in range(5):
                            _ = model(test_input)
                    
                    # Benchmark
                    times = []
                    with torch.no_grad():
                        for _ in range(20):
                            start_time = time.perf_counter()
                            output = model(test_input)
                            end_time = time.perf_counter()
                            times.append((end_time - start_time) * 1000)  # ms
                    
                    avg_time = np.mean(times)
                    std_time = np.std(times)
                    
                    results[model_key][test_key] = {
                        'avg_time_ms': avg_time,
                        'std_time_ms': std_time,
                        'min_time_ms': np.min(times),
                        'max_time_ms': np.max(times)
                    }
                    
                    # Basic performance check (should complete in reasonable time)
                    assert avg_time < 1000.0, f"Forward pass too slow: {avg_time:.2f}ms"
                    
                    # Variance check (should be relatively consistent)
                    cv = std_time / avg_time
                    assert cv < 0.5, f"Forward pass timing too variable: CV={cv:.2f}"
        
        # Store results for comparison
        print("\n=== FNO Forward Pass Performance ===")
        for model_key, model_results in results.items():
            print(f"\nModel: {model_key}")
            for test_key, timing in model_results.items():
                print(f"  {test_key}: {timing['avg_time_ms']:.2f}±{timing['std_time_ms']:.2f}ms")
    
    def test_fno_backward_pass_performance(self):
        """Benchmark FNO backward pass performance."""
        model = ProbabilisticFNO(
            input_dim=1, output_dim=1, modes=16, width=64, depth=4, spatial_dim=1
        )
        
        test_input = torch.randn(8, 1, 128, requires_grad=True)
        target = torch.randn(8, 1, 128)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Warmup
        for _ in range(5):
            optimizer.zero_grad()
            output = model(test_input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Benchmark backward pass
        times = []
        for _ in range(15):
            optimizer.zero_grad()
            
            forward_start = time.perf_counter()
            output = model(test_input)
            loss = criterion(output, target)
            forward_end = time.perf_counter()
            
            backward_start = time.perf_counter()
            loss.backward()
            backward_end = time.perf_counter()
            
            optimizer.step()
            
            forward_time = (forward_end - forward_start) * 1000
            backward_time = (backward_end - backward_start) * 1000
            
            times.append({
                'forward_ms': forward_time,
                'backward_ms': backward_time,
                'total_ms': forward_time + backward_time
            })
        
        avg_forward = np.mean([t['forward_ms'] for t in times])
        avg_backward = np.mean([t['backward_ms'] for t in times])
        avg_total = np.mean([t['total_ms'] for t in times])
        
        print(f"\n=== FNO Training Performance ===")
        print(f"Forward pass: {avg_forward:.2f}ms")
        print(f"Backward pass: {avg_backward:.2f}ms")
        print(f"Total step: {avg_total:.2f}ms")
        
        # Performance assertions
        assert avg_forward < 200.0, f"Forward pass too slow: {avg_forward:.2f}ms"
        assert avg_backward < 500.0, f"Backward pass too slow: {avg_backward:.2f}ms"
        assert avg_total < 600.0, f"Training step too slow: {avg_total:.2f}ms"
    
    def test_memory_efficiency(self):
        """Benchmark memory usage efficiency."""
        import gc
        
        process = psutil.Process(os.getpid())
        
        # Record baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024**2  # MB
        
        model = ProbabilisticFNO(
            input_dim=1, output_dim=1, modes=16, width=64, depth=4, spatial_dim=1
        )
        
        # Memory after model creation
        gc.collect()
        model_memory = process.memory_info().rss / 1024**2  # MB
        model_overhead = model_memory - baseline_memory
        
        # Create data and train
        inputs = torch.randn(64, 1, 128)
        targets = torch.randn(64, 1, 128)
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=8)
        
        # Memory during training
        gc.collect()
        pre_training_memory = process.memory_info().rss / 1024**2  # MB
        
        history = model.fit(dataloader, epochs=3, lr=1e-3, device="cpu")
        
        gc.collect()
        post_training_memory = process.memory_info().rss / 1024**2  # MB
        training_overhead = post_training_memory - pre_training_memory
        
        # Posterior fitting
        model.fit_posterior(dataloader)
        
        gc.collect()
        post_posterior_memory = process.memory_info().rss / 1024**2  # MB
        posterior_overhead = post_posterior_memory - post_training_memory
        
        print(f"\n=== Memory Usage Analysis ===")
        print(f"Baseline memory: {baseline_memory:.1f} MB")
        print(f"Model overhead: {model_overhead:.1f} MB")
        print(f"Training overhead: {training_overhead:.1f} MB")
        print(f"Posterior overhead: {posterior_overhead:.1f} MB")
        print(f"Total memory: {post_posterior_memory:.1f} MB")
        
        # Memory efficiency assertions
        assert model_overhead < 500.0, f"Model memory overhead too high: {model_overhead:.1f}MB"
        assert training_overhead < 1000.0, f"Training memory overhead too high: {training_overhead:.1f}MB"
        assert posterior_overhead < 500.0, f"Posterior memory overhead too high: {posterior_overhead:.1f}MB"
        
        # Cleanup
        del model, inputs, targets, dataset, dataloader
        gc.collect()
    
    def test_uncertainty_quantification_performance(self):
        """Benchmark uncertainty quantification performance."""
        model = ProbabilisticFNO(
            input_dim=1, output_dim=1, modes=8, width=32, depth=2, spatial_dim=1
        )
        
        # Quick training
        inputs = torch.randn(32, 1, 64)
        targets = torch.randn(32, 1, 64)
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=8)
        
        model.fit(dataloader, epochs=2, lr=1e-3, device="cpu")
        model.fit_posterior(dataloader)
        
        # Test different numbers of samples
        test_input = torch.randn(4, 1, 64)
        sample_counts = [1, 5, 10, 25, 50, 100]
        
        results = {}
        
        for num_samples in sample_counts:
            times = []
            
            # Warmup
            for _ in range(3):
                _, _ = model.predict_with_uncertainty(test_input, num_samples=num_samples)
            
            # Benchmark
            for _ in range(10):
                start_time = time.perf_counter()
                mean, std = model.predict_with_uncertainty(test_input, num_samples=num_samples)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # ms
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            results[num_samples] = {
                'avg_time_ms': avg_time,
                'std_time_ms': std_time,
                'time_per_sample_ms': avg_time / num_samples
            }
            
            # Performance check - should scale reasonably with sample count
            expected_max_time = num_samples * 10.0  # 10ms per sample max
            assert avg_time < expected_max_time, f"UQ too slow: {avg_time:.2f}ms for {num_samples} samples"
        
        print(f"\n=== Uncertainty Quantification Performance ===")
        for num_samples, timing in results.items():
            print(f"Samples: {num_samples:3d}, Time: {timing['avg_time_ms']:6.2f}ms, "
                  f"Per sample: {timing['time_per_sample_ms']:5.2f}ms")
        
        # Check scaling properties
        time_1_sample = results[1]['avg_time_ms']
        time_100_samples = results[100]['avg_time_ms']
        scaling_factor = time_100_samples / time_1_sample
        
        # Should scale sublinearly due to batch processing
        assert scaling_factor < 80, f"UQ scaling too poor: {scaling_factor:.1f}x for 100x samples"
    
    @pytest.mark.slow
    def test_large_scale_performance(self):
        """Benchmark performance on large-scale problems."""
        # Test with larger models and data
        large_configs = [
            {"batch_size": 1, "spatial_size": 512, "modes": 32, "width": 128},
            {"batch_size": 4, "spatial_size": 256, "modes": 16, "width": 64},
            {"batch_size": 16, "spatial_size": 128, "modes": 8, "width": 32},
        ]
        
        for config in large_configs:
            print(f"\n=== Testing config: {config} ===")
            
            model = ProbabilisticFNO(
                input_dim=1, output_dim=1, spatial_dim=1,
                modes=config["modes"], width=config["width"], depth=3
            )
            
            # Create large data
            test_input = torch.randn(config["batch_size"], 1, config["spatial_size"])
            
            # Measure forward pass
            model.eval()
            start_time = time.perf_counter()
            
            with torch.no_grad():
                output = model(test_input)
            
            forward_time = (time.perf_counter() - start_time) * 1000  # ms
            
            # Calculate memory footprint
            input_memory = test_input.numel() * test_input.element_size() / 1024**2  # MB
            output_memory = output.numel() * output.element_size() / 1024**2  # MB
            
            print(f"Forward time: {forward_time:.2f}ms")
            print(f"Input memory: {input_memory:.1f}MB")
            print(f"Output memory: {output_memory:.1f}MB")
            
            # Performance checks
            assert forward_time < 2000.0, f"Large scale forward too slow: {forward_time:.2f}ms"
            assert output.shape == test_input.shape, "Output shape mismatch"
            assert torch.all(torch.isfinite(output)), "Output contains non-finite values"
    
    def test_concurrent_model_performance(self):
        """Test performance with multiple concurrent models."""
        import threading
        import queue
        
        num_models = 4
        num_operations = 5
        results_queue = queue.Queue()
        
        def worker_thread(thread_id, results_queue):
            """Worker thread for concurrent testing."""
            thread_results = []
            
            model = ProbabilisticFNO(
                input_dim=1, output_dim=1, modes=8, width=32, depth=2, spatial_dim=1
            )
            
            test_input = torch.randn(4, 1, 64)
            
            for i in range(num_operations):
                start_time = time.perf_counter()
                
                with torch.no_grad():
                    output = model(test_input)
                
                end_time = time.perf_counter()
                operation_time = (end_time - start_time) * 1000  # ms
                
                thread_results.append({
                    'thread_id': thread_id,
                    'operation': i,
                    'time_ms': operation_time
                })
            
            results_queue.put(thread_results)
        
        # Start concurrent threads
        threads = []
        start_time = time.perf_counter()
        
        for i in range(num_models):
            thread = threading.Thread(target=worker_thread, args=(i, results_queue))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        total_time = (time.perf_counter() - start_time) * 1000  # ms
        
        # Collect results
        all_results = []
        while not results_queue.empty():
            all_results.extend(results_queue.get())
        
        # Analyze results
        operation_times = [r['time_ms'] for r in all_results]
        avg_operation_time = np.mean(operation_times)
        max_operation_time = np.max(operation_times)
        
        print(f"\n=== Concurrent Performance ===")
        print(f"Total time: {total_time:.2f}ms")
        print(f"Avg operation time: {avg_operation_time:.2f}ms")
        print(f"Max operation time: {max_operation_time:.2f}ms")
        print(f"Total operations: {len(all_results)}")
        
        # Performance assertions
        assert len(all_results) == num_models * num_operations, "Missing operations"
        assert avg_operation_time < 200.0, f"Concurrent operations too slow: {avg_operation_time:.2f}ms"
        assert max_operation_time < 500.0, f"Slowest operation too slow: {max_operation_time:.2f}ms"


@pytest.mark.benchmark
@pytest.mark.gpu
class TestGPUPerformance:
    """GPU-specific performance benchmarks."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_vs_cpu_performance(self):
        """Compare GPU vs CPU performance."""
        configs = [
            {"modes": 8, "width": 32, "depth": 2},
            {"modes": 16, "width": 64, "depth": 4},
        ]
        
        batch_sizes = [8, 32]
        spatial_size = 128
        
        for config in configs:
            for batch_size in batch_sizes:
                print(f"\n=== Config: {config}, Batch: {batch_size} ===")
                
                # CPU model
                cpu_model = ProbabilisticFNO(
                    input_dim=1, output_dim=1, spatial_dim=1, **config
                )
                cpu_input = torch.randn(batch_size, 1, spatial_size)
                
                # GPU model
                gpu_model = ProbabilisticFNO(
                    input_dim=1, output_dim=1, spatial_dim=1, **config
                ).cuda()
                gpu_input = torch.randn(batch_size, 1, spatial_size).cuda()
                
                # Warmup
                for _ in range(5):
                    with torch.no_grad():
                        _ = cpu_model(cpu_input)
                        _ = gpu_model(gpu_input)
                
                # Benchmark CPU
                cpu_times = []
                for _ in range(10):
                    start_time = time.perf_counter()
                    with torch.no_grad():
                        _ = cpu_model(cpu_input)
                    cpu_times.append((time.perf_counter() - start_time) * 1000)
                
                # Benchmark GPU
                gpu_times = []
                for _ in range(10):
                    torch.cuda.synchronize()
                    start_time = time.perf_counter()
                    with torch.no_grad():
                        _ = gpu_model(gpu_input)
                    torch.cuda.synchronize()
                    gpu_times.append((time.perf_counter() - start_time) * 1000)
                
                cpu_avg = np.mean(cpu_times)
                gpu_avg = np.mean(gpu_times)
                speedup = cpu_avg / gpu_avg
                
                print(f"CPU time: {cpu_avg:.2f}ms")
                print(f"GPU time: {gpu_avg:.2f}ms")
                print(f"GPU speedup: {speedup:.2f}x")
                
                # GPU should be faster for larger models/batches
                if config["width"] >= 64 or batch_size >= 16:
                    assert speedup > 1.0, f"GPU not faster than CPU: {speedup:.2f}x"
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_efficiency(self):
        """Test GPU memory usage efficiency."""
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        model = ProbabilisticFNO(
            input_dim=1, output_dim=1, modes=16, width=64, depth=4, spatial_dim=1
        ).cuda()
        
        model_memory = torch.cuda.memory_allocated() - initial_memory
        
        # Test with different batch sizes
        batch_sizes = [1, 4, 16, 64]
        spatial_size = 128
        
        memory_usage = {}
        
        for batch_size in batch_sizes:
            torch.cuda.empty_cache()
            pre_forward_memory = torch.cuda.memory_allocated()
            
            test_input = torch.randn(batch_size, 1, spatial_size).cuda()
            
            with torch.no_grad():
                output = model(test_input)
            
            peak_memory = torch.cuda.max_memory_allocated()
            forward_memory = peak_memory - pre_forward_memory
            
            memory_usage[batch_size] = {
                'forward_memory_mb': forward_memory / 1024**2,
                'per_sample_mb': forward_memory / batch_size / 1024**2
            }
            
            torch.cuda.reset_peak_memory_stats()
            del test_input, output
        
        print(f"\n=== GPU Memory Usage ===")
        print(f"Model memory: {model_memory / 1024**2:.1f}MB")
        
        for batch_size, usage in memory_usage.items():
            print(f"Batch {batch_size:2d}: {usage['forward_memory_mb']:6.1f}MB total, "
                  f"{usage['per_sample_mb']:5.1f}MB per sample")
        
        # Memory efficiency checks
        assert model_memory / 1024**2 < 1000, f"Model memory too high: {model_memory / 1024**2:.1f}MB"
        
        # Check memory scaling
        per_sample_memories = [usage['per_sample_mb'] for usage in memory_usage.values()]
        memory_variance = np.var(per_sample_memories)
        
        # Per-sample memory should be relatively consistent
        assert memory_variance < 1.0, f"Per-sample memory too variable: {memory_variance:.2f}"


@pytest.mark.benchmark
class TestScalabilityBenchmarks:
    """Scalability benchmarks for different problem sizes."""
    
    def test_spatial_resolution_scaling(self):
        """Test performance scaling with spatial resolution."""
        model = ProbabilisticFNO(
            input_dim=1, output_dim=1, modes=16, width=64, depth=4, spatial_dim=1
        )
        
        spatial_sizes = [32, 64, 128, 256, 512]
        batch_size = 4
        
        results = {}
        
        for spatial_size in spatial_sizes:
            test_input = torch.randn(batch_size, 1, spatial_size)
            
            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = model(test_input)
            
            # Benchmark
            times = []
            for _ in range(10):
                start_time = time.perf_counter()
                with torch.no_grad():
                    output = model(test_input)
                times.append((time.perf_counter() - start_time) * 1000)
            
            avg_time = np.mean(times)
            memory_mb = test_input.numel() * test_input.element_size() / 1024**2
            
            results[spatial_size] = {
                'avg_time_ms': avg_time,
                'memory_mb': memory_mb,
                'throughput_samples_per_sec': 1000 * batch_size / avg_time
            }
        
        print(f"\n=== Spatial Resolution Scaling ===")
        for size, result in results.items():
            print(f"Size {size:3d}: {result['avg_time_ms']:6.2f}ms, "
                  f"{result['memory_mb']:5.1f}MB, "
                  f"{result['throughput_samples_per_sec']:6.1f} samples/sec")
        
        # Check scaling properties
        time_32 = results[32]['avg_time_ms']
        time_512 = results[512]['avg_time_ms']
        size_ratio = 512 / 32  # 16x
        time_ratio = time_512 / time_32
        
        # Time should scale better than quadratically
        assert time_ratio < size_ratio**1.5, f"Poor spatial scaling: {time_ratio:.1f}x time for {size_ratio:.1f}x size"
    
    def test_batch_size_scaling(self):
        """Test performance scaling with batch size."""
        model = ProbabilisticFNO(
            input_dim=1, output_dim=1, modes=16, width=64, depth=4, spatial_dim=1
        )
        
        batch_sizes = [1, 2, 4, 8, 16, 32]
        spatial_size = 128
        
        results = {}
        
        for batch_size in batch_sizes:
            test_input = torch.randn(batch_size, 1, spatial_size)
            
            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    _ = model(test_input)
            
            # Benchmark
            times = []
            for _ in range(10):
                start_time = time.perf_counter()
                with torch.no_grad():
                    output = model(test_input)
                times.append((time.perf_counter() - start_time) * 1000)
            
            avg_time = np.mean(times)
            
            results[batch_size] = {
                'avg_time_ms': avg_time,
                'time_per_sample_ms': avg_time / batch_size,
                'throughput_samples_per_sec': 1000 * batch_size / avg_time
            }
        
        print(f"\n=== Batch Size Scaling ===")
        for batch_size, result in results.items():
            print(f"Batch {batch_size:2d}: {result['avg_time_ms']:6.2f}ms total, "
                  f"{result['time_per_sample_ms']:5.2f}ms per sample, "
                  f"{result['throughput_samples_per_sec']:6.1f} samples/sec")
        
        # Check batch processing efficiency
        time_per_sample_1 = results[1]['time_per_sample_ms']
        time_per_sample_32 = results[32]['time_per_sample_ms']
        efficiency_gain = time_per_sample_1 / time_per_sample_32
        
        # Larger batches should be more efficient
        assert efficiency_gain > 2.0, f"Poor batch efficiency: {efficiency_gain:.1f}x improvement"