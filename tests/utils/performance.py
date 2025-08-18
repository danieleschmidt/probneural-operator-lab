"""Performance testing utilities."""

import time
import torch
import psutil
import os
from typing import Callable, Dict, Any, Optional, List, Tuple
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    execution_time: float  # seconds
    peak_memory: float     # MB
    avg_memory: float      # MB
    cpu_percent: float     # percentage
    gpu_memory: Optional[float] = None  # MB


class PerformanceProfiler:
    """Context manager for profiling performance."""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.process = psutil.Process(os.getpid())
        self.memory_samples = []
        self.cpu_samples = []
        
    def __enter__(self):
        self.start_time = time.time()
        self._start_monitoring()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        execution_time = time.time() - self.start_time
        self._stop_monitoring()
        
        # Calculate metrics
        peak_memory = max(self.memory_samples) if self.memory_samples else 0
        avg_memory = sum(self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0
        avg_cpu = sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0
        
        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            torch.cuda.reset_peak_memory_stats()
        
        self.metrics = PerformanceMetrics(
            execution_time=execution_time,
            peak_memory=peak_memory,
            avg_memory=avg_memory,
            cpu_percent=avg_cpu,
            gpu_memory=gpu_memory
        )
    
    def _start_monitoring(self):
        """Start resource monitoring."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def _stop_monitoring(self):
        """Stop resource monitoring and collect final measurements."""
        mem_info = self.process.memory_info()
        self.memory_samples.append(mem_info.rss / 1024 / 1024)  # MB
        self.cpu_samples.append(self.process.cpu_percent())


@contextmanager
def time_it(name: str = "operation"):
    """Simple timing context manager."""
    start = time.time()
    print(f"Starting {name}...")
    try:
        yield
    finally:
        end = time.time()
        print(f"{name} completed in {end - start:.4f}s")


def benchmark_function(
    func: Callable, 
    *args, 
    n_runs: int = 10, 
    warmup_runs: int = 3,
    **kwargs
) -> Dict[str, float]:
    """Benchmark a function with multiple runs."""
    
    # Warmup runs
    for _ in range(warmup_runs):
        func(*args, **kwargs)
    
    # Actual benchmark runs
    times = []
    memory_usage = []
    
    for _ in range(n_runs):
        with PerformanceProfiler() as profiler:
            result = func(*args, **kwargs)
        
        times.append(profiler.metrics.execution_time)
        memory_usage.append(profiler.metrics.peak_memory)
    
    return {
        "mean_time": sum(times) / len(times),
        "std_time": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
        "min_time": min(times),
        "max_time": max(times),
        "mean_memory": sum(memory_usage) / len(memory_usage),
        "peak_memory": max(memory_usage),
    }


def profile_model_inference(model: torch.nn.Module, input_data: torch.Tensor, n_runs: int = 100) -> Dict[str, Any]:
    """Profile model inference performance."""
    model.eval()
    device = next(model.parameters()).device
    input_data = input_data.to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    
    for _ in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        
        with torch.no_grad():
            output = model(input_data)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end = time.time()
        times.append(end - start)
    
    batch_size = input_data.shape[0]
    
    return {
        "mean_batch_time": sum(times) / len(times),
        "mean_sample_time": sum(times) / len(times) / batch_size,
        "throughput_samples_per_sec": batch_size * len(times) / sum(times),
        "std_time": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
        "min_time": min(times),
        "max_time": max(times),
    }


def profile_training_step(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    input_data: torch.Tensor,
    target_data: torch.Tensor,
    n_runs: int = 50
) -> Dict[str, Any]:
    """Profile training step performance."""
    
    device = next(model.parameters()).device
    input_data = input_data.to(device)
    target_data = target_data.to(device)
    
    model.train()
    
    # Warmup
    for _ in range(5):
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target_data)
        loss.backward()
        optimizer.step()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    forward_times = []
    backward_times = []
    total_times = []
    
    for _ in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        total_start = time.time()
        
        # Forward pass
        forward_start = time.time()
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target_data)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        forward_end = time.time()
        
        # Backward pass
        backward_start = time.time()
        loss.backward()
        optimizer.step()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        backward_end = time.time()
        
        total_end = time.time()
        
        forward_times.append(forward_end - forward_start)
        backward_times.append(backward_end - backward_start)
        total_times.append(total_end - total_start)
    
    return {
        "mean_forward_time": sum(forward_times) / len(forward_times),
        "mean_backward_time": sum(backward_times) / len(backward_times),
        "mean_total_time": sum(total_times) / len(total_times),
        "forward_backward_ratio": sum(forward_times) / sum(backward_times),
    }


def measure_memory_usage(func: Callable) -> Tuple[Any, float]:
    """Measure peak memory usage of a function."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        result = func()
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        return result, peak_memory
    else:
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        result = func()
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        return result, mem_after - mem_before


def profile_memory_efficiency(model: torch.nn.Module, input_sizes: List[Tuple[int, ...]], device: torch.device) -> Dict[int, Dict[str, float]]:
    """Profile memory usage for different input sizes."""
    results = {}
    
    for size in input_sizes:
        batch_size = size[0]
        input_data = torch.randn(*size, device=device)
        
        def forward_pass():
            with torch.no_grad():
                return model(input_data)
        
        _, memory_usage = measure_memory_usage(forward_pass)
        
        results[batch_size] = {
            "total_memory_mb": memory_usage,
            "memory_per_sample_mb": memory_usage / batch_size if batch_size > 0 else 0,
            "input_size": size,
        }
    
    return results


class PerformanceAssertion:
    """Assert performance characteristics."""
    
    @staticmethod
    def assert_faster_than(actual_time: float, max_time: float, msg: Optional[str] = None):
        """Assert operation completes faster than threshold."""
        if msg is None:
            msg = f"Operation took {actual_time:.4f}s, expected < {max_time:.4f}s"
        assert actual_time < max_time, msg
    
    @staticmethod
    def assert_memory_usage_below(actual_memory: float, max_memory: float, msg: Optional[str] = None):
        """Assert memory usage is below threshold."""
        if msg is None:
            msg = f"Memory usage {actual_memory:.1f}MB, expected < {max_memory:.1f}MB"
        assert actual_memory < max_memory, msg
    
    @staticmethod
    def assert_throughput_above(actual_throughput: float, min_throughput: float, msg: Optional[str] = None):
        """Assert throughput is above threshold."""
        if msg is None:
            msg = f"Throughput {actual_throughput:.1f} samples/s, expected > {min_throughput:.1f}"
        assert actual_throughput > min_throughput, msg


def performance_test(
    max_time: Optional[float] = None,
    max_memory_mb: Optional[float] = None,
    min_throughput: Optional[float] = None
):
    """Decorator for performance testing."""
    def decorator(test_func):
        @wraps(test_func)
        def wrapper(*args, **kwargs):
            with PerformanceProfiler() as profiler:
                result = test_func(*args, **kwargs)
            
            metrics = profiler.metrics
            
            if max_time is not None:
                PerformanceAssertion.assert_faster_than(metrics.execution_time, max_time)
            
            if max_memory_mb is not None:
                PerformanceAssertion.assert_memory_usage_below(metrics.peak_memory, max_memory_mb)
            
            return result
        
        return wrapper
    return decorator


# Benchmarking utilities
def create_benchmark_suite(models: Dict[str, torch.nn.Module], test_cases: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Create comprehensive benchmark suite."""
    results = {}
    
    for model_name, model in models.items():
        results[model_name] = {}
        
        for test_name, input_data in test_cases.items():
            perf_metrics = profile_model_inference(model, input_data)
            results[model_name][test_name] = perf_metrics
    
    return results


def compare_performance(
    baseline: Dict[str, float], 
    current: Dict[str, float], 
    tolerance: float = 0.1
) -> Dict[str, Dict[str, Any]]:
    """Compare performance metrics against baseline."""
    comparison = {}
    
    for metric in baseline:
        if metric in current:
            baseline_val = baseline[metric]
            current_val = current[metric]
            
            if baseline_val != 0:
                relative_change = (current_val - baseline_val) / baseline_val
            else:
                relative_change = float('inf') if current_val != 0 else 0
            
            is_regression = abs(relative_change) > tolerance and relative_change > 0
            
            comparison[metric] = {
                "baseline": baseline_val,
                "current": current_val,
                "relative_change": relative_change,
                "is_regression": is_regression,
                "improvement": relative_change < 0
            }
    
    return comparison