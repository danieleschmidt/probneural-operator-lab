"""Performance monitoring and profiling utilities."""

import time
import torch
import psutil
import gc
from typing import Dict, Any, Optional, List, Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
import threading
import numpy as np


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    execution_time: float = 0.0
    memory_used: float = 0.0
    gpu_memory_used: float = 0.0
    cpu_utilization: float = 0.0
    throughput: float = 0.0
    batch_size: int = 0
    model_parameters: int = 0
    flops_estimate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'execution_time_ms': self.execution_time * 1000,
            'memory_used_mb': self.memory_used,
            'gpu_memory_used_mb': self.gpu_memory_used,
            'cpu_utilization_percent': self.cpu_utilization,
            'throughput_samples_per_sec': self.throughput,
            'batch_size': self.batch_size,
            'model_parameters': self.model_parameters,
            'estimated_gflops': self.flops_estimate / 1e9
        }


class PerformanceProfiler:
    """Comprehensive performance profiler for neural operators."""
    
    def __init__(self, enable_gpu_monitoring: bool = True):
        """Initialize profiler.
        
        Args:
            enable_gpu_monitoring: Whether to monitor GPU metrics
        """
        self.enable_gpu_monitoring = enable_gpu_monitoring and torch.cuda.is_available()
        self.metrics_history: List[PerformanceMetrics] = []
        self._start_time: Optional[float] = None
        self._start_memory: Optional[float] = None
        self._start_gpu_memory: Optional[float] = None
    
    @contextmanager
    def profile(self, operation_name: str = "operation"):
        """Context manager for profiling operations.
        
        Args:
            operation_name: Name of the operation being profiled
            
        Yields:
            PerformanceMetrics object that gets populated during execution
        """
        metrics = PerformanceMetrics()
        
        # Pre-execution cleanup and measurements
        gc.collect()
        if self.enable_gpu_monitoring:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        start_gpu_memory = self._get_gpu_memory_usage()
        
        try:
            yield metrics
        finally:
            # Post-execution measurements
            if self.enable_gpu_monitoring:
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            end_gpu_memory = self._get_gpu_memory_usage()
            
            # Calculate metrics
            metrics.execution_time = end_time - start_time
            metrics.memory_used = max(0, end_memory - start_memory)
            metrics.gpu_memory_used = max(0, end_gpu_memory - start_gpu_memory)
            metrics.cpu_utilization = psutil.cpu_percent()
            
            self.metrics_history.append(metrics)
    
    def profile_model_inference(self, model: torch.nn.Module, 
                              input_data: torch.Tensor,
                              num_warmup: int = 5,
                              num_iterations: int = 20) -> PerformanceMetrics:
        """Profile model inference performance.
        
        Args:
            model: PyTorch model to profile
            input_data: Input tensor or tuple of tensors
            num_warmup: Number of warmup iterations
            num_iterations: Number of timed iterations
            
        Returns:
            PerformanceMetrics with inference statistics
        """
        model.eval()
        device = next(model.parameters()).device
        
        # Move input data to model device
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.to(device)
        elif isinstance(input_data, (tuple, list)):
            input_data = [x.to(device) if isinstance(x, torch.Tensor) else x 
                         for x in input_data]
        
        # Warmup
        with torch.no_grad():
            for _ in range(num_warmup):
                if isinstance(input_data, (tuple, list)):
                    _ = model(*input_data)
                else:
                    _ = model(input_data)
        
        # Timed inference
        with self.profile("model_inference") as metrics:
            with torch.no_grad():
                times = []
                
                for _ in range(num_iterations):
                    if self.enable_gpu_monitoring:
                        torch.cuda.synchronize()
                    
                    start = time.perf_counter()
                    
                    if isinstance(input_data, (tuple, list)):
                        output = model(*input_data)
                    else:
                        output = model(input_data)
                    
                    if self.enable_gpu_monitoring:
                        torch.cuda.synchronize()
                    
                    end = time.perf_counter()
                    times.append(end - start)
                
                # Calculate statistics
                avg_time = np.mean(times)
                batch_size = input_data.shape[0] if isinstance(input_data, torch.Tensor) else input_data[0].shape[0]
                
                metrics.execution_time = avg_time
                metrics.throughput = batch_size / avg_time
                metrics.batch_size = batch_size
                metrics.model_parameters = sum(p.numel() for p in model.parameters())
                
                # Estimate FLOPs (rough approximation)
                metrics.flops_estimate = self._estimate_flops(model, input_data)
        
        return metrics
    
    def profile_training_step(self, model: torch.nn.Module,
                            optimizer: torch.optim.Optimizer,
                            criterion: torch.nn.Module,
                            input_data: torch.Tensor,
                            target_data: torch.Tensor) -> PerformanceMetrics:
        """Profile a complete training step.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            criterion: Loss function
            input_data: Input batch
            target_data: Target batch
            
        Returns:
            PerformanceMetrics for the training step
        """
        model.train()
        device = next(model.parameters()).device
        
        input_data = input_data.to(device)
        target_data = target_data.to(device)
        
        with self.profile("training_step") as metrics:
            # Forward pass
            optimizer.zero_grad()
            
            if hasattr(model, 'forward') and len(input_data.shape) > 2:
                # Handle different model interfaces
                if 'FNO' in model.__class__.__name__:
                    if input_data.ndim == 2:  # Add channel dimension
                        input_data = input_data.unsqueeze(1)
                    output = model(input_data)
                    if output.ndim > target_data.ndim:
                        output = output.squeeze(1)
                else:
                    output = model(input_data)
            else:
                output = model(input_data)
            
            # Loss computation
            loss = criterion(output, target_data)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Metrics
            metrics.batch_size = input_data.shape[0]
            metrics.model_parameters = sum(p.numel() for p in model.parameters())
        
        return metrics
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if not self.enable_gpu_monitoring:
            return 0.0
        
        return torch.cuda.memory_allocated() / 1024 / 1024
    
    def _estimate_flops(self, model: torch.nn.Module, input_data) -> float:
        """Rough FLOP estimation for neural operators."""
        if isinstance(input_data, (tuple, list)):
            input_size = input_data[0].numel()
        else:
            input_size = input_data.numel()
        
        param_count = sum(p.numel() for p in model.parameters())
        
        # Rough estimate: 2 FLOPs per parameter per sample
        return 2 * param_count * input_size
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics from all profiled operations."""
        if not self.metrics_history:
            return {}
        
        execution_times = [m.execution_time for m in self.metrics_history]
        memory_usage = [m.memory_used for m in self.metrics_history]
        gpu_memory_usage = [m.gpu_memory_used for m in self.metrics_history]
        throughputs = [m.throughput for m in self.metrics_history if m.throughput > 0]
        
        stats = {
            'total_operations': len(self.metrics_history),
            'avg_execution_time_ms': np.mean(execution_times) * 1000,
            'std_execution_time_ms': np.std(execution_times) * 1000,
            'avg_memory_used_mb': np.mean(memory_usage),
            'peak_memory_used_mb': np.max(memory_usage),
            'avg_gpu_memory_used_mb': np.mean(gpu_memory_usage),
            'peak_gpu_memory_used_mb': np.max(gpu_memory_usage),
        }
        
        if throughputs:
            stats.update({
                'avg_throughput_samples_per_sec': np.mean(throughputs),
                'peak_throughput_samples_per_sec': np.max(throughputs)
            })
        
        return stats
    
    def reset(self):
        """Reset metrics history."""
        self.metrics_history.clear()


class MemoryTracker:
    """Track memory usage during training and inference."""
    
    def __init__(self, track_gpu: bool = True):
        """Initialize memory tracker.
        
        Args:
            track_gpu: Whether to track GPU memory
        """
        self.track_gpu = track_gpu and torch.cuda.is_available()
        self.snapshots: List[Dict[str, float]] = []
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
    
    def take_snapshot(self, label: str = "") -> Dict[str, float]:
        """Take a memory usage snapshot.
        
        Args:
            label: Optional label for the snapshot
            
        Returns:
            Dictionary with memory usage statistics
        """
        snapshot = {
            'timestamp': time.time(),
            'label': label,
            'cpu_memory_mb': self._get_cpu_memory(),
        }
        
        if self.track_gpu:
            snapshot.update({
                'gpu_memory_allocated_mb': torch.cuda.memory_allocated() / 1024 / 1024,
                'gpu_memory_reserved_mb': torch.cuda.memory_reserved() / 1024 / 1024,
                'gpu_max_memory_allocated_mb': torch.cuda.max_memory_allocated() / 1024 / 1024,
            })
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous memory monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring:
            return
        
        self._monitoring = True
        
        def monitor():
            while self._monitoring:
                self.take_snapshot("continuous")
                time.sleep(interval)
        
        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous memory monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def _get_cpu_memory(self) -> float:
        """Get current CPU memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary."""
        if not self.snapshots:
            return {}
        
        cpu_memories = [s['cpu_memory_mb'] for s in self.snapshots]
        
        summary = {
            'peak_cpu_memory_mb': max(cpu_memories),
            'avg_cpu_memory_mb': np.mean(cpu_memories),
            'total_snapshots': len(self.snapshots)
        }
        
        if self.track_gpu:
            gpu_allocated = [s['gpu_memory_allocated_mb'] for s in self.snapshots]
            gpu_reserved = [s['gpu_memory_reserved_mb'] for s in self.snapshots]
            
            summary.update({
                'peak_gpu_allocated_mb': max(gpu_allocated),
                'peak_gpu_reserved_mb': max(gpu_reserved),
                'avg_gpu_allocated_mb': np.mean(gpu_allocated),
                'avg_gpu_reserved_mb': np.mean(gpu_reserved),
            })
        
        return summary
    
    def reset(self):
        """Reset all snapshots."""
        self.snapshots.clear()
        if self.track_gpu:
            torch.cuda.reset_peak_memory_stats()
    
    @contextmanager
    def track(self, operation_name: str = "operation"):
        """Context manager for tracking memory during an operation.
        
        Args:
            operation_name: Name of the operation
            
        Yields:
            Memory usage dictionary
        """
        start_snapshot = self.take_snapshot(f"{operation_name}_start")
        
        try:
            yield start_snapshot
        finally:
            end_snapshot = self.take_snapshot(f"{operation_name}_end")
            
            # Calculate memory delta
            memory_delta = end_snapshot['cpu_memory_mb'] - start_snapshot['cpu_memory_mb']
            print(f"{operation_name} memory delta: {memory_delta:.2f} MB")
            
            if self.track_gpu:
                gpu_delta = end_snapshot['gpu_memory_allocated_mb'] - start_snapshot['gpu_memory_allocated_mb']
                print(f"{operation_name} GPU memory delta: {gpu_delta:.2f} MB")