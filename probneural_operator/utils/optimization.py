"""Model and training optimization utilities."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import warnings


class ModelOptimizer:
    """Optimize neural operator models for better performance."""
    
    def __init__(self, model: nn.Module):
        """Initialize model optimizer.
        
        Args:
            model: PyTorch model to optimize
        """
        self.model = model
        self.original_model = None
    
    def apply_mixed_precision(self) -> nn.Module:
        """Apply automatic mixed precision for faster training.
        
        Returns:
            Model wrapped for AMP compatibility
        """
        # Enable mixed precision in the model
        if hasattr(self.model, '_apply_amp'):
            self.model._amp_enabled = True
        
        return self.model
    
    def apply_gradient_checkpointing(self) -> nn.Module:
        """Apply gradient checkpointing to reduce memory usage.
        
        Returns:
            Model with gradient checkpointing enabled
        """
        def checkpoint_wrapper(module):
            if hasattr(module, 'forward'):
                original_forward = module.forward
                
                def checkpointed_forward(*args, **kwargs):
                    if module.training:
                        return torch.utils.checkpoint.checkpoint(
                            original_forward, *args, **kwargs
                        )
                    else:
                        return original_forward(*args, **kwargs)
                
                module.forward = checkpointed_forward
        
        # Apply to specific layers that benefit from checkpointing
        for name, module in self.model.named_modules():
            if any(layer_type in name.lower() for layer_type in ['spectral', 'branch', 'trunk']):
                checkpoint_wrapper(module)
        
        return self.model
    
    def optimize_for_inference(self) -> nn.Module:
        """Optimize model for inference performance.
        
        Returns:
            Optimized model
        """
        self.model.eval()
        
        # Fuse operations where possible
        if hasattr(torch.jit, 'optimize_for_inference'):
            try:
                # Try to apply PyTorch JIT optimizations
                self.model = torch.jit.optimize_for_inference(
                    torch.jit.script(self.model)
                )
            except Exception as e:
                warnings.warn(f"Could not apply JIT optimization: {e}")
        
        return self.model
    
    def apply_pruning(self, sparsity: float = 0.1) -> nn.Module:
        """Apply structured pruning to reduce model size.
        
        Args:
            sparsity: Fraction of parameters to prune (0.0 to 1.0)
            
        Returns:
            Pruned model
        """
        try:
            import torch.nn.utils.prune as prune
            
            # Apply global magnitude pruning
            parameters_to_prune = []
            for name, module in self.model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    parameters_to_prune.append((module, 'weight'))
            
            if parameters_to_prune:
                prune.global_unstructured(
                    parameters_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=sparsity,
                )
                
                # Make pruning permanent
                for module, param in parameters_to_prune:
                    prune.remove(module, param)
            
        except ImportError:
            warnings.warn("Pruning requires PyTorch >= 1.4.0")
        
        return self.model
    
    def apply_quantization(self, quantization_type: str = "dynamic") -> nn.Module:
        """Apply quantization to reduce model size and improve inference speed.
        
        Args:
            quantization_type: Type of quantization ("dynamic", "static", "qat")
            
        Returns:
            Quantized model
        """
        if quantization_type == "dynamic":
            # Dynamic quantization - easiest to apply
            quantized_model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear, nn.Conv1d, nn.Conv2d},
                dtype=torch.qint8
            )
            return quantized_model
        
        elif quantization_type == "static":
            warnings.warn("Static quantization requires calibration data")
            return self.model
        
        else:
            warnings.warn(f"Unsupported quantization type: {quantization_type}")
            return self.model
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get comprehensive model statistics.
        
        Returns:
            Dictionary with model statistics
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        param_sizes = []
        param_types = []
        
        for name, param in self.model.named_parameters():
            param_sizes.append(param.numel())
            param_types.append(param.dtype)
        
        model_size_mb = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024
        
        stats = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params,
            'model_size_mb': model_size_mb,
            'parameter_dtypes': list(set(str(dt) for dt in param_types)),
            'largest_layer_params': max(param_sizes) if param_sizes else 0,
            'smallest_layer_params': min(param_sizes) if param_sizes else 0,
            'avg_layer_params': np.mean(param_sizes) if param_sizes else 0,
        }
        
        return stats
    
    def benchmark_inference(self, input_shape: Tuple[int, ...], 
                          device: str = "auto",
                          num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark inference performance.
        
        Args:
            input_shape: Shape of input tensor for benchmarking
            device: Device to run benchmark on
            num_iterations: Number of iterations for benchmarking
            
        Returns:
            Dictionary with benchmark results
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model.to(device)
        self.model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
        
        # Benchmark
        if device == "cuda":
            torch.cuda.synchronize()
        
        times = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
                end_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
                
                if device == "cuda":
                    start_time.record()
                    _ = self.model(dummy_input)
                    end_time.record()
                    torch.cuda.synchronize()
                    times.append(start_time.elapsed_time(end_time))
                else:
                    import time
                    start_time = time.perf_counter()
                    _ = self.model(dummy_input)
                    end_time = time.perf_counter()
                    times.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        batch_size = input_shape[0]
        throughput = batch_size / (avg_time / 1000)  # samples per second
        
        return {
            'avg_inference_time_ms': avg_time,
            'std_inference_time_ms': std_time,
            'min_inference_time_ms': min(times),
            'max_inference_time_ms': max(times),
            'throughput_samples_per_sec': throughput,
            'device': device,
            'batch_size': batch_size
        }


class DataLoaderOptimizer:
    """Optimize data loading for better training performance."""
    
    @staticmethod
    def optimize_dataloader(dataloader: DataLoader, 
                          device: str = "auto",
                          **kwargs) -> DataLoader:
        """Optimize dataloader for better performance.
        
        Args:
            dataloader: Original dataloader
            device: Target device
            **kwargs: Additional arguments for DataLoader
            
        Returns:
            Optimized dataloader
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Determine optimal settings
        num_workers = kwargs.get('num_workers', 0)
        if num_workers == 0 and device == "cuda":
            # Use multiple workers for GPU training
            import multiprocessing
            num_workers = min(4, multiprocessing.cpu_count())
        
        pin_memory = kwargs.get('pin_memory', device == "cuda")
        persistent_workers = kwargs.get('persistent_workers', num_workers > 0)
        
        # Create optimized dataloader
        optimized_loader = DataLoader(
            dataloader.dataset,
            batch_size=dataloader.batch_size,
            shuffle=kwargs.get('shuffle', True),
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=kwargs.get('drop_last', True),
            **{k: v for k, v in kwargs.items() 
               if k not in ['num_workers', 'pin_memory', 'persistent_workers', 'shuffle', 'drop_last']}
        )
        
        return optimized_loader
    
    @staticmethod
    def benchmark_dataloader(dataloader: DataLoader, 
                           num_batches: int = 50) -> Dict[str, float]:
        """Benchmark dataloader performance.
        
        Args:
            dataloader: DataLoader to benchmark
            num_batches: Number of batches to iterate through
            
        Returns:
            Dictionary with benchmark results
        """
        times = []
        batch_sizes = []
        
        import time
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            batch_start = time.perf_counter()
            
            # Measure time to load and process batch
            if isinstance(batch, (tuple, list)):
                batch_size = batch[0].shape[0]
                # Simulate moving to device
                _ = [x.to('cpu', non_blocking=True) if isinstance(x, torch.Tensor) else x 
                     for x in batch]
            else:
                batch_size = batch.shape[0]
                _ = batch.to('cpu', non_blocking=True)
            
            batch_end = time.perf_counter()
            times.append(batch_end - batch_start)
            batch_sizes.append(batch_size)
        
        if not times:
            return {}
        
        avg_time = np.mean(times)
        total_samples = sum(batch_sizes)
        throughput = total_samples / sum(times)
        
        return {
            'avg_batch_time_ms': avg_time * 1000,
            'std_batch_time_ms': np.std(times) * 1000,
            'throughput_samples_per_sec': throughput,
            'total_batches': len(times),
            'total_samples': total_samples,
            'avg_batch_size': np.mean(batch_sizes)
        }
    
    @staticmethod
    def create_prefetch_dataloader(dataloader: DataLoader, 
                                 device: str,
                                 queue_size: int = 2) -> 'PrefetchDataLoader':
        """Create a dataloader with GPU prefetching.
        
        Args:
            dataloader: Original dataloader
            device: Target device for prefetching
            queue_size: Size of prefetch queue
            
        Returns:
            PrefetchDataLoader instance
        """
        return PrefetchDataLoader(dataloader, device, queue_size)


class PrefetchDataLoader:
    """DataLoader with GPU prefetching for improved performance."""
    
    def __init__(self, dataloader: DataLoader, device: str, queue_size: int = 2):
        """Initialize prefetch dataloader.
        
        Args:
            dataloader: Original dataloader
            device: Target device
            queue_size: Number of batches to prefetch
        """
        self.dataloader = dataloader
        self.device = device
        self.queue_size = queue_size
        
    def __iter__(self):
        """Iterate with prefetching."""
        import queue
        import threading
        
        def producer(data_queue, stop_event):
            try:
                for batch in self.dataloader:
                    if stop_event.is_set():
                        break
                    
                    # Move batch to device
                    if isinstance(batch, (tuple, list)):
                        batch = [x.to(self.device, non_blocking=True) 
                               if isinstance(x, torch.Tensor) else x for x in batch]
                    else:
                        batch = batch.to(self.device, non_blocking=True)
                    
                    data_queue.put(batch)
                    
            except Exception as e:
                data_queue.put(e)
            finally:
                data_queue.put(None)  # Sentinel
        
        data_queue = queue.Queue(maxsize=self.queue_size)
        stop_event = threading.Event()
        
        producer_thread = threading.Thread(
            target=producer, 
            args=(data_queue, stop_event),
            daemon=True
        )
        producer_thread.start()
        
        try:
            while True:
                batch = data_queue.get()
                
                if batch is None:  # Sentinel
                    break
                elif isinstance(batch, Exception):
                    raise batch
                else:
                    yield batch
        finally:
            stop_event.set()
            producer_thread.join(timeout=1.0)
    
    def __len__(self):
        """Return length of original dataloader."""
        return len(self.dataloader)
    
    @property
    def batch_size(self):
        """Return batch size of original dataloader."""
        return self.dataloader.batch_size
    
    @property
    def dataset(self):
        """Return dataset of original dataloader."""
        return self.dataloader.dataset


def auto_tune_batch_size(model: nn.Module, 
                        dataloader: DataLoader,
                        max_batch_size: int = 1024,
                        device: str = "auto") -> int:
    """Automatically find the optimal batch size for training.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader to test with
        max_batch_size: Maximum batch size to test
        device: Device to run tests on
        
    Returns:
        Optimal batch size
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model.to(device)
    model.train()
    
    # Get a sample batch
    sample_batch = next(iter(dataloader))
    if isinstance(sample_batch, (tuple, list)):
        sample_input, sample_target = sample_batch
    else:
        sample_input = sample_batch
        sample_target = torch.randn_like(sample_input)
    
    original_batch_size = sample_input.shape[0]
    
    # Test different batch sizes
    batch_sizes = []
    current_size = 1
    
    while current_size <= max_batch_size:
        batch_sizes.append(current_size)
        current_size *= 2
    
    optimal_batch_size = original_batch_size
    
    for batch_size in batch_sizes:
        try:
            # Create test batch
            test_input = sample_input[:1].repeat(batch_size, *([1] * (sample_input.ndim - 1)))
            test_target = sample_target[:1].repeat(batch_size, *([1] * (sample_target.ndim - 1)))
            
            test_input = test_input.to(device)
            test_target = test_target.to(device)
            
            # Test forward and backward pass
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            optimizer.zero_grad()
            
            output = model(test_input)
            loss = nn.MSELoss()(output, test_target)
            loss.backward()
            
            optimal_batch_size = batch_size
            
            # Clear memory
            del test_input, test_target, output, loss
            torch.cuda.empty_cache() if device == "cuda" else None
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                break
            else:
                raise e
    
    return optimal_batch_size