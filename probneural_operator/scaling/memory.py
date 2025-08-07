"""Memory management and resource optimization utilities."""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import mmap
import os
import threading
import time
import gc
import weakref
from typing import Dict, Any, List, Optional, Tuple, Union, Iterator, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
import psutil


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    cpu_memory_mb: float = 0.0
    gpu_memory_mb: Dict[int, float] = None
    gpu_max_memory_mb: Dict[int, float] = None
    gpu_reserved_mb: Dict[int, float] = None
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.gpu_memory_mb is None:
            self.gpu_memory_mb = {}
        if self.gpu_max_memory_mb is None:
            self.gpu_max_memory_mb = {}
        if self.gpu_reserved_mb is None:
            self.gpu_reserved_mb = {}


class GradientCheckpointer:
    """Advanced gradient checkpointing for memory-efficient training."""
    
    def __init__(self, 
                 preserve_rng_state: bool = True,
                 pack_sequences: bool = True):
        """Initialize gradient checkpointer.
        
        Args:
            preserve_rng_state: Whether to preserve RNG state
            pack_sequences: Whether to pack sequences for better memory efficiency
        """
        self.preserve_rng_state = preserve_rng_state
        self.pack_sequences = pack_sequences
        self._checkpoint_functions = {}
        self._memory_savings = 0.0
    
    def checkpoint_sequential(self, 
                            sequential: nn.Sequential, 
                            segments: int,
                            input_tensor: torch.Tensor,
                            **kwargs) -> torch.Tensor:
        """Apply checkpointing to sequential module.
        
        Args:
            sequential: Sequential module
            segments: Number of segments to checkpoint
            input_tensor: Input tensor
            **kwargs: Additional arguments
            
        Returns:
            Output tensor
        """
        if segments == 1:
            return sequential(input_tensor)
        
        # Calculate memory before checkpointing
        start_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Split sequential into segments
        layers_per_segment = len(sequential) // segments
        remainder = len(sequential) % segments
        
        def run_segment(start_idx: int, end_idx: int):
            """Run a segment of the sequential module."""
            def segment_fn(x):
                for i in range(start_idx, end_idx):
                    x = sequential[i](x)
                return x
            return segment_fn
        
        # Apply checkpointing to each segment
        x = input_tensor
        start_idx = 0
        
        for segment_idx in range(segments):
            # Calculate segment size
            segment_size = layers_per_segment + (1 if segment_idx < remainder else 0)
            end_idx = start_idx + segment_size
            
            if segment_idx == segments - 1:
                # Don't checkpoint the last segment for better performance
                segment_fn = run_segment(start_idx, end_idx)
                x = segment_fn(x)
            else:
                # Checkpoint this segment
                segment_fn = run_segment(start_idx, end_idx)
                x = checkpoint.checkpoint(
                    segment_fn, 
                    x,
                    preserve_rng_state=self.preserve_rng_state,
                    **kwargs
                )
            
            start_idx = end_idx
        
        # Calculate memory savings
        end_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        self._memory_savings += (start_memory - end_memory) / 1024 / 1024  # MB
        
        return x
    
    def checkpoint_function(self, 
                          func: Callable,
                          *args,
                          **kwargs) -> Any:
        """Apply checkpointing to arbitrary function.
        
        Args:
            func: Function to checkpoint
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function output
        """
        return checkpoint.checkpoint(
            func,
            *args,
            preserve_rng_state=self.preserve_rng_state,
            **kwargs
        )
    
    def register_checkpoint_function(self, 
                                   name: str,
                                   func: Callable,
                                   trigger_condition: Optional[Callable] = None):
        """Register a function for automatic checkpointing.
        
        Args:
            name: Function name
            func: Function to checkpoint
            trigger_condition: Condition to trigger checkpointing
        """
        self._checkpoint_functions[name] = {
            'function': func,
            'trigger_condition': trigger_condition,
            'call_count': 0,
            'memory_saved': 0.0
        }
    
    def auto_checkpoint_forward(self, 
                               module: nn.Module,
                               memory_threshold_mb: float = 1000.0) -> nn.Module:
        """Automatically apply checkpointing to module forward pass.
        
        Args:
            module: Module to modify
            memory_threshold_mb: Memory threshold to trigger checkpointing
            
        Returns:
            Modified module with checkpointing
        """
        original_forward = module.forward
        
        def checkpointed_forward(*args, **kwargs):
            # Check memory usage
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated() / 1024 / 1024
                
                if current_memory > memory_threshold_mb and module.training:
                    return checkpoint.checkpoint(
                        original_forward,
                        *args,
                        preserve_rng_state=self.preserve_rng_state,
                        **kwargs
                    )
            
            return original_forward(*args, **kwargs)
        
        module.forward = checkpointed_forward
        return module
    
    def get_memory_savings(self) -> float:
        """Get total memory savings in MB."""
        return self._memory_savings


class MixedPrecisionManager:
    """Advanced mixed precision training with automatic loss scaling."""
    
    def __init__(self,
                 enabled: bool = True,
                 init_scale: float = 2.**16,
                 growth_factor: float = 2.0,
                 backoff_factor: float = 0.5,
                 growth_interval: int = 2000,
                 enabled_layers: Optional[List[str]] = None):
        """Initialize mixed precision manager.
        
        Args:
            enabled: Whether to enable mixed precision
            init_scale: Initial loss scale
            growth_factor: Factor to grow loss scale
            backoff_factor: Factor to reduce loss scale
            growth_interval: Interval to attempt growing scale
            enabled_layers: Specific layers to enable mixed precision for
        """
        self.enabled = enabled and torch.cuda.is_available()
        self.scaler = GradScaler(
            init_scale=init_scale,
            growth_factor=growth_factor,
            backoff_factor=backoff_factor,
            growth_interval=growth_interval,
            enabled=self.enabled
        ) if self.enabled else None
        
        self.enabled_layers = enabled_layers
        self._scale_history = deque(maxlen=1000)
        self._overflow_count = 0
        self._update_count = 0
        
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for backward pass."""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def unscale_gradients(self, optimizer: torch.optim.Optimizer):
        """Unscale gradients before gradient clipping."""
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
    
    def step_optimizer(self, optimizer: torch.optim.Optimizer) -> bool:
        """Step optimizer with automatic scaling.
        
        Returns:
            True if step was taken, False if skipped due to overflow
        """
        if self.scaler is not None:
            # Get scale before step
            old_scale = self.scaler.get_scale()
            
            # Step and update
            self.scaler.step(optimizer)
            self.scaler.update()
            
            # Track scaling
            new_scale = self.scaler.get_scale()
            self._scale_history.append(new_scale)
            self._update_count += 1
            
            # Check for overflow
            if new_scale < old_scale:
                self._overflow_count += 1
                return False
            
            return True
        else:
            optimizer.step()
            return True
    
    @torch.cuda.amp.autocast()
    def forward_with_autocast(self, model: nn.Module, *args, **kwargs):
        """Forward pass with automatic mixed precision."""
        return model(*args, **kwargs)
    
    def convert_model_to_half(self, 
                            model: nn.Module,
                            exclude_layers: Optional[List[str]] = None) -> nn.Module:
        """Convert model to half precision selectively.
        
        Args:
            model: Model to convert
            exclude_layers: Layer names to exclude from conversion
            
        Returns:
            Converted model
        """
        exclude_layers = exclude_layers or ['LayerNorm', 'BatchNorm', 'Embedding']
        
        def should_convert_layer(module):
            """Check if layer should be converted to half precision."""
            layer_type = type(module).__name__
            
            # Exclude certain layer types
            if any(exclude_type in layer_type for exclude_type in exclude_layers):
                return False
            
            # Include specific layers if specified
            if self.enabled_layers is not None:
                return any(enabled_type in layer_type for enabled_type in self.enabled_layers)
            
            # Default: convert linear and conv layers
            return isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d))
        
        for name, module in model.named_modules():
            if should_convert_layer(module):
                module.half()
        
        return model
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get mixed precision scaling statistics."""
        if not self._scale_history:
            return {}
        
        scales = list(self._scale_history)
        
        return {
            'current_scale': scales[-1] if scales else 0,
            'avg_scale': np.mean(scales),
            'min_scale': np.min(scales),
            'max_scale': np.max(scales),
            'overflow_count': self._overflow_count,
            'overflow_rate': self._overflow_count / max(1, self._update_count),
            'total_updates': self._update_count,
            'enabled': self.enabled
        }


class MemoryMappedDataset:
    """Memory-mapped dataset for efficient large dataset handling."""
    
    def __init__(self,
                 data_path: str,
                 index_path: Optional[str] = None,
                 dtype: np.dtype = np.float32,
                 shape: Optional[Tuple[int, ...]] = None,
                 cache_size: int = 1000):
        """Initialize memory-mapped dataset.
        
        Args:
            data_path: Path to memory-mapped data file
            index_path: Path to index file
            dtype: Data type
            shape: Shape of individual samples
            cache_size: Number of samples to cache in memory
        """
        self.data_path = data_path
        self.index_path = index_path or (data_path + '.idx')
        self.dtype = dtype
        self.shape = shape
        self.cache_size = cache_size
        
        self._mmap_file = None
        self._data_mmap = None
        self._index = None
        self._cache = {}
        self._cache_order = deque(maxlen=cache_size)
        self._lock = threading.Lock()
        
        self._initialize()
    
    def _initialize(self):
        """Initialize memory mapping and index."""
        # Open memory-mapped file
        if os.path.exists(self.data_path):
            self._mmap_file = open(self.data_path, 'r+b')
            self._data_mmap = mmap.mmap(self._mmap_file.fileno(), 0)
        
        # Load index
        if os.path.exists(self.index_path):
            with open(self.index_path, 'rb') as f:
                self._index = pickle.load(f)
        else:
            self._build_index()
    
    def _build_index(self):
        """Build index for the dataset."""
        if self._data_mmap is None:
            return
        
        # Simple index: assume fixed-size records
        if self.shape is not None:
            record_size = int(np.prod(self.shape) * np.dtype(self.dtype).itemsize)
            num_records = len(self._data_mmap) // record_size
            
            self._index = {
                'num_records': num_records,
                'record_size': record_size,
                'shape': self.shape,
                'dtype': str(self.dtype)
            }
            
            # Save index
            with open(self.index_path, 'wb') as f:
                pickle.dump(self._index, f)
    
    def __len__(self) -> int:
        """Get dataset length."""
        if self._index is not None:
            return self._index['num_records']
        return 0
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get item by index with caching."""
        with self._lock:
            # Check cache first
            if idx in self._cache:
                self._cache_order.remove(idx)
                self._cache_order.append(idx)
                return self._cache[idx]
            
            # Load from memory-mapped file
            data = self._load_sample(idx)
            
            # Add to cache
            if len(self._cache) >= self.cache_size:
                # Remove oldest item
                oldest_idx = self._cache_order[0]
                del self._cache[oldest_idx]
            
            self._cache[idx] = data
            self._cache_order.append(idx)
            
            return data
    
    def _load_sample(self, idx: int) -> torch.Tensor:
        """Load sample from memory-mapped file."""
        if self._data_mmap is None or self._index is None:
            raise ValueError("Dataset not properly initialized")
        
        record_size = self._index['record_size']
        shape = self._index['shape']
        dtype = np.dtype(self._index['dtype'])
        
        # Calculate offset
        offset = idx * record_size
        
        # Read data
        self._data_mmap.seek(offset)
        data_bytes = self._data_mmap.read(record_size)
        
        # Convert to numpy array
        data_array = np.frombuffer(data_bytes, dtype=dtype).reshape(shape)
        
        # Convert to tensor
        return torch.from_numpy(data_array.copy())
    
    def prefetch_batch(self, indices: List[int]):
        """Prefetch a batch of samples into cache."""
        def prefetch_worker():
            for idx in indices:
                if idx not in self._cache:
                    self.__getitem__(idx)
        
        # Run prefetching in background thread
        thread = threading.Thread(target=prefetch_worker)
        thread.start()
        return thread
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'cache_size': len(self._cache),
                'max_cache_size': self.cache_size,
                'cache_hit_rate': len(self._cache) / max(1, len(self._cache_order)),
                'cached_indices': list(self._cache.keys())
            }
    
    def close(self):
        """Close memory-mapped file."""
        if self._data_mmap is not None:
            self._data_mmap.close()
        if self._mmap_file is not None:
            self._mmap_file.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()
    
    @staticmethod
    def create_from_tensor(tensor: torch.Tensor, 
                          output_path: str) -> 'MemoryMappedDataset':
        """Create memory-mapped dataset from tensor.
        
        Args:
            tensor: Source tensor
            output_path: Output file path
            
        Returns:
            MemoryMappedDataset instance
        """
        # Convert tensor to numpy
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        data_array = tensor.numpy()
        
        # Write to file
        with open(output_path, 'wb') as f:
            data_array.tobytes()
        
        # Create index
        index = {
            'num_records': data_array.shape[0],
            'record_size': int(np.prod(data_array.shape[1:]) * data_array.dtype.itemsize),
            'shape': data_array.shape[1:],
            'dtype': str(data_array.dtype)
        }
        
        index_path = output_path + '.idx'
        with open(index_path, 'wb') as f:
            pickle.dump(index, f)
        
        return MemoryMappedDataset(
            data_path=output_path,
            index_path=index_path,
            dtype=data_array.dtype,
            shape=data_array.shape[1:]
        )


class MemoryPoolManager:
    """Advanced memory pool management for tensor allocation."""
    
    def __init__(self,
                 pool_size_mb: float = 512.0,
                 enable_cpu_pool: bool = True,
                 enable_gpu_pool: bool = True):
        """Initialize memory pool manager.
        
        Args:
            pool_size_mb: Size of memory pool in MB
            enable_cpu_pool: Whether to enable CPU memory pooling
            enable_gpu_pool: Whether to enable GPU memory pooling
        """
        self.pool_size_bytes = int(pool_size_mb * 1024 * 1024)
        self.enable_cpu_pool = enable_cpu_pool
        self.enable_gpu_pool = enable_gpu_pool and torch.cuda.is_available()
        
        # Memory pools organized by (device, dtype, size)
        self._pools = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self._pool_stats = defaultdict(lambda: {
            'allocated': 0,
            'freed': 0,
            'current_size': 0,
            'peak_size': 0,
            'hits': 0,
            'misses': 0
        })
        
        self._lock = threading.Lock()
        self._cleanup_thread = None
        self._cleanup_interval = 60.0  # seconds
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_worker():
            while True:
                time.sleep(self._cleanup_interval)
                self._periodic_cleanup()
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _periodic_cleanup(self):
        """Periodic cleanup of unused memory."""
        with self._lock:
            for device_pools in self._pools.values():
                for dtype_pools in device_pools.values():
                    for size, tensor_list in dtype_pools.items():
                        # Keep only recent tensors
                        if len(tensor_list) > 10:
                            # Remove older tensors
                            tensors_to_keep = tensor_list[-5:]
                            freed_count = len(tensor_list) - len(tensors_to_keep)
                            
                            dtype_pools[size] = tensors_to_keep
                            
                            # Update stats
                            device = next(iter(self._pools.keys()))
                            self._pool_stats[device]['freed'] += freed_count
    
    def get_tensor(self,
                  size: Tuple[int, ...],
                  dtype: torch.dtype = torch.float32,
                  device: Union[str, torch.device] = 'cpu') -> torch.Tensor:
        """Get tensor from pool or allocate new one.
        
        Args:
            size: Tensor size
            dtype: Data type
            device: Device
            
        Returns:
            Tensor from pool or newly allocated
        """
        if isinstance(device, str):
            device = torch.device(device)
        
        # Check if pooling is enabled for this device
        if (device.type == 'cpu' and not self.enable_cpu_pool) or \
           (device.type == 'cuda' and not self.enable_gpu_pool):
            return torch.zeros(size, dtype=dtype, device=device)
        
        with self._lock:
            device_key = f"{device.type}:{device.index if device.index is not None else 0}"
            size_key = tuple(size)
            
            # Try to get from pool
            pool = self._pools[device_key][dtype][size_key]
            
            if pool:
                tensor = pool.pop()
                tensor.zero_()  # Clear data
                
                self._pool_stats[device_key]['hits'] += 1
                return tensor
            else:
                # Allocate new tensor
                tensor = torch.zeros(size, dtype=dtype, device=device)
                
                self._pool_stats[device_key]['allocated'] += 1
                self._pool_stats[device_key]['misses'] += 1
                
                return tensor
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return tensor to pool for reuse.
        
        Args:
            tensor: Tensor to return to pool
        """
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        device = tensor.device
        device_key = f"{device.type}:{device.index if device.index is not None else 0}"
        
        # Check if pooling is enabled for this device
        if (device.type == 'cpu' and not self.enable_cpu_pool) or \
           (device.type == 'cuda' and not self.enable_gpu_pool):
            return
        
        with self._lock:
            size_key = tuple(tensor.shape)
            pool = self._pools[device_key][tensor.dtype][size_key]
            
            # Check pool size limits
            tensor_size = tensor.numel() * tensor.element_size()
            current_pool_size = sum(
                t.numel() * t.element_size() 
                for dtype_pools in self._pools[device_key].values()
                for size_pools in dtype_pools.values()
                for t in size_pools
            )
            
            if current_pool_size + tensor_size <= self.pool_size_bytes:
                pool.append(tensor.detach())
                
                # Update stats
                stats = self._pool_stats[device_key]
                stats['current_size'] = current_pool_size + tensor_size
                stats['peak_size'] = max(stats['peak_size'], stats['current_size'])
    
    def clear_pool(self, device: Optional[Union[str, torch.device]] = None):
        """Clear memory pool.
        
        Args:
            device: Specific device to clear (None for all)
        """
        with self._lock:
            if device is not None:
                if isinstance(device, str):
                    device = torch.device(device)
                device_key = f"{device.type}:{device.index if device.index is not None else 0}"
                
                if device_key in self._pools:
                    self._pools[device_key].clear()
                    self._pool_stats[device_key]['current_size'] = 0
            else:
                # Clear all pools
                self._pools.clear()
                for stats in self._pool_stats.values():
                    stats['current_size'] = 0
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            stats = {}
            
            for device_key, device_stats in self._pool_stats.items():
                hit_rate = device_stats['hits'] / max(1, device_stats['hits'] + device_stats['misses'])
                
                stats[device_key] = {
                    'allocated': device_stats['allocated'],
                    'freed': device_stats['freed'],
                    'current_size_mb': device_stats['current_size'] / 1024 / 1024,
                    'peak_size_mb': device_stats['peak_size'] / 1024 / 1024,
                    'pool_hit_rate': hit_rate,
                    'total_requests': device_stats['hits'] + device_stats['misses']
                }
            
            return stats
    
    def optimize_allocation_strategy(self):
        """Optimize memory allocation strategy based on usage patterns."""
        with self._lock:
            # Analyze usage patterns
            for device_key, device_pools in self._pools.items():
                stats = self._pool_stats[device_key]
                
                # If hit rate is low, reduce pool size
                hit_rate = stats['hits'] / max(1, stats['hits'] + stats['misses'])
                
                if hit_rate < 0.3:  # Less than 30% hit rate
                    # Reduce pool by keeping only most recently used tensors
                    for dtype_pools in device_pools.values():
                        for size_key, tensor_list in dtype_pools.items():
                            if len(tensor_list) > 5:
                                dtype_pools[size_key] = tensor_list[-3:]  # Keep 3 most recent
    
    def get_memory_usage(self) -> MemoryStats:
        """Get current memory usage statistics."""
        stats = MemoryStats()
        stats.timestamp = time.time()
        
        # CPU memory
        try:
            process = psutil.Process()
            stats.cpu_memory_mb = process.memory_info().rss / 1024 / 1024
        except:
            stats.cpu_memory_mb = 0.0
        
        # GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    stats.gpu_memory_mb[i] = torch.cuda.memory_allocated(i) / 1024 / 1024
                    stats.gpu_max_memory_mb[i] = torch.cuda.max_memory_allocated(i) / 1024 / 1024
                    stats.gpu_reserved_mb[i] = torch.cuda.memory_reserved(i) / 1024 / 1024
                except:
                    stats.gpu_memory_mb[i] = 0.0
                    stats.gpu_max_memory_mb[i] = 0.0
                    stats.gpu_reserved_mb[i] = 0.0
        
        return stats


def optimize_model_memory_usage(model: nn.Module,
                               input_shape: Tuple[int, ...],
                               checkpointing_segments: int = 4,
                               mixed_precision: bool = True,
                               memory_efficient_attention: bool = True) -> Tuple[nn.Module, Dict[str, Any]]:
    """Comprehensive memory optimization for neural operator models.
    
    Args:
        model: Model to optimize
        input_shape: Input tensor shape for analysis
        checkpointing_segments: Number of checkpointing segments
        mixed_precision: Whether to apply mixed precision
        memory_efficient_attention: Whether to optimize attention
        
    Returns:
        Tuple of optimized model and optimization stats
    """
    optimization_stats = {
        'original_memory_mb': 0.0,
        'optimized_memory_mb': 0.0,
        'memory_savings_mb': 0.0,
        'optimizations_applied': []
    }
    
    # Measure original memory usage
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        dummy_input = torch.randn(input_shape).cuda()
        model = model.cuda()
        
        # Forward pass to measure memory
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(dummy_input)
        
        optimization_stats['original_memory_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
    
    # Apply gradient checkpointing
    if checkpointing_segments > 1:
        checkpointer = GradientCheckpointer()
        
        # Find sequential layers to checkpoint
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential) and len(module) > checkpointing_segments:
                original_forward = module.forward
                
                def checkpointed_forward(x):
                    return checkpointer.checkpoint_sequential(module, checkpointing_segments, x)
                
                module.forward = checkpointed_forward
                optimization_stats['optimizations_applied'].append(f'checkpointing_{name}')
    
    # Apply mixed precision
    if mixed_precision:
        mp_manager = MixedPrecisionManager()
        model = mp_manager.convert_model_to_half(model)
        optimization_stats['optimizations_applied'].append('mixed_precision')
    
    # Optimize attention mechanisms
    if memory_efficient_attention:
        for name, module in model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                # Apply memory efficient attention if available
                if hasattr(module, 'enable_memory_efficient_attention'):
                    module.enable_memory_efficient_attention()
                    optimization_stats['optimizations_applied'].append(f'efficient_attention_{name}')
    
    # Measure optimized memory usage
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = model(dummy_input)
        
        optimization_stats['optimized_memory_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
        optimization_stats['memory_savings_mb'] = (
            optimization_stats['original_memory_mb'] - optimization_stats['optimized_memory_mb']
        )
    
    return model, optimization_stats


class MemoryMonitor:
    """Real-time memory monitoring and alerting."""
    
    def __init__(self,
                 warning_threshold_mb: float = 1000.0,
                 critical_threshold_mb: float = 2000.0,
                 monitoring_interval: float = 5.0):
        """Initialize memory monitor.
        
        Args:
            warning_threshold_mb: Warning threshold in MB
            critical_threshold_mb: Critical threshold in MB
            monitoring_interval: Monitoring interval in seconds
        """
        self.warning_threshold_mb = warning_threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.monitoring_interval = monitoring_interval
        
        self._monitoring = False
        self._monitor_thread = None
        self._callbacks = []
        self._memory_history = deque(maxlen=100)
        
    def add_callback(self, callback: Callable[[MemoryStats], None]):
        """Add callback for memory events."""
        self._callbacks.append(callback)
    
    def start_monitoring(self):
        """Start memory monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logging.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
        
        logging.info("Memory monitoring stopped")
    
    def _monitor_loop(self):
        """Memory monitoring loop."""
        pool_manager = MemoryPoolManager()
        
        while self._monitoring:
            try:
                # Get current memory stats
                stats = pool_manager.get_memory_usage()
                self._memory_history.append(stats)
                
                # Check thresholds
                total_gpu_memory = sum(stats.gpu_memory_mb.values()) if stats.gpu_memory_mb else 0
                total_memory = stats.cpu_memory_mb + total_gpu_memory
                
                if total_memory >= self.critical_threshold_mb:
                    logging.critical(f"Critical memory usage: {total_memory:.1f} MB")
                    self._trigger_emergency_cleanup()
                elif total_memory >= self.warning_threshold_mb:
                    logging.warning(f"High memory usage: {total_memory:.1f} MB")
                
                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(stats)
                    except Exception as e:
                        logging.error(f"Memory monitor callback error: {e}")
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logging.error(f"Memory monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _trigger_emergency_cleanup(self):
        """Trigger emergency memory cleanup."""
        logging.info("Triggering emergency memory cleanup")
        
        # Force garbage collection
        gc.collect()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear memory pools
        pool_manager = MemoryPoolManager()
        pool_manager.clear_pool()
    
    def get_memory_trend(self, window_minutes: int = 10) -> Dict[str, float]:
        """Get memory usage trend over time window."""
        if len(self._memory_history) < 2:
            return {}
        
        # Filter recent history
        current_time = time.time()
        window_seconds = window_minutes * 60
        recent_stats = [
            s for s in self._memory_history
            if current_time - s.timestamp <= window_seconds
        ]
        
        if len(recent_stats) < 2:
            return {}
        
        # Calculate trends
        cpu_memories = [s.cpu_memory_mb for s in recent_stats]
        gpu_memories = [sum(s.gpu_memory_mb.values()) for s in recent_stats]
        
        return {
            'cpu_memory_trend_mb_per_min': (cpu_memories[-1] - cpu_memories[0]) / window_minutes,
            'gpu_memory_trend_mb_per_min': (gpu_memories[-1] - gpu_memories[0]) / window_minutes,
            'avg_cpu_memory_mb': np.mean(cpu_memories),
            'avg_gpu_memory_mb': np.mean(gpu_memories),
            'peak_cpu_memory_mb': np.max(cpu_memories),
            'peak_gpu_memory_mb': np.max(gpu_memories)
        }