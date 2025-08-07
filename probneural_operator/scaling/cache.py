"""Performance optimization and caching utilities."""

import torch
import torch.nn as nn
import numpy as np
import time
import hashlib
import pickle
from typing import Dict, Any, Optional, Tuple, List, Union
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading
import weakref
import gc
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    data: torch.Tensor
    timestamp: float
    hit_count: int
    size_bytes: int
    key_hash: str


class PredictionCache:
    """Intelligent caching for model predictions with LRU eviction."""
    
    def __init__(self, 
                 max_size: int = 1000,
                 max_memory_mb: float = 512.0,
                 ttl_seconds: float = 3600.0,
                 enable_compression: bool = True):
        """Initialize prediction cache.
        
        Args:
            max_size: Maximum number of cache entries
            max_memory_mb: Maximum memory usage in MB
            ttl_seconds: Time-to-live for cache entries
            enable_compression: Whether to compress cached tensors
        """
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.ttl_seconds = ttl_seconds
        self.enable_compression = enable_compression
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._memory_usage = 0
        self._hit_count = 0
        self._miss_count = 0
        self._lock = threading.RLock()
        
        # Statistics
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'compressions': 0,
            'memory_saved_mb': 0.0
        }
    
    def _generate_key(self, 
                     model_state: Dict[str, Any], 
                     input_tensor: torch.Tensor,
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate cache key from model state and input."""
        # Create deterministic key from model parameters and input
        key_components = []
        
        # Model state hash (weights, architecture params)
        if model_state:
            model_hash = hashlib.md5(str(sorted(model_state.items())).encode()).hexdigest()
            key_components.append(f"model:{model_hash}")
        
        # Input tensor properties
        input_props = {
            'shape': tuple(input_tensor.shape),
            'dtype': str(input_tensor.dtype),
            'device': str(input_tensor.device),
            'hash': hashlib.md5(input_tensor.detach().cpu().numpy().tobytes()).hexdigest()[:16]
        }
        key_components.append(f"input:{input_props}")
        
        # Additional metadata
        if metadata:
            meta_hash = hashlib.md5(str(sorted(metadata.items())).encode()).hexdigest()[:8]
            key_components.append(f"meta:{meta_hash}")
        
        return "|".join(key_components)
    
    def _compress_tensor(self, tensor: torch.Tensor) -> bytes:
        """Compress tensor using efficient encoding."""
        if not self.enable_compression:
            return pickle.dumps(tensor)
        
        # Convert to numpy for better compression
        numpy_array = tensor.detach().cpu().numpy()
        
        # Use zlib compression
        import zlib
        compressed_data = zlib.compress(pickle.dumps(numpy_array), level=6)
        
        original_size = tensor.numel() * tensor.element_size()
        compressed_size = len(compressed_data)
        
        if compressed_size < original_size:
            self._stats['compressions'] += 1
            self._stats['memory_saved_mb'] += (original_size - compressed_size) / 1024 / 1024
            return compressed_data
        else:
            return pickle.dumps(tensor)
    
    def _decompress_tensor(self, compressed_data: bytes, device: str = "cpu") -> torch.Tensor:
        """Decompress tensor and move to device."""
        try:
            import zlib
            # Try to decompress
            numpy_array = pickle.loads(zlib.decompress(compressed_data))
            return torch.from_numpy(numpy_array).to(device)
        except:
            # Fallback to direct unpickling
            tensor = pickle.loads(compressed_data)
            return tensor.to(device)
    
    def get(self, 
           model_state: Dict[str, Any],
           input_tensor: torch.Tensor,
           metadata: Optional[Dict[str, Any]] = None) -> Optional[torch.Tensor]:
        """Get cached prediction if available."""
        with self._lock:
            key = self._generate_key(model_state, input_tensor, metadata)
            
            if key not in self._cache:
                self._stats['misses'] += 1
                return None
            
            entry = self._cache[key]
            
            # Check TTL
            if time.time() - entry.timestamp > self.ttl_seconds:
                self._evict_entry(key)
                self._stats['misses'] += 1
                return None
            
            # Update access order (LRU)
            self._cache.move_to_end(key)
            entry.hit_count += 1
            
            self._stats['hits'] += 1
            
            # Decompress and return
            device = str(input_tensor.device)
            return self._decompress_tensor(entry.data, device)
    
    def put(self,
           model_state: Dict[str, Any],
           input_tensor: torch.Tensor, 
           output_tensor: torch.Tensor,
           metadata: Optional[Dict[str, Any]] = None):
        """Cache prediction result."""
        with self._lock:
            key = self._generate_key(model_state, input_tensor, metadata)
            
            # Compress output tensor
            compressed_data = self._compress_tensor(output_tensor)
            size_bytes = len(compressed_data)
            
            # Create cache entry
            entry = CacheEntry(
                data=compressed_data,
                timestamp=time.time(),
                hit_count=0,
                size_bytes=size_bytes,
                key_hash=key
            )
            
            # Check if we need to evict entries
            self._ensure_capacity(size_bytes)
            
            # Add to cache
            self._cache[key] = entry
            self._memory_usage += size_bytes
    
    def _ensure_capacity(self, new_entry_size: int):
        """Ensure cache has capacity for new entry."""
        # Evict expired entries first
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time - entry.timestamp > self.ttl_seconds
        ]
        
        for key in expired_keys:
            self._evict_entry(key)
        
        # Evict LRU entries if needed
        while (len(self._cache) >= self.max_size or 
               self._memory_usage + new_entry_size > self.max_memory_bytes):
            
            if not self._cache:
                break
                
            # Remove least recently used entry
            lru_key = next(iter(self._cache))
            self._evict_entry(lru_key)
    
    def _evict_entry(self, key: str):
        """Evict entry from cache."""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._memory_usage -= entry.size_bytes
            self._stats['evictions'] += 1
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._memory_usage = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0.0
            
            return {
                'entries': len(self._cache),
                'memory_usage_mb': self._memory_usage / 1024 / 1024,
                'memory_limit_mb': self.max_memory_bytes / 1024 / 1024,
                'hit_rate': hit_rate,
                'total_hits': self._stats['hits'],
                'total_misses': self._stats['misses'],
                'total_evictions': self._stats['evictions'],
                'compressions_performed': self._stats['compressions'],
                'memory_saved_mb': self._stats['memory_saved_mb']
            }


class TensorOperationCache:
    """Cache for expensive tensor operations like matrix decompositions."""
    
    def __init__(self, max_size: int = 100):
        """Initialize tensor operation cache.
        
        Args:
            max_size: Maximum number of cached operations
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._lock = threading.Lock()
    
    def cached_svd(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Cached SVD decomposition."""
        key = self._tensor_key(tensor, "svd")
        
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            
            # Compute SVD
            u, s, v = torch.svd(tensor)
            result = (u, s, v)
            
            # Cache result
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            
            self._cache[key] = result
            return result
    
    def cached_eigendecomposition(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cached eigendecomposition."""
        key = self._tensor_key(tensor, "eigen")
        
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            
            # Compute eigendecomposition
            eigenvalues, eigenvectors = torch.symeig(tensor, eigenvectors=True)
            result = (eigenvalues, eigenvectors)
            
            # Cache result
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
            
            self._cache[key] = result
            return result
    
    def _tensor_key(self, tensor: torch.Tensor, operation: str) -> str:
        """Generate key for tensor operation."""
        tensor_hash = hashlib.md5(tensor.detach().cpu().numpy().tobytes()).hexdigest()[:16]
        return f"{operation}:{tensor_hash}:{tensor.shape}:{tensor.dtype}"


class AdaptiveBatchSizer:
    """Automatically optimize batch size based on system resources and performance."""
    
    def __init__(self, 
                 initial_batch_size: int = 32,
                 min_batch_size: int = 1,
                 max_batch_size: int = 1024,
                 target_memory_usage: float = 0.8,
                 performance_history_size: int = 10):
        """Initialize adaptive batch sizer.
        
        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum allowed batch size
            max_batch_size: Maximum allowed batch size
            target_memory_usage: Target GPU memory usage (0.0-1.0)
            performance_history_size: Number of performance measurements to keep
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_memory_usage = target_memory_usage
        self.performance_history_size = performance_history_size
        
        self._performance_history: List[Dict[str, float]] = []
        self._memory_history: List[float] = []
        self._adjustment_count = 0
        
    def update_performance(self, 
                         batch_size: int,
                         throughput: float,
                         memory_usage: float,
                         execution_time: float):
        """Update performance metrics and potentially adjust batch size."""
        
        perf_entry = {
            'batch_size': batch_size,
            'throughput': throughput,
            'memory_usage': memory_usage,
            'execution_time': execution_time,
            'efficiency': throughput / memory_usage if memory_usage > 0 else 0
        }
        
        self._performance_history.append(perf_entry)
        self._memory_history.append(memory_usage)
        
        # Keep only recent history
        if len(self._performance_history) > self.performance_history_size:
            self._performance_history = self._performance_history[-self.performance_history_size:]
            self._memory_history = self._memory_history[-self.performance_history_size:]
        
        # Decide on batch size adjustment
        new_batch_size = self._compute_optimal_batch_size()
        
        if new_batch_size != self.current_batch_size:
            self.current_batch_size = new_batch_size
            self._adjustment_count += 1
    
    def _compute_optimal_batch_size(self) -> int:
        """Compute optimal batch size based on recent performance."""
        if len(self._performance_history) < 2:
            return self.current_batch_size
        
        recent_memory = np.mean(self._memory_history[-3:])
        recent_performance = self._performance_history[-3:]
        
        # Memory-based adjustment
        if recent_memory > self.target_memory_usage:
            # Reduce batch size to free memory
            reduction_factor = 0.8
            suggested_size = int(self.current_batch_size * reduction_factor)
        elif recent_memory < self.target_memory_usage * 0.6:
            # Increase batch size to better utilize memory
            increase_factor = 1.2
            suggested_size = int(self.current_batch_size * increase_factor)
        else:
            # Performance-based adjustment
            efficiencies = [p['efficiency'] for p in recent_performance]
            
            if len(efficiencies) >= 2:
                if efficiencies[-1] < efficiencies[-2] * 0.95:
                    # Performance decreased, try smaller batch
                    suggested_size = int(self.current_batch_size * 0.9)
                elif efficiencies[-1] > efficiencies[-2] * 1.05:
                    # Performance improved, try larger batch
                    suggested_size = int(self.current_batch_size * 1.1)
                else:
                    suggested_size = self.current_batch_size
            else:
                suggested_size = self.current_batch_size
        
        # Clamp to bounds
        return max(self.min_batch_size, min(self.max_batch_size, suggested_size))
    
    def get_current_batch_size(self) -> int:
        """Get current optimal batch size."""
        return self.current_batch_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch sizing statistics."""
        if not self._performance_history:
            return {}
        
        recent_perf = self._performance_history[-5:]
        
        return {
            'current_batch_size': self.current_batch_size,
            'adjustments_made': self._adjustment_count,
            'avg_throughput': np.mean([p['throughput'] for p in recent_perf]),
            'avg_memory_usage': np.mean([p['memory_usage'] for p in recent_perf]),
            'avg_efficiency': np.mean([p['efficiency'] for p in recent_perf]),
            'performance_trend': 'improving' if len(recent_perf) >= 2 and 
                               recent_perf[-1]['efficiency'] > recent_perf[0]['efficiency']
                               else 'stable' if len(recent_perf) >= 2 else 'unknown'
        }


class MemoryOptimizer:
    """Advanced memory optimization for large model training and inference."""
    
    def __init__(self):
        """Initialize memory optimizer."""
        self._memory_pools: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self._allocation_stats = {
            'total_allocations': 0,
            'pool_hits': 0,
            'memory_saved_mb': 0.0
        }
        self._lock = threading.Lock()
    
    def get_tensor(self, 
                  shape: Tuple[int, ...], 
                  dtype: torch.dtype, 
                  device: torch.device) -> torch.Tensor:
        """Get tensor from memory pool or allocate new one."""
        pool_key = f"{shape}_{dtype}_{device}"
        
        with self._lock:
            if pool_key in self._memory_pools and self._memory_pools[pool_key]:
                # Reuse from pool
                tensor = self._memory_pools[pool_key].pop()
                tensor.zero_()  # Clear data
                self._allocation_stats['pool_hits'] += 1
                return tensor
            else:
                # Allocate new tensor
                tensor = torch.zeros(shape, dtype=dtype, device=device)
                self._allocation_stats['total_allocations'] += 1
                return tensor
    
    def return_tensor(self, tensor: torch.Tensor, max_pool_size: int = 10):
        """Return tensor to memory pool for reuse."""
        if tensor.numel() == 0:
            return
        
        pool_key = f"{tuple(tensor.shape)}_{tensor.dtype}_{tensor.device}"
        
        with self._lock:
            if len(self._memory_pools[pool_key]) < max_pool_size:
                # Add to pool for reuse
                self._memory_pools[pool_key].append(tensor.detach())
                memory_saved = tensor.numel() * tensor.element_size() / 1024 / 1024
                self._allocation_stats['memory_saved_mb'] += memory_saved
    
    def clear_pools(self):
        """Clear all memory pools."""
        with self._lock:
            self._memory_pools.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory optimization statistics."""
        with self._lock:
            pool_sizes = {key: len(tensors) for key, tensors in self._memory_pools.items()}
            
            return {
                'total_allocations': self._allocation_stats['total_allocations'],
                'pool_hits': self._allocation_stats['pool_hits'],
                'pool_hit_rate': (self._allocation_stats['pool_hits'] / 
                                max(1, self._allocation_stats['total_allocations'])),
                'memory_saved_mb': self._allocation_stats['memory_saved_mb'],
                'active_pools': len(self._memory_pools),
                'pool_sizes': pool_sizes
            }
    
    @staticmethod
    def optimize_model_memory(model: nn.Module) -> nn.Module:
        """Apply memory optimizations to model."""
        # Enable memory efficient attention if available
        for module in model.modules():
            if hasattr(module, 'enable_memory_efficient_attention'):
                module.enable_memory_efficient_attention()
        
        # Apply gradient checkpointing to large layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)) and hasattr(module, 'weight'):
                if module.weight.numel() > 1000000:  # Large layers
                    # Wrap in checkpoint
                    original_forward = module.forward
                    
                    def checkpointed_forward(*args, **kwargs):
                        if module.training:
                            return torch.utils.checkpoint.checkpoint(
                                original_forward, *args, **kwargs
                            )
                        else:
                            return original_forward(*args, **kwargs)
                    
                    module.forward = checkpointed_forward
        
        return model
    
    @staticmethod
    def memory_efficient_attention(query: torch.Tensor, 
                                 key: torch.Tensor, 
                                 value: torch.Tensor,
                                 chunk_size: int = 1024) -> torch.Tensor:
        """Memory efficient attention computation."""
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        if seq_len <= chunk_size:
            # Use standard attention for small sequences
            scores = torch.matmul(query, key.transpose(-2, -1)) / np.sqrt(head_dim)
            attn_weights = torch.softmax(scores, dim=-1)
            return torch.matmul(attn_weights, value)
        
        # Chunked attention for large sequences
        output = torch.zeros_like(query)
        
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            query_chunk = query[:, :, i:end_i, :]
            
            # Compute attention for this chunk against all keys/values
            scores_chunk = torch.matmul(query_chunk, key.transpose(-2, -1)) / np.sqrt(head_dim)
            attn_weights_chunk = torch.softmax(scores_chunk, dim=-1)
            output[:, :, i:end_i, :] = torch.matmul(attn_weights_chunk, value)
        
        return output