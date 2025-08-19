"""
Advanced performance optimization engine for probabilistic neural operators.

This module provides intelligent performance optimization including:
- Adaptive resource allocation
- Memory pool management  
- Computation graph optimization
- Dynamic load balancing
- Performance profiling and auto-tuning
"""

import time
import threading
import queue
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from collections import deque, defaultdict
import functools
import weakref

logger = logging.getLogger(__name__)

class OptimizationLevel(Enum):
    """Performance optimization levels."""
    NONE = "none"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"

class ResourceType(Enum):
    """Resource types for optimization."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"

@dataclass
class PerformanceMetric:
    """Performance metric tracking."""
    name: str
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    resource_type: ResourceType = ResourceType.CPU
    optimization_target: float = 0.0

@dataclass
class OptimizationHint:
    """Optimization hint for specific operations."""
    operation_name: str
    resource_type: ResourceType
    expected_load: float
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

class AdvancedMemoryPool:
    """Memory pool with intelligent allocation and garbage collection."""
    
    def __init__(self, 
                 initial_size: int = 1024 * 1024,  # 1MB
                 max_size: int = 1024 * 1024 * 100,  # 100MB
                 growth_factor: float = 2.0):
        """Initialize memory pool.
        
        Args:
            initial_size: Initial pool size in bytes
            max_size: Maximum pool size in bytes
            growth_factor: Growth factor when expanding pool
        """
        self.initial_size = initial_size
        self.max_size = max_size
        self.growth_factor = growth_factor
        
        # Memory management
        self.pool = bytearray(initial_size)
        self.allocated_blocks = {}
        self.free_blocks = [(0, initial_size)]
        self.current_size = initial_size
        self.total_allocated = 0
        
        # Statistics
        self.allocation_count = 0
        self.deallocation_count = 0
        self.fragmentation_events = 0
        
        # Thread safety
        self.lock = threading.Lock()
    
    def allocate(self, size: int) -> Optional[int]:
        """Allocate memory block from pool.
        
        Args:
            size: Size in bytes to allocate
            
        Returns:
            Offset in pool or None if allocation failed
        """
        with self.lock:
            # Find suitable free block
            best_fit_idx = -1
            best_fit_size = float('inf')
            
            for i, (offset, block_size) in enumerate(self.free_blocks):
                if block_size >= size and block_size < best_fit_size:
                    best_fit_idx = i
                    best_fit_size = block_size
            
            if best_fit_idx == -1:
                # Try to expand pool
                if self._expand_pool(size):
                    return self.allocate(size)  # Retry after expansion
                return None
            
            # Allocate from best fit block
            offset, block_size = self.free_blocks.pop(best_fit_idx)
            
            # If block is larger than needed, split it
            if block_size > size:
                remaining_offset = offset + size
                remaining_size = block_size - size
                self.free_blocks.append((remaining_offset, remaining_size))
                self.free_blocks.sort()  # Keep sorted for efficiency
            
            # Record allocation
            self.allocated_blocks[offset] = size
            self.total_allocated += size
            self.allocation_count += 1
            
            return offset
    
    def deallocate(self, offset: int) -> bool:
        """Deallocate memory block.
        
        Args:
            offset: Offset of block to deallocate
            
        Returns:
            True if successful, False otherwise
        """
        with self.lock:
            if offset not in self.allocated_blocks:
                return False
            
            size = self.allocated_blocks.pop(offset)
            self.total_allocated -= size
            self.deallocation_count += 1
            
            # Add to free blocks and merge adjacent blocks
            self.free_blocks.append((offset, size))
            self._merge_free_blocks()
            
            return True
    
    def _expand_pool(self, min_additional_size: int) -> bool:
        """Expand memory pool if possible."""
        if self.current_size >= self.max_size:
            return False
        
        # Calculate new size
        new_size = min(
            max(int(self.current_size * self.growth_factor), 
                self.current_size + min_additional_size),
            self.max_size
        )
        
        if new_size <= self.current_size:
            return False
        
        # Expand pool
        additional_size = new_size - self.current_size
        self.pool.extend(bytearray(additional_size))
        
        # Add new space to free blocks
        self.free_blocks.append((self.current_size, additional_size))
        self.current_size = new_size
        
        self._merge_free_blocks()
        return True
    
    def _merge_free_blocks(self):
        """Merge adjacent free blocks to reduce fragmentation."""
        if len(self.free_blocks) < 2:
            return
        
        self.free_blocks.sort()
        merged = []
        current_offset, current_size = self.free_blocks[0]
        
        for offset, size in self.free_blocks[1:]:
            if current_offset + current_size == offset:
                # Adjacent blocks, merge them
                current_size += size
            else:
                merged.append((current_offset, current_size))
                current_offset, current_size = offset, size
        
        merged.append((current_offset, current_size))
        
        if len(merged) < len(self.free_blocks):
            self.fragmentation_events += 1
        
        self.free_blocks = merged
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self.lock:
            fragmentation = len(self.free_blocks) / max(1, self.current_size // 1024)
            utilization = self.total_allocated / self.current_size if self.current_size > 0 else 0
            
            return {
                "current_size": self.current_size,
                "max_size": self.max_size,
                "total_allocated": self.total_allocated,
                "utilization": utilization,
                "fragmentation_score": fragmentation,
                "allocation_count": self.allocation_count,
                "deallocation_count": self.deallocation_count,
                "fragmentation_events": self.fragmentation_events,
                "free_blocks": len(self.free_blocks)
            }

class PerformanceProfiler:
    """Advanced performance profiler with intelligent insights."""
    
    def __init__(self, max_history: int = 10000):
        """Initialize performance profiler.
        
        Args:
            max_history: Maximum number of performance records to keep
        """
        self.max_history = max_history
        
        # Performance tracking
        self.metrics_history = defaultdict(deque)
        self.operation_times = defaultdict(list)
        self.resource_usage = defaultdict(deque)
        
        # Profiling state
        self.active_operations = {}
        self.profiling_enabled = True
        
        # Thread safety
        self.lock = threading.Lock()
    
    def start_operation(self, operation_name: str, metadata: Optional[Dict] = None):
        """Start profiling an operation."""
        if not self.profiling_enabled:
            return
        
        with self.lock:
            self.active_operations[operation_name] = {
                'start_time': time.time(),
                'metadata': metadata or {}
            }
    
    def end_operation(self, operation_name: str) -> Optional[float]:
        """End profiling an operation and return duration."""
        if not self.profiling_enabled or operation_name not in self.active_operations:
            return None
        
        with self.lock:
            start_info = self.active_operations.pop(operation_name)
            duration = time.time() - start_info['start_time']
            
            # Record operation time
            self.operation_times[operation_name].append(duration)
            
            # Limit history
            if len(self.operation_times[operation_name]) > self.max_history:
                self.operation_times[operation_name] = \
                    self.operation_times[operation_name][-self.max_history:]
            
            return duration
    
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric."""
        if not self.profiling_enabled:
            return
        
        with self.lock:
            self.metrics_history[metric.name].append(metric)
            
            # Limit history
            if len(self.metrics_history[metric.name]) > self.max_history:
                self.metrics_history[metric.name].popleft()
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        with self.lock:
            times = self.operation_times.get(operation_name, [])
            
            if not times:
                return {"operation": operation_name, "count": 0}
            
            return {
                "operation": operation_name,
                "count": len(times),
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
                "total_time": sum(times),
                "recent_times": times[-10:]  # Last 10 operations
            }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get all performance statistics."""
        with self.lock:
            all_stats = {}
            
            for operation_name in self.operation_times:
                all_stats[operation_name] = self.get_operation_stats(operation_name)
            
            return all_stats
    
    def identify_bottlenecks(self, threshold_percentile: float = 0.95) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        with self.lock:
            for operation_name, times in self.operation_times.items():
                if len(times) < 10:  # Need sufficient data
                    continue
                
                # Calculate percentile threshold
                sorted_times = sorted(times)
                threshold_idx = int(len(sorted_times) * threshold_percentile)
                threshold_time = sorted_times[threshold_idx]
                
                # Check if recent operations exceed threshold
                recent_times = times[-20:]  # Last 20 operations
                slow_operations = [t for t in recent_times if t > threshold_time]
                
                if len(slow_operations) > len(recent_times) * 0.3:  # >30% slow
                    bottlenecks.append({
                        "operation": operation_name,
                        "avg_time": sum(times) / len(times),
                        "threshold_time": threshold_time,
                        "slow_ratio": len(slow_operations) / len(recent_times),
                        "recommendation": f"Optimize {operation_name} - frequently exceeds performance threshold"
                    })
            
            # Sort by severity (slow ratio)
            bottlenecks.sort(key=lambda x: x['slow_ratio'], reverse=True)
            
            return bottlenecks

class AdaptiveOptimizer:
    """Adaptive optimizer that learns and adjusts based on performance patterns."""
    
    def __init__(self, 
                 optimization_level: OptimizationLevel = OptimizationLevel.BASIC,
                 learning_rate: float = 0.1):
        """Initialize adaptive optimizer.
        
        Args:
            optimization_level: Default optimization level
            learning_rate: Rate of adaptation (0.0 to 1.0)
        """
        self.optimization_level = optimization_level
        self.learning_rate = learning_rate
        
        # Components
        self.memory_pool = AdvancedMemoryPool()
        self.profiler = PerformanceProfiler()
        
        # Optimization state
        self.optimization_strategies = {}
        self.performance_baselines = {}
        self.adaptation_history = []
        
        # Adaptive parameters
        self.cache_hit_rates = defaultdict(float)
        self.load_patterns = defaultdict(list)
        self.resource_utilization = defaultdict(deque)
        
        # Control
        self.optimization_enabled = True
        self.lock = threading.Lock()
    
    def optimize_operation(self, 
                          operation_name: str,
                          operation_func: Callable,
                          *args, 
                          **kwargs) -> Any:
        """Optimize execution of an operation."""
        if not self.optimization_enabled:
            return operation_func(*args, **kwargs)
        
        # Start profiling
        self.profiler.start_operation(operation_name)
        
        try:
            # Apply optimizations based on learned patterns
            optimized_func = self._apply_optimizations(operation_name, operation_func)
            
            # Execute with optimizations
            result = optimized_func(*args, **kwargs)
            
            # Record successful execution
            execution_time = self.profiler.end_operation(operation_name)
            self._update_performance_baseline(operation_name, execution_time)
            
            return result
            
        except Exception as e:
            # Handle optimization failures gracefully
            logger.warning(f"Optimization failed for {operation_name}: {e}")
            
            # Fall back to original function
            self.profiler.end_operation(operation_name)
            return operation_func(*args, **kwargs)
    
    def _apply_optimizations(self, 
                           operation_name: str, 
                           operation_func: Callable) -> Callable:
        """Apply learned optimizations to an operation."""
        
        # Check if we have optimization strategies for this operation
        if operation_name not in self.optimization_strategies:
            self.optimization_strategies[operation_name] = self._learn_optimization_strategy(operation_name)
        
        strategy = self.optimization_strategies[operation_name]
        optimized_func = operation_func
        
        # Apply caching if beneficial
        if strategy.get('use_caching', False):
            optimized_func = self._add_caching(optimized_func, operation_name)
        
        # Apply memory pooling if beneficial
        if strategy.get('use_memory_pool', False):
            optimized_func = self._add_memory_pooling(optimized_func, operation_name)
        
        # Apply parallel execution if beneficial
        if strategy.get('use_parallel', False):
            optimized_func = self._add_parallelization(optimized_func, operation_name)
        
        return optimized_func
    
    def _learn_optimization_strategy(self, operation_name: str) -> Dict[str, Any]:
        """Learn optimization strategy for an operation."""
        
        # Get historical performance data
        stats = self.profiler.get_operation_stats(operation_name)
        
        if stats['count'] < 5:
            # Not enough data, use basic strategy
            return {
                'use_caching': False,
                'use_memory_pool': False,
                'use_parallel': False,
                'confidence': 0.1
            }
        
        # Analyze patterns
        avg_time = stats['avg_time']
        variability = (stats['max_time'] - stats['min_time']) / avg_time if avg_time > 0 else 0
        
        strategy = {
            'use_caching': avg_time > 0.1 and variability < 0.5,  # Slow, consistent operations
            'use_memory_pool': avg_time > 0.05,  # Moderately slow operations
            'use_parallel': avg_time > 1.0 and variability > 0.3,  # Slow, variable operations
            'confidence': min(stats['count'] / 100.0, 1.0)  # Confidence based on data points
        }
        
        return strategy
    
    def _add_caching(self, func: Callable, operation_name: str) -> Callable:
        """Add caching optimization to function."""
        cache = {}
        cache_hits = 0
        cache_misses = 0
        
        @functools.wraps(func)
        def cached_func(*args, **kwargs):
            nonlocal cache_hits, cache_misses
            
            # Create cache key (simple string representation)
            cache_key = str(args) + str(sorted(kwargs.items()))
            
            if cache_key in cache:
                cache_hits += 1
                self.cache_hit_rates[operation_name] = cache_hits / (cache_hits + cache_misses)
                return cache[cache_key]
            else:
                cache_misses += 1
                result = func(*args, **kwargs)
                
                # Limit cache size
                if len(cache) > 1000:
                    # Remove oldest entries (simple LRU approximation)
                    oldest_keys = list(cache.keys())[:100]
                    for key in oldest_keys:
                        del cache[key]
                
                cache[cache_key] = result
                self.cache_hit_rates[operation_name] = cache_hits / (cache_hits + cache_misses)
                return result
        
        return cached_func
    
    def _add_memory_pooling(self, func: Callable, operation_name: str) -> Callable:
        """Add memory pooling optimization to function."""
        
        @functools.wraps(func)
        def pooled_func(*args, **kwargs):
            # This is a placeholder for memory pooling integration
            # In a real implementation, this would use the memory pool for allocations
            return func(*args, **kwargs)
        
        return pooled_func
    
    def _add_parallelization(self, func: Callable, operation_name: str) -> Callable:
        """Add parallelization optimization to function."""
        
        @functools.wraps(func)
        def parallel_func(*args, **kwargs):
            # This is a placeholder for parallelization
            # In a real implementation, this would detect parallelizable work
            return func(*args, **kwargs)
        
        return parallel_func
    
    def _update_performance_baseline(self, operation_name: str, execution_time: Optional[float]):
        """Update performance baseline for an operation."""
        if execution_time is None:
            return
        
        with self.lock:
            if operation_name not in self.performance_baselines:
                self.performance_baselines[operation_name] = deque(maxlen=100)
            
            self.performance_baselines[operation_name].append(execution_time)
            
            # Adapt optimization strategies based on performance
            self._adapt_strategies(operation_name)
    
    def _adapt_strategies(self, operation_name: str):
        """Adapt optimization strategies based on performance feedback."""
        if operation_name not in self.performance_baselines:
            return
        
        baselines = list(self.performance_baselines[operation_name])
        if len(baselines) < 10:
            return
        
        # Calculate performance trend
        recent_performance = sum(baselines[-5:]) / 5
        historical_performance = sum(baselines[:-5]) / len(baselines[:-5])
        
        improvement_ratio = historical_performance / recent_performance if recent_performance > 0 else 1.0
        
        # Adapt strategy based on performance trend
        if operation_name in self.optimization_strategies:
            strategy = self.optimization_strategies[operation_name]
            
            # If performance improved, increase confidence in current strategies
            if improvement_ratio > 1.1:  # 10% improvement
                strategy['confidence'] = min(strategy['confidence'] * 1.1, 1.0)
            
            # If performance degraded, reduce confidence and try different strategies
            elif improvement_ratio < 0.9:  # 10% degradation
                strategy['confidence'] = max(strategy['confidence'] * 0.9, 0.1)
                
                # Try different optimization combinations
                if strategy['confidence'] < 0.5:
                    strategy['use_caching'] = not strategy['use_caching']
                    strategy['use_parallel'] = not strategy['use_parallel']
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        with self.lock:
            # Analyze performance bottlenecks
            bottlenecks = self.profiler.identify_bottlenecks()
            
            # Calculate overall metrics
            total_operations = sum(len(times) for times in self.profiler.operation_times.values())
            avg_cache_hit_rate = sum(self.cache_hit_rates.values()) / len(self.cache_hit_rates) if self.cache_hit_rates else 0
            
            # Memory pool statistics
            memory_stats = self.memory_pool.get_stats()
            
            return {
                "optimization_level": self.optimization_level.value,
                "total_operations_profiled": total_operations,
                "avg_cache_hit_rate": avg_cache_hit_rate,
                "performance_bottlenecks": bottlenecks,
                "memory_pool_stats": memory_stats,
                "optimization_strategies": dict(self.optimization_strategies),
                "cache_hit_rates": dict(self.cache_hit_rates),
                "recommendations": self._generate_recommendations()
            }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Check cache performance
        if self.cache_hit_rates:
            avg_hit_rate = sum(self.cache_hit_rates.values()) / len(self.cache_hit_rates)
            if avg_hit_rate < 0.3:
                recommendations.append("Consider adjusting caching strategies - low cache hit rate detected")
        
        # Check memory utilization
        memory_stats = self.memory_pool.get_stats()
        if memory_stats['utilization'] > 0.8:
            recommendations.append("High memory utilization detected - consider increasing memory pool size")
        
        if memory_stats['fragmentation_score'] > 0.5:
            recommendations.append("Memory fragmentation detected - consider periodic defragmentation")
        
        # Check for bottlenecks
        bottlenecks = self.profiler.identify_bottlenecks()
        if bottlenecks:
            recommendations.append(f"Performance bottlenecks detected in {len(bottlenecks)} operations")
        
        return recommendations

# Global optimizer instance
global_optimizer = AdaptiveOptimizer()

# Convenience functions
def optimize(operation_name: str):
    """Decorator for automatic optimization."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return global_optimizer.optimize_operation(operation_name, func, *args, **kwargs)
        return wrapper
    return decorator

def profile_operation(operation_name: str):
    """Decorator for operation profiling."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            global_optimizer.profiler.start_operation(operation_name)
            try:
                result = func(*args, **kwargs)
                global_optimizer.profiler.end_operation(operation_name)
                return result
            except Exception as e:
                global_optimizer.profiler.end_operation(operation_name)
                raise
        return wrapper
    return decorator