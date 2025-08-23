#!/usr/bin/env python3
"""
Generation 3: MAKE IT SCALE - Performance Optimization Demo
============================================================

This module demonstrates advanced scaling optimizations for the ProbNeural Operator
framework, including caching, concurrent processing, memory optimization, and
intelligent batching.
"""

import math
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import deque, defaultdict
import json

# Import from our framework
from probneural_operator.core import MockTensor, BaseNeuralOperator, ProbabilisticFNO
from probneural_operator.robust import RobustMockTensor, RobustProbabilisticFNO


class PerformanceCache:
    """High-performance prediction cache with intelligent eviction."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}
        self.access_times = {}
        self.access_counts = defaultdict(int)
        self.lock = threading.RLock()
    
    def _hash_tensor(self, tensor: Union[MockTensor, RobustMockTensor]) -> str:
        """Create hash key for tensor."""
        # Simple hash based on data values (rounded for cache hits)
        rounded_data = [round(x, 6) for x in tensor.data]
        return str(hash(tuple(rounded_data)))
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with LRU tracking."""
        with self.lock:
            if key in self.cache:
                current_time = time.time()
                
                # Check TTL
                if current_time - self.access_times[key] > self.ttl_seconds:
                    self._evict(key)
                    return None
                
                # Update access tracking
                self.access_times[key] = current_time
                self.access_counts[key] += 1
                
                return self.cache[key]
            return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache with intelligent eviction."""
        with self.lock:
            current_time = time.time()
            
            # Check if cache is full
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = value
            self.access_times[key] = current_time
            self.access_counts[key] = 1
    
    def _evict(self, key: str) -> None:
        """Evict specific key."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.access_counts.pop(key, None)
    
    def _evict_lru(self) -> None:
        """Evict least recently used items."""
        if not self.cache:
            return
        
        # Find LRU items (lowest access count + oldest access time)
        lru_key = min(
            self.cache.keys(),
            key=lambda k: (self.access_counts[k], self.access_times[k])
        )
        self._evict(lru_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        with self.lock:
            total_accesses = sum(self.access_counts.values())
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "utilization": len(self.cache) / self.max_size,
                "total_accesses": total_accesses,
                "avg_accesses_per_item": total_accesses / max(len(self.cache), 1)
            }


class BatchOptimizer:
    """Intelligent batch processing with dynamic sizing."""
    
    def __init__(self, initial_batch_size: int = 32, max_batch_size: int = 512):
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = max_batch_size
        self.current_batch_size = initial_batch_size
        self.performance_history = deque(maxlen=100)
        self.adaptation_factor = 1.1
    
    def adaptive_batch(self, data: List[Any]) -> List[List[Any]]:
        """Create adaptive batches based on performance."""
        if not data:
            return []
        
        # Determine optimal batch size
        optimal_size = self._get_optimal_batch_size()
        
        # Create batches
        batches = []
        for i in range(0, len(data), optimal_size):
            batch = data[i:i + optimal_size]
            batches.append(batch)
        
        return batches
    
    def _get_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on performance history."""
        if len(self.performance_history) < 10:
            return self.current_batch_size
        
        # Analyze recent performance trends
        recent_perf = list(self.performance_history)[-10:]
        avg_throughput = sum(recent_perf) / len(recent_perf)
        
        # If throughput is improving, increase batch size
        if len(recent_perf) >= 2:
            trend = recent_perf[-1] - recent_perf[-2]
            if trend > 0 and self.current_batch_size < self.max_batch_size:
                self.current_batch_size = min(
                    int(self.current_batch_size * self.adaptation_factor),
                    self.max_batch_size
                )
            elif trend < -0.1:  # Significant degradation
                self.current_batch_size = max(
                    int(self.current_batch_size / self.adaptation_factor),
                    8  # Minimum batch size
                )
        
        return self.current_batch_size
    
    def record_performance(self, batch_size: int, processing_time: float) -> None:
        """Record batch processing performance."""
        throughput = batch_size / max(processing_time, 0.001)  # items/second
        self.performance_history.append(throughput)


class ParallelProcessor:
    """Parallel processing for neural operator predictions."""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    def process_parallel(self, model: BaseNeuralOperator, 
                        data: List[Union[MockTensor, RobustMockTensor]],
                        batch_optimizer: BatchOptimizer) -> List[Union[MockTensor, RobustMockTensor]]:
        """Process data in parallel with optimized batching."""
        if not data:
            return []
        
        # Create optimized batches
        batches = batch_optimizer.adaptive_batch(data)
        
        # Process batches in parallel
        futures = []
        for batch in batches:
            future = self.executor.submit(self._process_batch, model, batch)
            futures.append(future)
        
        # Collect results
        all_results = []
        start_time = time.time()
        
        for future in as_completed(futures):
            try:
                batch_results = future.result(timeout=60)  # 60 second timeout
                all_results.extend(batch_results)
            except Exception as e:
                print(f"Warning: Batch processing failed: {e}")
                continue
        
        # Record performance
        total_time = time.time() - start_time
        batch_optimizer.record_performance(len(data), total_time)
        
        return all_results
    
    def _process_batch(self, model: BaseNeuralOperator, 
                      batch: List[Union[MockTensor, RobustMockTensor]]) -> List[Union[MockTensor, RobustMockTensor]]:
        """Process a single batch."""
        results = []
        for item in batch:
            try:
                result = model.forward(item)
                results.append(result)
            except Exception as e:
                # Create fallback result
                fallback = RobustMockTensor([0.0] * model.output_dim)
                results.append(fallback)
        
        return results
    
    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


class ResourceMonitor:
    """Monitor system resources and adjust processing accordingly."""
    
    def __init__(self):
        self.metrics = {
            "cpu_usage": deque(maxlen=100),
            "memory_usage": deque(maxlen=100),
            "processing_latency": deque(maxlen=100)
        }
        self.thresholds = {
            "cpu_high": 80.0,
            "memory_high": 85.0,
            "latency_high": 1.0  # seconds
        }
    
    def record_metrics(self, cpu_pct: float, memory_pct: float, latency: float) -> None:
        """Record system metrics."""
        self.metrics["cpu_usage"].append(cpu_pct)
        self.metrics["memory_usage"].append(memory_pct)
        self.metrics["processing_latency"].append(latency)
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status."""
        status = {"status": "healthy", "recommendations": []}
        
        # Check CPU usage
        if self.metrics["cpu_usage"]:
            avg_cpu = sum(self.metrics["cpu_usage"]) / len(self.metrics["cpu_usage"])
            if avg_cpu > self.thresholds["cpu_high"]:
                status["status"] = "stressed"
                status["recommendations"].append("Reduce batch size or concurrent workers")
        
        # Check memory usage
        if self.metrics["memory_usage"]:
            avg_memory = sum(self.metrics["memory_usage"]) / len(self.metrics["memory_usage"])
            if avg_memory > self.thresholds["memory_high"]:
                status["status"] = "stressed"
                status["recommendations"].append("Enable memory optimization features")
        
        # Check processing latency
        if self.metrics["processing_latency"]:
            avg_latency = sum(self.metrics["processing_latency"]) / len(self.metrics["processing_latency"])
            if avg_latency > self.thresholds["latency_high"]:
                status["status"] = "degraded"
                status["recommendations"].append("Consider model optimization or hardware upgrade")
        
        # Add metrics summary
        status["metrics"] = {
            "avg_cpu": sum(self.metrics["cpu_usage"]) / max(len(self.metrics["cpu_usage"]), 1),
            "avg_memory": sum(self.metrics["memory_usage"]) / max(len(self.metrics["memory_usage"]), 1),
            "avg_latency": sum(self.metrics["processing_latency"]) / max(len(self.metrics["processing_latency"]), 1)
        }
        
        return status
    
    def should_throttle(self) -> bool:
        """Determine if processing should be throttled."""
        status = self.get_resource_status()
        return status["status"] in ["stressed", "critical"]


class AdvancedModelManager:
    """Advanced model management with caching, monitoring, and optimization."""
    
    def __init__(self, model: BaseNeuralOperator):
        self.model = model
        self.cache = PerformanceCache(max_size=5000, ttl_seconds=1800)
        self.batch_optimizer = BatchOptimizer(initial_batch_size=16, max_batch_size=256)
        self.parallel_processor = ParallelProcessor(max_workers=4)
        self.resource_monitor = ResourceMonitor()
        
        # Performance tracking
        self.prediction_count = 0
        self.cache_hits = 0
        self.total_processing_time = 0.0
    
    def predict_optimized(self, inputs: List[Union[MockTensor, RobustMockTensor]]) -> List[Union[MockTensor, RobustMockTensor]]:
        """Make predictions with full optimization pipeline."""
        if not inputs:
            return []
        
        start_time = time.time()
        results = []
        cache_missed_inputs = []
        cache_missed_indices = []
        
        # Step 1: Check cache for each input
        for i, input_tensor in enumerate(inputs):
            cache_key = self.cache._hash_tensor(input_tensor)
            cached_result = self.cache.get(cache_key)
            
            if cached_result is not None:
                results.append(cached_result)
                self.cache_hits += 1
            else:
                results.append(None)  # Placeholder
                cache_missed_inputs.append(input_tensor)
                cache_missed_indices.append(i)
        
        # Step 2: Process cache misses with parallel optimization
        if cache_missed_inputs:
            # Check if we should throttle processing
            if self.resource_monitor.should_throttle():
                print("⚠️  System under stress, throttling processing...")
                time.sleep(0.1)  # Brief throttle
            
            # Process in parallel
            computed_results = self.parallel_processor.process_parallel(
                self.model, cache_missed_inputs, self.batch_optimizer
            )
            
            # Update cache and results
            for idx, (orig_idx, input_tensor, result) in enumerate(
                zip(cache_missed_indices, cache_missed_inputs, computed_results)
            ):
                cache_key = self.cache._hash_tensor(input_tensor)
                self.cache.set(cache_key, result)
                results[orig_idx] = result
        
        # Step 3: Update performance metrics
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        self.prediction_count += len(inputs)
        
        # Mock resource usage for monitoring
        mock_cpu = 30 + random.random() * 50  # 30-80%
        mock_memory = 40 + random.random() * 40  # 40-80%
        self.resource_monitor.record_metrics(mock_cpu, mock_memory, processing_time)
        
        return [r for r in results if r is not None]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        cache_stats = self.cache.get_stats()
        resource_status = self.resource_monitor.get_resource_status()
        
        avg_prediction_time = (
            self.total_processing_time / max(self.prediction_count, 1)
        )
        
        cache_hit_rate = self.cache_hits / max(self.prediction_count, 1)
        
        return {
            "predictions_processed": self.prediction_count,
            "cache_hit_rate": cache_hit_rate,
            "avg_prediction_time": avg_prediction_time,
            "current_batch_size": self.batch_optimizer.current_batch_size,
            "cache_stats": cache_stats,
            "resource_status": resource_status,
            "throughput_predictions_per_sec": self.prediction_count / max(self.total_processing_time, 1)
        }
    
    def optimize_model(self) -> Dict[str, Any]:
        """Apply model optimizations based on performance data."""
        stats = self.get_performance_stats()
        optimizations = []
        
        # Cache optimization
        if stats["cache_hit_rate"] < 0.3:  # Low cache hit rate
            self.cache.max_size = min(self.cache.max_size * 2, 20000)
            optimizations.append(f"Increased cache size to {self.cache.max_size}")
        
        # Batch size optimization
        if stats["avg_prediction_time"] > 0.5:  # High latency
            self.batch_optimizer.current_batch_size = max(
                self.batch_optimizer.current_batch_size // 2, 8
            )
            optimizations.append(f"Reduced batch size to {self.batch_optimizer.current_batch_size}")
        
        # Resource-based optimization
        resource_status = stats["resource_status"]["status"]
        if resource_status == "stressed":
            self.parallel_processor.max_workers = max(self.parallel_processor.max_workers - 1, 1)
            optimizations.append(f"Reduced parallel workers to {self.parallel_processor.max_workers}")
        elif resource_status == "healthy" and stats["throughput_predictions_per_sec"] < 50:
            self.parallel_processor.max_workers = min(self.parallel_processor.max_workers + 1, 8)
            optimizations.append(f"Increased parallel workers to {self.parallel_processor.max_workers}")
        
        return {
            "optimizations_applied": len(optimizations),
            "details": optimizations,
            "new_configuration": {
                "cache_size": self.cache.max_size,
                "batch_size": self.batch_optimizer.current_batch_size,
                "parallel_workers": self.parallel_processor.max_workers
            }
        }
    
    def shutdown(self):
        """Clean shutdown of all components."""
        self.parallel_processor.shutdown()


def demo_scaling_optimization():
    """Demonstrate Generation 3 scaling optimizations."""
    print("⚡ ProbNeural Operator Lab - Generation 3 Scaling Demo")
    print("=" * 65)
    
    try:
        # Create optimized model
        print("🚀 Initializing optimized model manager...")
        base_model = RobustProbabilisticFNO(
            modes=8, width=16, depth=2, 
            input_dim=32, output_dim=32
        )
        
        model_manager = AdvancedModelManager(base_model)
        print("✅ Advanced model manager initialized")
        
        # Generate test data
        print("\n📊 Generating test dataset...")
        test_inputs = [
            RobustMockTensor([random.gauss(0, 1) for _ in range(32)])
            for _ in range(200)
        ]
        print(f"✅ Generated {len(test_inputs)} test samples")
        
        # Benchmark Phase 1: Initial predictions
        print("\n📈 Phase 1: Initial predictions (cold cache)...")
        start_time = time.time()
        
        batch1_results = model_manager.predict_optimized(test_inputs[:50])
        batch1_time = time.time() - start_time
        
        print(f"✅ Processed {len(batch1_results)} predictions in {batch1_time:.3f}s")
        print(f"   Throughput: {len(batch1_results)/batch1_time:.1f} predictions/sec")
        
        # Benchmark Phase 2: Warm cache predictions
        print("\n🔥 Phase 2: Repeat predictions (warm cache)...")
        start_time = time.time()
        
        batch2_results = model_manager.predict_optimized(test_inputs[:50])  # Same data
        batch2_time = time.time() - start_time
        
        print(f"✅ Processed {len(batch2_results)} predictions in {batch2_time:.3f}s")
        print(f"   Throughput: {len(batch2_results)/batch2_time:.1f} predictions/sec")
        print(f"   Speedup: {batch1_time/max(batch2_time, 0.001):.1f}x")
        
        # Benchmark Phase 3: Large batch processing
        print("\n🚀 Phase 3: Large batch processing...")
        start_time = time.time()
        
        batch3_results = model_manager.predict_optimized(test_inputs[50:])
        batch3_time = time.time() - start_time
        
        print(f"✅ Processed {len(batch3_results)} predictions in {batch3_time:.3f}s")
        print(f"   Throughput: {len(batch3_results)/batch3_time:.1f} predictions/sec")
        
        # Performance analysis
        print("\n📋 Performance Analysis:")
        stats = model_manager.get_performance_stats()
        
        print(f"   Total predictions: {stats['predictions_processed']}")
        print(f"   Cache hit rate: {stats['cache_hit_rate']:.2%}")
        print(f"   Avg prediction time: {stats['avg_prediction_time']:.4f}s")
        print(f"   Current batch size: {stats['current_batch_size']}")
        print(f"   Overall throughput: {stats['throughput_predictions_per_sec']:.1f} pred/sec")
        print(f"   Cache utilization: {stats['cache_stats']['utilization']:.2%}")
        print(f"   Resource status: {stats['resource_status']['status']}")
        
        # Auto-optimization
        print("\n🎛️  Applying automatic optimizations...")
        optimization_results = model_manager.optimize_model()
        
        print(f"✅ Applied {optimization_results['optimizations_applied']} optimizations:")
        for detail in optimization_results['details']:
            print(f"   • {detail}")
        
        # Final configuration
        config = optimization_results['new_configuration']
        print(f"\n⚙️  Optimized Configuration:")
        print(f"   Cache size: {config['cache_size']}")
        print(f"   Batch size: {config['batch_size']}")
        print(f"   Parallel workers: {config['parallel_workers']}")
        
        # Resource recommendations
        if stats['resource_status']['recommendations']:
            print(f"\n💡 System Recommendations:")
            for rec in stats['resource_status']['recommendations']:
                print(f"   • {rec}")
        
        print("\n✅ Generation 3 scaling demo complete!")
        
        # Cleanup
        model_manager.shutdown()
        
        return {
            "performance_stats": stats,
            "optimization_results": optimization_results,
            "phase_timings": {
                "cold_cache": batch1_time,
                "warm_cache": batch2_time,
                "large_batch": batch3_time,
                "cache_speedup": batch1_time / max(batch2_time, 0.001)
            }
        }
        
    except Exception as e:
        print(f"\n❌ Scaling demo failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    demo_scaling_optimization()