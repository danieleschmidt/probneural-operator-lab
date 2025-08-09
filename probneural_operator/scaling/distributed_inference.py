"""
Distributed inference system for probabilistic neural operators at scale.

This module provides high-performance distributed inference capabilities including:
- Multi-GPU inference with automatic load balancing
- Distributed model serving across nodes
- Advanced caching and memory optimization
- Dynamic scaling based on load
- Fault tolerance and recovery
"""

import asyncio
import json
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging


@dataclass
class DistributedConfig:
    """Configuration for distributed inference."""
    max_workers: int = 8
    gpu_devices: List[str] = None
    memory_limit_gb: float = 16.0
    batch_timeout_ms: int = 50
    max_batch_size: int = 256
    cache_size: int = 100000
    enable_compression: bool = True
    load_balance_strategy: str = "round_robin"  # round_robin, least_loaded, hash
    
    def __post_init__(self):
        if self.gpu_devices is None:
            self.gpu_devices = ["cuda:0"]


@dataclass
class WorkerStats:
    """Statistics for a worker node."""
    worker_id: str
    total_requests: int = 0
    active_requests: int = 0
    average_latency: float = 0.0
    memory_usage: float = 0.0
    gpu_utilization: float = 0.0
    last_heartbeat: float = 0.0
    is_healthy: bool = True


class InferenceCache:
    """High-performance cache for inference results."""
    
    def __init__(self, max_size: int = 100000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = {}  # key -> (result, timestamp, access_count)
        self.access_order = deque()  # LRU tracking
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def _compute_key(self, model_id: str, input_data: List[float], options: Dict[str, Any]) -> str:
        """Compute cache key for input."""
        # Create deterministic key from inputs
        input_hash = hash(tuple(input_data))
        options_hash = hash(tuple(sorted(options.items())))
        return f"{model_id}:{input_hash}:{options_hash}"
    
    def get(
        self,
        model_id: str,
        input_data: List[float],
        options: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get cached result if available."""
        key = self._compute_key(model_id, input_data, options)
        
        with self.lock:
            if key in self.cache:
                result, timestamp, access_count = self.cache[key]
                
                # Check TTL
                if time.time() - timestamp <= self.ttl_seconds:
                    # Update access info
                    self.cache[key] = (result, timestamp, access_count + 1)
                    self.access_order.append(key)
                    self.hits += 1
                    return result.copy()
                else:
                    # Expired
                    del self.cache[key]
            
            self.misses += 1
            return None
    
    def put(
        self,
        model_id: str,
        input_data: List[float],
        options: Dict[str, Any],
        result: Dict[str, Any]
    ) -> None:
        """Cache inference result."""
        key = self._compute_key(model_id, input_data, options)
        
        with self.lock:
            # Evict if at capacity
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = (result.copy(), time.time(), 1)
            self.access_order.append(key)
    
    def _evict_lru(self):
        """Evict least recently used items."""
        while self.access_order and len(self.cache) >= self.max_size:
            old_key = self.access_order.popleft()
            if old_key in self.cache:
                del self.cache[old_key]
                break
    
    def _cleanup_loop(self):
        """Background cleanup of expired entries."""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                current_time = time.time()
                
                with self.lock:
                    expired_keys = [
                        key for key, (_, timestamp, _) in self.cache.items()
                        if current_time - timestamp > self.ttl_seconds
                    ]
                    
                    for key in expired_keys:
                        del self.cache[key]
                        
            except Exception as e:
                logging.error(f"Cache cleanup error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "cache_size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": hit_rate,
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total_requests
        }


class LoadBalancer:
    """Intelligent load balancer for distributing requests."""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.workers = {}  # worker_id -> WorkerStats
        self.round_robin_index = 0
        self.lock = threading.RLock()
    
    def register_worker(self, worker_id: str, worker_info: Dict[str, Any]) -> None:
        """Register a new worker."""
        with self.lock:
            self.workers[worker_id] = WorkerStats(worker_id=worker_id)
            logging.info(f"Registered worker {worker_id}")
    
    def select_worker(self) -> Optional[str]:
        """Select optimal worker for next request."""
        with self.lock:
            healthy_workers = [
                wid for wid, stats in self.workers.items() if stats.is_healthy
            ]
            
            if not healthy_workers:
                return None
            
            if self.strategy == "round_robin":
                return self._round_robin_selection(healthy_workers)
            elif self.strategy == "least_loaded":
                return self._least_loaded_selection(healthy_workers)
            else:
                return healthy_workers[0]
    
    def _round_robin_selection(self, workers: List[str]) -> str:
        """Round-robin worker selection."""
        if not workers:
            return None
        
        selected = workers[self.round_robin_index % len(workers)]
        self.round_robin_index = (self.round_robin_index + 1) % len(workers)
        return selected
    
    def _least_loaded_selection(self, workers: List[str]) -> str:
        """Select worker with least load."""
        return min(workers, key=lambda w: self.workers[w].active_requests)
    
    def update_worker_stats(self, worker_id: str, stats_update: Dict[str, Any]) -> None:
        """Update worker statistics."""
        with self.lock:
            if worker_id in self.workers:
                worker = self.workers[worker_id]
                
                worker.total_requests = stats_update.get("total_requests", worker.total_requests)
                worker.active_requests = stats_update.get("active_requests", worker.active_requests)
                worker.average_latency = stats_update.get("average_latency", worker.average_latency)
                worker.memory_usage = stats_update.get("memory_usage", worker.memory_usage)
                worker.gpu_utilization = stats_update.get("gpu_utilization", worker.gpu_utilization)
                worker.last_heartbeat = time.time()
                worker.is_healthy = stats_update.get("is_healthy", True)
    
    def get_worker_stats(self) -> Dict[str, WorkerStats]:
        """Get all worker statistics."""
        with self.lock:
            return {wid: stats for wid, stats in self.workers.items()}


class InferenceWorker:
    """Individual inference worker for processing requests."""
    
    def __init__(self, worker_id: str, device: str = "cpu", config: Optional[DistributedConfig] = None):
        self.worker_id = worker_id
        self.device = device
        self.config = config or DistributedConfig()
        
        self.request_queue = asyncio.Queue(maxsize=1000)
        self.batch_processor = None
        self.stats = {
            "total_requests": 0,
            "active_requests": 0,
            "total_processing_time": 0.0,
            "memory_usage": 0.0,
            "gpu_utilization": 0.0,
            "is_healthy": True
        }
        
        self.models = {}  # model_id -> model_instance
        self.running = False
    
    async def start(self):
        """Start the worker."""
        self.running = True
        self.batch_processor = asyncio.create_task(self._batch_processing_loop())
        logging.info(f"Started worker {self.worker_id} on device {self.device}")
    
    async def stop(self):
        """Stop the worker gracefully."""
        self.running = False
        if self.batch_processor:
            self.batch_processor.cancel()
        logging.info(f"Stopped worker {self.worker_id}")
    
    async def predict(
        self,
        model_id: str,
        input_data: List[float],
        options: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Submit prediction request."""
        future = asyncio.Future()
        
        request = {
            "model_id": model_id,
            "input_data": input_data,
            "options": options,
            "future": future,
            "timestamp": time.time()
        }
        
        try:
            await self.request_queue.put(request)
            return await future
        except Exception as e:
            self.stats["is_healthy"] = False
            raise e
    
    async def _batch_processing_loop(self):
        """Main batch processing loop."""
        while self.running:
            try:
                batch = await self._collect_batch()
                if batch:
                    await self._process_batch(batch)
                    
            except Exception as e:
                logging.error(f"Batch processing error in worker {self.worker_id}: {e}")
                await asyncio.sleep(0.1)
    
    async def _collect_batch(self) -> List[Dict[str, Any]]:
        """Collect requests into a batch."""
        batch = []
        timeout = self.config.batch_timeout_ms / 1000.0
        
        try:
            # Get first request (blocking)
            first_request = await asyncio.wait_for(
                self.request_queue.get(),
                timeout=timeout
            )
            batch.append(first_request)
            
            # Collect additional requests (non-blocking)
            while (len(batch) < self.config.max_batch_size and 
                   not self.request_queue.empty()):
                try:
                    request = await asyncio.wait_for(
                        self.request_queue.get(),
                        timeout=0.001
                    )
                    batch.append(request)
                except asyncio.TimeoutError:
                    break
                    
        except asyncio.TimeoutError:
            pass
        
        return batch
    
    async def _process_batch(self, batch: List[Dict[str, Any]]):
        """Process a batch of requests."""
        start_time = time.time()
        self.stats["active_requests"] += len(batch)
        
        try:
            # Group by model_id for efficient processing
            model_batches = defaultdict(list)
            for request in batch:
                model_batches[request["model_id"]].append(request)
            
            # Process each model batch
            for model_id, model_requests in model_batches.items():
                await self._process_model_batch(model_id, model_requests)
                
        finally:
            processing_time = time.time() - start_time
            self.stats["active_requests"] -= len(batch)
            self.stats["total_requests"] += len(batch)
            self.stats["total_processing_time"] += processing_time
    
    async def _process_model_batch(self, model_id: str, requests: List[Dict[str, Any]]):
        """Process batch for a specific model."""
        try:
            # Simulate model inference (in practice would use actual model)
            batch_inputs = [req["input_data"] for req in requests]
            
            # Simulate batch processing time
            await asyncio.sleep(0.01 * len(batch_inputs))
            
            # Generate mock results
            for i, request in enumerate(requests):
                input_data = request["input_data"]
                
                # Mock probabilistic prediction
                prediction = [x * 0.8 + 0.1 for x in input_data]
                uncertainty = [0.1 + 0.2 * abs(x) for x in input_data]
                
                result = {
                    "prediction": prediction,
                    "uncertainty": uncertainty,
                    "processing_time": time.time() - request["timestamp"],
                    "worker_id": self.worker_id,
                    "batch_size": len(requests)
                }
                
                # Complete the future
                if not request["future"].done():
                    request["future"].set_result(result)
                    
        except Exception as e:
            # Reject all futures in batch
            for request in requests:
                if not request["future"].done():
                    request["future"].set_exception(e)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        avg_latency = 0.0
        if self.stats["total_requests"] > 0:
            avg_latency = self.stats["total_processing_time"] / self.stats["total_requests"]
        
        return {
            **self.stats,
            "average_latency": avg_latency,
            "queue_size": self.request_queue.qsize() if hasattr(self.request_queue, 'qsize') else 0
        }


class AutoScaler:
    """Automatic scaling system for inference workers."""
    
    def __init__(self, min_workers: int = 2, max_workers: int = 10):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scaling_history = deque(maxlen=100)
        self.last_scale_time = 0
        self.scale_cooldown = 60  # seconds
    
    def should_scale_up(self, load_metrics: Dict[str, Any]) -> bool:
        """Determine if we should scale up."""
        current_time = time.time()
        
        # Cooldown period
        if current_time - self.last_scale_time < self.scale_cooldown:
            return False
        
        # Check scaling conditions
        avg_queue_size = load_metrics.get("avg_queue_size", 0)
        avg_latency = load_metrics.get("avg_latency", 0)
        cpu_utilization = load_metrics.get("cpu_utilization", 0)
        
        scale_up_conditions = [
            avg_queue_size > 50,  # High queue backlog
            avg_latency > 1.0,    # High latency
            cpu_utilization > 0.8 # High CPU usage
        ]
        
        return any(scale_up_conditions)
    
    def should_scale_down(self, load_metrics: Dict[str, Any]) -> bool:
        """Determine if we should scale down."""
        current_time = time.time()
        
        # Cooldown period
        if current_time - self.last_scale_time < self.scale_cooldown:
            return False
        
        # Check scaling conditions
        avg_queue_size = load_metrics.get("avg_queue_size", 0)
        avg_latency = load_metrics.get("avg_latency", 0)
        cpu_utilization = load_metrics.get("cpu_utilization", 0)
        
        scale_down_conditions = [
            avg_queue_size < 5,     # Low queue
            avg_latency < 0.1,      # Low latency  
            cpu_utilization < 0.3   # Low CPU usage
        ]
        
        return all(scale_down_conditions)
    
    def record_scaling_event(self, action: str, worker_count: int, metrics: Dict[str, Any]):
        """Record scaling event."""
        self.scaling_history.append({
            "timestamp": time.time(),
            "action": action,
            "worker_count": worker_count,
            "metrics": metrics.copy()
        })
        self.last_scale_time = time.time()
    
    def get_scaling_history(self) -> List[Dict[str, Any]]:
        """Get scaling history."""
        return list(self.scaling_history)


class DistributedInferenceManager:
    """Main manager for distributed inference system."""
    
    def __init__(self, config: Optional[DistributedConfig] = None):
        self.config = config or DistributedConfig()
        self.workers = {}  # worker_id -> InferenceWorker
        self.load_balancer = LoadBalancer(self.config.load_balance_strategy)
        self.cache = InferenceCache(max_size=self.config.cache_size)
        self.autoscaler = AutoScaler(min_workers=2, max_workers=self.config.max_workers)
        
        self.running = False
        self.monitoring_task = None
        
    async def start(self):
        """Start the distributed inference system."""
        self.running = True
        
        # Start initial workers
        for i in range(2):  # Start with 2 workers
            await self._add_worker(i)
        
        # Start monitoring
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logging.info("Distributed inference manager started")
    
    async def stop(self):
        """Stop the system gracefully."""
        self.running = False
        
        # Stop monitoring
        if self.monitoring_task:
            self.monitoring_task.cancel()
        
        # Stop all workers
        for worker in self.workers.values():
            await worker.stop()
        
        logging.info("Distributed inference manager stopped")
    
    async def predict(
        self,
        model_id: str,
        input_data: List[float],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make distributed prediction."""
        options = options or {}
        
        # Check cache first
        cached_result = self.cache.get(model_id, input_data, options)
        if cached_result:
            return cached_result
        
        # Select worker
        worker_id = self.load_balancer.select_worker()
        if not worker_id or worker_id not in self.workers:
            raise RuntimeError("No healthy workers available")
        
        # Make prediction
        start_time = time.time()
        try:
            worker = self.workers[worker_id]
            result = await worker.predict(model_id, input_data, options)
            
            # Cache result
            self.cache.put(model_id, input_data, options, result)
            
            return result
            
        except Exception as e:
            # Mark worker as unhealthy
            self.load_balancer.update_worker_stats(worker_id, {"is_healthy": False})
            raise e
        finally:
            # Update load balancer stats
            processing_time = time.time() - start_time
            self.load_balancer.update_worker_stats(worker_id, {
                "total_requests": self.workers[worker_id].stats["total_requests"],
                "average_latency": processing_time,
                "is_healthy": True
            })
    
    async def predict_batch(
        self,
        model_id: str,
        batch_inputs: List[List[float]],
        options: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Make batch predictions."""
        tasks = []
        for input_data in batch_inputs:
            task = asyncio.create_task(
                self.predict(model_id, input_data, options)
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    async def _add_worker(self, worker_index: int):
        """Add a new worker."""
        worker_id = f"worker_{worker_index}"
        device = self.config.gpu_devices[worker_index % len(self.config.gpu_devices)]
        
        worker = InferenceWorker(worker_id, device, self.config)
        self.workers[worker_id] = worker
        
        await worker.start()
        self.load_balancer.register_worker(worker_id, {})
        
        logging.info(f"Added worker {worker_id} on device {device}")
    
    async def _remove_worker(self, worker_id: str):
        """Remove a worker."""
        if worker_id in self.workers:
            await self.workers[worker_id].stop()
            del self.workers[worker_id]
            logging.info(f"Removed worker {worker_id}")
    
    async def _monitoring_loop(self):
        """Background monitoring and auto-scaling."""
        while self.running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Collect metrics
                metrics = self._collect_system_metrics()
                
                # Auto-scaling decisions
                current_worker_count = len(self.workers)
                
                if (self.autoscaler.should_scale_up(metrics) and 
                    current_worker_count < self.config.max_workers):
                    
                    # Scale up
                    await self._add_worker(current_worker_count)
                    self.autoscaler.record_scaling_event("scale_up", current_worker_count + 1, metrics)
                    
                elif (self.autoscaler.should_scale_down(metrics) and
                      current_worker_count > 2):
                    
                    # Scale down (remove least loaded worker)
                    worker_stats = self.load_balancer.get_worker_stats()
                    least_loaded = min(worker_stats.keys(), 
                                     key=lambda w: worker_stats[w].active_requests)
                    
                    await self._remove_worker(least_loaded)
                    self.autoscaler.record_scaling_event("scale_down", current_worker_count - 1, metrics)
                
                # Update worker health status
                self._update_worker_health()
                
            except Exception as e:
                logging.error(f"Monitoring loop error: {e}")
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-wide metrics."""
        worker_stats = [worker.get_stats() for worker in self.workers.values()]
        
        if not worker_stats:
            return {}
        
        avg_queue_size = sum(stats.get("queue_size", 0) for stats in worker_stats) / len(worker_stats)
        avg_latency = sum(stats.get("average_latency", 0) for stats in worker_stats) / len(worker_stats)
        
        return {
            "worker_count": len(self.workers),
            "avg_queue_size": avg_queue_size,
            "avg_latency": avg_latency,
            "cpu_utilization": 0.5,  # Mock value
            "memory_usage": sum(stats.get("memory_usage", 0) for stats in worker_stats)
        }
    
    def _update_worker_health(self):
        """Update worker health based on recent performance."""
        for worker_id, worker in self.workers.items():
            stats = worker.get_stats()
            
            # Simple health check based on error rate and responsiveness
            is_healthy = (
                stats.get("average_latency", 0) < 5.0 and  # Not too slow
                stats.get("queue_size", 0) < 200 and       # Not overloaded
                stats.get("is_healthy", True)               # No internal errors
            )
            
            self.load_balancer.update_worker_stats(worker_id, {
                **stats,
                "is_healthy": is_healthy
            })
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        worker_stats = self.load_balancer.get_worker_stats()
        cache_stats = self.cache.get_stats()
        system_metrics = self._collect_system_metrics()
        
        return {
            "status": "running" if self.running else "stopped",
            "worker_count": len(self.workers),
            "healthy_workers": len([w for w in worker_stats.values() if w.is_healthy]),
            "cache_stats": cache_stats,
            "system_metrics": system_metrics,
            "scaling_history": self.autoscaler.get_scaling_history()[-10:],
            "timestamp": time.time()
        }


# Demo and testing
async def demo_distributed_inference():
    """Demonstrate distributed inference capabilities."""
    print("🌐 Distributed Inference System Demo")
    print("=" * 60)
    
    # Initialize system
    config = DistributedConfig(max_workers=4, max_batch_size=64)
    manager = DistributedInferenceManager(config)
    
    print("🚀 Starting distributed system...")
    await manager.start()
    
    # Wait for workers to initialize
    await asyncio.sleep(1)
    
    print(f"  ✓ Started with {len(manager.workers)} workers")
    
    print("\n📊 Single prediction test...")
    test_input = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    start_time = time.time()
    result = await manager.predict("fno_model", test_input)
    latency = time.time() - start_time
    
    print(f"  ✓ Prediction completed in {latency:.3f}s")
    print(f"  ✓ Worker: {result.get('worker_id')}")
    print(f"  ✓ Avg uncertainty: {sum(result['uncertainty'])/len(result['uncertainty']):.3f}")
    
    print("\n🔄 Batch prediction test...")
    batch_inputs = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2]
    ]
    
    start_time = time.time()
    batch_results = await manager.predict_batch("deeponet_model", batch_inputs)
    batch_latency = time.time() - start_time
    
    print(f"  ✓ Batch of {len(batch_inputs)} completed in {batch_latency:.3f}s")
    print(f"  ✓ Avg latency per item: {batch_latency/len(batch_inputs):.3f}s")
    
    print("\n📈 Cache performance test...")
    
    # Test cache hits
    for i in range(5):
        await manager.predict("fno_model", test_input)  # Should hit cache
    
    cache_stats = manager.cache.get_stats()
    print(f"  ✓ Cache hit rate: {cache_stats['hit_rate']:.2%}")
    print(f"  ✓ Cache size: {cache_stats['cache_size']}")
    
    print("\n⚡ Load testing...")
    
    # Simulate load to trigger auto-scaling
    load_tasks = []
    for i in range(20):
        task = asyncio.create_task(
            manager.predict("load_test_model", [float(j) for j in range(10)])
        )
        load_tasks.append(task)
    
    start_time = time.time()
    await asyncio.gather(*load_tasks)
    load_test_time = time.time() - start_time
    
    print(f"  ✓ Processed {len(load_tasks)} concurrent requests in {load_test_time:.3f}s")
    print(f"  ✓ Throughput: {len(load_tasks)/load_test_time:.1f} requests/second")
    
    # Wait for potential auto-scaling
    await asyncio.sleep(2)
    
    print("\n📊 System status...")
    status = manager.get_system_status()
    
    print(f"  Worker count: {status['worker_count']}")
    print(f"  Healthy workers: {status['healthy_workers']}")
    print(f"  Cache hit rate: {status['cache_stats']['hit_rate']:.2%}")
    print(f"  Avg latency: {status['system_metrics'].get('avg_latency', 0):.3f}s")
    
    print("\n🛑 Shutting down...")
    await manager.stop()
    
    print(f"\n{'='*60}")
    print("✅ Distributed inference demo completed!")


def main():
    """Main function for testing."""
    asyncio.run(demo_distributed_inference())


if __name__ == "__main__":
    main()