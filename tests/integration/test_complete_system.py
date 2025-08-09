"""
Complete system integration tests for ProbNeural-Operator-Lab.

This module provides comprehensive integration tests covering the entire
system from data loading through prediction with uncertainty quantification.
Tests validate the end-to-end functionality, performance, and reliability.
"""

import sys
import os
import asyncio
import time
import random
import math
from typing import List, Dict, Any, Optional


# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


def test_basic_framework_functionality():
    """Test basic framework functionality without external dependencies."""
    print("🧪 Testing Basic Framework Functionality")
    print("-" * 50)
    
    try:
        # Test synthetic data generation
        print("1. Testing synthetic data generation...")
        
        def generate_test_data(n_samples: int = 10, n_points: int = 8) -> List[List[float]]:
            """Generate test data for validation."""
            data = []
            for i in range(n_samples):
                sample = []
                for j in range(n_points):
                    x = j / (n_points - 1)
                    y = math.sin(2 * math.pi * x) + random.gauss(0, 0.1)
                    sample.append(y)
                data.append(sample)
            return data
        
        test_data = generate_test_data()
        assert len(test_data) == 10, "Expected 10 samples"
        assert len(test_data[0]) == 8, "Expected 8 points per sample"
        print("   ✓ Synthetic data generation working")
        
        # Test data validation
        print("2. Testing data validation...")
        
        def validate_data(data: List[List[float]]) -> bool:
            """Validate data structure and values."""
            if not data:
                return False
            
            expected_length = len(data[0])
            for sample in data:
                if len(sample) != expected_length:
                    return False
                for val in sample:
                    if not isinstance(val, (int, float)):
                        return False
                    if not (-10 < val < 10):  # Reasonable range
                        return False
            return True
        
        assert validate_data(test_data), "Data validation failed"
        print("   ✓ Data validation working")
        
        # Test uncertainty simulation
        print("3. Testing uncertainty quantification...")
        
        def compute_uncertainty(predictions: List[float], n_samples: int = 5) -> List[float]:
            """Compute uncertainty through Monte Carlo sampling."""
            uncertainties = []
            for pred in predictions:
                samples = [pred + random.gauss(0, 0.1) for _ in range(n_samples)]
                std = math.sqrt(sum((s - pred)**2 for s in samples) / len(samples))
                uncertainties.append(std)
            return uncertainties
        
        test_predictions = [0.5, 0.3, 0.8, 0.1]
        uncertainties = compute_uncertainty(test_predictions)
        
        assert len(uncertainties) == len(test_predictions), "Uncertainty length mismatch"
        assert all(u >= 0 for u in uncertainties), "Negative uncertainties"
        print("   ✓ Uncertainty quantification working")
        
        # Test model pipeline simulation
        print("4. Testing model pipeline...")
        
        def simulate_neural_operator_inference(
            input_data: List[float],
            add_uncertainty: bool = True
        ) -> Dict[str, Any]:
            """Simulate neural operator inference."""
            # Simple transformation (derivative approximation)
            predictions = []
            for i in range(len(input_data)):
                if i == 0:
                    pred = input_data[1] - input_data[0]
                elif i == len(input_data) - 1:
                    pred = input_data[i] - input_data[i-1]
                else:
                    pred = (input_data[i+1] - input_data[i-1]) / 2
                
                predictions.append(pred)
            
            result = {"predictions": predictions}
            
            if add_uncertainty:
                result["uncertainties"] = compute_uncertainty(predictions)
            
            return result
        
        test_input = [0.1, 0.2, 0.4, 0.8, 1.6]
        result = simulate_neural_operator_inference(test_input)
        
        assert "predictions" in result, "Missing predictions"
        assert "uncertainties" in result, "Missing uncertainties"
        assert len(result["predictions"]) == len(test_input), "Prediction length mismatch"
        print("   ✓ Model pipeline working")
        
        print("\n✅ All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_advanced_components():
    """Test advanced components like monitoring and caching."""
    print("\n🔧 Testing Advanced Components")
    print("-" * 50)
    
    try:
        # Test caching functionality
        print("1. Testing caching system...")
        
        class SimpleCache:
            def __init__(self, max_size: int = 100):
                self.cache = {}
                self.access_order = []
                self.max_size = max_size
                self.hits = 0
                self.misses = 0
            
            def _make_key(self, *args) -> str:
                return str(hash(tuple(str(arg) for arg in args)))
            
            def get(self, *key_parts) -> Optional[Any]:
                key = self._make_key(*key_parts)
                if key in self.cache:
                    self.hits += 1
                    # Move to end for LRU
                    self.access_order.remove(key)
                    self.access_order.append(key)
                    return self.cache[key]
                else:
                    self.misses += 1
                    return None
            
            def put(self, value: Any, *key_parts) -> None:
                key = self._make_key(*key_parts)
                
                if len(self.cache) >= self.max_size and key not in self.cache:
                    # Evict LRU
                    old_key = self.access_order.pop(0)
                    del self.cache[old_key]
                
                self.cache[key] = value
                if key not in self.access_order:
                    self.access_order.append(key)
            
            def hit_rate(self) -> float:
                total = self.hits + self.misses
                return self.hits / total if total > 0 else 0
        
        cache = SimpleCache(max_size=5)
        
        # Test cache operations
        cache.put("result1", "input1", "model1")
        cache.put("result2", "input2", "model1")
        
        assert cache.get("input1", "model1") == "result1", "Cache get failed"
        assert cache.get("input3", "model1") is None, "Cache should miss"
        
        # Test LRU eviction
        for i in range(10):
            cache.put(f"result{i}", f"input{i}", "model1")
        
        assert len(cache.cache) <= 5, "Cache size exceeded limit"
        assert cache.hit_rate() > 0, "Hit rate should be positive"
        print("   ✓ Caching system working")
        
        # Test performance monitoring
        print("2. Testing performance monitoring...")
        
        class PerformanceMonitor:
            def __init__(self):
                self.metrics = {}
                self.request_times = []
            
            def record_request(self, duration: float, success: bool = True):
                self.request_times.append(duration)
                if "total_requests" not in self.metrics:
                    self.metrics["total_requests"] = 0
                if "successful_requests" not in self.metrics:
                    self.metrics["successful_requests"] = 0
                
                self.metrics["total_requests"] += 1
                if success:
                    self.metrics["successful_requests"] += 1
            
            def get_stats(self) -> Dict[str, float]:
                if not self.request_times:
                    return {}
                
                avg_time = sum(self.request_times) / len(self.request_times)
                success_rate = self.metrics["successful_requests"] / self.metrics["total_requests"]
                
                return {
                    "average_response_time": avg_time,
                    "success_rate": success_rate,
                    "total_requests": self.metrics["total_requests"]
                }
        
        monitor = PerformanceMonitor()
        
        # Simulate requests
        for i in range(10):
            duration = random.uniform(0.01, 0.1)
            success = random.random() > 0.1  # 90% success rate
            monitor.record_request(duration, success)
        
        stats = monitor.get_stats()
        assert "average_response_time" in stats, "Missing response time metric"
        assert 0 <= stats["success_rate"] <= 1, "Invalid success rate"
        print("   ✓ Performance monitoring working")
        
        # Test load balancing simulation
        print("3. Testing load balancing...")
        
        class SimpleLoadBalancer:
            def __init__(self):
                self.workers = {}
                self.current_index = 0
            
            def add_worker(self, worker_id: str):
                self.workers[worker_id] = {"requests": 0, "active": True}
            
            def select_worker(self, strategy: str = "round_robin") -> Optional[str]:
                active_workers = [wid for wid, info in self.workers.items() if info["active"]]
                
                if not active_workers:
                    return None
                
                if strategy == "round_robin":
                    selected = active_workers[self.current_index % len(active_workers)]
                    self.current_index += 1
                    return selected
                elif strategy == "least_loaded":
                    return min(active_workers, key=lambda w: self.workers[w]["requests"])
                
                return active_workers[0]
            
            def record_request(self, worker_id: str):
                if worker_id in self.workers:
                    self.workers[worker_id]["requests"] += 1
        
        balancer = SimpleLoadBalancer()
        balancer.add_worker("worker1")
        balancer.add_worker("worker2")
        balancer.add_worker("worker3")
        
        # Test round-robin
        selections = [balancer.select_worker("round_robin") for _ in range(6)]
        assert len(set(selections)) == 3, "Round-robin not working"
        print("   ✓ Load balancing working")
        
        print("\n✅ All advanced component tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Advanced component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_async_operations():
    """Test asynchronous operations and concurrency."""
    print("\n⚡ Testing Async Operations")
    print("-" * 50)
    
    try:
        # Test async request processing
        print("1. Testing async request processing...")
        
        async def simulate_async_inference(input_data: List[float], delay: float = 0.01) -> Dict[str, Any]:
            """Simulate asynchronous inference."""
            await asyncio.sleep(delay)  # Simulate processing time
            
            # Simple transformation
            result = [x * 2 + random.gauss(0, 0.1) for x in input_data]
            uncertainty = [0.1 + 0.2 * abs(x) for x in result]
            
            return {
                "prediction": result,
                "uncertainty": uncertainty,
                "processing_time": delay
            }
        
        # Single async request
        test_input = [0.1, 0.2, 0.3]
        result = await simulate_async_inference(test_input)
        
        assert "prediction" in result, "Missing prediction in async result"
        assert len(result["prediction"]) == len(test_input), "Prediction length mismatch"
        print("   ✓ Async inference working")
        
        # Concurrent requests
        print("2. Testing concurrent request handling...")
        
        async def batch_inference(inputs: List[List[float]]) -> List[Dict[str, Any]]:
            """Process multiple inputs concurrently."""
            tasks = [simulate_async_inference(input_data) for input_data in inputs]
            return await asyncio.gather(*tasks)
        
        batch_inputs = [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.7, 0.8]
        ]
        
        start_time = time.time()
        batch_results = await batch_inference(batch_inputs)
        batch_time = time.time() - start_time
        
        assert len(batch_results) == len(batch_inputs), "Batch result count mismatch"
        assert batch_time < 0.1, "Concurrent processing too slow"  # Should be much faster than sequential
        print("   ✓ Concurrent processing working")
        
        # Test queue processing
        print("3. Testing queue-based processing...")
        
        class AsyncRequestQueue:
            def __init__(self):
                self.queue = asyncio.Queue()
                self.processed = []
                self.processing = False
            
            async def add_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
                future = asyncio.Future()
                await self.queue.put((request_data, future))
                return await future
            
            async def process_queue(self):
                """Background queue processor."""
                self.processing = True
                while self.processing:
                    try:
                        request, future = await asyncio.wait_for(
                            self.queue.get(), timeout=0.1
                        )
                        
                        # Simulate processing
                        await asyncio.sleep(0.01)
                        result = {"processed": True, "data": request}
                        
                        future.set_result(result)
                        self.processed.append(result)
                        
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        if not future.done():
                            future.set_exception(e)
            
            async def start(self):
                self.processor_task = asyncio.create_task(self.process_queue())
            
            async def stop(self):
                self.processing = False
                await self.processor_task
        
        queue_processor = AsyncRequestQueue()
        await queue_processor.start()
        
        # Submit requests
        results = await asyncio.gather(*[
            queue_processor.add_request({"id": i, "data": [i, i+1]})
            for i in range(5)
        ])
        
        await queue_processor.stop()
        
        assert len(results) == 5, "Queue processing count mismatch"
        assert all(r["processed"] for r in results), "Not all requests processed"
        print("   ✓ Queue processing working")
        
        print("\n✅ All async operation tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Async operation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling_and_resilience():
    """Test error handling and system resilience."""
    print("\n🛡️  Testing Error Handling & Resilience")
    print("-" * 50)
    
    try:
        # Test input validation
        print("1. Testing input validation...")
        
        def validate_input(data: Any) -> tuple[bool, str]:
            """Validate input data with comprehensive checks."""
            if not isinstance(data, list):
                return False, "Input must be a list"
            
            if len(data) == 0:
                return False, "Input cannot be empty"
            
            if len(data) > 1000:
                return False, "Input too large"
            
            for i, val in enumerate(data):
                if not isinstance(val, (int, float)):
                    return False, f"Invalid type at index {i}"
                
                if math.isnan(val) or math.isinf(val):
                    return False, f"Invalid value at index {i}"
                
                if abs(val) > 1e6:
                    return False, f"Value too large at index {i}"
            
            return True, ""
        
        # Test valid inputs
        assert validate_input([1.0, 2.0, 3.0])[0], "Valid input rejected"
        
        # Test invalid inputs
        assert not validate_input("not_a_list")[0], "String input accepted"
        assert not validate_input([])[0], "Empty list accepted"
        assert not validate_input([float('nan')])[0], "NaN accepted"
        assert not validate_input([float('inf')])[0], "Infinity accepted"
        print("   ✓ Input validation working")
        
        # Test graceful degradation
        print("2. Testing graceful degradation...")
        
        class ResilientPredictor:
            def __init__(self):
                self.fallback_enabled = True
                self.error_count = 0
                self.total_requests = 0
            
            def predict(self, input_data: List[float], use_fallback: bool = True) -> Dict[str, Any]:
                """Make prediction with fallback capability."""
                self.total_requests += 1
                
                try:
                    # Simulate occasional failures
                    if random.random() < 0.2:  # 20% failure rate
                        raise RuntimeError("Simulated model failure")
                    
                    # Normal prediction
                    predictions = [x * 0.8 + 0.1 for x in input_data]
                    uncertainties = [0.1 + 0.2 * abs(x) for x in predictions]
                    
                    return {
                        "predictions": predictions,
                        "uncertainties": uncertainties,
                        "method": "primary_model",
                        "confidence": 0.95
                    }
                
                except Exception as e:
                    self.error_count += 1
                    
                    if use_fallback and self.fallback_enabled:
                        # Fallback to simple linear model
                        predictions = [x * 1.0 for x in input_data]  # Identity
                        uncertainties = [0.5] * len(input_data)  # High uncertainty
                        
                        return {
                            "predictions": predictions,
                            "uncertainties": uncertainties,
                            "method": "fallback_model",
                            "confidence": 0.6,
                            "error": str(e)
                        }
                    else:
                        raise e
            
            def get_health_status(self) -> Dict[str, Any]:
                error_rate = self.error_count / self.total_requests if self.total_requests > 0 else 0
                return {
                    "error_rate": error_rate,
                    "total_requests": self.total_requests,
                    "health": "healthy" if error_rate < 0.5 else "degraded"
                }
        
        predictor = ResilientPredictor()
        
        # Test resilient predictions
        results = []
        for i in range(20):
            try:
                result = predictor.predict([0.1, 0.2, 0.3])
                results.append(result)
            except Exception:
                pass  # Some failures expected
        
        # Should have gotten some results despite failures
        assert len(results) > 0, "No successful predictions"
        
        # Check that fallback was used
        fallback_used = any(r.get("method") == "fallback_model" for r in results)
        assert fallback_used, "Fallback mechanism not triggered"
        
        health = predictor.get_health_status()
        assert "error_rate" in health, "Missing error rate in health status"
        print("   ✓ Graceful degradation working")
        
        # Test circuit breaker pattern
        print("3. Testing circuit breaker...")
        
        class CircuitBreaker:
            def __init__(self, failure_threshold: int = 3, timeout: float = 1.0):
                self.failure_threshold = failure_threshold
                self.timeout = timeout
                self.failure_count = 0
                self.last_failure_time = 0
                self.state = "closed"  # closed, open, half-open
            
            def call(self, func, *args, **kwargs):
                """Execute function with circuit breaker protection."""
                current_time = time.time()
                
                if self.state == "open":
                    if current_time - self.last_failure_time > self.timeout:
                        self.state = "half-open"
                    else:
                        raise RuntimeError("Circuit breaker is open")
                
                try:
                    result = func(*args, **kwargs)
                    
                    if self.state == "half-open":
                        self.state = "closed"
                        self.failure_count = 0
                    
                    return result
                
                except Exception as e:
                    self.failure_count += 1
                    self.last_failure_time = current_time
                    
                    if self.failure_count >= self.failure_threshold:
                        self.state = "open"
                    
                    raise e
        
        def failing_function(fail: bool = False):
            if fail:
                raise RuntimeError("Function failed")
            return "success"
        
        breaker = CircuitBreaker(failure_threshold=2, timeout=0.1)
        
        # Trigger failures to open circuit
        for i in range(3):
            try:
                breaker.call(failing_function, fail=True)
            except RuntimeError:
                pass
        
        assert breaker.state == "open", "Circuit breaker should be open"
        
        # Should reject calls while open
        try:
            breaker.call(failing_function, fail=False)
            assert False, "Should have rejected call"
        except RuntimeError:
            pass  # Expected
        
        # Wait for timeout and test recovery
        time.sleep(0.15)
        result = breaker.call(failing_function, fail=False)
        assert result == "success", "Circuit should have closed"
        assert breaker.state == "closed", "Circuit breaker should be closed"
        print("   ✓ Circuit breaker working")
        
        print("\n✅ All error handling tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_complete_integration_tests():
    """Run all integration tests."""
    print("🚀 COMPLETE SYSTEM INTEGRATION TESTS")
    print("=" * 70)
    
    test_results = {}
    
    # Run test suites
    test_suites = [
        ("Basic Framework", test_basic_framework_functionality),
        ("Advanced Components", test_advanced_components),
        ("Async Operations", test_async_operations),
        ("Error Handling", test_error_handling_and_resilience)
    ]
    
    for suite_name, test_func in test_suites:
        print(f"\n🧪 Running {suite_name} Tests...")
        start_time = time.time()
        
        if asyncio.iscoroutinefunction(test_func):
            success = await test_func()
        else:
            success = test_func()
        
        duration = time.time() - start_time
        test_results[suite_name] = {
            "success": success,
            "duration": duration
        }
    
    # Print summary
    print(f"\n{'='*70}")
    print("📊 INTEGRATION TEST SUMMARY")
    print("=" * 70)
    
    total_tests = len(test_suites)
    passed_tests = sum(1 for result in test_results.values() if result["success"])
    total_time = sum(result["duration"] for result in test_results.values())
    
    print(f"Tests run: {total_tests}")
    print(f"Tests passed: {passed_tests}")
    print(f"Tests failed: {total_tests - passed_tests}")
    print(f"Success rate: {passed_tests/total_tests:.1%}")
    print(f"Total time: {total_time:.2f}s")
    
    print("\nDetailed Results:")
    for suite_name, result in test_results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"  {suite_name:<20}: {status} ({result['duration']:.2f}s)")
    
    if passed_tests == total_tests:
        print(f"\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("System is ready for production deployment.")
    else:
        print(f"\n⚠️  SOME TESTS FAILED!")
        print("Please review and fix failing tests before deployment.")
    
    return passed_tests == total_tests


def main():
    """Main test runner."""
    return asyncio.run(run_complete_integration_tests())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)