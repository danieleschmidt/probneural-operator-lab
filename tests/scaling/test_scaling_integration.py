"""Integration tests for scaling functionality."""

import pytest
import torch
import torch.nn as nn
import numpy as np
import time
import tempfile
import shutil
from unittest.mock import Mock, patch

from probneural_operator.scaling import (
    PredictionCache, AdaptiveBatchSizer, MemoryOptimizer,
    MultiGPUTrainer, ResourcePoolManager,
    AdvancedOptimizerFactory, OptimizerConfig,
    GradientCheckpointer, MixedPrecisionManager,
    ModelVersionManager, InferenceOptimizer, ModelMetadata
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_size=64, hidden_size=32, output_size=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.layers(x.flatten(1))


@pytest.fixture
def simple_model():
    """Create simple model for testing."""
    return SimpleModel()


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    inputs = torch.randn(100, 64)
    outputs = torch.randn(100, 64)
    return inputs, outputs


class TestCachingOptimization:
    """Test caching and optimization features."""
    
    def test_prediction_cache(self, simple_model, sample_data):
        """Test prediction caching."""
        cache = PredictionCache(max_size=10, max_memory_mb=32)
        inputs, _ = sample_data
        
        # Test cache miss
        result = cache.get({'model_hash': 'test'}, inputs[0])
        assert result is None
        
        # Generate and cache result
        with torch.no_grad():
            output = simple_model(inputs[0].unsqueeze(0))
        
        cache.put({'model_hash': 'test'}, inputs[0], output.squeeze(0))
        
        # Test cache hit
        cached_result = cache.get({'model_hash': 'test'}, inputs[0])
        assert cached_result is not None
        assert torch.allclose(cached_result, output.squeeze(0), atol=1e-6)
        
        # Test cache statistics
        stats = cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5
    
    def test_adaptive_batch_sizer(self):
        """Test adaptive batch sizing."""
        batch_sizer = AdaptiveBatchSizer(
            initial_batch_size=32,
            min_batch_size=8,
            max_batch_size=128
        )
        
        # Initial batch size
        assert batch_sizer.get_current_batch_size() == 32
        
        # High memory usage should reduce batch size
        batch_sizer.update_performance(32, 100.0, 0.9, 0.1)
        new_size = batch_sizer.get_current_batch_size()
        assert new_size <= 32
        
        # Low memory usage should increase batch size
        batch_sizer.update_performance(new_size, 150.0, 0.4, 0.05)
        newer_size = batch_sizer.get_current_batch_size()
        
        stats = batch_sizer.get_stats()
        assert 'current_batch_size' in stats
        assert 'adjustments_made' in stats
    
    def test_memory_optimizer(self):
        """Test memory optimizer."""
        optimizer = MemoryOptimizer()
        
        # Test tensor allocation and return
        tensor1 = optimizer.get_tensor((10, 20), torch.float32, 'cpu')
        assert tensor1.shape == (10, 20)
        assert tensor1.dtype == torch.float32
        
        optimizer.return_tensor(tensor1)
        
        # Second allocation should reuse from pool
        tensor2 = optimizer.get_tensor((10, 20), torch.float32, 'cpu')
        
        stats = optimizer.get_stats()
        assert stats['pool_hits'] >= 1
        assert stats['total_allocations'] >= 1


class TestDistributedTraining:
    """Test distributed training features."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_multi_gpu_trainer_single_gpu(self, simple_model, sample_data):
        """Test multi-GPU trainer with single GPU."""
        inputs, outputs = sample_data
        
        # Create trainer (will use DataParallel with single GPU)
        trainer = MultiGPUTrainer(simple_model, device_ids=[0])
        
        # Create data loader
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(inputs, outputs)
        dataloader = DataLoader(dataset, batch_size=16)
        
        # Create optimizer and criterion
        optimizer = torch.optim.Adam(trainer.model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Test training step
        stats = trainer.train_step(dataloader, optimizer, criterion)
        
        assert 'avg_loss' in stats
        assert 'training_time' in stats
        assert 'throughput' in stats
        assert stats['num_gpus_used'] == 1
    
    def test_resource_pool_manager(self):
        """Test resource pool manager."""
        manager = ResourcePoolManager()
        
        # Test GPU discovery
        gpu_count = len(manager.gpu_pool)
        assert gpu_count >= 0
        
        # Test resource stats
        stats = manager.get_resource_stats()
        assert 'total_gpus' in stats
        assert 'cpu_utilization_percent' in stats
        assert 'memory_usage_gb' in stats


class TestAdvancedOptimizers:
    """Test advanced optimization algorithms."""
    
    def test_optimizer_factory(self, simple_model):
        """Test optimizer factory."""
        # Test AdamW
        config = OptimizerConfig(optimizer_type="adamw", learning_rate=1e-3)
        optimizer = AdvancedOptimizerFactory.create_optimizer(simple_model, config)
        assert isinstance(optimizer, torch.optim.AdamW)
        
        # Test SGD
        config = OptimizerConfig(optimizer_type="sgd", learning_rate=1e-2)
        optimizer = AdvancedOptimizerFactory.create_optimizer(simple_model, config)
        assert isinstance(optimizer, torch.optim.SGD)
    
    def test_optimizer_recommendations(self):
        """Test optimizer recommendations."""
        recommendations = AdvancedOptimizerFactory.get_optimizer_recommendations(
            model_size=1000,
            dataset_size=10000,
            task_type="regression"
        )
        
        assert len(recommendations) > 0
        assert all(isinstance(rec, OptimizerConfig) for rec in recommendations)


class TestMemoryManagement:
    """Test memory management features."""
    
    def test_gradient_checkpointer(self, simple_model):
        """Test gradient checkpointing."""
        checkpointer = GradientCheckpointer()
        
        # Test checkpointing a sequential module
        seq_model = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 64)
        )
        
        input_tensor = torch.randn(4, 64)
        
        # Apply checkpointing
        output = checkpointer.checkpoint_sequential(seq_model, 2, input_tensor)
        assert output.shape == (4, 64)
    
    def test_mixed_precision_manager(self):
        """Test mixed precision manager."""
        mp_manager = MixedPrecisionManager(enabled=False)  # Disable for CPU testing
        
        # Test scaling stats (should work even when disabled)
        stats = mp_manager.get_scaling_stats()
        assert 'enabled' in stats
        assert stats['enabled'] == False
    
    def test_memory_mapped_dataset_creation(self):
        """Test memory-mapped dataset creation."""
        # Create temporary data
        data = torch.randn(100, 10)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = f"{temp_dir}/test_data.bin"
            
            # This would normally create a memory-mapped dataset
            # For testing, we just verify the interface exists
            from probneural_operator.scaling.memory import MemoryMappedDataset
            
            # Test dataset creation interface
            assert hasattr(MemoryMappedDataset, 'create_from_tensor')


class TestModelServing:
    """Test model serving features."""
    
    def test_model_version_manager(self, simple_model):
        """Test model version management."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ModelVersionManager(temp_dir)
            
            # Create metadata
            metadata = ModelMetadata(
                model_id="test_model",
                name="Test Model",
                version="1.0",
                description="Test model for unit tests",
                input_shape=(64,),
                output_shape=(64,),
                model_type="SimpleModel",
                created_time=time.time(),
                updated_time=time.time()
            )
            
            # Register model
            success = manager.register_model(simple_model, "test_model", "1.0", metadata)
            assert success
            
            # Load model
            loaded_model = manager.load_model("test_model", "1.0")
            assert loaded_model is not None
            
            # Test model info
            info = manager.get_model_info("test_model")
            assert info is not None
            assert info['model_id'] == "test_model"
            assert info['version'] == "1.0"
            
            # List models
            models = manager.list_models()
            assert len(models) == 1
            assert models[0]['model_id'] == "test_model"
    
    def test_inference_optimizer(self, simple_model):
        """Test inference optimization."""
        optimizer = InferenceOptimizer()
        
        # Create example input
        example_input = torch.randn(1, 64)
        
        # Test optimization
        optimized_model = optimizer.optimize_model(
            simple_model, example_input, optimization_level="basic"
        )
        
        # Model should still work
        with torch.no_grad():
            output = optimized_model(example_input)
            assert output.shape == (1, 64)
        
        # Test benchmarking
        benchmark = optimizer.benchmark_model(optimized_model, example_input, num_iterations=10)
        
        assert 'avg_inference_time_ms' in benchmark
        assert 'throughput_samples_per_sec' in benchmark
        assert benchmark['avg_inference_time_ms'] > 0
        assert benchmark['throughput_samples_per_sec'] > 0


class TestIntegration:
    """Integration tests combining multiple features."""
    
    def test_end_to_end_scaling_pipeline(self, simple_model, sample_data):
        """Test complete scaling pipeline."""
        inputs, outputs = sample_data
        
        # 1. Setup memory optimization
        memory_optimizer = MemoryOptimizer()
        
        # 2. Setup caching
        cache = PredictionCache(max_size=20)
        
        # 3. Setup adaptive batching
        batch_sizer = AdaptiveBatchSizer()
        
        # 4. Create optimized version of model
        inference_optimizer = InferenceOptimizer()
        example_input = inputs[0].unsqueeze(0)
        optimized_model = inference_optimizer.optimize_model(simple_model, example_input)
        
        # 5. Test inference with all optimizations
        model_state = {'model_hash': 'optimized_v1'}
        
        for i in range(10):
            test_input = inputs[i % 3]  # Repeat inputs for cache testing
            
            # Check cache first
            cached_result = cache.get(model_state, test_input)
            
            if cached_result is None:
                # Generate prediction
                with torch.no_grad():
                    result = optimized_model(test_input.unsqueeze(0))
                
                # Cache result
                cache.put(model_state, test_input, result.squeeze(0))
            else:
                result = cached_result.unsqueeze(0)
            
            assert result.shape == (1, 64)
        
        # 6. Verify cache performance
        cache_stats = cache.get_stats()
        assert cache_stats['hits'] > 0  # Should have some cache hits
        
        # 7. Test memory optimization
        tensor = memory_optimizer.get_tensor((10, 64), torch.float32, 'cpu')
        memory_optimizer.return_tensor(tensor)
        
        memory_stats = memory_optimizer.get_stats()
        assert memory_stats['pool_hits'] >= 1
    
    def test_model_serving_with_optimization(self, simple_model):
        """Test model serving with optimization features."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup model version manager
            version_manager = ModelVersionManager(temp_dir)
            
            # Register model
            metadata = ModelMetadata(
                model_id="serving_test",
                name="Serving Test Model",
                version="1.0",
                description="Model for serving tests",
                input_shape=(64,),
                output_shape=(64,),
                model_type="SimpleModel",
                created_time=time.time(),
                updated_time=time.time()
            )
            
            version_manager.register_model(simple_model, "serving_test", "1.0", metadata)
            
            # Load and optimize model
            loaded_model = version_manager.load_model("serving_test", "1.0")
            assert loaded_model is not None
            
            # Test inference
            test_input = torch.randn(1, 64)
            with torch.no_grad():
                output = loaded_model(test_input)
            
            assert output.shape == (1, 64)
            
            # Test performance recording
            version_manager.record_request("serving_test", 0.05, success=True)
            
            # Get updated model info
            info = version_manager.get_model_info("serving_test")
            assert info['request_count'] == 1
            assert info['avg_response_time_ms'] == 50.0


@pytest.mark.performance
class TestPerformance:
    """Performance tests for scaling features."""
    
    def test_cache_performance(self, simple_model):
        """Test cache performance under load."""
        cache = PredictionCache(max_size=100, max_memory_mb=64)
        
        # Generate test data
        inputs = [torch.randn(64) for _ in range(50)]
        model_state = {'model_hash': 'perf_test'}
        
        # Warm up cache
        for input_tensor in inputs[:25]:
            with torch.no_grad():
                output = simple_model(input_tensor.unsqueeze(0))
            cache.put(model_state, input_tensor, output.squeeze(0))
        
        # Test cache hit performance
        start_time = time.time()
        
        for _ in range(1000):
            # Random input selection to test cache hits
            input_tensor = inputs[np.random.randint(0, 25)]
            result = cache.get(model_state, input_tensor)
            assert result is not None
        
        cache_time = time.time() - start_time
        
        # Test direct inference performance
        start_time = time.time()
        
        for _ in range(1000):
            input_tensor = inputs[np.random.randint(0, 25)]
            with torch.no_grad():
                result = simple_model(input_tensor.unsqueeze(0))
        
        inference_time = time.time() - start_time
        
        # Cache should be faster
        assert cache_time < inference_time
        
        logging.info(f"Cache time: {cache_time:.3f}s, Inference time: {inference_time:.3f}s")
        logging.info(f"Speedup: {inference_time / cache_time:.2f}x")
    
    def test_optimization_effectiveness(self, simple_model):
        """Test that optimizations actually improve performance."""
        optimizer = InferenceOptimizer()
        example_input = torch.randn(4, 64)
        
        # Benchmark original model
        original_benchmark = optimizer.benchmark_model(simple_model, example_input, num_iterations=50)
        
        # Optimize model
        optimized_model = optimizer.optimize_model(simple_model, example_input, "standard")
        
        # Benchmark optimized model  
        optimized_benchmark = optimizer.benchmark_model(optimized_model, example_input, num_iterations=50)
        
        # Check that optimization didn't break functionality
        with torch.no_grad():
            original_output = simple_model(example_input)
            optimized_output = optimized_model(example_input)
        
        # Outputs should be close (within numerical precision)
        assert torch.allclose(original_output, optimized_output, atol=1e-4)
        
        logging.info(f"Original latency: {original_benchmark['avg_inference_time_ms']:.2f}ms")
        logging.info(f"Optimized latency: {optimized_benchmark['avg_inference_time_ms']:.2f}ms")
        
        # At minimum, optimization shouldn't significantly slow down inference
        assert optimized_benchmark['avg_inference_time_ms'] <= original_benchmark['avg_inference_time_ms'] * 1.2


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__])