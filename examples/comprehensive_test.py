#!/usr/bin/env python3
"""
Comprehensive Test Suite for ProbNeural-Operator-Lab

This script tests all major components:
1. Data generation and loading
2. FNO and DeepONet models
3. Training and inference
4. Performance monitoring
5. Model optimization
6. Error handling
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
import traceback
from pathlib import Path

# Import all components
from probneural_operator import ProbabilisticFNO
from probneural_operator.models import ProbabilisticDeepONet
from probneural_operator.data.datasets import BurgersDataset, NavierStokesDataset
from probneural_operator.utils.performance import PerformanceProfiler, MemoryTracker
from probneural_operator.utils.optimization import ModelOptimizer, DataLoaderOptimizer


def test_component(name: str, test_func):
    """Helper to run individual tests with error handling."""
    print(f"\n{'='*20} Testing {name} {'='*20}")
    try:
        start_time = time.time()
        result = test_func()
        end_time = time.time()
        print(f"✅ {name} test PASSED ({end_time - start_time:.2f}s)")
        return True, result
    except Exception as e:
        print(f"❌ {name} test FAILED: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False, None


def test_data_loading():
    """Test data loading and generation."""
    print("Testing Burgers dataset...")
    
    # Test Burgers dataset
    burgers_dataset = BurgersDataset(
        data_path='/tmp/test_burgers.h5',
        split='train',
        resolution=32,
        time_steps=10,
        viscosity=0.01
    )
    
    assert len(burgers_dataset) > 0, "Dataset should not be empty"
    
    sample_input, sample_output = burgers_dataset[0]
    assert sample_input.shape == (32,), f"Expected input shape (32,), got {sample_input.shape}"
    assert sample_output.shape == (10, 32), f"Expected output shape (10, 32), got {sample_output.shape}"
    
    # Test DataLoader
    dataloader = DataLoader(burgers_dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))
    batch_input, batch_output = batch
    
    assert batch_input.shape == (4, 32), f"Expected batch input shape (4, 32), got {batch_input.shape}"
    assert batch_output.shape == (4, 10, 32), f"Expected batch output shape (4, 10, 32), got {batch_output.shape}"
    
    print(f"Dataset size: {len(burgers_dataset)}")
    print(f"Sample shapes: input {sample_input.shape}, output {sample_output.shape}")
    
    return {"dataset_size": len(burgers_dataset), "sample_shapes": (sample_input.shape, sample_output.shape)}


def test_fno_model():
    """Test FNO model functionality."""
    print("Testing ProbabilisticFNO...")
    
    model = ProbabilisticFNO(
        input_dim=1,
        output_dim=1,
        modes=8,
        width=32,
        depth=2,
        spatial_dim=1
    )
    
    # Test forward pass
    batch_size = 4
    spatial_size = 32
    x = torch.randn(batch_size, 1, spatial_size)
    
    with torch.no_grad():
        output = model(x)
    
    assert output.shape == (batch_size, 1, spatial_size), f"Expected output shape {(batch_size, 1, spatial_size)}, got {output.shape}"
    
    # Test parameter count
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0, "Model should have parameters"
    
    # Test configuration retrieval
    config = model.get_config()
    assert 'modes' in config, "Config should contain 'modes'"
    assert config['modes'] == 8, f"Expected modes=8, got {config['modes']}"
    
    print(f"Model parameters: {param_count:,}")
    print(f"Forward pass successful: {x.shape} -> {output.shape}")
    
    return {"param_count": param_count, "forward_shape": output.shape}


def test_deeponet_model():
    """Test DeepONet model functionality."""
    print("Testing ProbabilisticDeepONet...")
    
    model = ProbabilisticDeepONet(
        branch_dim=32,
        trunk_dim=1,
        output_dim=1,
        branch_layers=[64, 64],
        trunk_layers=[64, 64],
        activation='tanh'
    )
    
    # Test forward pass
    batch_size = 4
    branch_input = torch.randn(batch_size, 32)
    trunk_input = torch.randn(batch_size, 16, 1)
    
    with torch.no_grad():
        output = model(branch_input, trunk_input)
    
    expected_shape = (batch_size, 16, 1)
    assert output.shape == expected_shape, f"Expected output shape {expected_shape}, got {output.shape}"
    
    # Test parameter count
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0, "Model should have parameters"
    
    print(f"Model parameters: {param_count:,}")
    print(f"Forward pass successful: branch {branch_input.shape} + trunk {trunk_input.shape} -> {output.shape}")
    
    return {"param_count": param_count, "forward_shape": output.shape}


def test_training_loop():
    """Test training functionality."""
    print("Testing training loops...")
    
    # Create small dataset
    dataset = BurgersDataset(
        data_path='/tmp/test_training.h5',
        split='train',
        resolution=16,
        time_steps=5,
        viscosity=0.01
    )
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Test FNO training
    fno_model = ProbabilisticFNO(
        input_dim=1,
        output_dim=1,
        modes=4,
        width=16,
        depth=1,
        spatial_dim=1
    )
    
    fno_history = fno_model.fit(
        train_loader=dataloader,
        epochs=2,
        lr=1e-2,
        device='cpu'
    )
    
    assert 'train_loss' in fno_history, "Training history should contain 'train_loss'"
    assert len(fno_history['train_loss']) == 2, f"Expected 2 epochs, got {len(fno_history['train_loss'])}"
    
    # Test DeepONet training
    deeponet_model = ProbabilisticDeepONet(
        branch_dim=16,
        trunk_dim=1,
        output_dim=1,
        branch_layers=[32, 32],
        trunk_layers=[32, 32]
    )
    
    deeponet_history = deeponet_model.fit(
        train_loader=dataloader,
        epochs=2,
        lr=1e-2,
        device='cpu'
    )
    
    assert 'train_loss' in deeponet_history, "Training history should contain 'train_loss'"
    assert len(deeponet_history['train_loss']) == 2, f"Expected 2 epochs, got {len(deeponet_history['train_loss'])}"
    
    print(f"FNO final loss: {fno_history['train_loss'][-1]:.6f}")
    print(f"DeepONet final loss: {deeponet_history['train_loss'][-1]:.6f}")
    
    return {
        "fno_loss": fno_history['train_loss'][-1],
        "deeponet_loss": deeponet_history['train_loss'][-1]
    }


def test_performance_monitoring():
    """Test performance monitoring utilities."""
    print("Testing performance monitoring...")
    
    # Create simple model for testing
    model = ProbabilisticFNO(
        input_dim=1,
        output_dim=1,
        modes=4,
        width=16,
        depth=1,
        spatial_dim=1
    )
    
    # Test PerformanceProfiler
    profiler = PerformanceProfiler(enable_gpu_monitoring=False)
    
    # Profile inference
    input_data = torch.randn(8, 1, 32)
    metrics = profiler.profile_model_inference(
        model=model,
        input_data=input_data,
        num_warmup=2,
        num_iterations=5
    )
    
    assert metrics.execution_time > 0, "Execution time should be positive"
    assert metrics.throughput > 0, "Throughput should be positive"
    assert metrics.batch_size == 8, f"Expected batch size 8, got {metrics.batch_size}"
    
    # Test MemoryTracker
    memory_tracker = MemoryTracker(track_gpu=False)
    
    with memory_tracker.track("test_operation"):
        _ = model(input_data)
    
    memory_summary = memory_tracker.get_memory_summary()
    assert 'peak_cpu_memory_mb' in memory_summary, "Memory summary should contain CPU memory info"
    
    print(f"Inference time: {metrics.execution_time*1000:.2f}ms")
    print(f"Throughput: {metrics.throughput:.1f} samples/sec")
    print(f"Peak memory: {memory_summary['peak_cpu_memory_mb']:.1f} MB")
    
    return {
        "inference_time_ms": metrics.execution_time * 1000,
        "throughput": metrics.throughput,
        "peak_memory_mb": memory_summary['peak_cpu_memory_mb']
    }


def test_model_optimization():
    """Test model optimization utilities."""
    print("Testing model optimization...")
    
    model = ProbabilisticFNO(
        input_dim=1,
        output_dim=1,
        modes=8,
        width=32,
        depth=2,
        spatial_dim=1
    )
    
    optimizer = ModelOptimizer(model)
    
    # Test model statistics
    stats = optimizer.get_model_statistics()
    assert 'total_parameters' in stats, "Stats should contain total parameters"
    assert 'model_size_mb' in stats, "Stats should contain model size"
    assert stats['total_parameters'] > 0, "Model should have parameters"
    
    # Test inference benchmark
    benchmark_results = optimizer.benchmark_inference(
        input_shape=(4, 1, 32),
        device='cpu',
        num_iterations=10
    )
    
    assert 'avg_inference_time_ms' in benchmark_results, "Benchmark should contain timing info"
    assert 'throughput_samples_per_sec' in benchmark_results, "Benchmark should contain throughput"
    assert benchmark_results['avg_inference_time_ms'] > 0, "Inference time should be positive"
    
    print(f"Model size: {stats['model_size_mb']:.2f} MB")
    print(f"Parameters: {stats['total_parameters']:,}")
    print(f"Benchmark time: {benchmark_results['avg_inference_time_ms']:.2f}ms")
    
    return {
        "model_size_mb": stats['model_size_mb'],
        "total_parameters": stats['total_parameters'],
        "benchmark_time_ms": benchmark_results['avg_inference_time_ms']
    }


def test_error_handling():
    """Test error handling and edge cases."""
    print("Testing error handling...")
    
    # Test invalid spatial dimension
    try:
        _ = ProbabilisticFNO(
            input_dim=1,
            output_dim=1,
            modes=8,
            width=32,
            depth=2,
            spatial_dim=4  # Invalid
        )
        assert False, "Should have raised ValueError for invalid spatial_dim"
    except ValueError:
        pass  # Expected
    
    # Test empty dataloader
    from torch.utils.data import TensorDataset
    empty_dataset = TensorDataset(torch.empty(0, 16), torch.empty(0, 16))
    empty_loader = DataLoader(empty_dataset, batch_size=1)
    
    model = ProbabilisticFNO(
        input_dim=1,
        output_dim=1,
        modes=4,
        width=16,
        depth=1,
        spatial_dim=1
    )
    
    # Should handle empty dataloader gracefully
    history = model.fit(
        train_loader=empty_loader,
        epochs=1,
        device='cpu'
    )
    
    assert 'train_loss' in history, "Should return history even with empty loader"
    assert len(history['train_loss']) == 1, "Should have one epoch entry"
    
    print("Error handling tests passed")
    
    return {"empty_loader_handled": True}


def run_comprehensive_test():
    """Run all tests and generate summary report."""
    print("🚀 Starting Comprehensive Test Suite for ProbNeural-Operator-Lab")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Define all tests
    tests = [
        ("Data Loading", test_data_loading),
        ("FNO Model", test_fno_model),
        ("DeepONet Model", test_deeponet_model),
        ("Training Loop", test_training_loop),
        ("Performance Monitoring", test_performance_monitoring),
        ("Model Optimization", test_model_optimization),
        ("Error Handling", test_error_handling),
    ]
    
    # Run all tests
    results = {}
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        success, result = test_component(test_name, test_func)
        results[test_name] = {"success": success, "result": result}
        if success:
            passed_tests += 1
    
    # Generate summary report
    print(f"\n{'='*60}")
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print(f"{'='*60}")
    
    print(f"Tests Passed: {passed_tests}/{total_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! The framework is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    # Detailed results
    print(f"\n{'='*60}")
    print("📋 DETAILED RESULTS")
    print(f"{'='*60}")
    
    for test_name, result in results.items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        print(f"{test_name:<25} {status}")
        
        if result["success"] and result["result"]:
            # Print key metrics if available
            data = result["result"]
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        if 'time' in key.lower() or 'ms' in key.lower():
                            print(f"  {key}: {value:.2f}")
                        elif 'mb' in key.lower():
                            print(f"  {key}: {value:.1f}")
                        elif isinstance(value, int) and value > 1000:
                            print(f"  {key}: {value:,}")
                        else:
                            print(f"  {key}: {value}")
    
    # Performance summary
    if results["Performance Monitoring"]["success"]:
        perf_data = results["Performance Monitoring"]["result"]
        print(f"\n🚀 PERFORMANCE HIGHLIGHTS")
        print(f"Inference Speed: {perf_data['inference_time_ms']:.2f}ms")
        print(f"Throughput: {perf_data['throughput']:.1f} samples/sec")
        print(f"Memory Usage: {perf_data['peak_memory_mb']:.1f} MB")
    
    if results["Model Optimization"]["success"]:
        opt_data = results["Model Optimization"]["result"]
        print(f"Model Efficiency: {opt_data['model_size_mb']:.1f} MB, {opt_data['total_parameters']:,} params")
    
    print(f"\n{'='*60}")
    print("🎯 RECOMMENDATIONS")
    print(f"{'='*60}")
    
    if passed_tests == total_tests:
        print("• Framework is ready for production use")
        print("• All core components are working correctly")
        print("• Performance monitoring is functional")
        print("• Error handling is robust")
    else:
        print("• Review failed test cases above")
        print("• Ensure all dependencies are properly installed")
        print("• Check for version compatibility issues")
    
    print("\n📚 Next Steps:")
    print("• Run examples/neural_operator_comparison.py for detailed benchmarks")
    print("• Try examples/basic_training_example.py for end-to-end workflow") 
    print("• Explore different PDE types and model configurations")
    print("• Apply to your specific scientific computing problems")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)