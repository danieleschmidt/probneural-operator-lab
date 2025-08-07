#!/usr/bin/env python3
"""
Comprehensive demonstration of ProbNeural-Operator-Lab Generation 3 scaling features.

This script demonstrates all the high-performance computing and scaling capabilities:
1. Performance optimization and caching
2. Multi-GPU distributed training
3. Auto-scaling and load balancing
4. Advanced optimizers
5. Memory management
6. Model serving
7. Comprehensive monitoring
"""

import os
import sys
import time
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Any, List, Optional

# Add probneural_operator to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from probneural_operator.models.fno import FourierNeuralOperator
from probneural_operator.scaling import (
    PredictionCache, AdaptiveBatchSizer, MemoryOptimizer,
    MultiGPUTrainer, ResourcePoolManager, AutoScaler, LoadBalancer, ResourceMonitor,
    AdvancedOptimizerFactory, OptimizerConfig, LearningRateScheduler, GradientManager,
    GradientCheckpointer, MixedPrecisionManager, MemoryPoolManager,
    ModelServer, ModelVersionManager, InferenceOptimizer, ModelMetadata,
    CheckpointManager
)


def setup_logging():
    """Setup comprehensive logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('scaling_demo.log')
        ]
    )


def create_synthetic_dataset(num_samples: int = 1000, 
                           input_dim: int = 64,
                           output_dim: int = 64) -> tuple:
    """Create synthetic dataset for demonstration."""
    logging.info(f"Creating synthetic dataset: {num_samples} samples, {input_dim}D input, {output_dim}D output")
    
    # Generate synthetic PDE-like data
    x = torch.linspace(0, 1, input_dim)
    X, Y = torch.meshgrid(x, x, indexing='ij')
    
    inputs = []
    outputs = []
    
    for i in range(num_samples):
        # Generate random coefficients for synthetic PDE
        a = np.random.uniform(0.1, 2.0)
        b = np.random.uniform(0.1, 2.0)
        
        # Input: initial condition
        u0 = torch.sin(a * np.pi * X) * torch.cos(b * np.pi * Y)
        
        # Output: solution after some evolution
        u_t = u0 * torch.exp(-0.1 * (a**2 + b**2) * np.pi**2)
        
        inputs.append(u0.unsqueeze(0))  # Add channel dimension
        outputs.append(u_t.unsqueeze(0))
    
    input_tensor = torch.stack(inputs)
    output_tensor = torch.stack(outputs)
    
    logging.info(f"Dataset created: inputs {input_tensor.shape}, outputs {output_tensor.shape}")
    
    return input_tensor, output_tensor


def demonstrate_caching_optimization():
    """Demonstrate performance optimization and caching."""
    logging.info("=== DEMONSTRATING CACHING & OPTIMIZATION ===")
    
    # Create model and data
    model = FourierNeuralOperator(
        n_modes_x=16, n_modes_y=16,
        hidden_channels=32,
        in_channels=1, out_channels=1
    )
    
    inputs, _ = create_synthetic_dataset(100, 64, 64)
    
    # Initialize caching system
    cache = PredictionCache(max_size=50, max_memory_mb=128)
    memory_optimizer = MemoryOptimizer()
    
    logging.info("Testing prediction caching...")
    
    # Test caching performance
    model_state = {'model_hash': 'test_model_v1'}
    
    cache_hits = 0
    cache_misses = 0
    
    for i in range(10):
        test_input = inputs[i % 5]  # Repeat some inputs to test caching
        
        # Try to get from cache
        cached_result = cache.get(model_state, test_input)
        
        if cached_result is not None:
            cache_hits += 1
            logging.info(f"Cache hit for input {i}")
        else:
            cache_misses += 1
            # Generate prediction
            with torch.no_grad():
                result = model(test_input.unsqueeze(0))
            
            # Store in cache
            cache.put(model_state, test_input, result.squeeze(0))
            logging.info(f"Cache miss for input {i}, result cached")
    
    # Print cache statistics
    stats = cache.get_stats()
    logging.info(f"Cache performance: {cache_hits} hits, {cache_misses} misses")
    logging.info(f"Cache stats: {stats}")
    
    # Demonstrate adaptive batch sizing
    logging.info("Testing adaptive batch sizing...")
    batch_sizer = AdaptiveBatchSizer(initial_batch_size=16, min_batch_size=4, max_batch_size=64)
    
    for epoch in range(5):
        # Simulate training with varying performance
        batch_size = batch_sizer.get_current_batch_size()
        
        # Simulate some training metrics
        throughput = np.random.uniform(50, 100)  # samples per second
        memory_usage = np.random.uniform(0.4, 0.9)  # memory usage ratio
        execution_time = batch_size / throughput
        
        batch_sizer.update_performance(batch_size, throughput, memory_usage, execution_time)
        
        stats = batch_sizer.get_stats()
        logging.info(f"Epoch {epoch}: batch_size={batch_size}, throughput={throughput:.1f}, {stats['performance_trend']}")


def demonstrate_distributed_training():
    """Demonstrate multi-GPU and distributed training."""
    logging.info("=== DEMONSTRATING DISTRIBUTED TRAINING ===")
    
    if not torch.cuda.is_available():
        logging.info("GPU not available, skipping distributed training demo")
        return
    
    # Create model and data
    model = FourierNeuralOperator(
        n_modes_x=16, n_modes_y=16,
        hidden_channels=64,
        in_channels=1, out_channels=1
    )
    
    inputs, outputs = create_synthetic_dataset(200, 64, 64)
    dataset = TensorDataset(inputs, outputs)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Setup multi-GPU training
    if torch.cuda.device_count() > 1:
        logging.info(f"Setting up multi-GPU training on {torch.cuda.device_count()} GPUs")
        
        multi_gpu_trainer = MultiGPUTrainer(model)
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Train for a few steps
        stats = multi_gpu_trainer.train_step(dataloader, optimizer, criterion, use_mixed_precision=True)
        
        logging.info(f"Multi-GPU training stats: {stats}")
        
        # Get memory statistics
        memory_stats = multi_gpu_trainer.get_memory_stats()
        logging.info(f"GPU memory usage: {memory_stats}")
    else:
        logging.info("Only 1 GPU available, skipping multi-GPU demo")
    
    # Demonstrate resource pool management
    logging.info("Testing resource pool management...")
    resource_manager = ResourcePoolManager()
    
    # Try to allocate GPU
    gpu_id = resource_manager.allocate_gpu(memory_gb=2.0)
    if gpu_id is not None:
        logging.info(f"Allocated GPU {gpu_id}")
        
        # Do some work...
        time.sleep(1)
        
        # Release GPU
        resource_manager.release_gpu(gpu_id)
        logging.info(f"Released GPU {gpu_id}")
    
    # Get resource stats
    resource_stats = resource_manager.get_resource_stats()
    logging.info(f"Resource statistics: {resource_stats}")


def demonstrate_autoscaling():
    """Demonstrate auto-scaling and load balancing."""
    logging.info("=== DEMONSTRATING AUTO-SCALING ===")
    
    # Setup resource monitoring
    monitor = ResourceMonitor(monitoring_interval=2.0)
    monitor.start_monitoring()
    
    # Setup auto-scaler
    from probneural_operator.scaling.autoscale import ScalingPolicy
    
    policy = ScalingPolicy(
        cpu_scale_up_threshold=70.0,
        cpu_scale_down_threshold=30.0,
        min_instances=1,
        max_instances=5,
        cooldown_period_seconds=10.0  # Short cooldown for demo
    )
    
    autoscaler = AutoScaler(policy, monitor)
    autoscaler.start()
    
    # Setup load balancer
    load_balancer = LoadBalancer(balancing_strategy="round_robin")
    
    # Register some mock instances
    for i in range(3):
        load_balancer.register_instance(
            f"instance_{i}",
            f"http://localhost:800{i}",
            weight=1.0
        )
    
    # Simulate some requests
    logging.info("Simulating request load...")
    
    for i in range(10):
        # Get next instance from load balancer
        instance = load_balancer.get_next_instance()
        if instance:
            logging.info(f"Request {i} routed to {instance['id']}")
            
            # Simulate request processing
            request_id = monitor.record_request_start()
            time.sleep(0.1)  # Simulate processing time
            monitor.record_request_end(request_id, success=True)
            
            # Record completion
            load_balancer.record_request_completion(
                instance['id'], 
                0.1,  # response time
                success=True
            )
    
    # Get current metrics and scaling status
    time.sleep(3)  # Allow monitoring to collect data
    
    current_metrics = monitor.get_current_metrics()
    if current_metrics:
        logging.info(f"Current metrics: CPU={current_metrics.cpu_percent:.1f}%, "
                    f"Memory={current_metrics.memory_percent:.1f}%, "
                    f"Requests={current_metrics.active_requests}")
    
    logging.info(f"Current instances: {autoscaler.get_current_instances()}")
    
    # Get load balancer stats
    lb_stats = load_balancer.get_instance_stats()
    logging.info(f"Load balancer stats: {lb_stats}")
    
    # Cleanup
    autoscaler.stop()
    monitor.stop_monitoring()


def demonstrate_advanced_optimizers():
    """Demonstrate advanced optimization algorithms."""
    logging.info("=== DEMONSTRATING ADVANCED OPTIMIZERS ===")
    
    # Create model
    model = FourierNeuralOperator(
        n_modes_x=16, n_modes_y=16,
        hidden_channels=32,
        in_channels=1, out_channels=1
    )
    
    # Test different optimizer configurations
    optimizer_configs = [
        OptimizerConfig(optimizer_type="adamw", learning_rate=1e-3, weight_decay=1e-4),
        OptimizerConfig(optimizer_type="lion", learning_rate=1e-4, weight_decay=1e-2),
        OptimizerConfig(optimizer_type="lars", learning_rate=1e-3, momentum=0.9)
    ]
    
    inputs, outputs = create_synthetic_dataset(100, 64, 64)
    dataset = TensorDataset(inputs, outputs)
    
    for config in optimizer_configs:
        logging.info(f"Testing optimizer: {config.optimizer_type}")
        
        # Create optimizer
        optimizer = AdvancedOptimizerFactory.create_optimizer(model, config)
        
        # Create learning rate scheduler
        scheduler = LearningRateScheduler.create_scheduler(
            optimizer, 
            "warmup_cosine",
            warmup_epochs=5,
            total_epochs=20
        )
        
        # Setup gradient management
        gradient_manager = GradientManager(max_grad_norm=1.0, accumulation_steps=2)
        
        # Train for a few steps
        model.train()
        criterion = nn.MSELoss()
        
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        for epoch in range(3):
            total_loss = 0.0
            for batch_idx, (batch_inputs, batch_targets) in enumerate(dataloader):
                loss = criterion(model(batch_inputs), batch_targets)
                
                # Use gradient manager for accumulation
                updated = gradient_manager.accumulate_gradients(loss, optimizer, model)
                
                if updated:
                    scheduler.step()
                
                total_loss += loss.item()
                
                if batch_idx >= 5:  # Only do a few batches for demo
                    break
            
            # Get gradient statistics
            grad_stats = gradient_manager.get_gradient_stats()
            logging.info(f"Epoch {epoch}, Loss: {total_loss:.4f}, "
                        f"Grad norm: {grad_stats.get('recent_gradient_norm', 0):.4f}")


def demonstrate_memory_management():
    """Demonstrate memory management and optimization."""
    logging.info("=== DEMONSTRATING MEMORY MANAGEMENT ===")
    
    # Create larger model for memory demo
    model = FourierNeuralOperator(
        n_modes_x=32, n_modes_y=32,
        hidden_channels=128,
        in_channels=1, out_channels=1
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Setup memory management
    checkpointer = GradientCheckpointer()
    mp_manager = MixedPrecisionManager(enabled=torch.cuda.is_available())
    pool_manager = MemoryPoolManager()
    
    logging.info("Testing gradient checkpointing...")
    
    # Apply gradient checkpointing to model
    for name, module in model.named_modules():
        if isinstance(module, nn.Sequential) and len(module) > 2:
            checkpointer.auto_checkpoint_forward(module, memory_threshold_mb=500)
            logging.info(f"Applied checkpointing to {name}")
    
    # Test memory pool
    logging.info("Testing memory pool management...")
    
    tensors = []
    for i in range(10):
        tensor = pool_manager.get_tensor((32, 64, 64), torch.float32, 'cuda' if torch.cuda.is_available() else 'cpu')
        tensors.append(tensor)
    
    # Return tensors to pool
    for tensor in tensors:
        pool_manager.return_tensor(tensor)
    
    # Get pool statistics
    pool_stats = pool_manager.get_pool_stats()
    logging.info(f"Memory pool stats: {pool_stats}")
    
    # Test mixed precision
    if torch.cuda.is_available():
        logging.info("Testing mixed precision training...")
        
        inputs, outputs = create_synthetic_dataset(50, 64, 64)
        inputs, outputs = inputs.cuda(), outputs.cuda()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        for i in range(3):
            optimizer.zero_grad()
            
            # Forward pass with autocast
            with torch.cuda.amp.autocast():
                pred = model(inputs[:8])  # Small batch
                loss = criterion(pred, outputs[:8])
            
            # Scale and backward
            scaled_loss = mp_manager.scale_loss(loss)
            scaled_loss.backward()
            
            # Unscale and clip gradients
            mp_manager.unscale_gradients(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Step optimizer
            updated = mp_manager.step_optimizer(optimizer)
            
            if updated:
                logging.info(f"Step {i}: loss={loss.item():.4f}")
        
        # Get scaling stats
        scaling_stats = mp_manager.get_scaling_stats()
        logging.info(f"Mixed precision stats: {scaling_stats}")


def demonstrate_model_serving():
    """Demonstrate model serving capabilities."""
    logging.info("=== DEMONSTRATING MODEL SERVING ===")
    
    # Create and train a simple model
    model = FourierNeuralOperator(
        n_modes_x=16, n_modes_y=16,
        hidden_channels=32,
        in_channels=1, out_channels=1
    )
    
    # Create version manager
    version_manager = ModelVersionManager("./model_storage")
    
    # Register model
    metadata = ModelMetadata(
        model_id="fno_demo",
        name="Demo FNO Model",
        version="1.0",
        description="Demonstration Fourier Neural Operator",
        input_shape=(1, 64, 64),
        output_shape=(1, 64, 64),
        model_type="FourierNeuralOperator",
        created_time=time.time(),
        updated_time=time.time(),
        tags=["demo", "fno"]
    )
    
    success = version_manager.register_model(model, "fno_demo", "1.0", metadata)
    logging.info(f"Model registration: {'successful' if success else 'failed'}")
    
    # Test model loading
    loaded_model = version_manager.load_model("fno_demo", "1.0")
    if loaded_model is not None:
        logging.info("Model loaded successfully")
        
        # Test inference optimization
        optimizer = InferenceOptimizer()
        
        # Create example input
        example_input = torch.randn(1, 1, 64, 64)
        
        # Optimize model
        optimized_model = optimizer.optimize_model(loaded_model, example_input, "standard")
        
        # Benchmark performance
        benchmark_results = optimizer.benchmark_model(optimized_model, example_input, num_iterations=50)
        logging.info(f"Benchmark results: {benchmark_results}")
    
    # Get model information
    model_info = version_manager.get_model_info("fno_demo")
    logging.info(f"Model info: {model_info}")
    
    # Demonstrate checkpoint management
    logging.info("Testing checkpoint management...")
    
    checkpoint_manager = CheckpointManager("./checkpoints")
    
    # Save a checkpoint
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': {},
        'loss': 0.5
    }
    
    checkpoint_path = checkpoint_manager.save_checkpoint(state, epoch=1, is_best=True)
    logging.info(f"Saved checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    loaded_state = checkpoint_manager.load_checkpoint(load_best=True)
    if loaded_state:
        logging.info(f"Loaded checkpoint from epoch {loaded_state['epoch']}")


def run_comprehensive_demo():
    """Run comprehensive demonstration of all scaling features."""
    logging.info("Starting ProbNeural-Operator-Lab Generation 3 Scaling Demo")
    logging.info("=" * 60)
    
    try:
        # Run individual demonstrations
        demonstrate_caching_optimization()
        demonstrate_distributed_training()
        demonstrate_autoscaling()
        demonstrate_advanced_optimizers()
        demonstrate_memory_management()
        demonstrate_model_serving()
        
        logging.info("=" * 60)
        logging.info("All scaling demonstrations completed successfully!")
        
        # Print summary
        logging.info("\nSCALING FEATURES DEMONSTRATED:")
        logging.info("✓ Performance Optimization & Caching")
        logging.info("✓ Multi-GPU & Distributed Training")
        logging.info("✓ Auto-Scaling & Load Balancing")
        logging.info("✓ Advanced Optimization Algorithms")
        logging.info("✓ Memory Management & Resource Optimization")
        logging.info("✓ Model Serving & Version Management")
        
    except Exception as e:
        logging.error(f"Demo failed with error: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ProbNeural-Operator-Lab Scaling Demo")
    parser.add_argument("--feature", type=str, choices=[
        "all", "caching", "distributed", "autoscaling", 
        "optimizers", "memory", "serving"
    ], default="all", help="Which feature to demonstrate")
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    if args.feature == "all":
        run_comprehensive_demo()
    elif args.feature == "caching":
        demonstrate_caching_optimization()
    elif args.feature == "distributed":
        demonstrate_distributed_training()
    elif args.feature == "autoscaling":
        demonstrate_autoscaling()
    elif args.feature == "optimizers":
        demonstrate_advanced_optimizers()
    elif args.feature == "memory":
        demonstrate_memory_management()
    elif args.feature == "serving":
        demonstrate_model_serving()


if __name__ == "__main__":
    main()