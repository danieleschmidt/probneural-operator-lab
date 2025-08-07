"""
Comprehensive Usage Guide for ProbNeural-Operator-Lab

This example demonstrates the complete workflow from configuration to training
to uncertainty quantification and model evaluation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the framework
from probneural_operator.models import ProbabilisticFNO
from probneural_operator.utils import (
    setup_logging, create_default_config, create_monitoring_suite,
    run_comprehensive_diagnostics, secure_operation,
    create_training_logger, validate_config_compatibility
)


def generate_burgers_data(n_samples=200, spatial_size=64, viscosity=0.01):
    """
    Generate synthetic 1D Burgers' equation data.
    
    The Burgers' equation is: ∂u/∂t + u∂u/∂x = ν∂²u/∂x² 
    
    Args:
        n_samples: Number of samples to generate
        spatial_size: Spatial discretization points
        viscosity: Viscosity parameter
        
    Returns:
        Tuple of (initial_conditions, solutions)
    """
    logger.info(f"Generating {n_samples} Burgers' equation samples")
    
    # Spatial grid
    x = np.linspace(0, 2*np.pi, spatial_size, endpoint=False)
    dx = x[1] - x[0]
    
    # Time parameters
    dt = 0.001
    t_final = 1.0
    nt = int(t_final / dt)
    
    initial_conditions = []
    solutions = []
    
    for i in range(n_samples):
        # Random initial condition (smooth random function)
        np.random.seed(i)  # Reproducible
        
        # Sum of random Fourier modes
        u0 = np.zeros(spatial_size)
        for k in range(1, 6):  # First 5 modes
            amplitude = np.random.normal(0, 1/k)
            phase = np.random.uniform(0, 2*np.pi)
            u0 += amplitude * np.sin(k * x + phase)
        
        # Solve using finite difference
        u = u0.copy()
        for _ in range(nt):
            # Compute derivatives
            du_dx = np.gradient(u, dx, edge_order=2)
            d2u_dx2 = np.gradient(du_dx, dx, edge_order=2)
            
            # Burgers' equation update
            u = u - dt * u * du_dx + dt * viscosity * d2u_dx2
        
        initial_conditions.append(u0)
        solutions.append(u)
    
    # Convert to tensors
    initial_conditions = torch.tensor(np.array(initial_conditions), dtype=torch.float32)
    solutions = torch.tensor(np.array(solutions), dtype=torch.float32)
    
    logger.info(f"Generated data shapes: IC {initial_conditions.shape}, Sol {solutions.shape}")
    return initial_conditions, solutions


def comprehensive_workflow():
    """Demonstrates the complete workflow."""
    
    # 1. SETUP AND CONFIGURATION
    logger.info("=== STEP 1: Setup and Configuration ===")
    
    # Create output directory
    output_dir = Path("./comprehensive_example_output")
    output_dir.mkdir(exist_ok=True)
    
    # Set up logging
    setup_logging(log_dir=str(output_dir / "logs"))
    
    # Create default configuration
    config = create_default_config("fno")
    config.name = "comprehensive_burgers_example"
    config.model.modes = 16
    config.model.width = 64
    config.model.depth = 4
    config.model.spatial_dim = 1
    config.model.learning_rate = 0.001
    config.training.epochs = 50
    config.training.batch_size = 16
    config.training.early_stopping = True
    config.training.patience = 10
    config.posterior.method = "laplace"
    config.posterior.prior_precision = 1.0
    config.output_dir = str(output_dir)
    
    # Validate configuration
    warnings = validate_config_compatibility(config)
    if warnings:
        logger.warning(f"Configuration warnings: {warnings}")
    
    config.validate()
    logger.info("Configuration validated successfully")
    
    # 2. DATA GENERATION AND PREPARATION
    logger.info("=== STEP 2: Data Generation and Preparation ===")
    
    # Generate synthetic Burgers' equation data
    initial_conditions, solutions = generate_burgers_data(
        n_samples=200, spatial_size=64, viscosity=0.01
    )
    
    # Split data
    n_train = 140
    n_val = 30
    n_test = 30
    
    train_ic = initial_conditions[:n_train]
    train_sol = solutions[:n_train]
    val_ic = initial_conditions[n_train:n_train+n_val]
    val_sol = solutions[n_train:n_train+n_val]
    test_ic = initial_conditions[n_train+n_val:]
    test_sol = solutions[n_train+n_val:]
    
    # Create data loaders
    train_dataset = TensorDataset(train_ic, train_sol)
    val_dataset = TensorDataset(val_ic, val_sol)
    test_dataset = TensorDataset(test_ic, test_sol)
    
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)
    
    logger.info(f"Data split: {n_train} train, {n_val} val, {n_test} test")
    
    # 3. MODEL CREATION AND DIAGNOSTICS
    logger.info("=== STEP 3: Model Creation and Diagnostics ===")
    
    # Create probabilistic FNO model
    model = ProbabilisticFNO(
        input_dim=config.model.input_dim,
        output_dim=config.model.output_dim,
        modes=config.model.modes,
        width=config.model.width,
        depth=config.model.depth,
        spatial_dim=config.model.spatial_dim,
        posterior_type=config.posterior.method,
        prior_precision=config.posterior.prior_precision
    )
    
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Run comprehensive diagnostics
    sample_input = train_ic[:2].unsqueeze(1)  # Add channel dimension
    sample_target = train_sol[:2].unsqueeze(1)
    
    diagnostics = run_comprehensive_diagnostics(model, sample_input, sample_target)
    
    logger.info(f"Diagnostics: {diagnostics['summary']}")
    if diagnostics['summary']['critical'] > 0:
        logger.error("Critical issues found in diagnostics!")
        return
    
    # 4. MONITORING SETUP
    logger.info("=== STEP 4: Monitoring Setup ===")
    
    # Create monitoring suite
    monitoring = create_monitoring_suite("comprehensive_example")
    training_logger = create_training_logger("comprehensive_example")
    
    # Start monitoring
    monitoring['resource_monitor'].start()
    monitoring['health_monitor'].start()
    
    # 5. TRAINING WITH COMPREHENSIVE MONITORING
    logger.info("=== STEP 5: Training with Monitoring ===")
    
    try:
        with secure_operation("training", max_memory_gb=8.0) as security:
            # Start training logging
            training_logger.start_training(
                total_epochs=config.training.epochs,
                model_config=config.model.to_dict()
            )
            
            # Train the model
            history = model.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=config.training.epochs,
                lr=config.model.learning_rate,
                device="cpu"  # Use CPU for compatibility
            )
            
            # Log training completion
            training_logger.training_complete({
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
                'best_val_loss': min(history['val_loss']) if history['val_loss'] else None
            })
            
            logger.info(f"Training completed - Final loss: {history['train_loss'][-1]:.6f}")
    
    finally:
        # Stop monitoring
        monitoring['resource_monitor'].stop()
        monitoring['health_monitor'].stop()
    
    # 6. POSTERIOR FITTING FOR UNCERTAINTY QUANTIFICATION
    logger.info("=== STEP 6: Posterior Fitting ===")
    
    # Fit posterior approximation
    model.fit_posterior(train_loader, val_loader)
    logger.info("Posterior approximation fitted successfully")
    
    # 7. UNCERTAINTY QUANTIFICATION AND EVALUATION
    logger.info("=== STEP 7: Uncertainty Quantification and Evaluation ===")
    
    model.eval()
    
    # Test uncertainty quantification on test set
    test_metrics = []
    uncertainties = []
    
    with torch.no_grad():
        for test_input, test_target in test_loader:
            test_input_reshaped = test_input.unsqueeze(1)  # Add channel dimension
            
            # Get predictions with uncertainty
            mean_pred, std_pred = model.predict_with_uncertainty(
                test_input_reshaped, num_samples=50, return_std=True
            )
            
            # Remove channel dimension for comparison
            mean_pred = mean_pred.squeeze(1)
            std_pred = std_pred.squeeze(1)
            
            # Calculate metrics
            mse = torch.mean((mean_pred - test_target) ** 2, dim=-1)
            mae = torch.mean(torch.abs(mean_pred - test_target), dim=-1)
            
            # Uncertainty metrics
            avg_uncertainty = torch.mean(std_pred, dim=-1)
            
            test_metrics.extend([(mse[i].item(), mae[i].item()) for i in range(len(mse))])
            uncertainties.extend(avg_uncertainty.tolist())
    
    # Calculate overall metrics
    avg_mse = np.mean([m[0] for m in test_metrics])
    avg_mae = np.mean([m[1] for m in test_metrics])
    avg_uncertainty = np.mean(uncertainties)
    
    logger.info(f"Test Results:")
    logger.info(f"  Average MSE: {avg_mse:.6f}")
    logger.info(f"  Average MAE: {avg_mae:.6f}")
    logger.info(f"  Average Uncertainty: {avg_uncertainty:.6f}")
    
    # 8. VISUALIZATION AND ANALYSIS
    logger.info("=== STEP 8: Visualization and Analysis ===")
    
    # Create visualizations
    create_visualizations(model, test_loader, output_dir)
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    # 9. FINAL DIAGNOSTICS AND HEALTH CHECK
    logger.info("=== STEP 9: Final Diagnostics ===")
    
    # Run final diagnostics
    final_diagnostics = run_comprehensive_diagnostics(model, sample_input, sample_target)
    logger.info(f"Final diagnostics: {final_diagnostics['summary']}")
    
    # Save results
    results = {
        'config': config.to_dict(),
        'training_history': history,
        'test_metrics': {
            'mse': avg_mse,
            'mae': avg_mae,
            'uncertainty': avg_uncertainty
        },
        'diagnostics': {
            'initial': diagnostics['summary'],
            'final': final_diagnostics['summary']
        }
    }
    
    import json
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Comprehensive workflow completed! Results saved to {output_dir}")
    
    return model, results


def create_visualizations(model, test_loader, output_dir):
    """Create comprehensive visualizations."""
    
    # Get a test sample
    test_input, test_target = next(iter(test_loader))
    test_input_reshaped = test_input[:4].unsqueeze(1)  # First 4 samples
    test_target = test_target[:4]
    
    # Get predictions with uncertainty
    with torch.no_grad():
        mean_pred, std_pred = model.predict_with_uncertainty(
            test_input_reshaped, num_samples=100, return_std=True
        )
    
    mean_pred = mean_pred.squeeze(1)
    std_pred = std_pred.squeeze(1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    x = np.linspace(0, 2*np.pi, test_input.shape[1])
    
    for i in range(4):
        # Top row: Predictions vs True
        axes[0, i].plot(x, test_input[i].numpy(), 'b--', label='Initial Condition', alpha=0.7)
        axes[0, i].plot(x, test_target[i].numpy(), 'g-', label='True Solution', linewidth=2)
        axes[0, i].plot(x, mean_pred[i].numpy(), 'r-', label='Predicted Mean', linewidth=2)
        
        # Uncertainty bands
        axes[0, i].fill_between(
            x,
            (mean_pred[i] - 2*std_pred[i]).numpy(),
            (mean_pred[i] + 2*std_pred[i]).numpy(),
            alpha=0.3, color='red', label='2σ Uncertainty'
        )
        
        axes[0, i].set_title(f'Sample {i+1}: Prediction vs Truth')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        
        # Bottom row: Uncertainty analysis
        error = torch.abs(mean_pred[i] - test_target[i])
        axes[1, i].plot(x, error.numpy(), 'orange', label='Absolute Error', linewidth=2)
        axes[1, i].plot(x, std_pred[i].numpy(), 'purple', label='Predicted Uncertainty', linewidth=2)
        axes[1, i].set_title(f'Sample {i+1}: Error vs Uncertainty')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "predictions_with_uncertainty.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Visualizations saved")


def plot_training_history(history, output_dir):
    """Plot training history."""
    
    plt.figure(figsize=(12, 4))
    
    # Training and validation loss
    plt.subplot(1, 2, 1)
    epochs = range(1, len(history['train_loss']) + 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    if history.get('val_loss'):
        plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Training time per epoch (if available)
    if 'epoch_times' in history:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['epoch_times'], 'g-', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.title('Training Time per Epoch')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Training history plot saved")


def quick_start_example():
    """Quick start example for new users."""
    
    logger.info("=== QUICK START EXAMPLE ===")
    
    # 1. Create model
    model = ProbabilisticFNO(
        input_dim=1,
        output_dim=1,
        modes=8,
        width=32,
        depth=2,
        spatial_dim=1
    )
    
    # 2. Generate simple synthetic data
    x = torch.randn(32, 64)  # 32 samples, 64 spatial points
    y = torch.sin(x) + 0.1 * torch.randn_like(x)  # Simple transformation
    
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 3. Train model
    history = model.fit(dataloader, epochs=10, lr=0.001)
    
    # 4. Fit posterior for uncertainty
    model.fit_posterior(dataloader)
    
    # 5. Make predictions with uncertainty
    test_x = torch.randn(4, 64)
    test_x_reshaped = test_x.unsqueeze(1)
    
    mean, std = model.predict_with_uncertainty(test_x_reshaped, num_samples=20)
    
    logger.info(f"Quick start completed!")
    logger.info(f"Prediction mean shape: {mean.shape}")
    logger.info(f"Prediction std shape: {std.shape}")
    logger.info(f"Average uncertainty: {torch.mean(std):.4f}")
    
    return model


if __name__ == "__main__":
    print("ProbNeural-Operator-Lab Comprehensive Usage Guide")
    print("=" * 60)
    
    # Run quick start example
    print("\n1. Running Quick Start Example...")
    quick_start_model = quick_start_example()
    
    # Run comprehensive workflow
    print("\n2. Running Comprehensive Workflow...")
    try:
        comprehensive_model, results = comprehensive_workflow()
        print(f"\nWorkflow completed successfully!")
        print(f"Final test MSE: {results['test_metrics']['mse']:.6f}")
        print(f"Final test uncertainty: {results['test_metrics']['uncertainty']:.6f}")
    except Exception as e:
        logger.error(f"Comprehensive workflow failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nUsage guide completed! Check the output directory for detailed results.")