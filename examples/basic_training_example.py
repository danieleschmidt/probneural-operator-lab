#!/usr/bin/env python3
"""
Basic training example for ProbNeural-Operator-Lab.

This example demonstrates:
1. Loading/generating synthetic Burgers equation data
2. Training a Probabilistic FNO model
3. Making predictions with uncertainty quantification
4. Evaluating model performance
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import ProbNeural-Operator-Lab components
from probneural_operator import ProbabilisticFNO
from probneural_operator.data.datasets import BurgersDataset
from probneural_operator.posteriors import LinearizedLaplace


def main():
    """Main training and evaluation loop."""
    print("=== ProbNeural-Operator-Lab: Basic Training Example ===\n")
    
    # Configuration
    config = {
        'data': {
            'resolution': 128,
            'time_steps': 50,
            'viscosity': 0.01,
            'batch_size': 8
        },
        'model': {
            'input_dim': 1,  # Initial condition
            'output_dim': 1,  # Solution at final time
            'modes': 16,
            'width': 64,
            'depth': 4,
            'spatial_dim': 1
        },
        'training': {
            'epochs': 50,
            'lr': 1e-3,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    }
    
    print(f"Using device: {config['training']['device']}")
    
    # Step 1: Create datasets
    print("\n1. Creating datasets...")
    
    train_dataset = BurgersDataset(
        data_path='/tmp/burgers_train.h5',
        split='train',
        resolution=config['data']['resolution'],
        time_steps=config['data']['time_steps'],
        viscosity=config['data']['viscosity']
    )
    
    val_dataset = BurgersDataset(
        data_path='/tmp/burgers_val.h5', 
        split='val',
        resolution=config['data']['resolution'],
        time_steps=config['data']['time_steps'],
        viscosity=config['data']['viscosity']
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['data']['batch_size'], 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False
    )
    
    # Step 2: Create model
    print("\n2. Creating Probabilistic FNO model...")
    
    model = ProbabilisticFNO(
        input_dim=config['model']['input_dim'],
        output_dim=config['model']['output_dim'],
        modes=config['model']['modes'],
        width=config['model']['width'],
        depth=config['model']['depth'],
        spatial_dim=config['model']['spatial_dim'],
        posterior_type='laplace',
        prior_precision=1.0
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 3: Train model
    print("\n3. Training model...")
    
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        lr=config['training']['lr'],
        device=config['training']['device']
    )
    
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    
    # Step 4: Fit posterior for uncertainty quantification
    print("\n4. Fitting posterior for uncertainty quantification...")
    
    try:
        model.fit_posterior(train_loader, val_loader)
        print("Posterior fitted successfully!")
        uncertainty_enabled = True
    except Exception as e:
        print(f"Warning: Could not fit posterior: {e}")
        print("Continuing without uncertainty quantification...")
        uncertainty_enabled = False
    
    # Step 5: Evaluate model
    print("\n5. Evaluating model...")
    
    model.eval()
    test_losses = []
    predictions = []
    targets = []
    uncertainties = [] if uncertainty_enabled else None
    
    with torch.no_grad():
        for batch_idx, (inputs, outputs) in enumerate(val_loader):
            inputs = inputs.to(config['training']['device'])
            targets_batch = outputs[:, -1].to(config['training']['device'])  # Final time step
            
            if uncertainty_enabled:
                try:
                    pred_mean, pred_std = model.predict_with_uncertainty(
                        inputs.unsqueeze(1),  # Add channel dimension
                        return_std=True,
                        num_samples=50
                    )
                    pred_mean = pred_mean.squeeze(1)  # Remove channel dimension
                    pred_std = pred_std.squeeze(1)
                    uncertainties.extend(pred_std.cpu().numpy())
                except:
                    # Fallback to regular prediction
                    pred_mean = model.predict(inputs.unsqueeze(1)).squeeze(1)
                    uncertainty_enabled = False
            else:
                pred_mean = model.predict(inputs.unsqueeze(1)).squeeze(1)
            
            # Compute loss
            loss = nn.MSELoss()(pred_mean, targets_batch)
            test_losses.append(loss.item())
            
            predictions.extend(pred_mean.cpu().numpy())
            targets.extend(targets_batch.cpu().numpy())
            
            if batch_idx >= 10:  # Limit evaluation for demo
                break
    
    avg_test_loss = np.mean(test_losses)
    print(f"Average test loss: {avg_test_loss:.6f}")
    
    # Step 6: Visualize results
    print("\n6. Creating visualizations...")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History')
    plt.legend()
    plt.yscale('log')
    
    # Plot predictions vs targets
    plt.subplot(1, 2, 2)
    predictions_flat = np.array(predictions).flatten()
    targets_flat = np.array(targets).flatten()
    
    plt.scatter(targets_flat, predictions_flat, alpha=0.5, s=1)
    min_val = min(targets_flat.min(), predictions_flat.min())
    max_val = max(targets_flat.max(), predictions_flat.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictions vs Truth')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/tmp/training_results.png', dpi=150, bbox_inches='tight')
    print("Training results saved to /tmp/training_results.png")
    
    # Plot sample prediction with uncertainty
    if uncertainty_enabled and uncertainties:
        plt.figure(figsize=(10, 6))
        
        # Get a sample from validation set
        sample_input, sample_target = val_dataset[0]
        sample_input = sample_input.unsqueeze(0).unsqueeze(1).to(config['training']['device'])
        
        with torch.no_grad():
            pred_mean, pred_std = model.predict_with_uncertainty(
                sample_input, return_std=True, num_samples=100
            )
            
        x = np.linspace(0, 1, config['data']['resolution'])
        pred_mean = pred_mean.squeeze().cpu().numpy()
        pred_std = pred_std.squeeze().cpu().numpy()
        sample_target = sample_target[-1].numpy()  # Final time step
        
        plt.plot(x, sample_target, 'b-', label='True Solution', linewidth=2)
        plt.plot(x, pred_mean, 'r--', label='Mean Prediction', linewidth=2)
        plt.fill_between(x, pred_mean - 2*pred_std, pred_mean + 2*pred_std, 
                        alpha=0.3, color='red', label='95% Confidence')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.title('Prediction with Uncertainty Quantification')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig('/tmp/uncertainty_prediction.png', dpi=150, bbox_inches='tight')
        print("Uncertainty prediction saved to /tmp/uncertainty_prediction.png")
    
    # Step 7: Summary
    print("\n=== Training Summary ===")
    print(f"Dataset: Burgers equation with {config['data']['resolution']} spatial points")
    print(f"Model: Probabilistic FNO with {config['model']['modes']} modes, width {config['model']['width']}")
    print(f"Training: {config['training']['epochs']} epochs, learning rate {config['training']['lr']}")
    print(f"Final validation loss: {history['val_loss'][-1]:.6f}")
    print(f"Test loss: {avg_test_loss:.6f}")
    print(f"Uncertainty quantification: {'Enabled' if uncertainty_enabled else 'Disabled'}")
    
    print("\n✅ Training completed successfully!")
    print("\nNext steps:")
    print("- Try different PDE types (NavierStokes, Darcy, Heat)")
    print("- Experiment with different neural operator architectures")
    print("- Use active learning for efficient data collection")
    print("- Apply to real-world scientific computing problems")


if __name__ == "__main__":
    main()