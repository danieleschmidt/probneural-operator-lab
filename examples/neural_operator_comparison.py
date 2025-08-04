#!/usr/bin/env python3
"""
Neural Operator Comparison: FNO vs DeepONet

This example demonstrates and compares:
1. Probabilistic Fourier Neural Operator (FNO)
2. Probabilistic Deep Operator Network (DeepONet)

Both models are trained on the same Burgers equation dataset and compared
for accuracy and uncertainty quantification.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

# Import ProbNeural-Operator-Lab components
from probneural_operator import ProbabilisticFNO
from probneural_operator.models import ProbabilisticDeepONet
from probneural_operator.data.datasets import BurgersDataset


def compare_neural_operators():
    """Compare FNO and DeepONet on Burgers equation."""
    print("=== Neural Operator Comparison: FNO vs DeepONet ===\n")
    
    # Configuration
    config = {
        'data': {
            'resolution': 64,
            'time_steps': 20,
            'viscosity': 0.01,
            'batch_size': 16
        },
        'training': {
            'epochs': 20,
            'lr': 1e-3,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    }
    
    print(f"Using device: {config['training']['device']}")
    print(f"Dataset: Burgers equation with resolution {config['data']['resolution']}")
    print(f"Training: {config['training']['epochs']} epochs, lr {config['training']['lr']}")
    
    # Step 1: Create datasets
    print("\n1. Creating datasets...")
    
    train_dataset = BurgersDataset(
        data_path='/tmp/burgers_compare_train.h5',
        split='train',
        resolution=config['data']['resolution'],
        time_steps=config['data']['time_steps'],
        viscosity=config['data']['viscosity']
    )
    
    val_dataset = BurgersDataset(
        data_path='/tmp/burgers_compare_val.h5',
        split='val',
        resolution=config['data']['resolution'],
        time_steps=config['data']['time_steps'],
        viscosity=config['data']['viscosity']
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False)
    
    # Step 2: Create models
    print("\n2. Creating models...")
    
    # Fourier Neural Operator
    fno_model = ProbabilisticFNO(
        input_dim=1,
        output_dim=1,
        modes=16,
        width=64,
        depth=4,
        spatial_dim=1,
        posterior_type='laplace',
        prior_precision=1.0
    )
    
    # Deep Operator Network
    deeponet_model = ProbabilisticDeepONet(
        branch_dim=config['data']['resolution'],
        trunk_dim=1,
        output_dim=1,
        branch_layers=[128, 128, 128],
        trunk_layers=[128, 128, 128],
        activation='tanh',
        posterior_type='laplace',
        prior_precision=1.0
    )
    
    fno_params = sum(p.numel() for p in fno_model.parameters())
    deeponet_params = sum(p.numel() for p in deeponet_model.parameters())
    
    print(f"FNO parameters: {fno_params:,}")
    print(f"DeepONet parameters: {deeponet_params:,}")
    
    # Step 3: Train models
    print("\n3. Training models...")
    
    models = {
        'FNO': fno_model,
        'DeepONet': deeponet_model
    }
    
    histories = {}
    training_times = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        start_time = time.time()
        
        history = model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config['training']['epochs'],
            lr=config['training']['lr'],
            device=config['training']['device']
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        histories[name] = history
        training_times[name] = training_time
        
        print(f"{name} training completed in {training_time:.1f}s")
        print(f"{name} final train loss: {history['train_loss'][-1]:.6f}")
        print(f"{name} final val loss: {history['val_loss'][-1]:.6f}")
    
    # Step 4: Evaluate models
    print("\n4. Evaluating models...")
    
    results = {}
    
    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        model.eval()
        
        test_losses = []
        predictions = []
        targets = []
        inference_times = []
        
        with torch.no_grad():
            for batch_idx, (inputs, outputs) in enumerate(val_loader):
                inputs = inputs.to(config['training']['device'])
                targets_batch = outputs[:, -1].to(config['training']['device'])  # Final time step
                
                # Time inference
                start_time = time.time()
                
                if name == 'FNO':
                    # FNO expects channel dimension
                    pred = model.predict(inputs.unsqueeze(1)).squeeze(1)
                else:
                    # DeepONet - use its predict method with grid
                    batch_size = inputs.shape[0]
                    n_points = targets_batch.shape[-1]
                    grid = torch.linspace(0, 1, n_points).unsqueeze(0).unsqueeze(-1)
                    grid = grid.expand(batch_size, -1, -1).to(config['training']['device'])
                    pred = model(inputs, grid).squeeze(-1)
                
                end_time = time.time()
                inference_times.append(end_time - start_time)
                
                # Compute loss
                loss = nn.MSELoss()(pred, targets_batch)
                test_losses.append(loss.item())
                
                predictions.extend(pred.cpu().numpy())
                targets.extend(targets_batch.cpu().numpy())
                
                if batch_idx >= 20:  # Limit evaluation for demo
                    break
        
        avg_test_loss = np.mean(test_losses)
        avg_inference_time = np.mean(inference_times)
        
        results[name] = {
            'test_loss': avg_test_loss,
            'inference_time': avg_inference_time,
            'training_time': training_times[name],
            'params': fno_params if name == 'FNO' else deeponet_params,
            'predictions': np.array(predictions),
            'targets': np.array(targets)
        }
        
        print(f"{name} test loss: {avg_test_loss:.6f}")
        print(f"{name} avg inference time: {avg_inference_time*1000:.2f}ms")
    
    # Step 5: Create visualizations
    print("\n5. Creating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Training history comparison
    ax = axes[0, 0]
    for name, history in histories.items():
        ax.plot(history['train_loss'], label=f'{name} Train', linestyle='-')
        ax.plot(history['val_loss'], label=f'{name} Val', linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training History Comparison')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # Performance comparison
    ax = axes[0, 1]
    names = list(results.keys())
    test_losses = [results[name]['test_loss'] for name in names]
    inference_times = [results[name]['inference_time']*1000 for name in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    ax2 = ax.twinx()
    bars1 = ax.bar(x - width/2, test_losses, width, label='Test Loss', alpha=0.8)
    bars2 = ax2.bar(x + width/2, inference_times, width, label='Inference Time (ms)', alpha=0.8, color='orange')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Test Loss', color='blue')
    ax2.set_ylabel('Inference Time (ms)', color='orange')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    
    # Add value labels on bars
    for bar, val in zip(bars1, test_losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.05, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    for bar, val in zip(bars2, inference_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + bar.get_height()*0.05,
                 f'{val:.1f}ms', ha='center', va='bottom', fontsize=10)
    
    # Model complexity comparison  
    ax = axes[0, 2]
    params = [results[name]['params'] for name in names]
    training_times = [results[name]['training_time'] for name in names]
    
    colors = ['blue', 'orange']
    scatter = ax.scatter(params, training_times, c=colors, s=100, alpha=0.7)
    
    for i, name in enumerate(names):
        ax.annotate(name, (params[i], training_times[i]), 
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('Number of Parameters')
    ax.set_ylabel('Training Time (s)')
    ax.set_title('Model Complexity vs Training Time')
    ax.grid(True, alpha=0.3)
    
    # Prediction comparison on sample data
    sample_idx = 0
    for i, name in enumerate(names):
        ax = axes[1, i]
        
        pred = results[name]['predictions'][sample_idx]
        target = results[name]['targets'][sample_idx]
        x_coords = np.linspace(0, 1, len(pred))
        
        ax.plot(x_coords, target, 'b-', label='True Solution', linewidth=2)
        ax.plot(x_coords, pred, 'r--', label='Prediction', linewidth=2)
        ax.fill_between(x_coords, pred-0.1, pred+0.1, alpha=0.3, color='red', label='Uncertainty (placeholder)')
        
        ax.set_xlabel('x')
        ax.set_ylabel('u(x)')
        ax.set_title(f'{name} Sample Prediction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Compute and display error
        error = np.mean((pred - target)**2)
        ax.text(0.05, 0.95, f'MSE: {error:.4f}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = "Model Comparison Summary\\n\\n"
    
    best_accuracy = min(results.keys(), key=lambda x: results[x]['test_loss'])
    best_speed = min(results.keys(), key=lambda x: results[x]['inference_time'])
    most_efficient = min(results.keys(), key=lambda x: results[x]['params'])
    
    summary_text += f"Best Accuracy: {best_accuracy}\\n"
    summary_text += f"  Test Loss: {results[best_accuracy]['test_loss']:.6f}\\n\\n"
    summary_text += f"Fastest Inference: {best_speed}\\n"
    summary_text += f"  Time: {results[best_speed]['inference_time']*1000:.2f}ms\\n\\n"
    summary_text += f"Most Parameter Efficient: {most_efficient}\\n"
    summary_text += f"  Parameters: {results[most_efficient]['params']:,}\\n\\n"
    
    summary_text += "Trade-offs:\\n"
    for name in names:
        r = results[name]
        summary_text += f"{name}:\\n"
        summary_text += f"  Accuracy: {r['test_loss']:.6f}\\n"
        summary_text += f"  Speed: {r['inference_time']*1000:.1f}ms\\n"
        summary_text += f"  Params: {r['params']:,}\\n\\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/tmp/neural_operator_comparison.png', dpi=150, bbox_inches='tight')
    print("Comparison results saved to /tmp/neural_operator_comparison.png")
    
    # Step 6: Print detailed comparison
    print("\n=== Detailed Comparison Results ===")
    print(f"{'Metric':<20} {'FNO':<15} {'DeepONet':<15} {'Winner':<10}")
    print("-" * 65)
    
    metrics = [
        ('Test Loss', 'test_loss', 'lower'),
        ('Inference Time (ms)', 'inference_time', 'lower', 1000),
        ('Training Time (s)', 'training_time', 'lower'),
        ('Parameters', 'params', 'lower')
    ]
    
    for metric_name, key, direction, *scale in metrics:
        scale = scale[0] if scale else 1
        fno_val = results['FNO'][key] * scale
        deeponet_val = results['DeepONet'][key] * scale
        
        if direction == 'lower':
            winner = 'FNO' if fno_val < deeponet_val else 'DeepONet'
        else:
            winner = 'FNO' if fno_val > deeponet_val else 'DeepONet'
        
        if key == 'params':
            print(f"{metric_name:<20} {fno_val:,.0f}{'':>4} {deeponet_val:,.0f}{'':>4} {winner:<10}")
        elif scale == 1000:
            print(f"{metric_name:<20} {fno_val:.2f}{'':>9} {deeponet_val:.2f}{'':>9} {winner:<10}")
        else:
            print(f"{metric_name:<20} {fno_val:.6f}{'':>5} {deeponet_val:.6f}{'':>5} {winner:<10}")
    
    print("\n=== Recommendations ===")
    print("• FNO: Better for problems with translational symmetry and periodic domains")
    print("• DeepONet: Better for general operator learning with arbitrary geometries")
    print("• Both models support uncertainty quantification for reliable predictions")
    print("• Choose based on your specific problem characteristics and constraints")
    
    print("\n✅ Neural operator comparison completed successfully!")


if __name__ == "__main__":
    compare_neural_operators()