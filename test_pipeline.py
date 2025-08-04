#!/usr/bin/env python3
"""Quick test of the complete pipeline."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Import ProbNeural-Operator-Lab components
from probneural_operator import ProbabilisticFNO
from probneural_operator.data.datasets import BurgersDataset

def main():
    """Test the complete pipeline."""
    print("=== Testing ProbNeural-Operator-Lab Pipeline ===\n")
    
    # Step 1: Create small dataset for quick test
    print("1. Creating small test dataset...")
    
    train_dataset = BurgersDataset(
        data_path='/tmp/burgers_test.h5',
        split='train',
        resolution=64,    # Reduced resolution
        time_steps=20,    # Reduced time steps
        viscosity=0.01
    )
    
    print(f"Dataset size: {len(train_dataset)}")
    print(f"Input shape: {train_dataset[0][0].shape}")
    print(f"Output shape: {train_dataset[0][1].shape}")
    
    # Create data loader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    
    # Step 2: Create model
    print("\n2. Creating Probabilistic FNO model...")
    
    model = ProbabilisticFNO(
        input_dim=1,
        output_dim=1,
        modes=8,
        width=32,
        depth=2,
        spatial_dim=1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Step 3: Test forward pass
    print("\n3. Testing forward pass...")
    
    sample_batch = next(iter(train_loader))
    inputs, targets = sample_batch
    print(f"Batch input shape: {inputs.shape}")
    print(f"Batch target shape: {targets.shape}")
    
    # Add channel dimension for FNO
    inputs_with_channel = inputs.unsqueeze(1)
    print(f"Inputs with channel: {inputs_with_channel.shape}")
    
    with torch.no_grad():
        outputs = model(inputs_with_channel)
        print(f"Model output shape: {outputs.shape}")
    
    # Step 4: Train for a few epochs
    print("\n4. Training for 3 epochs...")
    
    history = model.fit(
        train_loader=train_loader,
        val_loader=None,
        epochs=3,
        lr=1e-3,
        device='cpu'
    )
    
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    
    # Step 5: Test prediction
    print("\n5. Testing prediction...")
    
    model.eval()
    with torch.no_grad():
        test_input = inputs[:1].unsqueeze(1)  # Single sample with channel
        prediction = model.predict(test_input)
        print(f"Prediction shape: {prediction.shape}")
        print(f"Prediction range: [{prediction.min():.3f}, {prediction.max():.3f}]")
    
    print("\n✅ Pipeline test completed successfully!")
    print("\nNext: Run full training example with: python3 examples/basic_training_example.py")

if __name__ == "__main__":
    main()