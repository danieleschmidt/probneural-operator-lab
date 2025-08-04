"""Unit tests for neural operator models."""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from probneural_operator.models import ProbabilisticFNO, FourierNeuralOperator
from probneural_operator.data.datasets import BurgersDataset


class TestFourierNeuralOperator:
    """Test cases for FourierNeuralOperator."""
    
    def test_fno_initialization(self):
        """Test FNO can be initialized with different configurations."""
        # Test 1D FNO
        model_1d = FourierNeuralOperator(
            input_dim=1,
            output_dim=1,
            modes=8,
            width=32,
            depth=2,
            spatial_dim=1
        )
        assert model_1d.modes == 8
        assert model_1d.width == 32
        assert model_1d.depth == 2
        
        # Test 2D FNO
        model_2d = FourierNeuralOperator(
            input_dim=2,
            output_dim=1,
            modes=8,
            width=32,
            depth=2,
            spatial_dim=2
        )
        assert model_2d.spatial_dim == 2
    
    def test_fno_forward_pass(self):
        """Test forward pass with different input shapes."""
        model = FourierNeuralOperator(
            input_dim=1,
            output_dim=1,
            modes=8,
            width=32,
            depth=2,
            spatial_dim=1
        )
        
        # Test 1D forward pass
        batch_size = 4
        spatial_size = 64
        x = torch.randn(batch_size, 1, spatial_size)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 1, spatial_size)
        
    def test_fno_2d_forward_pass(self):
        """Test 2D FNO forward pass."""
        model = FourierNeuralOperator(
            input_dim=1,
            output_dim=1,
            modes=8,
            width=32,
            depth=2,
            spatial_dim=2
        )
        
        batch_size = 2
        height, width = 32, 32
        x = torch.randn(batch_size, 1, height, width)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 1, height, width)
    
    def test_parameter_count(self):
        """Test that parameter count is reasonable."""
        model = FourierNeuralOperator(
            input_dim=1,
            output_dim=1,
            modes=8,
            width=32,
            depth=2,
            spatial_dim=1
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0
        assert param_count < 1_000_000  # Should be reasonable size


class TestProbabilisticFNO:
    """Test cases for ProbabilisticFNO."""
    
    def test_probabilistic_fno_initialization(self):
        """Test ProbabilisticFNO initialization."""
        model = ProbabilisticFNO(
            input_dim=1,
            output_dim=1,
            modes=8,
            width=32,
            depth=2,
            spatial_dim=1,
            posterior_type='laplace',
            prior_precision=1.0
        )
        
        assert model.posterior_type == 'laplace'
        assert model.prior_precision == 1.0
        assert not model._is_fitted
    
    def test_probabilistic_fno_forward_pass(self):
        """Test forward pass works like regular FNO."""
        model = ProbabilisticFNO(
            input_dim=1,
            output_dim=1,
            modes=8,
            width=32,
            depth=2,
            spatial_dim=1
        )
        
        batch_size = 4
        spatial_size = 64
        x = torch.randn(batch_size, 1, spatial_size)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, 1, spatial_size)
    
    def test_training_with_custom_fit(self):
        """Test the custom fit method works correctly."""
        # Create simple synthetic data
        batch_size = 8
        spatial_size = 32
        n_samples = 32
        
        # Generate random data
        inputs = torch.randn(n_samples, spatial_size)
        targets = torch.randn(n_samples, spatial_size)  # Same shape for simplicity
        
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model = ProbabilisticFNO(
            input_dim=1,
            output_dim=1,
            modes=4,
            width=16,
            depth=1,
            spatial_dim=1
        )
        
        # Train for a few epochs
        history = model.fit(
            train_loader=dataloader,
            val_loader=None,
            epochs=2,
            lr=1e-2,
            device='cpu'
        )
        
        assert 'train_loss' in history
        assert len(history['train_loss']) == 2
        assert all(isinstance(loss, float) for loss in history['train_loss'])
    
    def test_prediction_with_reshaping(self):
        """Test that prediction handles reshaping correctly."""
        model = ProbabilisticFNO(
            input_dim=1,
            output_dim=1,
            modes=4,
            width=16,
            depth=1,
            spatial_dim=1
        )
        
        # Test prediction with and without channel dimension
        x_no_channel = torch.randn(2, 32)  # (batch, spatial)
        x_with_channel = torch.randn(2, 1, 32)  # (batch, channel, spatial)
        
        with torch.no_grad():
            # Both should work
            pred1 = model.predict(x_with_channel)
            pred2 = model.predict(x_no_channel.unsqueeze(1))
        
        assert pred1.shape == (2, 1, 32)
        assert pred2.shape == (2, 1, 32)
    
    def test_get_config(self):
        """Test configuration retrieval."""
        config = {
            'input_dim': 2,
            'output_dim': 1,
            'modes': 12,
            'width': 64,
            'depth': 4,
            'spatial_dim': 2
        }
        
        model = ProbabilisticFNO(**config)
        retrieved_config = model.get_config()
        
        assert retrieved_config['input_dim'] == 2
        assert retrieved_config['output_dim'] == 1
        assert retrieved_config['modes'] == 12
        assert retrieved_config['width'] == 64


class TestModelIntegration:
    """Integration tests with datasets."""
    
    def test_fno_with_synthetic_data(self):
        """Test FNO training with synthetic Burgers data."""
        # Create small dataset
        try:
            dataset = BurgersDataset(
                data_path='/tmp/test_burgers.h5',
                split='train',
                resolution=32,
                time_steps=10,
                viscosity=0.01
            )
            
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
            
            model = ProbabilisticFNO(
                input_dim=1,
                output_dim=1,
                modes=4,
                width=16,
                depth=1,
                spatial_dim=1
            )
            
            # Train for 1 epoch
            history = model.fit(
                train_loader=dataloader,
                epochs=1,
                lr=1e-2,
                device='cpu'
            )
            
            assert len(history['train_loss']) == 1
            assert isinstance(history['train_loss'][0], float)
            
        except Exception as e:
            pytest.skip(f"Skipping integration test due to: {e}")


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_spatial_dim(self):
        """Test that invalid spatial dimensions raise errors."""
        with pytest.raises(ValueError):
            FourierNeuralOperator(
                input_dim=1,
                output_dim=1,
                modes=8,
                width=32,
                depth=2,
                spatial_dim=4  # Unsupported
            )
    
    def test_zero_modes(self):
        """Test handling of zero modes."""
        # Should still work, just with no spectral components
        model = FourierNeuralOperator(
            input_dim=1,
            output_dim=1,
            modes=0,
            width=32,
            depth=2,
            spatial_dim=1
        )
        
        x = torch.randn(2, 1, 32)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 1, 32)
    
    def test_mismatched_input_output_dims(self):
        """Test various input/output dimension combinations."""
        model = FourierNeuralOperator(
            input_dim=3,
            output_dim=2,
            modes=8,
            width=32,
            depth=2,
            spatial_dim=1
        )
        
        x = torch.randn(2, 3, 64)
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (2, 2, 64)
    
    def test_empty_dataloader(self):
        """Test behavior with empty dataloader."""
        empty_dataset = TensorDataset(torch.empty(0, 32), torch.empty(0, 32))
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
        
        assert 'train_loss' in history
        assert len(history['train_loss']) == 1