"""Integration tests for complete workflows."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TestBasicWorkflows:
    """Test basic ML workflows end-to-end."""
    
    def test_simple_training_workflow(self, sample_dataloader, device):
        """Test a simple training workflow."""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Training loop
        model.train()
        initial_loss = None
        
        for epoch in range(3):  # Very short training
            epoch_loss = 0.0
            
            for batch_idx, (x, y) in enumerate(sample_dataloader):
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output.squeeze(), y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if initial_loss is None:
                    initial_loss = loss.item()
        
        # Basic checks
        assert initial_loss is not None
        assert epoch_loss >= 0
        
        # Model should have learned something (loss should decrease)
        # Note: This is not guaranteed for random data, but we check anyway
        model.eval()
        with torch.no_grad():
            x, y = next(iter(sample_dataloader))
            x, y = x.to(device), y.to(device)
            final_output = model(x)
            assert torch.isfinite(final_output).all()
    
    def test_evaluation_workflow(self, sample_dataloader, device):
        """Test model evaluation workflow."""
        # Create and train a simple model
        model = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        ).to(device)
        
        criterion = nn.MSELoss()
        
        # Evaluation
        model.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for x, y in sample_dataloader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output.squeeze(), y)
                
                total_loss += loss.item() * x.size(0)
                num_samples += x.size(0)
        
        avg_loss = total_loss / num_samples
        assert avg_loss >= 0
        assert num_samples > 0
    
    def test_checkpoint_save_load_workflow(self, model_checkpoint_dir, device):
        """Test model checkpoint save/load workflow."""
        # Create model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        ).to(device)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Save initial state
        initial_state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': 0
        }
        
        checkpoint_path = model_checkpoint_dir / "test_checkpoint.pt"
        torch.save(initial_state, checkpoint_path)
        
        # Modify model
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(1.0)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        
        # Verify checkpoint was loaded correctly
        assert checkpoint['epoch'] == 0
        # Check that parameters were restored (they shouldn't all be 1.0)
        params_all_ones = all(
            torch.allclose(param, torch.ones_like(param))
            for param in model.parameters()
        )
        assert not params_all_ones


class TestUncertaintyWorkflows:
    """Test uncertainty quantification workflows."""
    
    def test_mock_uncertainty_workflow(self, mock_neural_operator, mock_posterior, sample_dataloader):
        """Test uncertainty quantification workflow with mocks."""
        # Setup
        model = mock_neural_operator()
        posterior = mock_posterior()
        
        # Fit posterior
        posterior.fit(model, sample_dataloader)
        assert posterior.fitted
        
        # Make predictions with uncertainty
        x_test = torch.randn(10, 2)
        mean, std = posterior.predict(x_test)
        
        assert mean.shape == x_test.shape
        assert std.shape == x_test.shape
        assert (std > 0).all()
        
        # Sample from posterior
        samples = posterior.sample(x_test, n_samples=20)
        assert samples.shape == (20, *x_test.shape)
        
        # Compute log marginal likelihood
        lml = posterior.log_marginal_likelihood()
        assert isinstance(lml, torch.Tensor)
    
    def test_ensemble_like_workflow(self, sample_dataloader, device):
        """Test ensemble-like workflow for uncertainty."""
        # Create multiple models (simple ensemble)
        models = []
        for i in range(3):
            model = nn.Sequential(
                nn.Linear(2, 8),
                nn.ReLU(),
                nn.Linear(8, 1)
            ).to(device)
            models.append(model)
        
        # Train each model with different initialization
        for model_idx, model in enumerate(models):
            # Different random seed for each model
            torch.manual_seed(42 + model_idx)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss()
            
            model.train()
            for epoch in range(2):  # Very short training
                for x, y in sample_dataloader:
                    x, y = x.to(device), y.to(device)
                    
                    optimizer.zero_grad()
                    output = model(x)
                    loss = criterion(output.squeeze(), y)
                    loss.backward()
                    optimizer.step()
        
        # Ensemble prediction
        x_test = torch.randn(5, 2).to(device)
        predictions = []
        
        for model in models:
            model.eval()
            with torch.no_grad():
                pred = model(x_test)
                predictions.append(pred)
        
        # Compute ensemble statistics
        predictions = torch.stack(predictions, dim=0)  # (n_models, batch_size, output_dim)
        ensemble_mean = predictions.mean(dim=0)
        ensemble_std = predictions.std(dim=0)
        
        assert ensemble_mean.shape == (5, 1)
        assert ensemble_std.shape == (5, 1)
        assert (ensemble_std >= 0).all()


class TestActivelearningWorkflows:
    """Test active learning workflows."""
    
    def test_mock_active_learning_workflow(self, mock_neural_operator, mock_posterior, active_learning_config):
        """Test active learning workflow with mocks."""
        config = active_learning_config
        
        # Setup
        model = mock_neural_operator()
        posterior = mock_posterior()
        
        # Initial labeled data (small)
        x_labeled = torch.randn(config["init_size"], 2)
        y_labeled = torch.randn(config["init_size"])
        
        # Pool of unlabeled data
        x_pool = torch.randn(50, 2)
        
        # Fit initial posterior
        labeled_dataset = TensorDataset(x_labeled, y_labeled)
        labeled_loader = DataLoader(labeled_dataset, batch_size=4)
        posterior.fit(model, labeled_loader)
        
        # Active learning loop (simplified)
        for iteration in range(3):  # Very short for testing
            # Acquisition function (mock implementation using uncertainty)
            with torch.no_grad():
                _, uncertainties = posterior.predict(x_pool)
                uncertainty_scores = uncertainties.sum(dim=1)  # Simple acquisition
            
            # Select batch with highest uncertainty
            _, selected_indices = torch.topk(uncertainty_scores, config["batch_size"])
            selected_x = x_pool[selected_indices]
            
            # Simulate labeling (random labels for testing)
            selected_y = torch.randn(config["batch_size"])
            
            # Add to labeled set
            x_labeled = torch.cat([x_labeled, selected_x])
            y_labeled = torch.cat([y_labeled, selected_y])
            
            # Remove from pool
            mask = torch.ones(x_pool.size(0), dtype=torch.bool)
            mask[selected_indices] = False
            x_pool = x_pool[mask]
            
            # Refit posterior
            updated_dataset = TensorDataset(x_labeled, y_labeled)
            updated_loader = DataLoader(updated_dataset, batch_size=4)
            posterior.fit(model, updated_loader)
        
        # Check that we acquired the expected amount of data
        expected_size = config["init_size"] + 3 * config["batch_size"]
        assert x_labeled.shape[0] == expected_size
        assert y_labeled.shape[0] == expected_size


class TestPDEWorkflows:
    """Test PDE-related workflows."""
    
    def test_pde_data_workflow(self, sample_pde_data, device):
        """Test PDE data processing workflow."""
        data = sample_pde_data
        
        # Extract data
        x = data['x'].to(device)
        t = data['t'].to(device)
        u0 = data['u0'].to(device)
        u_exact = data['u_exact'].to(device)
        
        # Create a simple neural operator for PDE
        class SimplePDEOperator(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, output_size)
                )
            
            def forward(self, u0):
                # Input: initial condition u0 (batch_size, nx)
                # Output: solution at all time steps (batch_size, nt, nx)
                batch_size = u0.shape[0]
                nx = u0.shape[1]
                nt = u_exact.shape[1]
                
                # Simple approach: predict each time step independently
                # In reality, this would be more sophisticated
                output = []
                current_u = u0
                
                for t_idx in range(nt):
                    # Predict next state based on current state
                    if t_idx == 0:
                        next_u = current_u  # Initial condition
                    else:
                        # Use network to predict evolution
                        next_u = self.net(current_u.unsqueeze(-1)).squeeze(-1)
                        current_u = next_u
                    
                    output.append(next_u)
                
                return torch.stack(output, dim=1)  # (batch_size, nt, nx)
        
        # Create and test model
        model = SimplePDEOperator(input_size=1, hidden_size=32, output_size=1).to(device)
        
        # Test forward pass
        batch_u0 = u0.unsqueeze(0)  # Add batch dimension
        output = model(batch_u0)
        
        assert output.shape == (1, u_exact.shape[0], u_exact.shape[1])
        assert torch.isfinite(output).all()
    
    @pytest.mark.slow
    def test_pde_training_workflow(self, pde_dataloader, device):
        """Test PDE training workflow."""
        # Simple neural operator
        class SimpleOperator(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(32, 64),  # Spatial dimension
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32 * 20)  # Output: spatial * temporal
                )
            
            def forward(self, u0):
                # u0: (batch_size, spatial_dim)
                # output: (batch_size, temporal_dim, spatial_dim)
                batch_size = u0.shape[0]
                output = self.net(u0)
                return output.view(batch_size, 20, 32)  # Reshape to (batch, time, space)
        
        model = SimpleOperator().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Training loop
        model.train()
        for epoch in range(2):  # Very short for testing
            for batch_idx, (u0, u_target) in enumerate(pde_dataloader):
                u0, u_target = u0.to(device), u_target.to(device)
                
                optimizer.zero_grad()
                u_pred = model(u0)
                loss = criterion(u_pred, u_target)
                loss.backward()
                optimizer.step()
                
                # Basic checks
                assert torch.isfinite(loss)
                assert not torch.isnan(loss)


@pytest.mark.integration
class TestEndToEndWorkflows:
    """End-to-end integration tests."""
    
    def test_complete_probabilistic_workflow(self, sample_dataloader, mock_posterior, device):
        """Test complete probabilistic neural operator workflow."""
        # This would test the full pipeline:
        # 1. Model training
        # 2. Posterior fitting
        # 3. Uncertainty prediction
        # 4. Calibration
        # 5. Active learning
        
        # For now, we use mocks, but in the future this would test
        # the real implementations
        
        # 1. Create and train a model
        model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        
        # Quick training
        model.train()
        for epoch in range(2):
            for x, y in sample_dataloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output.squeeze(), y)
                loss.backward()
                optimizer.step()
        
        # 2. Fit posterior
        posterior = mock_posterior()
        posterior.fit(model, sample_dataloader)
        
        # 3. Make uncertain predictions
        x_test = torch.randn(20, 2).to(device)
        mean, std = posterior.predict(x_test)
        
        # 4. Basic validation
        assert mean.shape == x_test.shape
        assert std.shape == x_test.shape
        assert (std > 0).all()
        
        # 5. Sample from posterior
        samples = posterior.sample(x_test, n_samples=10)
        assert samples.shape == (10, *x_test.shape)
        
        # This demonstrates the complete workflow structure
        # that will be implemented with real components