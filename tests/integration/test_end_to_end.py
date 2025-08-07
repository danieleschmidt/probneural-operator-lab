"""End-to-end integration tests."""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from probneural_operator.models import ProbabilisticFNO
from probneural_operator.utils import (
    create_default_config, setup_logging, create_monitoring_suite,
    run_comprehensive_diagnostics, secure_operation
)


@pytest.mark.integration
class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    def test_basic_training_workflow(self, sample_dataloader, temp_dir):
        """Test basic model training workflow."""
        # Set up logging
        setup_logging(log_dir=str(temp_dir / "logs"))
        
        # Create model from config
        config = create_default_config("fno")
        config.model.modes = 4
        config.model.width = 16
        config.model.depth = 2
        config.training.epochs = 3
        config.output_dir = str(temp_dir / "outputs")
        
        # Initialize model
        model = ProbabilisticFNO(
            input_dim=config.model.input_dim,
            output_dim=config.model.output_dim,
            modes=config.model.modes,
            width=config.model.width,
            depth=config.model.depth,
            spatial_dim=config.model.spatial_dim
        )
        
        # Run diagnostics before training
        sample_input = next(iter(sample_dataloader))[0][:2]  # Small batch for testing
        sample_target = next(iter(sample_dataloader))[1][:2]
        
        diagnostics = run_comprehensive_diagnostics(
            model, sample_input, sample_target
        )
        
        assert diagnostics['summary']['total_checks'] > 0
        assert diagnostics['summary']['overall_status'] in ['healthy', 'warning']
        
        # Train model
        with secure_operation("training", max_memory_gb=8.0) as security:
            history = model.fit(
                train_loader=sample_dataloader,
                epochs=config.training.epochs,
                lr=config.model.learning_rate,
                device="cpu"
            )
        
        # Verify training completed
        assert len(history["train_loss"]) == config.training.epochs
        assert all(isinstance(loss, float) for loss in history["train_loss"])
        assert "total_time" in history
        
        # Test prediction
        test_input = sample_input
        predictions = model.predict(test_input)
        
        assert predictions.shape == test_input.shape
        assert torch.all(torch.isfinite(predictions))
    
    def test_probabilistic_workflow(self, sample_dataloader):
        """Test probabilistic neural operator workflow."""
        # Create probabilistic model
        model = ProbabilisticFNO(
            input_dim=1,
            output_dim=1,
            modes=4,
            width=16,
            depth=2,
            spatial_dim=1,
            posterior_type="laplace"
        )
        
        # Train model
        history = model.fit(
            train_loader=sample_dataloader,
            epochs=2,
            lr=1e-3,
            device="cpu"
        )
        
        assert len(history["train_loss"]) == 2
        
        # Fit posterior
        model.fit_posterior(sample_dataloader)
        assert model._is_fitted
        
        # Test uncertainty quantification
        test_input = next(iter(sample_dataloader))[0][:2]
        
        mean, std = model.predict_with_uncertainty(test_input, num_samples=10)
        
        assert mean.shape == test_input.shape
        assert std.shape == test_input.shape
        assert torch.all(std >= 0)  # Standard deviation should be non-negative
        assert torch.all(torch.isfinite(mean))
        assert torch.all(torch.isfinite(std))
        
        # Test sampling
        samples = model.sample_predictions(test_input, num_samples=5)
        assert samples.shape[0] == 5  # Number of samples
        assert samples.shape[1:] == test_input.shape
    
    def test_monitoring_integration(self, sample_dataloader):
        """Test integration with monitoring systems."""
        # Set up monitoring suite
        monitoring = create_monitoring_suite("integration_test")
        
        # Start monitoring
        monitoring['resource_monitor'].start()
        monitoring['health_monitor'].start()
        monitoring['training_monitor'].start_training()
        
        try:
            # Create and train model
            model = ProbabilisticFNO(
                input_dim=1, output_dim=1, modes=4, width=16, depth=2
            )
            
            # Train with monitoring
            for epoch in range(3):
                model.train()
                total_loss = 0.0
                num_batches = 0
                
                for batch_data, batch_targets in sample_dataloader:
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_data)
                    loss = torch.nn.functional.mse_loss(outputs, batch_targets)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
                
                # Log to training monitor
                monitoring['training_monitor'].log_epoch(epoch, avg_loss)
            
            # Check monitoring results
            training_summary = monitoring['training_monitor'].get_training_summary()
            assert 'best_loss' in training_summary
            assert training_summary['best_loss'] < float('inf')
            
            health_report = monitoring['health_monitor'].get_health_report()
            assert 'resource_usage' in health_report
            assert 'recent_alerts' in health_report
            
        finally:
            # Stop monitoring
            monitoring['resource_monitor'].stop()
            monitoring['health_monitor'].stop()
    
    def test_error_recovery_workflow(self, problematic_tensors):
        """Test error recovery and graceful handling."""
        model = ProbabilisticFNO(
            input_dim=1, output_dim=1, modes=4, width=16, depth=2
        )
        
        # Test with problematic inputs
        from probneural_operator.utils.exceptions import (
            NumericalStabilityError, TensorValidationError
        )
        
        # NaN input should raise appropriate error
        with pytest.raises((NumericalStabilityError, TensorValidationError)):
            model.predict(problematic_tensors["nan"])
        
        # Infinite input should raise appropriate error
        with pytest.raises((NumericalStabilityError, TensorValidationError)):
            model.predict(problematic_tensors["inf"])
        
        # Normal tensor should work fine
        normal_prediction = model.predict(problematic_tensors["normal"])
        assert torch.all(torch.isfinite(normal_prediction))
    
    def test_configuration_driven_workflow(self, temp_dir):
        """Test workflow driven by configuration files."""
        from probneural_operator.utils.config import ConfigManager, ExperimentConfig, FNOConfig
        
        # Create configuration
        config = ExperimentConfig(
            name="config_driven_test",
            model=FNOConfig(
                input_dim=1, output_dim=1, modes=8, width=32, depth=3,
                learning_rate=0.001, dropout=0.1
            ),
            output_dir=str(temp_dir / "config_outputs"),
            seed=42
        )
        
        # Save configuration
        config_manager = ConfigManager()
        config_path = temp_dir / "test_config.yaml"
        config_manager.save_config(config, config_path)
        
        # Load configuration
        loaded_config = config_manager.load_config(config_path)
        assert loaded_config.name == config.name
        
        # Create model from loaded config
        model = ProbabilisticFNO(
            input_dim=loaded_config.model.input_dim,
            output_dim=loaded_config.model.output_dim,
            modes=loaded_config.model.modes,
            width=loaded_config.model.width,
            depth=loaded_config.model.depth
        )
        
        # Create synthetic data
        torch.manual_seed(loaded_config.seed)
        inputs = torch.randn(32, 1, 64)
        targets = torch.randn(32, 1, 64)
        dataset = TensorDataset(inputs, targets)
        dataloader = DataLoader(dataset, batch_size=8)
        
        # Train model
        history = model.fit(
            train_loader=dataloader,
            epochs=2,
            lr=loaded_config.model.learning_rate,
            device="cpu"
        )
        
        assert len(history["train_loss"]) == 2
    
    @pytest.mark.slow
    def test_performance_workflow(self, sample_dataloader, performance_baseline):
        """Test performance benchmarks and regression."""
        import time
        
        # Create model
        model = ProbabilisticFNO(
            input_dim=1, output_dim=1, modes=16, width=64, depth=4
        )
        
        # Measure forward pass time
        test_input = next(iter(sample_dataloader))[0]
        
        model.eval()
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(10):  # Average over multiple runs
                _ = model(test_input)
        
        avg_forward_time = (time.time() - start_time) / 10 * 1000  # ms
        
        # Check against baseline (allow 50% tolerance)
        assert avg_forward_time < performance_baseline["fno_forward_time_ms"] * 1.5
        
        # Measure training performance
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        
        model.train()
        start_time = time.time()
        
        for batch_data, batch_targets in sample_dataloader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            break  # Just measure one batch
        
        backward_time = (time.time() - start_time) * 1000  # ms
        
        # Check backward pass performance
        assert backward_time < performance_baseline["fno_backward_time_ms"] * 2.0


@pytest.mark.integration
class TestSystemIntegration:
    """Test system-level integration scenarios."""
    
    def test_memory_management_integration(self, sample_dataloader):
        """Test memory management across components."""
        from probneural_operator.utils.security import MemoryMonitor
        
        monitor = MemoryMonitor(max_memory_gb=4.0)
        
        with monitor.monitor_operation("integration_test"):
            # Create multiple models to test memory usage
            models = []
            for i in range(3):
                model = ProbabilisticFNO(
                    input_dim=1, output_dim=1, modes=8, width=32, depth=2
                )
                models.append(model)
                
                # Train briefly
                history = model.fit(
                    train_loader=sample_dataloader,
                    epochs=1,
                    lr=1e-3,
                    device="cpu"
                )
                
                assert len(history["train_loss"]) == 1
            
            # Check memory usage didn't explode
            memory_info = monitor.check_memory_usage("final_check")
            assert memory_info['usage_fraction'] < 0.9  # Less than 90% of limit
    
    def test_logging_integration(self, temp_dir, sample_dataloader):
        """Test logging system integration."""
        from probneural_operator.utils.logging_config import (
            setup_logging, create_training_logger
        )
        
        log_dir = temp_dir / "integration_logs"
        setup_logging(log_dir=str(log_dir))
        
        # Create training logger
        training_logger = create_training_logger("integration_test")
        
        # Start training with logging
        model = ProbabilisticFNO(
            input_dim=1, output_dim=1, modes=4, width=16, depth=2
        )
        
        model_config = {
            'input_dim': 1, 'output_dim': 1, 'modes': 4, 'width': 16, 'depth': 2
        }
        
        training_logger.start_training(total_epochs=2, model_config=model_config)
        
        # Simulate training
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        
        for epoch in range(2):
            model.train()
            epoch_loss = 0.0
            num_batches = 0
            
            epoch_start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            epoch_end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            
            import time
            start_time = time.time()
            
            for batch_data, batch_targets in sample_dataloader:
                optimizer.zero_grad()
                outputs = model(batch_data)
                loss = criterion(outputs, batch_targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            epoch_time = time.time() - start_time
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            
            # Log epoch
            training_logger.log_epoch(
                epoch=epoch,
                train_loss=avg_loss,
                epoch_time=epoch_time
            )
        
        # Complete training
        training_logger.training_complete({'final_loss': avg_loss})
        
        # Verify logs were created
        assert log_dir.exists()
        log_files = list(log_dir.glob("*.log"))
        assert len(log_files) > 0
    
    def test_security_integration(self):
        """Test security system integration."""
        from probneural_operator.utils.security import (
            TensorSecurityValidator, sanitize_tensor, secure_operation
        )
        
        validator = TensorSecurityValidator(
            max_tensor_size_gb=0.1,  # Small limit for testing
            max_dimensions=5
        )
        
        # Test with valid tensor
        valid_tensor = torch.randn(10, 10)
        sanitized = sanitize_tensor(valid_tensor, validator)
        assert torch.allclose(valid_tensor, sanitized)
        
        # Test with problematic tensor
        large_tensor = torch.randn(1000, 1000)  # Too large
        
        with pytest.raises(Exception):  # Should raise security error
            validator.validate_tensor(large_tensor, "large_tensor")
        
        # Test secure operation context
        with secure_operation("security_test", max_memory_gb=2.0) as security:
            model = ProbabilisticFNO(
                input_dim=1, output_dim=1, modes=4, width=8, depth=1
            )
            
            test_input = torch.randn(2, 1, 32)
            output = model(test_input)
            
            # Security context should provide monitoring
            assert 'memory_monitor' in security
            assert 'tensor_validator' in security


@pytest.mark.integration
@pytest.mark.slow
class TestLongRunningIntegration:
    """Test long-running integration scenarios."""
    
    def test_extended_training_workflow(self, sample_dataloader):
        """Test extended training with all monitoring."""
        # This test runs longer training to catch convergence issues
        model = ProbabilisticFNO(
            input_dim=1, output_dim=1, modes=8, width=32, depth=3
        )
        
        # Set up comprehensive monitoring
        monitoring = create_monitoring_suite("extended_training")
        monitoring['resource_monitor'].start()
        monitoring['training_monitor'].start_training()
        
        try:
            # Extended training
            history = model.fit(
                train_loader=sample_dataloader,
                epochs=20,  # Longer training
                lr=1e-3,
                device="cpu"
            )
            
            # Verify training progressed
            assert len(history["train_loss"]) == 20
            
            # Check for convergence (loss should generally decrease)
            early_loss = np.mean(history["train_loss"][:5])
            late_loss = np.mean(history["train_loss"][-5:])
            assert late_loss <= early_loss  # Should improve or stay stable
            
            # Test posterior fitting
            model.fit_posterior(sample_dataloader)
            
            # Test uncertainty quantification after extended training
            test_input = next(iter(sample_dataloader))[0][:1]
            mean, std = model.predict_with_uncertainty(test_input, num_samples=20)
            
            # Uncertainties should be reasonable after training
            assert torch.all(std > 0)
            assert torch.all(std < 10.0)  # Not too large
            
        finally:
            monitoring['resource_monitor'].stop()
    
    def test_memory_leak_detection(self, sample_dataloader):
        """Test for memory leaks during extended operation."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Record initial memory
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform repeated operations
        for i in range(5):
            model = ProbabilisticFNO(
                input_dim=1, output_dim=1, modes=4, width=16, depth=2
            )
            
            # Train briefly
            model.fit(sample_dataloader, epochs=2, lr=1e-3, device="cpu")
            
            # Fit posterior
            model.fit_posterior(sample_dataloader)
            
            # Make predictions
            test_input = next(iter(sample_dataloader))[0][:2]
            _ = model.predict_with_uncertainty(test_input, num_samples=10)
            
            # Clean up
            del model
            gc.collect()
        
        # Check final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB)
        assert memory_increase < 500, f"Potential memory leak: {memory_increase:.1f}MB increase"