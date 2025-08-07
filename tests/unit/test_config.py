"""Unit tests for configuration management."""

import pytest
import json
import yaml
from pathlib import Path

from probneural_operator.utils.config import (
    ModelConfig, FNOConfig, DeepONetConfig, PosteriorConfig,
    TrainingConfig, ActiveLearningConfig, ExperimentConfig,
    ConfigManager, ConfigFormat, load_config, save_config,
    create_default_config, validate_config_compatibility
)
from probneural_operator.utils.exceptions import ConfigurationError


class TestModelConfigs:
    """Test model configuration classes."""
    
    def test_model_config_valid(self):
        """Test valid model configuration."""
        config = ModelConfig(
            input_dim=2,
            output_dim=1,
            learning_rate=0.001,
            dropout=0.1
        )
        
        config.validate()  # Should not raise
        assert config.input_dim == 2
        assert config.output_dim == 1
        assert config.learning_rate == 0.001
    
    def test_model_config_invalid(self):
        """Test invalid model configuration."""
        config = ModelConfig(input_dim=-1)  # Invalid
        
        with pytest.raises(ConfigurationError):
            config.validate()
        
        config = ModelConfig(learning_rate=0.0)  # Invalid (exclusive minimum)
        with pytest.raises(ConfigurationError):
            config.validate()
        
        config = ModelConfig(activation="invalid_activation")
        with pytest.raises(ConfigurationError):
            config.validate()
    
    def test_fno_config_valid(self):
        """Test valid FNO configuration."""
        config = FNOConfig(
            input_dim=1,
            output_dim=1,
            modes=16,
            width=64,
            depth=4,
            spatial_dim=2
        )
        
        config.validate()
        assert config.modes == 16
        assert config.spatial_dim == 2
    
    def test_fno_config_invalid(self):
        """Test invalid FNO configuration."""
        config = FNOConfig(modes=0)  # Invalid
        with pytest.raises(ConfigurationError):
            config.validate()
        
        config = FNOConfig(spatial_dim=5)  # Invalid (too high)
        with pytest.raises(ConfigurationError):
            config.validate()
    
    def test_deeponet_config_valid(self):
        """Test valid DeepONet configuration."""
        config = DeepONetConfig(
            input_dim=1,
            output_dim=1,
            branch_layers=[128, 128, 128],
            trunk_layers=[128, 128],
            basis_functions=100
        )
        
        config.validate()
        assert len(config.branch_layers) == 3
        assert config.basis_functions == 100
    
    def test_deeponet_config_invalid(self):
        """Test invalid DeepONet configuration."""
        config = DeepONetConfig(basis_functions=0)  # Invalid
        with pytest.raises(ConfigurationError):
            config.validate()
        
        config = DeepONetConfig(branch_layers=[4])  # Too small layers
        with pytest.raises(ConfigurationError):
            config.validate()


class TestPosteriorConfig:
    """Test posterior configuration."""
    
    def test_posterior_config_valid(self):
        """Test valid posterior configurations."""
        # Laplace
        config = PosteriorConfig(method="laplace", prior_precision=1.0)
        config.validate()
        
        # Variational
        config = PosteriorConfig(method="variational", num_samples=50)
        config.validate()
        
        # Ensemble
        config = PosteriorConfig(method="ensemble", ensemble_size=10)
        config.validate()
    
    def test_posterior_config_invalid(self):
        """Test invalid posterior configurations."""
        config = PosteriorConfig(method="invalid_method")
        with pytest.raises(ConfigurationError):
            config.validate()
        
        config = PosteriorConfig(prior_precision=0.0)  # Invalid (exclusive minimum)
        with pytest.raises(ConfigurationError):
            config.validate()


class TestTrainingConfig:
    """Test training configuration."""
    
    def test_training_config_valid(self):
        """Test valid training configuration."""
        config = TrainingConfig(
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            patience=10
        )
        
        config.validate()
        assert config.epochs == 100
        assert config.validation_split == 0.2
    
    def test_training_config_invalid(self):
        """Test invalid training configuration."""
        config = TrainingConfig(epochs=0)  # Invalid
        with pytest.raises(ConfigurationError):
            config.validate()
        
        config = TrainingConfig(validation_split=0.8)  # Too high
        with pytest.raises(ConfigurationError):
            config.validate()


class TestExperimentConfig:
    """Test experiment configuration."""
    
    def test_experiment_config_valid(self, sample_config):
        """Test valid experiment configuration."""
        sample_config.validate()  # Should not raise
        assert sample_config.name == "test_experiment"
    
    def test_experiment_config_to_dict(self, sample_config):
        """Test conversion to dictionary."""
        config_dict = sample_config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert "name" in config_dict
        assert "model" in config_dict
        assert "training" in config_dict
    
    def test_experiment_config_invalid(self):
        """Test invalid experiment configuration."""
        config = ExperimentConfig(
            model=FNOConfig(input_dim=-1),  # Invalid model config
        )
        
        with pytest.raises(ConfigurationError):
            config.validate()


class TestConfigManager:
    """Test configuration manager."""
    
    def test_create_default_config(self):
        """Test creating default configurations."""
        manager = ConfigManager()
        
        # FNO config
        fno_config = manager.create_default_config("fno")
        assert isinstance(fno_config.model, FNOConfig)
        fno_config.validate()
        
        # DeepONet config
        deeponet_config = manager.create_default_config("deeponet")
        assert isinstance(deeponet_config.model, DeepONetConfig)
        deeponet_config.validate()
    
    def test_save_and_load_yaml_config(self, sample_config, temp_dir):
        """Test saving and loading YAML configuration."""
        manager = ConfigManager()
        config_path = temp_dir / "test_config.yaml"
        
        # Save configuration
        manager.save_config(sample_config, config_path, ConfigFormat.YAML)
        assert config_path.exists()
        
        # Load configuration
        loaded_config = manager.load_config(config_path, ConfigFormat.YAML)
        
        assert loaded_config.name == sample_config.name
        assert loaded_config.model.input_dim == sample_config.model.input_dim
        assert loaded_config.training.epochs == sample_config.training.epochs
    
    def test_save_and_load_json_config(self, sample_config, temp_dir):
        """Test saving and loading JSON configuration."""
        manager = ConfigManager()
        config_path = temp_dir / "test_config.json"
        
        # Save configuration
        manager.save_config(sample_config, config_path, ConfigFormat.JSON)
        assert config_path.exists()
        
        # Load configuration
        loaded_config = manager.load_config(config_path, ConfigFormat.JSON)
        
        assert loaded_config.name == sample_config.name
        assert loaded_config.model.input_dim == sample_config.model.input_dim
    
    def test_auto_detect_format(self, sample_config, temp_dir):
        """Test automatic format detection."""
        manager = ConfigManager()
        
        # YAML file
        yaml_path = temp_dir / "config.yaml"
        manager.save_config(sample_config, yaml_path, ConfigFormat.YAML)
        loaded_yaml = manager.load_config(yaml_path)  # Auto-detect
        assert loaded_yaml.name == sample_config.name
        
        # JSON file
        json_path = temp_dir / "config.json"
        manager.save_config(sample_config, json_path, ConfigFormat.JSON)
        loaded_json = manager.load_config(json_path)  # Auto-detect
        assert loaded_json.name == sample_config.name
    
    def test_config_caching(self, sample_config, temp_dir):
        """Test configuration caching."""
        manager = ConfigManager()
        config_path = temp_dir / "cached_config.yaml"
        
        # Save and load config
        manager.save_config(sample_config, config_path)
        loaded_config1 = manager.load_config(config_path)
        
        # Check if config is cached
        cached_config = manager.get_cached_config(config_path)
        assert cached_config is not None
        assert cached_config.name == sample_config.name
        
        # Clear cache
        manager.clear_cache()
        assert manager.get_cached_config(config_path) is None
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent configuration file."""
        manager = ConfigManager()
        
        with pytest.raises(ConfigurationError, match="not found"):
            manager.load_config("nonexistent_config.yaml")
    
    def test_unsupported_format(self, sample_config, temp_dir):
        """Test unsupported format handling."""
        manager = ConfigManager()
        config_path = temp_dir / "config.unsupported"
        
        with pytest.raises(ConfigurationError, match="Cannot auto-detect format"):
            manager.load_config(config_path)


class TestEnvironmentOverrides:
    """Test environment-specific configuration overrides."""
    
    def test_development_environment(self, sample_config):
        """Test development environment overrides."""
        manager = ConfigManager()
        manager.environment = "development"
        
        # Apply overrides
        overridden_config = manager._apply_environment_overrides(sample_config)
        
        # Check if development overrides are applied
        assert overridden_config.training.epochs == 10  # Development override
        assert overridden_config.output_dir == "./dev_outputs"
    
    def test_testing_environment(self, sample_config):
        """Test testing environment overrides."""
        manager = ConfigManager()
        manager.environment = "testing"
        
        overridden_config = manager._apply_environment_overrides(sample_config)
        
        assert overridden_config.training.epochs == 2  # Testing override
        assert overridden_config.training.batch_size == 4
    
    def test_production_environment(self, sample_config):
        """Test production environment overrides."""
        manager = ConfigManager()
        manager.environment = "production"
        
        overridden_config = manager._apply_environment_overrides(sample_config)
        
        assert overridden_config.training.save_checkpoints == True
        assert overridden_config.output_dir == "./prod_outputs"


class TestConfigCompatibility:
    """Test configuration compatibility validation."""
    
    def test_valid_compatibility(self):
        """Test valid configuration compatibility."""
        config = ExperimentConfig(
            model=FNOConfig(modes=16, batch_norm=False),
            training=TrainingConfig(batch_size=32),
            posterior=PosteriorConfig(method="laplace")
        )
        
        warnings = validate_config_compatibility(config)
        assert len(warnings) == 0
    
    def test_batch_norm_warning(self):
        """Test batch norm with small batch size warning."""
        config = ExperimentConfig(
            model=FNOConfig(batch_norm=True),
            training=TrainingConfig(batch_size=2)  # Small batch size
        )
        
        warnings = validate_config_compatibility(config)
        assert any("batch normalization" in warning.lower() for warning in warnings)
    
    def test_active_learning_warnings(self):
        """Test active learning compatibility warnings."""
        config = ExperimentConfig(
            active_learning=ActiveLearningConfig(
                budget=20,  # Smaller than initial_size
                initial_size=50
            )
        )
        
        warnings = validate_config_compatibility(config)
        assert any("budget smaller" in warning.lower() for warning in warnings)


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_global_load_config(self, sample_config, temp_dir):
        """Test global load_config function."""
        config_path = temp_dir / "global_test.yaml"
        
        # Save using manager
        manager = ConfigManager()
        manager.save_config(sample_config, config_path)
        
        # Load using global function
        loaded_config = load_config(config_path)
        assert loaded_config.name == sample_config.name
    
    def test_global_save_config(self, sample_config, temp_dir):
        """Test global save_config function."""
        config_path = temp_dir / "global_save_test.yaml"
        
        # Save using global function
        save_config(sample_config, config_path)
        assert config_path.exists()
        
        # Verify by loading
        loaded_config = load_config(config_path)
        assert loaded_config.name == sample_config.name
    
    def test_create_default_config_function(self):
        """Test create_default_config function."""
        config = create_default_config("fno")
        assert isinstance(config.model, FNOConfig)
        config.validate()


class TestComplexConfigurations:
    """Test complex configuration scenarios."""
    
    def test_full_experiment_config(self):
        """Test a full, complex experiment configuration."""
        config = ExperimentConfig(
            name="complex_experiment",
            description="A complex experiment with all components",
            tags=["fno", "active_learning", "2d"],
            model=FNOConfig(
                input_dim=2,
                output_dim=1,
                modes=32,
                width=128,
                depth=6,
                spatial_dim=2,
                dropout=0.1
            ),
            posterior=PosteriorConfig(
                method="variational",
                num_samples=50,
                kl_weight=0.05
            ),
            training=TrainingConfig(
                epochs=200,
                batch_size=16,
                early_stopping=True,
                patience=20,
                device="cuda"
            ),
            active_learning=ActiveLearningConfig(
                acquisition_function="bald",
                budget=1000,
                batch_size=20,
                initial_size=100
            ),
            dataset_type="darcy",
            seed=12345
        )
        
        # Should validate without issues
        config.validate()
        
        # Check all components are properly configured
        assert config.model.spatial_dim == 2
        assert config.posterior.method == "variational"
        assert config.active_learning.budget == 1000
    
    def test_config_serialization_roundtrip(self, temp_dir):
        """Test full serialization roundtrip."""
        # Create complex config
        original_config = ExperimentConfig(
            name="roundtrip_test",
            model=DeepONetConfig(
                branch_layers=[64, 64, 64],
                trunk_layers=[32, 32],
                basis_functions=50
            ),
            posterior=PosteriorConfig(method="ensemble", ensemble_size=7),
            training=TrainingConfig(epochs=50, batch_size=8),
            tags=["test", "roundtrip"]
        )
        
        manager = ConfigManager()
        
        # Save as YAML
        yaml_path = temp_dir / "roundtrip.yaml"
        manager.save_config(original_config, yaml_path, ConfigFormat.YAML)
        
        # Load back
        loaded_config = manager.load_config(yaml_path)
        
        # Verify all fields match
        assert loaded_config.name == original_config.name
        assert loaded_config.model.basis_functions == original_config.model.basis_functions
        assert loaded_config.posterior.ensemble_size == original_config.posterior.ensemble_size
        assert loaded_config.training.epochs == original_config.training.epochs
        assert loaded_config.tags == original_config.tags


@pytest.mark.property
def test_config_property_based(config_strategy):
    """Property-based tests for configuration validation."""
    try:
        from hypothesis import given
        
        @given(config_strategy)
        def test_generated_configs_validate(config):
            """Generated configs should always validate."""
            config.validate()  # Should not raise
        
        test_generated_configs_validate()
        
    except ImportError:
        pytest.skip("Hypothesis not available for property-based testing")