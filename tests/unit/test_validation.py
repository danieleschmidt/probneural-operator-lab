"""Unit tests for validation utilities."""

import pytest
import torch
import numpy as np

from probneural_operator.utils.validation import (
    validate_tensor_shape, validate_tensor_dtype, validate_tensor_finite,
    validate_parameter_bounds, validate_integer_parameter, validate_float_parameter,
    check_numerical_stability, validate_training_data, safe_inversion,
    ValidationError, NumericalStabilityError, ParameterBoundsError,
    TensorValidationError, ValidationContext
)


class TestTensorValidation:
    """Test tensor validation functions."""
    
    def test_validate_tensor_shape_valid(self):
        """Test valid tensor shape validation."""
        tensor = torch.randn(10, 5)
        
        # Should not raise
        validate_tensor_shape(tensor, expected_shape=(10, 5))
        validate_tensor_shape(tensor, min_dims=2, max_dims=3)
        validate_tensor_shape(tensor, expected_shape=(10, None))  # Ignore second dimension
    
    def test_validate_tensor_shape_invalid(self):
        """Test invalid tensor shape validation."""
        tensor = torch.randn(10, 5)
        
        with pytest.raises(TensorValidationError, match="must have 3 dimensions"):
            validate_tensor_shape(tensor, expected_shape=(10, 5, 3))
        
        with pytest.raises(TensorValidationError, match="must have at least 3 dimensions"):
            validate_tensor_shape(tensor, min_dims=3)
        
        with pytest.raises(TensorValidationError, match="must have at most 1 dimensions"):
            validate_tensor_shape(tensor, max_dims=1)
    
    def test_validate_tensor_dtype_valid(self):
        """Test valid tensor dtype validation."""
        tensor_float = torch.randn(5, 5, dtype=torch.float32)
        tensor_int = torch.randint(0, 10, (5, 5), dtype=torch.int64)
        
        validate_tensor_dtype(tensor_float, torch.float32)
        validate_tensor_dtype(tensor_float, [torch.float32, torch.float64])
        validate_tensor_dtype(tensor_int, torch.int64)
    
    def test_validate_tensor_dtype_invalid(self):
        """Test invalid tensor dtype validation."""
        tensor = torch.randn(5, 5, dtype=torch.float32)
        
        with pytest.raises(TensorValidationError, match="must have dtype"):
            validate_tensor_dtype(tensor, torch.int64)
        
        with pytest.raises(TensorValidationError, match="must have dtype"):
            validate_tensor_dtype(tensor, [torch.int32, torch.int64])
    
    def test_validate_tensor_finite_valid(self):
        """Test valid finite tensor validation."""
        tensor = torch.randn(5, 5)
        
        # Should not raise
        validate_tensor_finite(tensor)
    
    def test_validate_tensor_finite_invalid(self, problematic_tensors):
        """Test invalid finite tensor validation."""
        with pytest.raises(NumericalStabilityError, match="contains.*NaN"):
            validate_tensor_finite(problematic_tensors["nan"])
        
        with pytest.raises(NumericalStabilityError, match="contains.*infinite"):
            validate_tensor_finite(problematic_tensors["inf"])


class TestParameterValidation:
    """Test parameter validation functions."""
    
    def test_validate_parameter_bounds_valid(self):
        """Test valid parameter bounds."""
        validate_parameter_bounds(5.0, "test_param", min_value=0.0, max_value=10.0)
        validate_parameter_bounds(0.0, "test_param", min_value=0.0)
        validate_parameter_bounds(10.0, "test_param", max_value=10.0)
    
    def test_validate_parameter_bounds_invalid(self):
        """Test invalid parameter bounds."""
        with pytest.raises(ParameterBoundsError, match="must be >= 0"):
            validate_parameter_bounds(-1.0, "test_param", min_value=0.0)
        
        with pytest.raises(ParameterBoundsError, match="must be <= 10"):
            validate_parameter_bounds(11.0, "test_param", max_value=10.0)
        
        with pytest.raises(ParameterBoundsError, match="must be > 0"):
            validate_parameter_bounds(0.0, "test_param", min_value=0.0, exclusive_min=True)
    
    def test_validate_integer_parameter(self):
        """Test integer parameter validation."""
        assert validate_integer_parameter(5, "test") == 5
        assert validate_integer_parameter(5.0, "test") == 5
        assert validate_integer_parameter(np.int32(5), "test") == 5
        
        with pytest.raises(ParameterBoundsError):
            validate_integer_parameter("5", "test")
        
        with pytest.raises(ParameterBoundsError):
            validate_integer_parameter(-1, "test", min_value=0)
    
    def test_validate_float_parameter(self):
        """Test float parameter validation."""
        assert validate_float_parameter(5.5, "test") == 5.5
        assert validate_float_parameter(5, "test") == 5.0
        assert validate_float_parameter(np.float64(5.5), "test") == 5.5
        
        with pytest.raises(ParameterBoundsError):
            validate_float_parameter("5.5", "test")
        
        with pytest.raises(ParameterBoundsError):
            validate_float_parameter(float('nan'), "test")
        
        with pytest.raises(ParameterBoundsError):
            validate_float_parameter(-1.0, "test", min_value=0.0, exclusive_min=True)


class TestNumericalStability:
    """Test numerical stability checks."""
    
    def test_check_numerical_stability_healthy(self):
        """Test numerical stability check on healthy matrices."""
        # Well-conditioned matrix
        A = torch.eye(5) + 0.1 * torch.randn(5, 5)
        
        diagnostics = check_numerical_stability({"matrix_A": A})
        
        assert "matrix_A" in diagnostics
        assert "condition_number" in diagnostics["matrix_A"]
        assert diagnostics["matrix_A"]["condition_number"] < 1e10
    
    def test_check_numerical_stability_ill_conditioned(self):
        """Test numerical stability check on ill-conditioned matrix."""
        # Create ill-conditioned matrix
        U, _, Vt = torch.svd(torch.randn(5, 5))
        # Very small singular values
        S = torch.tensor([1.0, 0.1, 0.01, 0.001, 1e-15])
        A = U @ torch.diag(S) @ Vt
        
        with pytest.raises(NumericalStabilityError):
            check_numerical_stability({"ill_conditioned": A})
    
    def test_safe_inversion_cholesky(self):
        """Test safe matrix inversion with Cholesky decomposition."""
        # Create positive definite matrix
        A = torch.randn(5, 10)
        psd_matrix = A.T @ A + 0.1 * torch.eye(10)
        
        inv_matrix = safe_inversion(psd_matrix, method="cholesky")
        
        # Check if inversion is approximately correct
        product = psd_matrix @ inv_matrix
        identity = torch.eye(10)
        assert torch.allclose(product, identity, atol=1e-4)
    
    def test_safe_inversion_lu(self):
        """Test safe matrix inversion with LU decomposition."""
        A = torch.randn(5, 5)
        A = A + torch.eye(5)  # Make it well-conditioned
        
        inv_matrix = safe_inversion(A, method="lu")
        
        product = A @ inv_matrix
        identity = torch.eye(5)
        assert torch.allclose(product, identity, atol=1e-4)
    
    def test_safe_inversion_svd(self):
        """Test safe matrix inversion with SVD."""
        A = torch.randn(5, 5)
        A = A + torch.eye(5)
        
        inv_matrix = safe_inversion(A, method="svd")
        
        product = A @ inv_matrix
        identity = torch.eye(5)
        assert torch.allclose(product, identity, atol=1e-4)


class TestTrainingDataValidation:
    """Test training data validation."""
    
    def test_validate_training_data_valid(self):
        """Test valid training data."""
        inputs = torch.randn(100, 10)
        targets = torch.randn(100, 1)
        
        # Should not raise
        validate_training_data(inputs, targets)
    
    def test_validate_training_data_mismatched_batch_size(self):
        """Test mismatched batch sizes."""
        inputs = torch.randn(100, 10)
        targets = torch.randn(50, 1)  # Different batch size
        
        with pytest.raises(ValidationError, match="batch size"):
            validate_training_data(inputs, targets)
    
    def test_validate_training_data_insufficient_samples(self):
        """Test insufficient number of samples."""
        inputs = torch.randn(2, 10)
        targets = torch.randn(2, 1)
        
        with pytest.raises(ValidationError, match="Need at least"):
            validate_training_data(inputs, targets, min_samples=5)
    
    def test_validate_training_data_problematic_values(self, problematic_tensors):
        """Test training data with problematic values."""
        normal = torch.randn(10, 5)
        
        with pytest.raises(NumericalStabilityError):
            validate_training_data(problematic_tensors["nan"], normal)
        
        with pytest.raises(NumericalStabilityError):
            validate_training_data(normal, problematic_tensors["inf"])


class TestValidationContext:
    """Test validation context manager."""
    
    def test_validation_context_strict_mode(self):
        """Test validation context in strict mode."""
        with pytest.raises(ValidationError):
            with ValidationContext(strict=True) as ctx:
                ctx.validate(True, "This should pass")
                ctx.validate(False, "This should fail")  # This should raise
    
    def test_validation_context_non_strict_mode(self):
        """Test validation context in non-strict mode."""
        with ValidationContext(strict=False) as ctx:
            ctx.validate(True, "This should pass")
            ctx.validate(False, "This should fail", warning=False)
            ctx.validate(False, "This is a warning", warning=True)
        
        summary = ctx.get_summary()
        assert len(summary["errors"]) == 1
        assert len(summary["warnings"]) == 1
    
    def test_validation_context_warnings(self, capture_warnings):
        """Test validation context warnings."""
        with ValidationContext(strict=False) as ctx:
            ctx.validate(False, "This is a warning", warning=True)
        
        assert len(capture_warnings) == 1
        assert "This is a warning" in str(capture_warnings[0].message)


@pytest.mark.property
def test_tensor_validation_properties(tensor_strategy):
    """Property-based tests for tensor validation."""
    try:
        from hypothesis import given
        
        @given(tensor_strategy)
        def test_finite_tensors_pass_validation(tensor):
            """Any finite tensor should pass finite validation."""
            if torch.all(torch.isfinite(tensor)):
                validate_tensor_finite(tensor)  # Should not raise
        
        test_finite_tensors_pass_validation()
        
    except ImportError:
        pytest.skip("Hypothesis not available for property-based testing")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_tensor_validation(self):
        """Test validation with empty tensors."""
        empty_tensor = torch.empty(0, 5)
        
        # Shape validation should work
        validate_tensor_shape(empty_tensor, min_dims=2)
        
        # Finite validation should pass for empty tensors
        validate_tensor_finite(empty_tensor)
    
    def test_scalar_tensor_validation(self):
        """Test validation with scalar tensors."""
        scalar = torch.tensor(5.0)
        
        validate_tensor_shape(scalar, expected_shape=())
        validate_tensor_finite(scalar)
    
    def test_very_large_matrices(self):
        """Test validation with very large matrices."""
        # This tests memory and performance considerations
        large_matrix = torch.randn(1000, 1000)
        
        # Should handle large matrices without issues
        validate_tensor_finite(large_matrix)
        validate_tensor_shape(large_matrix, min_dims=2)
    
    def test_boundary_parameter_values(self):
        """Test parameter validation at boundary values."""
        # Test exactly at boundaries
        validate_parameter_bounds(0.0, "test", min_value=0.0)
        validate_parameter_bounds(1.0, "test", max_value=1.0)
        
        # Test exclusive boundaries
        with pytest.raises(ParameterBoundsError):
            validate_parameter_bounds(0.0, "test", min_value=0.0, exclusive_min=True)
        
        with pytest.raises(ParameterBoundsError):
            validate_parameter_bounds(1.0, "test", max_value=1.0, exclusive_max=True)