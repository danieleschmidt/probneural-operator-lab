"""Test basic imports and package structure."""

import pytest


def test_package_import():
    """Test that main package can be imported."""
    import probneural_operator
    assert probneural_operator.__version__ == "0.1.0"


def test_submodule_imports():
    """Test that submodules can be imported."""
    from probneural_operator import models
    from probneural_operator import posteriors
    from probneural_operator import active
    from probneural_operator import calibration
    
    # Verify modules are accessible
    assert hasattr(models, '__all__')
    assert hasattr(posteriors, '__all__')
    assert hasattr(active, '__all__')
    assert hasattr(calibration, '__all__')


def test_main_exports():
    """Test that main exports are available."""
    import probneural_operator
    
    # These will be implemented later, so we just check the names exist
    expected_exports = [
        "ProbabilisticFNO",
        "ProbabilisticDeepONet", 
        "ProbabilisticGNO",
        "LinearizedLaplace",
        "VariationalPosterior",
        "DeepEnsemble",
        "ActiveLearner",
        "TemperatureScaling"
    ]
    
    for export in expected_exports:
        assert export in probneural_operator.__all__