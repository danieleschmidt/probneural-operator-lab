"""
ProbNeural-Operator-Lab: Probabilistic Neural Operators with Active Learning.

Framework for probabilistic neural operators with linearized Laplace approximation 
and active learning capabilities. Implements ICML 2025's "Linearization → Probabilistic NO" 
approach for uncertainty-aware PDE solving with neural operators.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"

# Core imports for easy access
from probneural_operator.models import (
    ProbabilisticFNO,
    ProbabilisticDeepONet,
    ProbabilisticGNO
)
from probneural_operator.posteriors import (
    LinearizedLaplace,
    VariationalPosterior,
    DeepEnsemble
)
from probneural_operator.active import ActiveLearner
from probneural_operator.calibration import TemperatureScaling

__all__ = [
    "ProbabilisticFNO",
    "ProbabilisticDeepONet", 
    "ProbabilisticGNO",
    "LinearizedLaplace",
    "VariationalPosterior",
    "DeepEnsemble",
    "ActiveLearner",
    "TemperatureScaling"
]