"""
ProbNeural-Operator-Lab: Probabilistic Neural Operators with Uncertainty Quantification.

Extends FNO/DeepONet with Bayesian uncertainty via linearized Laplace approximation.

Key classes (from probabilistic_operator module):
    ProbabilisticFNO1D      - FNO-1D + diagonal Laplace posterior
    LaplacePosterior        - diagonal GGN precision; predictive variance
    CalibrationEvaluator    - interval coverage at multiple confidence levels
    ActiveLearningSelector  - select most uncertain pool points for labeling
"""

__version__ = "1.0.0"
__author__ = "Daniel Schmidt"

# Primary public API
from probneural_operator.probabilistic_operator import (
    ProbabilisticFNO1D,
    LaplacePosterior,
    CalibrationEvaluator,
    ActiveLearningSelector,
    SpectralConv1d,
    FNO1DLayer,
)

__all__ = [
    "ProbabilisticFNO1D",
    "LaplacePosterior",
    "CalibrationEvaluator",
    "ActiveLearningSelector",
    "SpectralConv1d",
    "FNO1DLayer",
]
