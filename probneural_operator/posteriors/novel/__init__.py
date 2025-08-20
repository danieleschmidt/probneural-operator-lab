"""Novel uncertainty quantification methods for neural operators.

This module implements cutting-edge uncertainty quantification techniques that advance
beyond traditional Laplace approximation and variational inference:

1. Sparse Gaussian Process Neural Operator (SGPNO) - Scalable GP-based uncertainty
2. Normalizing Flow Posterior Approximation - Complex posterior geometries
3. Physics-Informed Conformal Prediction - PDE-aware uncertainty bounds
4. Meta-Learning Uncertainty Estimator (MLUE) - Rapid adaptation to new domains
5. Information-Theoretic Active Learning - Mutual information networks

Authors: TERRAGON Research Lab
Date: 2025-01-22
"""

from .sparse_gp_neural_operator import SparseGaussianProcessNeuralOperator
from .normalizing_flow_posterior import NormalizingFlowPosterior
from .physics_informed_conformal import PhysicsInformedConformalPredictor
from .meta_learning_uncertainty import MetaLearningUncertaintyEstimator
from .information_theoretic_active import InformationTheoreticActiveLearner

__all__ = [
    "SparseGaussianProcessNeuralOperator",
    "NormalizingFlowPosterior", 
    "PhysicsInformedConformalPredictor",
    "MetaLearningUncertaintyEstimator",
    "InformationTheoreticActiveLearner"
]