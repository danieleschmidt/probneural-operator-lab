"""Posterior approximation methods for uncertainty quantification.

This module provides a comprehensive suite of uncertainty quantification methods
for neural operators, including both established approaches and cutting-edge
novel methods developed by TERRAGON Research Lab.

Established Methods:
- LinearizedLaplace: Linearized Laplace approximation
- HierarchicalLaplaceApproximation: Multi-level Laplace approximation
- AdaptiveUncertaintyScaler: Adaptive uncertainty scaling

Novel Methods (Generation 4 - Research Breakthroughs):
- SparseGaussianProcessNeuralOperator: Scalable GP-based uncertainty
- NormalizingFlowPosterior: Complex posterior geometry modeling
- PhysicsInformedConformalPredictor: PDE-aware uncertainty bounds
- MetaLearningUncertaintyEstimator: Rapid domain adaptation
- InformationTheoreticActiveLearner: Optimal data selection

Authors: TERRAGON Research Lab
Date: 2025-01-22
"""

from .laplace import LinearizedLaplace, HierarchicalLaplaceApproximation
from .base import PosteriorApproximation, get_posterior
from .adaptive_uncertainty import AdaptiveUncertaintyScaler, UncertaintyScalingNetwork, MultiModalAdaptiveScaler

# Import novel methods
from .novel import (
    SparseGaussianProcessNeuralOperator,
    NormalizingFlowPosterior,
    PhysicsInformedConformalPredictor,
    MetaLearningUncertaintyEstimator,
    InformationTheoreticActiveLearner
)

# Placeholders for future implementations
class VariationalPosterior:
    """Placeholder for VariationalPosterior - to be implemented."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("VariationalPosterior not yet implemented")

class DeepEnsemble:
    """Placeholder for DeepEnsemble - to be implemented."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("DeepEnsemble not yet implemented")

__all__ = [
    # Base classes
    "PosteriorApproximation",
    "get_posterior",
    
    # Established methods
    "LinearizedLaplace", 
    "HierarchicalLaplaceApproximation",
    "AdaptiveUncertaintyScaler",
    "UncertaintyScalingNetwork",
    "MultiModalAdaptiveScaler",
    
    # Novel methods (Generation 4)
    "SparseGaussianProcessNeuralOperator",
    "NormalizingFlowPosterior",
    "PhysicsInformedConformalPredictor", 
    "MetaLearningUncertaintyEstimator",
    "InformationTheoreticActiveLearner",
    
    # Placeholders
    "VariationalPosterior", 
    "DeepEnsemble"
]