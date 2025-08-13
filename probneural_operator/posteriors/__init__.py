"""Posterior approximation methods for uncertainty quantification."""

from .laplace import LinearizedLaplace, HierarchicalLaplaceApproximation
from .base import PosteriorApproximation, get_posterior
from .adaptive_uncertainty import AdaptiveUncertaintyScaler, UncertaintyScalingNetwork, MultiModalAdaptiveScaler

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
    "PosteriorApproximation",
    "get_posterior",
    "LinearizedLaplace", 
    "HierarchicalLaplaceApproximation",
    "AdaptiveUncertaintyScaler",
    "UncertaintyScalingNetwork",
    "MultiModalAdaptiveScaler",
    "VariationalPosterior", 
    "DeepEnsemble"
]