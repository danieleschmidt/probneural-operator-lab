"""Factory for creating posterior approximations."""

from typing import Any, Dict

import torch.nn as nn

from .posterior import PosteriorApproximation


def get_posterior(posterior_type: str, 
                 model: nn.Module,
                 **kwargs) -> PosteriorApproximation:
    """Factory function to create posterior approximations.
    
    Args:
        posterior_type: Type of posterior ("laplace", "variational", "ensemble")
        model: Neural network model
        **kwargs: Additional arguments for posterior
        
    Returns:
        Configured posterior approximation
    """
    if posterior_type == "laplace":
        from ..laplace import LinearizedLaplace
        return LinearizedLaplace(model, **kwargs)
    elif posterior_type == "variational":
        from ..variational import VariationalPosterior
        return VariationalPosterior(model, **kwargs)
    elif posterior_type == "ensemble":
        from ..ensemble import DeepEnsemble
        return DeepEnsemble(model, **kwargs)
    else:
        raise ValueError(f"Unknown posterior type: {posterior_type}")


def register_posterior(name: str, posterior_class: type) -> None:
    """Register a new posterior approximation method.
    
    Args:
        name: Name of the posterior method
        posterior_class: Class implementing PosteriorApproximation
    """
    # Store in a registry for dynamic loading
    _POSTERIOR_REGISTRY[name] = posterior_class


# Global registry for posterior methods
_POSTERIOR_REGISTRY = {}