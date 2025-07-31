"""Posterior approximation methods for uncertainty quantification."""

from .laplace import LinearizedLaplace
from .variational import VariationalPosterior
from .ensemble import DeepEnsemble

__all__ = ["LinearizedLaplace", "VariationalPosterior", "DeepEnsemble"]