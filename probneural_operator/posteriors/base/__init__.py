"""Base classes for posterior approximation."""

from .posterior import PosteriorApproximation
from .factory import get_posterior

__all__ = [
    "PosteriorApproximation",
    "get_posterior"
]