"""Neural operator models with uncertainty quantification."""

from .fno import FourierNeuralOperator, ProbabilisticFNO
from .base import NeuralOperator, ProbabilisticNeuralOperator

# Placeholders for future implementations
class ProbabilisticDeepONet:
    """Placeholder for ProbabilisticDeepONet - to be implemented."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("ProbabilisticDeepONet not yet implemented")

class ProbabilisticGNO:
    """Placeholder for ProbabilisticGNO - to be implemented."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("ProbabilisticGNO not yet implemented")

__all__ = [
    "NeuralOperator",
    "ProbabilisticNeuralOperator",
    "FourierNeuralOperator", 
    "ProbabilisticFNO",
    "ProbabilisticDeepONet",
    "ProbabilisticGNO"
]