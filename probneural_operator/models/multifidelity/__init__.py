"""Multi-fidelity neural operators for uncertainty quantification.

This module implements novel multi-fidelity neural operators that can learn from
data at multiple levels of fidelity (e.g., coarse vs. fine mesh simulations)
and propagate uncertainty across fidelity levels.

Key innovations:
- Hierarchical uncertainty propagation
- Cross-fidelity transfer learning
- Optimal fidelity selection for active learning
- Multi-scale Bayesian inference
"""

from .multifidelity_fno import MultiFidelityFNO
from .fidelity_transfer import FidelityTransferLayer
from .uncertainty_propagation import MultiFidelityUncertaintyPropagation

__all__ = [
    "MultiFidelityFNO",
    "FidelityTransferLayer", 
    "MultiFidelityUncertaintyPropagation"
]