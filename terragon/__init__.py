"""
Terragon Autonomous SDLC System
==============================

Progressive quality gates and autonomous development lifecycle management.
Implements the 3-generation enhancement strategy with adaptive intelligence.
"""

from .quality_gates import (
    QualityGateFramework,
    QualityGateResult,
    Generation1Gates,
    Generation2Gates, 
    Generation3Gates,
    ResearchQualityGates,
    ContinuousQualityMonitor,
    AdaptiveQualityController,
    run_quality_gates
)

from .integration import (
    TeragonSDLCIntegration,
    setup_terragon_integration,
    generate_integration_summary
)

__version__ = "1.0.0"

__all__ = [
    # Core quality gates
    "QualityGateFramework",
    "QualityGateResult",
    "run_quality_gates",
    
    # Generation gates
    "Generation1Gates",
    "Generation2Gates", 
    "Generation3Gates",
    "ResearchQualityGates",
    
    # Advanced features
    "ContinuousQualityMonitor",
    "AdaptiveQualityController",
    
    # Integration
    "TeragonSDLCIntegration",
    "setup_terragon_integration",
    "generate_integration_summary",
]