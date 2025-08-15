"""
Terragon Progressive Quality Gates System
=========================================

Autonomous SDLC progressive quality gate implementation for continuous 
value delivery with adaptive intelligence and automatic quality enforcement.

This module implements the 3-generation progressive enhancement strategy:
- Generation 1: MAKE IT WORK (Simple)  
- Generation 2: MAKE IT ROBUST (Reliable)
- Generation 3: MAKE IT SCALE (Optimized)

Each generation has mandatory quality gates that must pass before progression.
"""

from .core import QualityGateFramework, QualityGateResult, run_quality_gates
from .generations import Generation1Gates, Generation2Gates, Generation3Gates
from .research import ResearchQualityGates
from .monitoring import ContinuousQualityMonitor
from .adaptive import AdaptiveQualityController

__all__ = [
    "QualityGateFramework",
    "QualityGateResult", 
    "run_quality_gates",
    "Generation1Gates",
    "Generation2Gates",
    "Generation3Gates",
    "ResearchQualityGates",
    "ContinuousQualityMonitor",
    "AdaptiveQualityController",
]

__version__ = "1.0.0"