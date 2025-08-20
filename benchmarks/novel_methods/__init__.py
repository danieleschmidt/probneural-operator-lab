"""Benchmarking framework for novel uncertainty quantification methods.

This module provides comprehensive benchmarking and evaluation tools for the
novel uncertainty quantification methods implemented in the repository.

Modules:
- novel_benchmark_suite: Main benchmarking framework
- theoretical_validation: Theoretical validation and convergence tests  
- performance_comparison: Performance comparison tools
- calibration_metrics: Advanced calibration and reliability metrics
- visualization_tools: Visualization and plotting utilities

Authors: TERRAGON Research Lab
Date: 2025-01-22
"""

from .novel_benchmark_suite import (
    NovelMethodsBenchmarkSuite,
    NovelBenchmarkConfig,
    ComparisonResult,
    run_comprehensive_novel_benchmark
)

from .theoretical_validation import (
    TheoreticalValidator,
    ConvergenceTest,
    BayesianConsistencyTest,
    PhysicsConsistencyTest
)

from .performance_comparison import (
    PerformanceComparator,
    ScalabilityAnalyzer,
    ComputationalProfiler
)

__all__ = [
    "NovelMethodsBenchmarkSuite",
    "NovelBenchmarkConfig", 
    "ComparisonResult",
    "run_comprehensive_novel_benchmark",
    "TheoreticalValidator",
    "ConvergenceTest",
    "BayesianConsistencyTest", 
    "PhysicsConsistencyTest",
    "PerformanceComparator",
    "ScalabilityAnalyzer",
    "ComputationalProfiler"
]