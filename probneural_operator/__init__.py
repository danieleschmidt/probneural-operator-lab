"""
ProbNeural-Operator-Lab: Probabilistic Neural Operators with Active Learning.

Framework for probabilistic neural operators with linearized Laplace approximation 
and active learning capabilities. Implements ICML 2025's "Linearization → Probabilistic NO" 
approach for uncertainty-aware PDE solving with neural operators.

Generation 3 Enhancement: High-Performance Scaling & Production Deployment
- Intelligent caching and performance optimization
- Multi-GPU and distributed training support
- Auto-scaling and dynamic load balancing
- Advanced optimization algorithms (AdamW, Lion, LARS)
- Memory management with gradient checkpointing
- HPC integration (SLURM, MPI) 
- Production-ready model serving with REST API
- Comprehensive performance monitoring
"""

__version__ = "0.3.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@example.com"

# Core imports for easy access
from probneural_operator.models import (
    ProbabilisticFNO,
    ProbabilisticDeepONet,
    ProbabilisticGNO
)
from probneural_operator.posteriors import (
    LinearizedLaplace,
    VariationalPosterior,
    DeepEnsemble
)
from probneural_operator.active import ActiveLearner
from probneural_operator.calibration import TemperatureScaling

# Generation 3 Scaling imports
from probneural_operator.scaling import (
    # Performance Optimization & Caching
    PredictionCache,
    TensorOperationCache,
    AdaptiveBatchSizer,
    MemoryOptimizer,
    
    # Concurrent Processing & Resource Pooling
    MultiGPUTrainer,
    DistributedTrainer,
    ResourcePoolManager,
    AsyncDataLoader,
    
    # Auto-Scaling & Load Balancing
    AutoScaler,
    LoadBalancer,
    ResourceMonitor,
    ElasticBatchProcessor,
    
    # Advanced Optimization Algorithms
    AdvancedOptimizerFactory,
    LearningRateScheduler,
    GradientManager,
    HyperparameterOptimizer,
    
    # Memory Management & Resource Optimization
    GradientCheckpointer,
    MixedPrecisionManager,
    MemoryMappedDataset,
    MemoryPoolManager,
    
    # High-Performance Computing Integration
    SLURMIntegration,
    MPIDistributedTrainer,
    CheckpointManager,
    JobScheduler,
    
    # Production Deployment & Serving
    ModelServer,
    ModelVersionManager,
    InferenceOptimizer,
    ContainerManager,
)

__all__ = [
    # Core Generation 1 & 2
    "ProbabilisticFNO",
    "ProbabilisticDeepONet", 
    "ProbabilisticGNO",
    "LinearizedLaplace",
    "VariationalPosterior",
    "DeepEnsemble",
    "ActiveLearner",
    "TemperatureScaling",
    
    # Generation 3 Scaling Features
    # Performance Optimization & Caching
    "PredictionCache",
    "TensorOperationCache", 
    "AdaptiveBatchSizer",
    "MemoryOptimizer",
    
    # Concurrent Processing & Resource Pooling
    "MultiGPUTrainer",
    "DistributedTrainer",
    "ResourcePoolManager",
    "AsyncDataLoader",
    
    # Auto-Scaling & Load Balancing
    "AutoScaler",
    "LoadBalancer", 
    "ResourceMonitor",
    "ElasticBatchProcessor",
    
    # Advanced Optimization Algorithms
    "AdvancedOptimizerFactory",
    "LearningRateScheduler",
    "GradientManager",
    "HyperparameterOptimizer",
    
    # Memory Management & Resource Optimization
    "GradientCheckpointer",
    "MixedPrecisionManager",
    "MemoryMappedDataset",
    "MemoryPoolManager",
    
    # High-Performance Computing Integration
    "SLURMIntegration",
    "MPIDistributedTrainer",
    "CheckpointManager",
    "JobScheduler",
    
    # Production Deployment & Serving
    "ModelServer",
    "ModelVersionManager",
    "InferenceOptimizer",
    "ContainerManager",
]