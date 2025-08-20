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

# Core imports for easy access with graceful degradation
import warnings

try:
    from probneural_operator.models import (
        ProbabilisticFNO,
        ProbabilisticDeepONet,
        ProbabilisticGNO
    )
except ImportError as e:
    warnings.warn(f"Models not available: {e}", ImportWarning)
    ProbabilisticFNO = ProbabilisticDeepONet = ProbabilisticGNO = None

try:
    from probneural_operator.posteriors import (
        LinearizedLaplace,
        VariationalPosterior,
        DeepEnsemble
    )
except ImportError as e:
    warnings.warn(f"Posteriors not available: {e}", ImportWarning)
    LinearizedLaplace = VariationalPosterior = DeepEnsemble = None

try:
    from probneural_operator.active import ActiveLearner
except ImportError as e:
    warnings.warn(f"Active learning not available: {e}", ImportWarning)
    ActiveLearner = None

try:
    from probneural_operator.calibration import TemperatureScaling
except ImportError as e:
    warnings.warn(f"Calibration not available: {e}", ImportWarning)
    TemperatureScaling = None

# Generation 3 Scaling imports
try:
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
except ImportError as e:
    warnings.warn(f"Scaling modules not available: {e}", ImportWarning)
    # Set all scaling modules to None
    PredictionCache = TensorOperationCache = AdaptiveBatchSizer = MemoryOptimizer = None
    MultiGPUTrainer = DistributedTrainer = ResourcePoolManager = AsyncDataLoader = None
    AutoScaler = LoadBalancer = ResourceMonitor = ElasticBatchProcessor = None
    AdvancedOptimizerFactory = LearningRateScheduler = GradientManager = HyperparameterOptimizer = None
    GradientCheckpointer = MixedPrecisionManager = MemoryMappedDataset = MemoryPoolManager = None
    SLURMIntegration = MPIDistributedTrainer = CheckpointManager = JobScheduler = None
    ModelServer = ModelVersionManager = InferenceOptimizer = ContainerManager = None

# Utility functions
def get_version():
    """Get package version."""
    return __version__

def check_dependencies():
    """Check which dependencies are available."""
    deps = {}
    
    try:
        import torch
        deps['torch'] = torch.__version__
    except ImportError:
        deps['torch'] = None
    
    try:
        import numpy
        deps['numpy'] = numpy.__version__
    except ImportError:
        deps['numpy'] = None
        
    try:
        import scipy
        deps['scipy'] = scipy.__version__
    except ImportError:
        deps['scipy'] = None
    
    return deps

# Build dynamic __all__ based on what's available
__all__ = ["get_version", "check_dependencies"]

# Add available modules to __all__
if ProbabilisticFNO is not None:
    __all__.extend(["ProbabilisticFNO", "ProbabilisticDeepONet", "ProbabilisticGNO"])

if LinearizedLaplace is not None:
    __all__.extend(["LinearizedLaplace", "VariationalPosterior", "DeepEnsemble"])

if ActiveLearner is not None:
    __all__.append("ActiveLearner")

if TemperatureScaling is not None:
    __all__.append("TemperatureScaling")

# Add scaling modules if available
scaling_modules = [
    "PredictionCache", "TensorOperationCache", "AdaptiveBatchSizer", "MemoryOptimizer",
    "MultiGPUTrainer", "DistributedTrainer", "ResourcePoolManager", "AsyncDataLoader",
    "AutoScaler", "LoadBalancer", "ResourceMonitor", "ElasticBatchProcessor",
    "AdvancedOptimizerFactory", "LearningRateScheduler", "GradientManager", "HyperparameterOptimizer",
    "GradientCheckpointer", "MixedPrecisionManager", "MemoryMappedDataset", "MemoryPoolManager",
    "SLURMIntegration", "MPIDistributedTrainer", "CheckpointManager", "JobScheduler",
    "ModelServer", "ModelVersionManager", "InferenceOptimizer", "ContainerManager"
]

for module in scaling_modules:
    if globals().get(module) is not None:
        __all__.append(module)