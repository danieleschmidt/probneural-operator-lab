"""Production deployment and serving utilities for neural operators."""

import torch
import torch.nn as nn
from torch.jit import script, trace
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import time
import asyncio
import threading
import json
import logging
import hashlib
import uuid
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import docker
import yaml
import os
import shutil
from pathlib import Path
import subprocess
from collections import deque, defaultdict
import aiohttp
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import pickle
import weakref


@dataclass
class ModelMetadata:
    """Metadata for deployed models."""
    model_id: str
    name: str
    version: str
    description: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    model_type: str
    created_time: float
    updated_time: float
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# Pydantic models for API
class PredictionRequest(BaseModel):
    """Request model for predictions."""
    input_data: List[List[float]] = Field(..., description="Input data as nested list")
    model_id: Optional[str] = Field(None, description="Specific model ID to use")
    version: Optional[str] = Field(None, description="Model version to use")
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional options")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: List[List[float]] = Field(..., description="Model prediction")
    model_id: str = Field(..., description="Model ID used")
    version: str = Field(..., description="Model version used")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    uncertainty: Optional[List[List[float]]] = Field(None, description="Prediction uncertainty")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ModelInfo(BaseModel):
    """Model information response."""
    model_id: str
    name: str
    version: str
    description: str
    input_shape: List[int]
    output_shape: List[int]
    model_type: str
    created_time: float
    updated_time: float
    performance_metrics: Dict[str, float]
    tags: List[str]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: float
    version: str
    models_loaded: int
    memory_usage_mb: float
    gpu_memory_mb: Dict[int, float]


class InferenceOptimizer:
    """Optimize models for inference performance."""
    
    def __init__(self):
        """Initialize inference optimizer."""
        self._optimized_models = {}
        self._optimization_stats = {}
    
    def optimize_model(self, 
                      model: nn.Module,
                      example_input: torch.Tensor,
                      optimization_level: str = "standard") -> nn.Module:
        """Optimize model for inference.
        
        Args:
            model: Model to optimize
            example_input: Example input for tracing
            optimization_level: Optimization level ("basic", "standard", "aggressive")
            
        Returns:
            Optimized model
        """
        model_id = str(id(model))
        start_time = time.time()
        
        optimized_model = model
        optimizations_applied = []
        
        try:
            # Set to evaluation mode
            optimized_model.eval()
            optimizations_applied.append("eval_mode")
            
            if optimization_level in ["standard", "aggressive"]:
                # Try TorchScript tracing
                try:
                    with torch.no_grad():
                        traced_model = trace(optimized_model, example_input)
                        optimized_model = traced_model
                        optimizations_applied.append("torchscript_trace")
                except Exception as e:
                    logging.warning(f"TorchScript tracing failed: {e}")
                    
                    # Fallback to scripting
                    try:
                        scripted_model = script(optimized_model)
                        optimized_model = scripted_model
                        optimizations_applied.append("torchscript_script")
                    except Exception as e:
                        logging.warning(f"TorchScript scripting failed: {e}")
            
            if optimization_level == "aggressive":
                # Additional aggressive optimizations
                try:
                    # Fuse operations
                    if hasattr(torch.jit, 'optimize_for_inference'):
                        optimized_model = torch.jit.optimize_for_inference(optimized_model)
                        optimizations_applied.append("jit_optimize_inference")
                except Exception as e:
                    logging.warning(f"JIT optimization failed: {e}")
                
                # Quantization (if supported)
                try:
                    if hasattr(torch.quantization, 'quantize_dynamic'):
                        optimized_model = torch.quantization.quantize_dynamic(
                            optimized_model,
                            {nn.Linear, nn.Conv1d, nn.Conv2d},
                            dtype=torch.qint8
                        )
                        optimizations_applied.append("dynamic_quantization")
                except Exception as e:
                    logging.warning(f"Quantization failed: {e}")
            
            # Warm up model
            with torch.no_grad():
                for _ in range(5):
                    _ = optimized_model(example_input)
            optimizations_applied.append("warmup")
            
            optimization_time = time.time() - start_time
            
            self._optimization_stats[model_id] = {
                'optimizations_applied': optimizations_applied,
                'optimization_time': optimization_time,
                'optimization_level': optimization_level
            }
            
            logging.info(f"Model optimized in {optimization_time:.3f}s: {', '.join(optimizations_applied)}")
            
            return optimized_model
            
        except Exception as e:
            logging.error(f"Model optimization failed: {e}")
            return model
    
    def benchmark_model(self, 
                       model: nn.Module, 
                       example_input: torch.Tensor,
                       num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark model performance.
        
        Args:
            model: Model to benchmark
            example_input: Example input
            num_iterations: Number of iterations
            
        Returns:
            Performance metrics
        """
        model.eval()
        device = next(model.parameters()).device
        example_input = example_input.to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(example_input)
        
        # Synchronize
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.perf_counter()
                
                _ = model(example_input)
                
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
        
        # Calculate statistics
        times = np.array(times) * 1000  # Convert to ms
        
        return {
            'avg_inference_time_ms': float(np.mean(times)),
            'std_inference_time_ms': float(np.std(times)),
            'min_inference_time_ms': float(np.min(times)),
            'max_inference_time_ms': float(np.max(times)),
            'p50_inference_time_ms': float(np.percentile(times, 50)),
            'p95_inference_time_ms': float(np.percentile(times, 95)),
            'p99_inference_time_ms': float(np.percentile(times, 99)),
            'throughput_samples_per_sec': 1000.0 / np.mean(times),
            'batch_size': example_input.shape[0]
        }


class ModelVersionManager:
    """Manage multiple versions of deployed models."""
    
    def __init__(self, storage_path: str = "./model_versions"):
        """Initialize model version manager.
        
        Args:
            storage_path: Path to store model versions
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self._models = {}  # {model_id: {version: model_info}}
        self._loaded_models = {}  # {(model_id, version): model}
        self._model_metadata = {}  # {model_id: ModelMetadata}
        self._default_versions = {}  # {model_id: version}
        
        # Performance tracking
        self._request_stats = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'errors': 0
        })
        
        # Load existing models
        self._load_existing_models()
    
    def _load_existing_models(self):
        """Load existing models from storage."""
        if not self.storage_path.exists():
            return
        
        for model_dir in self.storage_path.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_id = model_dir.name
            metadata_file = model_dir / "metadata.json"
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    self._model_metadata[model_id] = ModelMetadata(**metadata)
                    
                    # Load version information
                    versions = {}
                    for version_dir in model_dir.iterdir():
                        if version_dir.is_dir() and version_dir.name.startswith('v'):
                            version = version_dir.name
                            model_file = version_dir / "model.pth"
                            
                            if model_file.exists():
                                versions[version] = {
                                    'path': str(model_file),
                                    'created_time': model_file.stat().st_mtime
                                }
                    
                    if versions:
                        self._models[model_id] = versions
                        # Set default to latest version
                        latest_version = max(versions.keys(), key=lambda v: versions[v]['created_time'])
                        self._default_versions[model_id] = latest_version
                    
                except Exception as e:
                    logging.error(f"Failed to load model metadata for {model_id}: {e}")
    
    def register_model(self,
                      model: nn.Module,
                      model_id: str,
                      version: str,
                      metadata: ModelMetadata,
                      save_to_disk: bool = True) -> bool:
        """Register a new model version.
        
        Args:
            model: PyTorch model
            model_id: Unique model identifier
            version: Model version
            metadata: Model metadata
            save_to_disk: Whether to save model to disk
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create storage directories
            model_dir = self.storage_path / model_id
            version_dir = model_dir / f"v{version}"
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            if save_to_disk:
                model_file = version_dir / "model.pth"
                torch.save(model.state_dict(), model_file)
                
                # Save model architecture if possible
                try:
                    script_model = torch.jit.script(model)
                    script_file = version_dir / "model_script.pt"
                    script_model.save(str(script_file))
                except Exception as e:
                    logging.warning(f"Could not save scripted model: {e}")
            
            # Update in-memory structures
            if model_id not in self._models:
                self._models[model_id] = {}
            
            self._models[model_id][version] = {
                'path': str(version_dir / "model.pth"),
                'created_time': time.time()
            }
            
            self._model_metadata[model_id] = metadata
            
            # Set as default if first version or explicitly requested
            if model_id not in self._default_versions:
                self._default_versions[model_id] = version
            
            # Load into memory
            self._loaded_models[(model_id, version)] = model
            
            # Save metadata
            metadata_file = model_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            logging.info(f"Registered model {model_id} version {version}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to register model {model_id} version {version}: {e}")
            return False
    
    def load_model(self, 
                  model_id: str, 
                  version: Optional[str] = None,
                  model_class: Optional[type] = None) -> Optional[nn.Module]:
        """Load a specific model version.
        
        Args:
            model_id: Model identifier
            version: Model version (default version if None)
            model_class: Model class for instantiation
            
        Returns:
            Loaded model or None if not found
        """
        if version is None:
            version = self._default_versions.get(model_id)
        
        if not version or model_id not in self._models:
            return None
        
        # Check if already loaded
        model_key = (model_id, version)
        if model_key in self._loaded_models:
            return self._loaded_models[model_key]
        
        # Load from disk
        if version not in self._models[model_id]:
            return None
        
        model_path = self._models[model_id][version]['path']
        
        try:
            # Try to load scripted model first
            script_path = str(Path(model_path).parent / "model_script.pt")
            if os.path.exists(script_path):
                model = torch.jit.load(script_path)
                self._loaded_models[model_key] = model
                return model
            
            # Fallback to state dict loading
            if model_class is not None:
                model = model_class()
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
                model.eval()
                
                self._loaded_models[model_key] = model
                return model
            
            # Last resort: try to load raw state dict
            state_dict = torch.load(model_path, map_location='cpu')
            logging.warning(f"Loaded raw state dict for {model_id} v{version}")
            return state_dict
            
        except Exception as e:
            logging.error(f"Failed to load model {model_id} version {version}: {e}")
            return None
    
    def set_default_version(self, model_id: str, version: str) -> bool:
        """Set default version for a model.
        
        Args:
            model_id: Model identifier
            version: Version to set as default
            
        Returns:
            True if successful, False otherwise
        """
        if model_id in self._models and version in self._models[model_id]:
            self._default_versions[model_id] = version
            logging.info(f"Set default version for {model_id} to {version}")
            return True
        return False
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model information.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model information dictionary
        """
        if model_id not in self._model_metadata:
            return None
        
        metadata = self._model_metadata[model_id]
        versions = list(self._models.get(model_id, {}).keys())
        default_version = self._default_versions.get(model_id)
        
        stats = self._request_stats[model_id]
        avg_response_time = stats['total_time'] / max(1, stats['count'])
        error_rate = stats['errors'] / max(1, stats['count'])
        
        return {
            **metadata.to_dict(),
            'versions': versions,
            'default_version': default_version,
            'request_count': stats['count'],
            'avg_response_time_ms': avg_response_time * 1000,
            'error_rate': error_rate
        }
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models.
        
        Returns:
            List of model information
        """
        models = []
        for model_id in self._model_metadata:
            info = self.get_model_info(model_id)
            if info:
                models.append(info)
        
        return models
    
    def unload_model(self, model_id: str, version: Optional[str] = None):
        """Unload model from memory.
        
        Args:
            model_id: Model identifier
            version: Model version (all versions if None)
        """
        if version is None:
            # Unload all versions
            keys_to_remove = [key for key in self._loaded_models.keys() if key[0] == model_id]
            for key in keys_to_remove:
                del self._loaded_models[key]
        else:
            model_key = (model_id, version)
            if model_key in self._loaded_models:
                del self._loaded_models[model_key]
        
        logging.info(f"Unloaded model {model_id}" + (f" version {version}" if version else " (all versions)"))
    
    def delete_model(self, model_id: str, version: Optional[str] = None) -> bool:
        """Delete model from disk and memory.
        
        Args:
            model_id: Model identifier
            version: Model version (all versions if None)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if version is None:
                # Delete entire model
                model_dir = self.storage_path / model_id
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                
                # Remove from memory structures
                self._models.pop(model_id, None)
                self._model_metadata.pop(model_id, None)
                self._default_versions.pop(model_id, None)
                
                # Unload from memory
                self.unload_model(model_id)
                
                logging.info(f"Deleted model {model_id}")
            else:
                # Delete specific version
                version_dir = self.storage_path / model_id / f"v{version}"
                if version_dir.exists():
                    shutil.rmtree(version_dir)
                
                # Remove from memory structures
                if model_id in self._models:
                    self._models[model_id].pop(version, None)
                    
                    # If this was the default version, set a new default
                    if self._default_versions.get(model_id) == version:
                        remaining_versions = list(self._models[model_id].keys())
                        if remaining_versions:
                            latest = max(remaining_versions, key=lambda v: self._models[model_id][v]['created_time'])
                            self._default_versions[model_id] = latest
                        else:
                            self._default_versions.pop(model_id, None)
                
                # Unload from memory
                self.unload_model(model_id, version)
                
                logging.info(f"Deleted model {model_id} version {version}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to delete model {model_id}" + (f" version {version}" if version else "") + f": {e}")
            return False
    
    def record_request(self, model_id: str, response_time: float, success: bool = True):
        """Record request statistics.
        
        Args:
            model_id: Model identifier
            response_time: Response time in seconds
            success: Whether request was successful
        """
        stats = self._request_stats[model_id]
        stats['count'] += 1
        stats['total_time'] += response_time
        
        if not success:
            stats['errors'] += 1
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics.
        
        Returns:
            Memory usage information
        """
        total_models = sum(len(versions) for versions in self._models.values())
        loaded_models = len(self._loaded_models)
        
        # Estimate memory usage
        total_memory_mb = 0
        for model in self._loaded_models.values():
            if hasattr(model, 'parameters'):
                model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
                total_memory_mb += model_size
        
        return {
            'total_models': total_models,
            'loaded_models': loaded_models,
            'estimated_memory_mb': total_memory_mb,
            'models_in_memory': list(self._loaded_models.keys())
        }


class ModelServer:
    """High-performance REST API server for neural operator inference."""
    
    def __init__(self,
                 version_manager: ModelVersionManager,
                 host: str = "0.0.0.0",
                 port: int = 8000,
                 workers: int = 1,
                 enable_auth: bool = False,
                 auth_token: Optional[str] = None):
        """Initialize model server.
        
        Args:
            version_manager: Model version manager
            host: Server host
            port: Server port
            workers: Number of worker processes
            enable_auth: Whether to enable authentication
            auth_token: Authentication token
        """
        self.version_manager = version_manager
        self.host = host
        self.port = port
        self.workers = workers
        self.enable_auth = enable_auth
        self.auth_token = auth_token or "default-token"
        
        # Create FastAPI app
        self.app = FastAPI(
            title="ProbNeural-Operator Model Server",
            description="High-performance serving for probabilistic neural operators",
            version="1.0.0"
        )
        
        # Add middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Setup routes
        self._setup_routes()
        
        # Performance tracking
        self._request_count = 0
        self._error_count = 0
        self._response_times = deque(maxlen=1000)
        
        # Inference optimizer
        self.optimizer = InferenceOptimizer()
    
    def _setup_routes(self):
        """Setup API routes."""
        
        # Authentication dependency
        security = HTTPBearer() if self.enable_auth else None
        
        def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
            if self.enable_auth:
                if not credentials or credentials.credentials != self.auth_token:
                    raise HTTPException(status_code=401, detail="Invalid authentication token")
        
        auth_dependency = [Depends(verify_token)] if self.enable_auth else []
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            memory_usage = 0.0
            gpu_memory = {}
            
            try:
                import psutil
                process = psutil.Process()
                memory_usage = process.memory_info().rss / 1024 / 1024  # MB
                
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        gpu_memory[i] = torch.cuda.memory_allocated(i) / 1024 / 1024
            except:
                pass
            
            return HealthResponse(
                status="healthy",
                timestamp=time.time(),
                version="1.0.0",
                models_loaded=len(self.version_manager._loaded_models),
                memory_usage_mb=memory_usage,
                gpu_memory_mb=gpu_memory
            )
        
        @self.app.post("/predict", response_model=PredictionResponse, dependencies=auth_dependency)
        async def predict(request: PredictionRequest):
            """Make prediction with model."""
            start_time = time.time()
            
            try:
                self._request_count += 1
                
                # Get model
                model_id = request.model_id
                version = request.version
                
                if not model_id:
                    # Use first available model
                    models = self.version_manager.list_models()
                    if not models:
                        raise HTTPException(status_code=404, detail="No models available")
                    model_id = models[0]['model_id']
                
                model = self.version_manager.load_model(model_id, version)
                if model is None:
                    raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
                
                # Convert input data
                input_tensor = torch.tensor(request.input_data, dtype=torch.float32)
                
                # Move to GPU if available
                device = next(model.parameters()).device
                input_tensor = input_tensor.to(device)
                
                # Make prediction
                model.eval()
                with torch.no_grad():
                    if hasattr(model, 'predict_with_uncertainty'):
                        prediction, uncertainty = model.predict_with_uncertainty(input_tensor)
                        uncertainty = uncertainty.cpu().numpy().tolist()
                    else:
                        prediction = model(input_tensor)
                        uncertainty = None
                
                prediction = prediction.cpu().numpy().tolist()
                
                # Calculate response time
                inference_time = (time.time() - start_time) * 1000
                self._response_times.append(inference_time)
                
                # Record statistics
                used_version = version or self.version_manager._default_versions.get(model_id, "unknown")
                self.version_manager.record_request(model_id, inference_time / 1000, True)
                
                return PredictionResponse(
                    prediction=prediction,
                    model_id=model_id,
                    version=used_version,
                    inference_time_ms=inference_time,
                    uncertainty=uncertainty,
                    metadata=request.options
                )
                
            except HTTPException:
                raise
            except Exception as e:
                self._error_count += 1
                inference_time = (time.time() - start_time) * 1000
                
                if 'model_id' in locals():
                    self.version_manager.record_request(model_id, inference_time / 1000, False)
                
                logging.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/models", response_model=List[ModelInfo])
        async def list_models():
            """List available models."""
            models = self.version_manager.list_models()
            return [ModelInfo(**model) for model in models]
        
        @self.app.get("/models/{model_id}", response_model=ModelInfo)
        async def get_model_info(model_id: str):
            """Get information about a specific model."""
            info = self.version_manager.get_model_info(model_id)
            if info is None:
                raise HTTPException(status_code=404, detail="Model not found")
            return ModelInfo(**info)
        
        @self.app.delete("/models/{model_id}", dependencies=auth_dependency)
        async def delete_model(model_id: str, version: Optional[str] = None):
            """Delete a model or model version."""
            success = self.version_manager.delete_model(model_id, version)
            if not success:
                raise HTTPException(status_code=404, detail="Model not found")
            return {"message": f"Model {model_id}" + (f" version {version}" if version else "") + " deleted"}
        
        @self.app.get("/stats")
        async def get_stats():
            """Get server statistics."""
            avg_response_time = np.mean(list(self._response_times)) if self._response_times else 0
            error_rate = self._error_count / max(1, self._request_count)
            
            return {
                'total_requests': self._request_count,
                'error_count': self._error_count,
                'error_rate': error_rate,
                'avg_response_time_ms': avg_response_time,
                'memory_usage': self.version_manager.get_memory_usage()
            }
    
    def run(self, debug: bool = False):
        """Run the server.
        
        Args:
            debug: Whether to run in debug mode
        """
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            workers=self.workers if not debug else 1,
            log_level="info" if not debug else "debug",
            access_log=True
        )


class ContainerManager:
    """Manage containerized deployments of model servers."""
    
    def __init__(self, docker_registry: Optional[str] = None):
        """Initialize container manager.
        
        Args:
            docker_registry: Docker registry URL
        """
        self.docker_registry = docker_registry
        
        # Check Docker availability
        try:
            self.docker_client = docker.from_env()
            self.docker_available = True
        except Exception as e:
            logging.warning(f"Docker not available: {e}")
            self.docker_client = None
            self.docker_available = False
        
        self._running_containers = {}
    
    def build_image(self,
                   dockerfile_path: str,
                   image_name: str,
                   build_context: str = ".") -> bool:
        """Build Docker image for model server.
        
        Args:
            dockerfile_path: Path to Dockerfile
            image_name: Name for the image
            build_context: Build context path
            
        Returns:
            True if successful, False otherwise
        """
        if not self.docker_available:
            logging.error("Docker not available")
            return False
        
        try:
            logging.info(f"Building Docker image {image_name}")
            
            image, logs = self.docker_client.images.build(
                path=build_context,
                dockerfile=dockerfile_path,
                tag=image_name,
                rm=True
            )
            
            # Log build output
            for log in logs:
                if 'stream' in log:
                    logging.info(log['stream'].strip())
            
            logging.info(f"Successfully built image {image_name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to build image {image_name}: {e}")
            return False
    
    def create_dockerfile(self,
                         output_path: str,
                         base_image: str = "pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime",
                         requirements_file: Optional[str] = None,
                         model_files: Optional[List[str]] = None,
                         server_script: str = "server.py") -> str:
        """Create Dockerfile for model server.
        
        Args:
            output_path: Path to save Dockerfile
            base_image: Base Docker image
            requirements_file: Requirements file path
            model_files: List of model files to copy
            server_script: Server script name
            
        Returns:
            Path to created Dockerfile
        """
        dockerfile_content = f"""# Auto-generated Dockerfile for ProbNeural-Operator Model Server
FROM {base_image}

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
"""
        
        if requirements_file:
            dockerfile_content += f"""COPY {requirements_file} .
RUN pip install --no-cache-dir -r {os.path.basename(requirements_file)}
"""
        
        dockerfile_content += """
# Install probneural-operator
COPY . .
RUN pip install -e .

"""
        
        if model_files:
            dockerfile_content += "# Copy model files\n"
            for model_file in model_files:
                dockerfile_content += f"COPY {model_file} ./models/\n"
            dockerfile_content += "\n"
        
        dockerfile_content += f"""# Copy server script
COPY {server_script} .

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \\
  CMD curl -f http://localhost:8000/health || exit 1

# Run server
CMD ["python", "{server_script}", "--host", "0.0.0.0", "--port", "8000"]
"""
        
        with open(output_path, 'w') as f:
            f.write(dockerfile_content)
        
        logging.info(f"Created Dockerfile: {output_path}")
        return output_path
    
    def run_container(self,
                     image_name: str,
                     container_name: str,
                     port: int = 8000,
                     gpu_support: bool = True,
                     environment: Optional[Dict[str, str]] = None,
                     volumes: Optional[Dict[str, Dict[str, str]]] = None) -> Optional[str]:
        """Run model server container.
        
        Args:
            image_name: Docker image name
            container_name: Container name
            port: Port to expose
            gpu_support: Whether to enable GPU support
            environment: Environment variables
            volumes: Volume mounts
            
        Returns:
            Container ID if successful, None otherwise
        """
        if not self.docker_available:
            logging.error("Docker not available")
            return None
        
        try:
            # Container configuration
            config = {
                'image': image_name,
                'name': container_name,
                'ports': {8000: port},
                'detach': True,
                'environment': environment or {},
                'volumes': volumes or {}
            }
            
            # Add GPU support if requested and available
            if gpu_support:
                try:
                    # Check for nvidia-docker support
                    config['runtime'] = 'nvidia'
                    config['environment']['NVIDIA_VISIBLE_DEVICES'] = 'all'
                except:
                    logging.warning("GPU support not available")
            
            # Run container
            container = self.docker_client.containers.run(**config)
            container_id = container.id
            
            self._running_containers[container_name] = {
                'id': container_id,
                'image': image_name,
                'port': port,
                'created_time': time.time()
            }
            
            logging.info(f"Started container {container_name} (ID: {container_id[:12]})")
            return container_id
            
        except Exception as e:
            logging.error(f"Failed to run container {container_name}: {e}")
            return None
    
    def stop_container(self, container_name: str) -> bool:
        """Stop running container.
        
        Args:
            container_name: Container name to stop
            
        Returns:
            True if successful, False otherwise
        """
        if not self.docker_available:
            return False
        
        try:
            container = self.docker_client.containers.get(container_name)
            container.stop()
            container.remove()
            
            if container_name in self._running_containers:
                del self._running_containers[container_name]
            
            logging.info(f"Stopped container {container_name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to stop container {container_name}: {e}")
            return False
    
    def get_container_logs(self, container_name: str, tail: int = 100) -> Optional[str]:
        """Get container logs.
        
        Args:
            container_name: Container name
            tail: Number of lines to return
            
        Returns:
            Container logs or None if error
        """
        if not self.docker_available:
            return None
        
        try:
            container = self.docker_client.containers.get(container_name)
            logs = container.logs(tail=tail, decode=True)
            return logs
            
        except Exception as e:
            logging.error(f"Failed to get logs for {container_name}: {e}")
            return None
    
    def list_containers(self) -> List[Dict[str, Any]]:
        """List running containers.
        
        Returns:
            List of container information
        """
        containers = []
        
        for name, info in self._running_containers.items():
            try:
                container = self.docker_client.containers.get(name)
                status = container.status
                
                containers.append({
                    'name': name,
                    'id': info['id'][:12],
                    'image': info['image'],
                    'port': info['port'],
                    'status': status,
                    'created_time': info['created_time']
                })
            except:
                # Container might have been removed externally
                containers.append({
                    'name': name,
                    'id': info['id'][:12],
                    'image': info['image'],
                    'port': info['port'],
                    'status': 'unknown',
                    'created_time': info['created_time']
                })
        
        return containers
    
    def create_deployment_config(self,
                                image_name: str,
                                replicas: int = 3,
                                output_path: str = "deployment.yaml") -> str:
        """Create Kubernetes deployment configuration.
        
        Args:
            image_name: Docker image name
            replicas: Number of replicas
            output_path: Output file path
            
        Returns:
            Path to created deployment file
        """
        deployment_config = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': 'probneural-operator-server',
                'labels': {'app': 'probneural-operator-server'}
            },
            'spec': {
                'replicas': replicas,
                'selector': {
                    'matchLabels': {'app': 'probneural-operator-server'}
                },
                'template': {
                    'metadata': {
                        'labels': {'app': 'probneural-operator-server'}
                    },
                    'spec': {
                        'containers': [{
                            'name': 'model-server',
                            'image': image_name,
                            'ports': [{'containerPort': 8000}],
                            'resources': {
                                'limits': {
                                    'memory': '4Gi',
                                    'cpu': '2',
                                    'nvidia.com/gpu': '1'
                                },
                                'requests': {
                                    'memory': '2Gi',
                                    'cpu': '1'
                                }
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': 8000
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5
                            }
                        }]
                    }
                }
            }
        }
        
        service_config = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': 'probneural-operator-service'
            },
            'spec': {
                'selector': {'app': 'probneural-operator-server'},
                'ports': [{
                    'protocol': 'TCP',
                    'port': 80,
                    'targetPort': 8000
                }],
                'type': 'LoadBalancer'
            }
        }
        
        # Write YAML files
        with open(output_path, 'w') as f:
            yaml.dump(deployment_config, f, default_flow_style=False)
            f.write("---\n")
            yaml.dump(service_config, f, default_flow_style=False)
        
        logging.info(f"Created Kubernetes deployment config: {output_path}")
        return output_path