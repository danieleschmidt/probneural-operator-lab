#!/usr/bin/env python3
"""
Production-ready model server for ProbNeural-Operator-Lab.

This script demonstrates how to deploy neural operator models in a production environment
with high-performance serving, auto-scaling, and comprehensive monitoring.
"""

import os
import sys
import asyncio
import argparse
import logging
import signal
import time
from pathlib import Path
from typing import Optional

# Add probneural_operator to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from probneural_operator.models.fno import FourierNeuralOperator
from probneural_operator.scaling import (
    ModelServer, ModelVersionManager, InferenceOptimizer, ModelMetadata,
    ResourceMonitor, AutoScaler, LoadBalancer, ResourcePoolManager
)
from probneural_operator.scaling.autoscale import ScalingPolicy


class ProductionModelServer:
    """Production-ready model server with scaling and monitoring."""
    
    def __init__(self, 
                 model_storage_path: str = "./production_models",
                 host: str = "0.0.0.0",
                 port: int = 8000,
                 enable_auth: bool = False,
                 auth_token: Optional[str] = None,
                 enable_monitoring: bool = True,
                 enable_autoscaling: bool = False):
        """Initialize production server.
        
        Args:
            model_storage_path: Path to store model versions
            host: Server host
            port: Server port
            enable_auth: Enable authentication
            auth_token: Authentication token
            enable_monitoring: Enable resource monitoring
            enable_autoscaling: Enable auto-scaling
        """
        self.model_storage_path = Path(model_storage_path)
        self.host = host
        self.port = port
        self.enable_auth = enable_auth
        self.auth_token = auth_token
        self.enable_monitoring = enable_monitoring
        self.enable_autoscaling = enable_autoscaling
        
        # Initialize components
        self.version_manager = ModelVersionManager(str(self.model_storage_path))
        self.model_server = None
        self.resource_monitor = None
        self.autoscaler = None
        self.load_balancer = None
        
        # Setup logging
        self._setup_logging()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.running = False
    
    def _setup_logging(self):
        """Setup production logging."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(logs_dir / "production_server.log"),
                logging.FileHandler(logs_dir / "errors.log")
            ]
        )
        
        # Set specific log levels
        logging.getLogger("uvicorn").setLevel(logging.WARNING)
        logging.getLogger("fastapi").setLevel(logging.WARNING)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logging.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
    
    def load_demo_models(self):
        """Load demonstration models for testing."""
        logging.info("Loading demonstration models...")
        
        # Create and register demo FNO model
        fno_model = FourierNeuralOperator(
            n_modes_x=16, n_modes_y=16,
            hidden_channels=64,
            in_channels=1, out_channels=1
        )
        
        fno_metadata = ModelMetadata(
            model_id="fno_production",
            name="Production FNO Model",
            version="1.0",
            description="Production-ready Fourier Neural Operator for PDE solving",
            input_shape=(1, 64, 64),
            output_shape=(1, 64, 64),
            model_type="FourierNeuralOperator",
            created_time=time.time(),
            updated_time=time.time(),
            performance_metrics={
                "avg_inference_time_ms": 15.0,
                "throughput_samples_per_sec": 200.0,
                "memory_usage_mb": 256.0
            },
            tags=["production", "fno", "pde-solver"]
        )
        
        success = self.version_manager.register_model(
            fno_model, "fno_production", "1.0", fno_metadata, save_to_disk=True
        )
        
        if success:
            logging.info("Successfully registered FNO production model")
        else:
            logging.error("Failed to register FNO production model")
        
        # Create and register a smaller fast model
        fast_model = FourierNeuralOperator(
            n_modes_x=8, n_modes_y=8,
            hidden_channels=32,
            in_channels=1, out_channels=1
        )
        
        fast_metadata = ModelMetadata(
            model_id="fno_fast",
            name="Fast FNO Model",
            version="1.0", 
            description="Fast Fourier Neural Operator for low-latency inference",
            input_shape=(1, 64, 64),
            output_shape=(1, 64, 64),
            model_type="FourierNeuralOperator",
            created_time=time.time(),
            updated_time=time.time(),
            performance_metrics={
                "avg_inference_time_ms": 8.0,
                "throughput_samples_per_sec": 400.0,
                "memory_usage_mb": 128.0
            },
            tags=["production", "fno", "fast", "low-latency"]
        )
        
        success = self.version_manager.register_model(
            fast_model, "fno_fast", "1.0", fast_metadata, save_to_disk=True
        )
        
        if success:
            logging.info("Successfully registered fast FNO model")
        else:
            logging.error("Failed to register fast FNO model")
        
        # Optimize models for inference
        logging.info("Optimizing models for production inference...")
        optimizer = InferenceOptimizer()
        
        for model_id in ["fno_production", "fno_fast"]:
            model = self.version_manager.load_model(model_id, "1.0")
            if model is not None:
                # Create example input for optimization
                example_input = torch.randn(1, 1, 64, 64)
                
                # Optimize model
                optimized_model = optimizer.optimize_model(
                    model, example_input, optimization_level="aggressive"
                )
                
                # Re-register optimized model as new version
                metadata = self.version_manager.get_model_info(model_id)
                if metadata:
                    optimized_metadata = ModelMetadata(**metadata)
                    optimized_metadata.version = "1.1"
                    optimized_metadata.description += " (Optimized)"
                    optimized_metadata.updated_time = time.time()
                    
                    success = self.version_manager.register_model(
                        optimized_model, model_id, "1.1", optimized_metadata, save_to_disk=True
                    )
                    
                    if success:
                        # Set optimized version as default
                        self.version_manager.set_default_version(model_id, "1.1")
                        logging.info(f"Registered optimized version of {model_id}")
        
        # List all loaded models
        models = self.version_manager.list_models()
        logging.info(f"Loaded {len(models)} model versions:")
        for model_info in models:
            logging.info(f"  - {model_info['model_id']} v{model_info['version']}: {model_info['name']}")
    
    def setup_monitoring(self):
        """Setup resource monitoring and auto-scaling."""
        if not self.enable_monitoring:
            return
        
        logging.info("Setting up resource monitoring...")
        
        # Initialize resource monitor
        self.resource_monitor = ResourceMonitor(
            monitoring_interval=10.0,  # Monitor every 10 seconds
            history_size=200  # Keep 200 samples (33 minutes of history)
        )
        
        # Setup monitoring callbacks
        def log_high_resource_usage(metrics):
            """Log when resource usage is high."""
            if metrics.cpu_percent > 80:
                logging.warning(f"High CPU usage: {metrics.cpu_percent:.1f}%")
            
            if metrics.memory_percent > 85:
                logging.warning(f"High memory usage: {metrics.memory_percent:.1f}%")
            
            if metrics.gpu_memory_percent:
                for gpu_id, usage in metrics.gpu_memory_percent.items():
                    if usage > 90:
                        logging.warning(f"High GPU {gpu_id} memory usage: {usage:.1f}%")
        
        self.resource_monitor.add_callback(log_high_resource_usage)
        
        # Start monitoring
        self.resource_monitor.start_monitoring()
        
        # Setup auto-scaling if enabled
        if self.enable_autoscaling:
            logging.info("Setting up auto-scaling...")
            
            # Configure scaling policy
            policy = ScalingPolicy(
                cpu_scale_up_threshold=70.0,
                cpu_scale_down_threshold=30.0,
                memory_scale_up_threshold=80.0,
                memory_scale_down_threshold=40.0,
                latency_scale_up_threshold_ms=500.0,
                error_rate_scale_up_threshold=0.05,
                min_instances=1,
                max_instances=8,
                scale_up_count=2,
                scale_down_count=1,
                cooldown_period_seconds=60.0
            )
            
            # Initialize auto-scaler
            self.autoscaler = AutoScaler(policy, self.resource_monitor)
            self.autoscaler.start()
            
            # Setup load balancer
            self.load_balancer = LoadBalancer(
                balancing_strategy="performance",  # Route to best performing instances
                health_check_interval=30.0
            )
            
            # Register initial instance (this server)
            self.load_balancer.register_instance(
                "primary_server",
                f"http://{self.host}:{self.port}",
                weight=1.0
            )
            
            self.load_balancer.start_health_checks()
    
    def start(self):
        """Start the production server."""
        logging.info("Starting ProbNeural-Operator Production Server...")
        
        # Load models
        self.load_demo_models()
        
        # Setup monitoring
        self.setup_monitoring()
        
        # Create model server
        self.model_server = ModelServer(
            version_manager=self.version_manager,
            host=self.host,
            port=self.port,
            workers=1,  # Single worker for demo
            enable_auth=self.enable_auth,
            auth_token=self.auth_token
        )
        
        # Add custom routes for monitoring
        self._add_monitoring_routes()
        
        self.running = True
        
        logging.info(f"Server starting on http://{self.host}:{self.port}")
        logging.info("Available endpoints:")
        logging.info("  - POST /predict - Make predictions")
        logging.info("  - GET /health - Health check")
        logging.info("  - GET /models - List models")
        logging.info("  - GET /stats - Server statistics")
        logging.info("  - GET /monitoring/metrics - Resource metrics")
        logging.info("  - GET /monitoring/scaling - Scaling status")
        
        try:
            # Run server
            self.model_server.run(debug=False)
        except KeyboardInterrupt:
            logging.info("Received keyboard interrupt")
            self.shutdown()
        except Exception as e:
            logging.error(f"Server error: {e}")
            self.shutdown()
            raise
    
    def _add_monitoring_routes(self):
        """Add monitoring and management routes."""
        app = self.model_server.app
        
        @app.get("/monitoring/metrics")
        async def get_metrics():
            """Get current resource metrics."""
            if not self.resource_monitor:
                return {"error": "Monitoring not enabled"}
            
            current_metrics = self.resource_monitor.get_current_metrics()
            if current_metrics:
                return {
                    "timestamp": current_metrics.timestamp,
                    "cpu_percent": current_metrics.cpu_percent,
                    "memory_percent": current_metrics.memory_percent,
                    "gpu_memory_percent": current_metrics.gpu_memory_percent,
                    "active_requests": current_metrics.active_requests,
                    "request_latency_ms": current_metrics.request_latency_ms,
                    "throughput_rps": current_metrics.throughput_rps,
                    "error_rate": current_metrics.error_rate
                }
            else:
                return {"error": "No metrics available"}
        
        @app.get("/monitoring/scaling")
        async def get_scaling_status():
            """Get auto-scaling status."""
            if not self.autoscaler:
                return {"error": "Auto-scaling not enabled"}
            
            return {
                "current_instances": self.autoscaler.get_current_instances(),
                "scaling_history": self.autoscaler.get_scaling_history()[-10:],  # Last 10 actions
                "policy": {
                    "min_instances": self.autoscaler.policy.min_instances,
                    "max_instances": self.autoscaler.policy.max_instances,
                    "cpu_scale_up_threshold": self.autoscaler.policy.cpu_scale_up_threshold,
                    "cpu_scale_down_threshold": self.autoscaler.policy.cpu_scale_down_threshold
                }
            }
        
        @app.get("/monitoring/load_balancer")
        async def get_load_balancer_status():
            """Get load balancer status."""
            if not self.load_balancer:
                return {"error": "Load balancer not enabled"}
            
            return {
                "instances": self.load_balancer.get_instance_stats(),
                "strategy": self.load_balancer.balancing_strategy
            }
        
        @app.post("/admin/scale")
        async def manual_scale(target_instances: int):
            """Manually scale instances."""
            if not self.autoscaler:
                return {"error": "Auto-scaling not enabled"}
            
            success = self.autoscaler.force_scaling(target_instances)
            return {"success": success, "target_instances": target_instances}
        
        @app.get("/admin/optimize_model/{model_id}")
        async def optimize_model(model_id: str):
            """Optimize a model for better performance."""
            model = self.version_manager.load_model(model_id)
            if model is None:
                return {"error": "Model not found"}
            
            try:
                optimizer = InferenceOptimizer()
                example_input = torch.randn(1, 1, 64, 64)
                
                # Benchmark original model
                original_benchmark = optimizer.benchmark_model(model, example_input, num_iterations=20)
                
                # Optimize model
                optimized_model = optimizer.optimize_model(model, example_input, "aggressive")
                
                # Benchmark optimized model
                optimized_benchmark = optimizer.benchmark_model(optimized_model, example_input, num_iterations=20)
                
                # Calculate improvement
                speedup = original_benchmark['avg_inference_time_ms'] / optimized_benchmark['avg_inference_time_ms']
                
                return {
                    "model_id": model_id,
                    "optimization_successful": True,
                    "speedup": f"{speedup:.2f}x",
                    "original_latency_ms": original_benchmark['avg_inference_time_ms'],
                    "optimized_latency_ms": optimized_benchmark['avg_inference_time_ms'],
                    "original_throughput": original_benchmark['throughput_samples_per_sec'],
                    "optimized_throughput": optimized_benchmark['throughput_samples_per_sec']
                }
            
            except Exception as e:
                return {"error": f"Optimization failed: {str(e)}"}
    
    def shutdown(self):
        """Gracefully shutdown the server."""
        if not self.running:
            return
        
        logging.info("Shutting down production server...")
        
        self.running = False
        
        # Stop auto-scaling
        if self.autoscaler:
            self.autoscaler.stop()
            logging.info("Auto-scaler stopped")
        
        # Stop load balancer
        if self.load_balancer:
            self.load_balancer.stop_health_checks()
            logging.info("Load balancer stopped")
        
        # Stop resource monitoring
        if self.resource_monitor:
            self.resource_monitor.stop_monitoring()
            logging.info("Resource monitoring stopped")
        
        logging.info("Production server shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ProbNeural-Operator Production Server")
    
    # Server configuration
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--model-storage", type=str, default="./production_models", 
                       help="Model storage path")
    
    # Security
    parser.add_argument("--enable-auth", action="store_true", help="Enable authentication")
    parser.add_argument("--auth-token", type=str, help="Authentication token")
    
    # Monitoring and scaling
    parser.add_argument("--enable-monitoring", action="store_true", default=True,
                       help="Enable resource monitoring")
    parser.add_argument("--enable-autoscaling", action="store_true", 
                       help="Enable auto-scaling")
    
    # Development mode
    parser.add_argument("--dev", action="store_true", help="Development mode")
    
    args = parser.parse_args()
    
    # Create and start server
    server = ProductionModelServer(
        model_storage_path=args.model_storage,
        host=args.host,
        port=args.port,
        enable_auth=args.enable_auth,
        auth_token=args.auth_token,
        enable_monitoring=args.enable_monitoring,
        enable_autoscaling=args.enable_autoscaling
    )
    
    try:
        server.start()
    except KeyboardInterrupt:
        logging.info("Server interrupted by user")
    except Exception as e:
        logging.error(f"Server failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())