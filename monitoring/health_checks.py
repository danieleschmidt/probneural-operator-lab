"""Health check system for ProbNeural Operator Lab."""

import time
import threading
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import psutil

from probneural_operator.utils.logging import get_logger


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    duration: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "duration": self.duration
        }


class HealthCheck:
    """Base class for health checks."""
    
    def __init__(self, name: str, interval: float = 60.0, timeout: float = 30.0):
        self.name = name
        self.interval = interval
        self.timeout = timeout
        self.logger = get_logger(f"health.{name}")
    
    def check(self) -> HealthCheckResult:
        """Perform the health check."""
        start_time = time.time()
        
        try:
            status, message, details = self._check_impl()
            duration = time.time() - start_time
            
            result = HealthCheckResult(
                name=self.name,
                status=status,
                message=message,
                details=details,
                timestamp=datetime.utcnow(),
                duration=duration
            )
            
            self.logger.debug(f"Health check completed: {status.value} - {message}")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            error_details = {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
            
            result = HealthCheckResult(
                name=self.name,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                details=error_details,
                timestamp=datetime.utcnow(),
                duration=duration
            )
            
            self.logger.error(f"Health check failed: {str(e)}")
            return result
    
    def _check_impl(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        """Implementation of the health check logic."""
        raise NotImplementedError


class SystemResourcesHealthCheck(HealthCheck):
    """Check system resource usage."""
    
    def __init__(self, cpu_threshold: float = 90.0, memory_threshold: float = 90.0, 
                 disk_threshold: float = 95.0, **kwargs):
        super().__init__("system_resources", **kwargs)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.disk_threshold = disk_threshold
    
    def _check_impl(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        details = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": memory.used / 1024**3,
            "memory_available_gb": memory.available / 1024**3,
            "disk_percent": disk.percent,
            "disk_used_gb": disk.used / 1024**3,
            "disk_free_gb": disk.free / 1024**3
        }
        
        issues = []
        
        if cpu_percent > self.cpu_threshold:
            issues.append(f"High CPU usage: {cpu_percent:.1f}%")
        
        if memory.percent > self.memory_threshold:
            issues.append(f"High memory usage: {memory.percent:.1f}%")
        
        if disk.percent > self.disk_threshold:
            issues.append(f"High disk usage: {disk.percent:.1f}%")
        
        if issues:
            status = HealthStatus.CRITICAL if any("High" in issue for issue in issues) else HealthStatus.WARNING
            message = "; ".join(issues)
        else:
            status = HealthStatus.HEALTHY
            message = "System resources within normal limits"
        
        return status, message, details


class GPUHealthCheck(HealthCheck):
    """Check GPU availability and health."""
    
    def __init__(self, memory_threshold: float = 90.0, **kwargs):
        super().__init__("gpu_health", **kwargs)
        self.memory_threshold = memory_threshold
    
    def _check_impl(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        if not torch.cuda.is_available():
            return HealthStatus.WARNING, "CUDA not available", {"cuda_available": False}
        
        details = {
            "cuda_available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "devices": []
        }
        
        issues = []
        
        for i in range(torch.cuda.device_count()):
            device = torch.device(f"cuda:{i}")
            device_name = torch.cuda.get_device_name(i)
            
            # Memory information
            allocated = torch.cuda.memory_allocated(device)
            reserved = torch.cuda.memory_reserved(device)
            
            # Get total memory (requires a CUDA operation)
            torch.cuda.empty_cache()  # Clear cache to get accurate reading
            total_memory = torch.cuda.get_device_properties(device).total_memory
            
            memory_percent = (allocated / total_memory) * 100
            
            device_info = {
                "index": i,
                "name": device_name,
                "memory_allocated_gb": allocated / 1024**3,
                "memory_reserved_gb": reserved / 1024**3,
                "memory_total_gb": total_memory / 1024**3,
                "memory_percent": memory_percent
            }
            
            details["devices"].append(device_info)
            
            if memory_percent > self.memory_threshold:
                issues.append(f"GPU {i} high memory usage: {memory_percent:.1f}%")
        
        if issues:
            status = HealthStatus.WARNING
            message = "; ".join(issues)
        else:
            status = HealthStatus.HEALTHY
            message = f"All {torch.cuda.device_count()} GPU(s) healthy"
        
        return status, message, details


class ModelHealthCheck(HealthCheck):
    """Check model loading and inference capability."""
    
    def __init__(self, model_factory: Callable[[], torch.nn.Module], 
                 test_input_shape: Tuple[int, ...], **kwargs):
        super().__init__("model_health", **kwargs)
        self.model_factory = model_factory
        self.test_input_shape = test_input_shape
    
    def _check_impl(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Create model
            model = self.model_factory()
            model = model.to(device)
            model.eval()
            
            # Test inference
            test_input = torch.randn(*self.test_input_shape, device=device)
            
            with torch.no_grad():
                start_time = time.time()
                output = model(test_input)
                inference_time = time.time() - start_time
            
            # Validate output
            if torch.isnan(output).any():
                return HealthStatus.CRITICAL, "Model output contains NaN values", {
                    "inference_time": inference_time,
                    "output_shape": list(output.shape),
                    "has_nan": True
                }
            
            if torch.isinf(output).any():
                return HealthStatus.WARNING, "Model output contains infinite values", {
                    "inference_time": inference_time,
                    "output_shape": list(output.shape),
                    "has_inf": True
                }
            
            details = {
                "inference_time": inference_time,
                "input_shape": list(test_input.shape),
                "output_shape": list(output.shape),
                "device": str(device),
                "model_parameters": sum(p.numel() for p in model.parameters()),
                "output_mean": output.mean().item(),
                "output_std": output.std().item()
            }
            
            return HealthStatus.HEALTHY, "Model inference successful", details
            
        except Exception as e:
            return HealthStatus.CRITICAL, f"Model health check failed: {str(e)}", {
                "error": str(e),
                "device": str(device)
            }


class DataHealthCheck(HealthCheck):
    """Check data loading and preprocessing."""
    
    def __init__(self, data_loader_factory: Callable[[], Any], **kwargs):
        super().__init__("data_health", **kwargs)
        self.data_loader_factory = data_loader_factory
    
    def _check_impl(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        try:
            data_loader = self.data_loader_factory()
            
            # Test loading a few batches
            batch_count = 0
            total_samples = 0
            load_times = []
            
            for i, batch in enumerate(data_loader):
                if i >= 3:  # Only test first 3 batches
                    break
                
                start_time = time.time()
                
                # Basic validation
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    x, y = batch[0], batch[1]
                    
                    # Check for NaN/Inf
                    if torch.isnan(x).any() or torch.isnan(y).any():
                        return HealthStatus.CRITICAL, "Data contains NaN values", {
                            "batch_index": i,
                            "x_has_nan": torch.isnan(x).any().item(),
                            "y_has_nan": torch.isnan(y).any().item()
                        }
                    
                    total_samples += x.shape[0]
                
                load_time = time.time() - start_time
                load_times.append(load_time)
                batch_count += 1
            
            if batch_count == 0:
                return HealthStatus.CRITICAL, "No data batches loaded", {}
            
            avg_load_time = sum(load_times) / len(load_times)
            
            details = {
                "batches_tested": batch_count,
                "total_samples": total_samples,
                "avg_load_time": avg_load_time,
                "max_load_time": max(load_times),
                "min_load_time": min(load_times)
            }
            
            # Check for slow loading
            if avg_load_time > 5.0:  # 5 seconds per batch is quite slow
                return HealthStatus.WARNING, f"Slow data loading: {avg_load_time:.2f}s per batch", details
            
            return HealthStatus.HEALTHY, f"Data loading healthy: {batch_count} batches tested", details
            
        except Exception as e:
            return HealthStatus.CRITICAL, f"Data health check failed: {str(e)}", {
                "error": str(e)
            }


class StorageHealthCheck(HealthCheck):
    """Check storage and file system health."""
    
    def __init__(self, required_paths: List[Path], min_free_space_gb: float = 1.0, **kwargs):
        super().__init__("storage_health", **kwargs)
        self.required_paths = [Path(p) for p in required_paths]
        self.min_free_space_gb = min_free_space_gb
    
    def _check_impl(self) -> Tuple[HealthStatus, str, Dict[str, Any]]:
        issues = []
        details = {"paths": [], "disk_usage": {}}
        
        # Check required paths
        for path in self.required_paths:
            path_info = {
                "path": str(path),
                "exists": path.exists(),
                "is_dir": path.is_dir() if path.exists() else None,
                "is_file": path.is_file() if path.exists() else None,
                "size": path.stat().st_size if path.exists() and path.is_file() else None
            }
            
            if not path.exists():
                issues.append(f"Required path does not exist: {path}")
                path_info["status"] = "missing"
            elif not (path.is_dir() or path.is_file()):
                issues.append(f"Path is neither file nor directory: {path}")
                path_info["status"] = "invalid"
            else:
                path_info["status"] = "ok"
            
            details["paths"].append(path_info)
        
        # Check disk space
        try:
            disk_usage = psutil.disk_usage('/')
            free_space_gb = disk_usage.free / 1024**3
            
            details["disk_usage"] = {
                "total_gb": disk_usage.total / 1024**3,
                "used_gb": disk_usage.used / 1024**3,
                "free_gb": free_space_gb,
                "percent_used": (disk_usage.used / disk_usage.total) * 100
            }
            
            if free_space_gb < self.min_free_space_gb:
                issues.append(f"Low disk space: {free_space_gb:.2f}GB free")
        
        except Exception as e:
            issues.append(f"Failed to check disk usage: {str(e)}")
        
        if issues:
            status = HealthStatus.CRITICAL if any("does not exist" in issue or "Low disk space" in issue for issue in issues) else HealthStatus.WARNING
            message = "; ".join(issues)
        else:
            status = HealthStatus.HEALTHY
            message = "Storage health check passed"
        
        return status, message, details


class HealthMonitor:
    """Central health monitoring system."""
    
    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.results: Dict[str, HealthCheckResult] = {}
        self.logger = get_logger("health_monitor")
        self._monitoring = False
        self._monitor_thread = None
        self._stop_event = threading.Event()
    
    def add_check(self, health_check: HealthCheck):
        """Add a health check to the monitor."""
        self.checks[health_check.name] = health_check
        self.logger.info(f"Added health check: {health_check.name}")
    
    def remove_check(self, name: str):
        """Remove a health check from the monitor."""
        if name in self.checks:
            del self.checks[name]
            if name in self.results:
                del self.results[name]
            self.logger.info(f"Removed health check: {name}")
    
    def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check."""
        if name not in self.checks:
            raise ValueError(f"Health check '{name}' not found")
        
        health_check = self.checks[name]
        result = health_check.check()
        self.results[name] = result
        
        return result
    
    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks."""
        results = {}
        
        for name in self.checks:
            try:
                result = self.run_check(name)
                results[name] = result
                
                # Log critical issues
                if result.status == HealthStatus.CRITICAL:
                    self.logger.error(f"CRITICAL: {name} - {result.message}")
                elif result.status == HealthStatus.WARNING:
                    self.logger.warning(f"WARNING: {name} - {result.message}")
                
            except Exception as e:
                self.logger.error(f"Failed to run health check '{name}': {str(e)}")
        
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in self.results.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def start_monitoring(self, interval: float = 60.0):
        """Start continuous health monitoring."""
        if self._monitoring:
            self.logger.warning("Health monitoring already running")
            return
        
        self._monitoring = True
        self._stop_event.clear()
        
        def monitor_loop():
            """Main monitoring loop."""
            self.logger.info("Health monitoring started")
            
            while not self._stop_event.wait(interval):
                try:
                    self.run_all_checks()
                    overall_status = self.get_overall_status()
                    self.logger.info(f"Health check cycle completed - Overall status: {overall_status.value}")
                    
                except Exception as e:
                    self.logger.error(f"Error in health monitoring loop: {str(e)}")
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        self._stop_event.set()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        self.logger.info("Health monitoring stopped")
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        overall_status = self.get_overall_status()
        
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": overall_status.value,
            "checks": {}
        }
        
        for name, result in self.results.items():
            report["checks"][name] = result.to_dict()
        
        return report
    
    def save_health_report(self, output_dir: Optional[Path] = None):
        """Save health report to file."""
        if output_dir is None:
            output_dir = Path("monitoring")
        
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_file = output_dir / f"health_report_{timestamp}.json"
        
        report = self.get_health_report()
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Health report saved to {report_file}")
        return report_file


# Global health monitor instance
_global_monitor = HealthMonitor()


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance."""
    return _global_monitor


def add_default_health_checks():
    """Add default health checks to the global monitor."""
    monitor = get_health_monitor()
    
    # System resources check
    monitor.add_check(SystemResourcesHealthCheck())
    
    # GPU health check
    monitor.add_check(GPUHealthCheck())
    
    # Storage health check (basic paths)
    basic_paths = [Path("logs"), Path("data"), Path("models")]
    monitor.add_check(StorageHealthCheck(basic_paths))