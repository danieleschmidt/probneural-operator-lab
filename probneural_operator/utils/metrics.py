"""Metrics collection and monitoring for ProbNeural Operator Lab."""

import time
import threading
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from contextlib import contextmanager

import torch
import numpy as np
import psutil


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: datetime
    name: str
    value: float
    tags: Dict[str, str]
    unit: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "name": self.name,
            "value": self.value,
            "tags": self.tags,
            "unit": self.unit
        }


class MetricsCollector:
    """Central metrics collection system."""
    
    def __init__(self, buffer_size: int = 10000, flush_interval: int = 60):
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self._metrics_buffer = deque(maxlen=buffer_size)
        self._aggregated_metrics = defaultdict(list)
        self._lock = threading.Lock()
        self._auto_flush = True
        self._last_flush = time.time()
        
        # System metrics tracking
        self._system_metrics_enabled = False
        self._system_metrics_thread = None
        self._stop_system_metrics = threading.Event()
        
        # Callbacks for metric events
        self._callbacks: List[Callable[[MetricPoint], None]] = []
    
    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, unit: str = ""):
        """Record a metric value."""
        if tags is None:
            tags = {}
        
        metric_point = MetricPoint(
            timestamp=datetime.utcnow(),
            name=name,
            value=value,
            tags=tags,
            unit=unit
        )
        
        with self._lock:
            self._metrics_buffer.append(metric_point)
            self._aggregated_metrics[name].append(metric_point)
        
        # Trigger callbacks
        for callback in self._callbacks:
            try:
                callback(metric_point)
            except Exception as e:
                # Don't let callback errors break metrics collection
                print(f"Warning: Metrics callback error: {e}")
        
        # Auto-flush if needed
        if self._auto_flush and time.time() - self._last_flush > self.flush_interval:
            self.flush()
    
    def record_timing(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing metric."""
        self.record(name, duration, tags, unit="seconds")
    
    def record_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Record a counter metric."""
        self.record(name, value, tags, unit="count")
    
    def record_gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None, unit: str = ""):
        """Record a gauge metric."""
        self.record(name, value, tags, unit=unit)
    
    @contextmanager
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.record_timing(name, duration, tags)
    
    def increment_counter(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Increment a counter by 1."""
        self.record_counter(name, 1, tags)
    
    def get_metrics(self, name: Optional[str] = None, 
                   since: Optional[datetime] = None) -> List[MetricPoint]:
        """Get metrics from buffer."""
        with self._lock:
            if name is None:
                metrics = list(self._metrics_buffer)
            else:
                metrics = self._aggregated_metrics.get(name, [])
            
            if since is not None:
                metrics = [m for m in metrics if m.timestamp >= since]
            
            return metrics
    
    def get_metric_summary(self, name: str, 
                          since: Optional[datetime] = None) -> Dict[str, float]:
        """Get statistical summary of a metric."""
        metrics = self.get_metrics(name, since)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "median": sorted(values)[len(values) // 2],
            "std": float(np.std(values)) if len(values) > 1 else 0.0
        }
    
    def add_callback(self, callback: Callable[[MetricPoint], None]):
        """Add a callback for metric events."""
        self._callbacks.append(callback)
    
    def flush(self, output_dir: Optional[Path] = None):
        """Flush metrics to storage."""
        if output_dir is None:
            output_dir = Path("metrics")
        
        output_dir.mkdir(exist_ok=True)
        
        with self._lock:
            if not self._metrics_buffer:
                return
            
            # Save to JSON
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            json_file = output_dir / f"metrics_{timestamp}.json"
            
            metrics_data = [metric.to_dict() for metric in self._metrics_buffer]
            
            with open(json_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            # Save to CSV for easier analysis
            csv_file = output_dir / f"metrics_{timestamp}.csv"
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "name", "value", "tags", "unit"])
                
                for metric in self._metrics_buffer:
                    writer.writerow([
                        metric.timestamp.isoformat(),
                        metric.name,
                        metric.value,
                        json.dumps(metric.tags),
                        metric.unit
                    ])
            
            self._last_flush = time.time()
            
            print(f"Flushed {len(self._metrics_buffer)} metrics to {output_dir}")
    
    def start_system_monitoring(self, interval: float = 5.0):
        """Start collecting system metrics."""
        if self._system_metrics_enabled:
            return
        
        self._system_metrics_enabled = True
        self._stop_system_metrics.clear()
        
        def collect_system_metrics():
            """Collect system metrics in background thread."""
            while not self._stop_system_metrics.wait(interval):
                try:
                    # CPU metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.record_gauge("system.cpu.percent", cpu_percent, unit="percent")
                    
                    # Memory metrics
                    memory = psutil.virtual_memory()
                    self.record_gauge("system.memory.percent", memory.percent, unit="percent")
                    self.record_gauge("system.memory.used", memory.used / 1024**3, unit="GB")
                    self.record_gauge("system.memory.available", memory.available / 1024**3, unit="GB")
                    
                    # Disk metrics
                    disk = psutil.disk_usage('/')
                    self.record_gauge("system.disk.percent", disk.percent, unit="percent")
                    self.record_gauge("system.disk.used", disk.used / 1024**3, unit="GB")
                    self.record_gauge("system.disk.free", disk.free / 1024**3, unit="GB")
                    
                    # GPU metrics (if available)
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            device = torch.device(f"cuda:{i}")
                            
                            # Memory metrics
                            allocated = torch.cuda.memory_allocated(device) / 1024**3
                            reserved = torch.cuda.memory_reserved(device) / 1024**3
                            
                            tags = {"gpu": str(i)}
                            self.record_gauge("gpu.memory.allocated", allocated, tags, unit="GB")
                            self.record_gauge("gpu.memory.reserved", reserved, tags, unit="GB")
                            
                            # Utilization (if nvidia-ml-py is available)
                            try:
                                import pynvml
                                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                                self.record_gauge("gpu.utilization", utilization.gpu, tags, unit="percent")
                                self.record_gauge("gpu.memory.utilization", utilization.memory, tags, unit="percent")
                            except (ImportError, Exception):
                                pass
                
                except Exception as e:
                    print(f"Error collecting system metrics: {e}")
        
        self._system_metrics_thread = threading.Thread(target=collect_system_metrics, daemon=True)
        self._system_metrics_thread.start()
    
    def stop_system_monitoring(self):
        """Stop collecting system metrics."""
        if not self._system_metrics_enabled:
            return
        
        self._system_metrics_enabled = False
        self._stop_system_metrics.set()
        
        if self._system_metrics_thread:
            self._system_metrics_thread.join(timeout=5)
    
    def clear(self):
        """Clear all metrics."""
        with self._lock:
            self._metrics_buffer.clear()
            self._aggregated_metrics.clear()


class ExperimentTracker:
    """Track experiment metrics and metadata."""
    
    def __init__(self, experiment_name: str, metrics_collector: Optional[MetricsCollector] = None):
        self.experiment_name = experiment_name
        self.metrics = metrics_collector or MetricsCollector()
        self.start_time = datetime.utcnow()
        self.metadata = {}
        self.status = "running"
        
        # Record experiment start
        self.metrics.record("experiment.started", 1, {"experiment": experiment_name})
    
    def set_metadata(self, **kwargs):
        """Set experiment metadata."""
        self.metadata.update(kwargs)
    
    def record_metric(self, name: str, value: float, step: Optional[int] = None, **tags):
        """Record an experiment metric."""
        experiment_tags = {"experiment": self.experiment_name}
        experiment_tags.update(tags)
        
        if step is not None:
            experiment_tags["step"] = str(step)
        
        self.metrics.record(f"experiment.{name}", value, experiment_tags)
    
    def record_loss(self, loss: float, step: Optional[int] = None, phase: str = "train"):
        """Record training/validation loss."""
        self.record_metric("loss", loss, step=step, phase=phase)
    
    def record_accuracy(self, accuracy: float, step: Optional[int] = None, phase: str = "train"):
        """Record accuracy metric."""
        self.record_metric("accuracy", accuracy, step=step, phase=phase)
    
    def record_uncertainty_metrics(self, metrics_dict: Dict[str, float], step: Optional[int] = None):
        """Record uncertainty quantification metrics."""
        for name, value in metrics_dict.items():
            self.record_metric(f"uncertainty.{name}", value, step=step)
    
    def record_model_metrics(self, model: torch.nn.Module):
        """Record model-specific metrics."""
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.record_metric("model.parameters.total", param_count)
        self.record_metric("model.parameters.trainable", trainable_params)
        self.record_metric("model.parameters.frozen", param_count - trainable_params)
    
    def finish(self, status: str = "completed"):
        """Mark experiment as finished."""
        self.status = status
        duration = (datetime.utcnow() - self.start_time).total_seconds()
        
        self.metrics.record("experiment.finished", 1, {
            "experiment": self.experiment_name,
            "status": status
        })
        self.metrics.record("experiment.duration", duration, {
            "experiment": self.experiment_name
        }, unit="seconds")
    
    def save_summary(self, output_dir: Optional[Path] = None):
        """Save experiment summary."""
        if output_dir is None:
            output_dir = Path("experiments")
        
        output_dir.mkdir(exist_ok=True)
        
        summary = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.utcnow().isoformat(),
            "duration": (datetime.utcnow() - self.start_time).total_seconds(),
            "status": self.status,
            "metadata": self.metadata,
            "metrics_summary": {}
        }
        
        # Add metrics summaries
        experiment_metrics = [m for m in self.metrics.get_metrics() 
                            if m.tags.get("experiment") == self.experiment_name]
        
        metric_names = set(m.name for m in experiment_metrics)
        for name in metric_names:
            name_metrics = [m for m in experiment_metrics if m.name == name]
            if name_metrics:
                values = [m.value for m in name_metrics]
                summary["metrics_summary"][name] = {
                    "count": len(values),
                    "final_value": values[-1] if values else None,
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values)
                }
        
        summary_file = output_dir / f"{self.experiment_name}_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Experiment summary saved to {summary_file}")


# Global metrics collector instance
_global_metrics = MetricsCollector()


def get_global_metrics() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return _global_metrics


def record_metric(name: str, value: float, tags: Optional[Dict[str, str]] = None, unit: str = ""):
    """Record a metric using the global collector."""
    _global_metrics.record(name, value, tags, unit)


def timer(name: str, tags: Optional[Dict[str, str]] = None):
    """Context manager for timing using the global collector."""
    return _global_metrics.timer(name, tags)


def start_system_monitoring(interval: float = 5.0):
    """Start system monitoring using the global collector."""
    _global_metrics.start_system_monitoring(interval)


def stop_system_monitoring():
    """Stop system monitoring."""
    _global_metrics.stop_system_monitoring()


# Convenience decorators
def measure_time(metric_name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator to measure function execution time."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with timer(metric_name, tags):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def count_calls(metric_name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator to count function calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            _global_metrics.increment_counter(metric_name, tags)
            return func(*args, **kwargs)
        return wrapper
    return decorator