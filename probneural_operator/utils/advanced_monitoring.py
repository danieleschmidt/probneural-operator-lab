"""
Advanced monitoring and observability for probabilistic neural operators.

This module provides comprehensive monitoring, logging, and observability
features including distributed tracing, metrics collection, alerting,
and performance profiling for production deployments.
"""

import json
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
import logging


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: float
    labels: Dict[str, str]


@dataclass
class Alert:
    """Alert definition."""
    name: str
    condition: Callable[[float], bool]
    message: str
    severity: str
    cooldown: float = 300.0  # 5 minutes
    last_triggered: float = 0.0


class MetricsCollector:
    """Collect and aggregate metrics for monitoring."""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_seconds = retention_hours * 3600
        self.metrics = defaultdict(deque)  # metric_name -> deque of MetricPoints
        self.lock = threading.RLock()
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def record_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a single metric value."""
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            labels=labels or {}
        )
        
        with self.lock:
            self.metrics[name].append(point)
    
    def record_histogram(
        self,
        name: str,
        value: float,
        buckets: List[float],
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record histogram metric (for latencies, sizes, etc.)."""
        # Record the actual value
        self.record_metric(f"{name}_value", value, labels)
        
        # Record bucket counts
        for bucket in buckets:
            bucket_name = f"{name}_bucket_le_{bucket}"
            if value <= bucket:
                self.record_metric(bucket_name, 1, labels)
            else:
                self.record_metric(bucket_name, 0, labels)
    
    def record_counter(
        self,
        name: str,
        increment: float = 1.0,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record counter increment."""
        self.record_metric(name, increment, labels)
    
    def get_metric_summary(
        self,
        name: str,
        window_seconds: int = 300
    ) -> Dict[str, float]:
        """Get statistical summary of a metric over time window."""
        cutoff_time = time.time() - window_seconds
        
        with self.lock:
            if name not in self.metrics:
                return {}
            
            recent_values = [
                point.value for point in self.metrics[name]
                if point.timestamp > cutoff_time
            ]
            
            if not recent_values:
                return {}
            
            recent_values.sort()
            n = len(recent_values)
            
            return {
                "count": n,
                "sum": sum(recent_values),
                "min": recent_values[0],
                "max": recent_values[-1],
                "mean": sum(recent_values) / n,
                "p50": recent_values[n // 2],
                "p90": recent_values[int(n * 0.9)],
                "p95": recent_values[int(n * 0.95)],
                "p99": recent_values[int(n * 0.99)]
            }
    
    def get_time_series(
        self,
        name: str,
        window_seconds: int = 3600
    ) -> List[Dict[str, Any]]:
        """Get time series data for a metric."""
        cutoff_time = time.time() - window_seconds
        
        with self.lock:
            if name not in self.metrics:
                return []
            
            return [
                {
                    "timestamp": point.timestamp,
                    "value": point.value,
                    "labels": point.labels
                }
                for point in self.metrics[name]
                if point.timestamp > cutoff_time
            ]
    
    def list_metrics(self) -> List[str]:
        """List all metric names."""
        with self.lock:
            return list(self.metrics.keys())
    
    def _cleanup_loop(self):
        """Background cleanup of old metrics."""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                cutoff_time = time.time() - self.retention_seconds
                
                with self.lock:
                    for name, points in self.metrics.items():
                        while points and points[0].timestamp < cutoff_time:
                            points.popleft()
                            
            except Exception as e:
                logging.error(f"Metrics cleanup error: {e}")


class AlertManager:
    """Manage alerts based on metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.alerts = {}
        self.alert_history = deque(maxlen=1000)
        self.handlers = []
        
    def register_alert(self, alert: Alert) -> None:
        """Register a new alert."""
        self.alerts[alert.name] = alert
        
    def add_handler(self, handler: Callable[[str, Dict[str, Any]], None]) -> None:
        """Add alert handler function."""
        self.handlers.append(handler)
    
    def check_alerts(self) -> List[Dict[str, Any]]:
        """Check all alerts and trigger if necessary."""
        triggered = []
        current_time = time.time()
        
        for alert_name, alert in self.alerts.items():
            # Skip if in cooldown
            if current_time - alert.last_triggered < alert.cooldown:
                continue
                
            try:
                # Get recent metric value
                summary = self.metrics.get_metric_summary(alert_name, window_seconds=60)
                if not summary:
                    continue
                
                # Check condition
                metric_value = summary.get("mean", 0)
                if alert.condition(metric_value):
                    alert.last_triggered = current_time
                    
                    alert_event = {
                        "alert": alert_name,
                        "message": alert.message,
                        "severity": alert.severity,
                        "value": metric_value,
                        "timestamp": current_time
                    }
                    
                    triggered.append(alert_event)
                    self.alert_history.append(alert_event)
                    
                    # Call handlers
                    for handler in self.handlers:
                        try:
                            handler(alert_name, alert_event)
                        except Exception as e:
                            logging.error(f"Alert handler error: {e}")
                            
            except Exception as e:
                logging.error(f"Alert check error for {alert_name}: {e}")
        
        return triggered
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history."""
        cutoff = time.time() - hours * 3600
        return [
            alert for alert in self.alert_history
            if alert["timestamp"] > cutoff
        ]


class DistributedTracer:
    """Simple distributed tracing implementation."""
    
    def __init__(self):
        self.traces = {}
        self.spans = deque(maxlen=10000)
        self.current_trace = threading.local()
    
    def start_trace(self, operation_name: str) -> str:
        """Start a new trace."""
        trace_id = f"trace_{int(time.time() * 1000000)}"
        span_id = f"span_{trace_id}_0"
        
        trace_data = {
            "trace_id": trace_id,
            "operation": operation_name,
            "start_time": time.time(),
            "spans": {},
            "root_span": span_id
        }
        
        self.traces[trace_id] = trace_data
        self.current_trace.trace_id = trace_id
        self.current_trace.span_counter = 0
        
        # Start root span
        self.start_span(operation_name, span_id)
        
        return trace_id
    
    def start_span(
        self,
        operation_name: str,
        span_id: Optional[str] = None,
        parent_span: Optional[str] = None
    ) -> str:
        """Start a new span within current trace."""
        if not hasattr(self.current_trace, 'trace_id'):
            return self.start_trace(operation_name)
        
        if not span_id:
            self.current_trace.span_counter += 1
            span_id = f"span_{self.current_trace.trace_id}_{self.current_trace.span_counter}"
        
        span_data = {
            "span_id": span_id,
            "trace_id": self.current_trace.trace_id,
            "operation": operation_name,
            "parent_span": parent_span,
            "start_time": time.time(),
            "end_time": None,
            "duration": None,
            "tags": {},
            "logs": []
        }
        
        if self.current_trace.trace_id in self.traces:
            self.traces[self.current_trace.trace_id]["spans"][span_id] = span_data
        
        return span_id
    
    def finish_span(self, span_id: str, tags: Optional[Dict[str, Any]] = None) -> None:
        """Finish a span."""
        if not hasattr(self.current_trace, 'trace_id'):
            return
            
        trace_id = self.current_trace.trace_id
        if trace_id not in self.traces:
            return
            
        trace = self.traces[trace_id]
        if span_id not in trace["spans"]:
            return
        
        span = trace["spans"][span_id]
        span["end_time"] = time.time()
        span["duration"] = span["end_time"] - span["start_time"]
        
        if tags:
            span["tags"].update(tags)
        
        # Copy completed span to global collection
        self.spans.append(span.copy())
    
    def add_span_log(self, span_id: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """Add log entry to span."""
        if not hasattr(self.current_trace, 'trace_id'):
            return
            
        trace_id = self.current_trace.trace_id
        if trace_id not in self.traces or span_id not in self.traces[trace_id]["spans"]:
            return
        
        log_entry = {
            "timestamp": time.time(),
            "message": message,
            "data": data or {}
        }
        
        self.traces[trace_id]["spans"][span_id]["logs"].append(log_entry)
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get complete trace data."""
        return self.traces.get(trace_id)
    
    def get_recent_spans(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent spans across all traces."""
        return list(self.spans)[-limit:]


class PerformanceProfiler:
    """Profile performance characteristics."""
    
    def __init__(self):
        self.profiles = deque(maxlen=1000)
        
    def profile_function(self, func_name: str):
        """Decorator to profile function execution."""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                memory_before = self._get_memory_usage()
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)
                    raise
                finally:
                    end_time = time.time()
                    memory_after = self._get_memory_usage()
                    
                    profile_data = {
                        "function": func_name,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": end_time - start_time,
                        "memory_before": memory_before,
                        "memory_after": memory_after,
                        "memory_delta": memory_after - memory_before,
                        "success": success,
                        "error": error,
                        "args_count": len(args),
                        "kwargs_count": len(kwargs)
                    }
                    
                    self.profiles.append(profile_data)
                
                return result
            return wrapper
        return decorator
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage (simplified)."""
        # In real implementation, would use psutil or similar
        return 0.0
    
    def get_function_stats(self, func_name: str) -> Dict[str, Any]:
        """Get performance statistics for a function."""
        func_profiles = [p for p in self.profiles if p["function"] == func_name]
        
        if not func_profiles:
            return {}
        
        durations = [p["duration"] for p in func_profiles if p["success"]]
        memory_deltas = [p["memory_delta"] for p in func_profiles if p["success"]]
        error_count = len([p for p in func_profiles if not p["success"]])
        
        return {
            "call_count": len(func_profiles),
            "success_count": len(durations),
            "error_count": error_count,
            "error_rate": error_count / len(func_profiles) if func_profiles else 0,
            "avg_duration": sum(durations) / len(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "avg_memory_delta": sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0
        }


class ObservabilityManager:
    """Central manager for all observability features."""
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.alerts = AlertManager(self.metrics)
        self.tracer = DistributedTracer()
        self.profiler = PerformanceProfiler()
        
        self._setup_default_alerts()
    
    def _setup_default_alerts(self):
        """Setup default alerts for common issues."""
        default_alerts = [
            Alert(
                name="high_error_rate",
                condition=lambda x: x > 0.05,  # 5% error rate
                message="Error rate exceeding 5%",
                severity="warning"
            ),
            Alert(
                name="high_response_time",
                condition=lambda x: x > 1.0,  # 1 second
                message="Average response time exceeding 1 second",
                severity="warning"
            ),
            Alert(
                name="low_throughput", 
                condition=lambda x: x < 10,  # 10 requests per second
                message="Throughput below 10 RPS",
                severity="info"
            )
        ]
        
        for alert in default_alerts:
            self.alerts.register_alert(alert)
    
    def record_request(
        self,
        operation: str,
        duration: float,
        success: bool,
        model_id: Optional[str] = None
    ) -> None:
        """Record request metrics with proper labels."""
        labels = {"operation": operation}
        if model_id:
            labels["model_id"] = model_id
            
        # Record basic metrics
        self.metrics.record_metric("request_duration", duration, labels)
        self.metrics.record_counter("requests_total", 1.0, labels)
        
        if not success:
            self.metrics.record_counter("errors_total", 1.0, labels)
        
        # Record histogram for latency analysis
        latency_buckets = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
        self.metrics.record_histogram("request_duration_histogram", duration, latency_buckets, labels)
    
    def record_model_metrics(
        self,
        model_id: str,
        prediction_time: float,
        uncertainty_score: float,
        batch_size: int
    ) -> None:
        """Record model-specific metrics."""
        labels = {"model_id": model_id}
        
        self.metrics.record_metric("model_prediction_time", prediction_time, labels)
        self.metrics.record_metric("model_uncertainty_score", uncertainty_score, labels)
        self.metrics.record_metric("model_batch_size", batch_size, labels)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        dashboard = {
            "timestamp": time.time(),
            "metrics": {},
            "alerts": self.alerts.get_alert_history(hours=1),
            "recent_spans": self.tracer.get_recent_spans(limit=50)
        }
        
        # Key metrics summaries
        key_metrics = [
            "request_duration", "requests_total", "errors_total",
            "model_prediction_time", "model_uncertainty_score"
        ]
        
        for metric_name in key_metrics:
            summary = self.metrics.get_metric_summary(metric_name, window_seconds=300)
            if summary:
                dashboard["metrics"][metric_name] = summary
        
        return dashboard
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        for metric_name in self.metrics.list_metrics():
            if metric_name.endswith("_total"):  # Counter metrics
                time_series = self.metrics.get_time_series(metric_name, window_seconds=60)
                if time_series:
                    latest = time_series[-1]
                    lines.append(f"# TYPE {metric_name} counter")
                    lines.append(f"{metric_name} {latest['value']}")
            
            else:  # Gauge metrics
                summary = self.metrics.get_metric_summary(metric_name, window_seconds=60)
                if summary:
                    lines.append(f"# TYPE {metric_name} gauge")
                    lines.append(f"{metric_name} {summary['mean']}")
        
        return "\n".join(lines)


# Example usage and testing
def demo_observability():
    """Demonstrate observability features."""
    print("🔍 Advanced Monitoring and Observability Demo")
    print("=" * 60)
    
    # Initialize observability
    obs = ObservabilityManager()
    
    print("📊 Recording sample metrics...")
    
    # Simulate some requests
    import random
    for i in range(20):
        duration = random.uniform(0.1, 2.0)
        success = random.random() > 0.1  # 90% success rate
        model_id = random.choice(["fno_model", "deeponet_model"])
        
        obs.record_request("predict", duration, success, model_id)
        
        if success:
            obs.record_model_metrics(
                model_id=model_id,
                prediction_time=duration * 0.8,
                uncertainty_score=random.uniform(0.1, 0.5),
                batch_size=random.randint(1, 32)
            )
    
    print("  ✓ Recorded 20 sample requests")
    
    # Check alerts
    print("\n🚨 Checking alerts...")
    triggered_alerts = obs.alerts.check_alerts()
    print(f"  ✓ {len(triggered_alerts)} alerts triggered")
    
    # Get dashboard data
    print("\n📈 Generating dashboard data...")
    dashboard = obs.get_dashboard_data()
    print(f"  ✓ Dashboard with {len(dashboard['metrics'])} metric summaries")
    print(f"  ✓ {len(dashboard['alerts'])} recent alerts")
    print(f"  ✓ {len(dashboard['recent_spans'])} recent spans")
    
    # Export Prometheus metrics
    print("\n📤 Exporting Prometheus metrics...")
    prometheus_output = obs.export_prometheus_metrics()
    print(f"  ✓ Exported {len(prometheus_output.split(chr(10)))} metric lines")
    
    # Performance profiling demo
    print("\n⚡ Performance profiling demo...")
    
    @obs.profiler.profile_function("demo_function")
    def demo_function(x):
        time.sleep(x * 0.01)  # Simulate work
        return x * 2
    
    # Call profiled function
    for i in range(5):
        demo_function(random.uniform(0.1, 1.0))
    
    stats = obs.profiler.get_function_stats("demo_function")
    print(f"  ✓ Function called {stats.get('call_count', 0)} times")
    print(f"  ✓ Average duration: {stats.get('avg_duration', 0):.4f}s")
    
    print(f"\n{'='*60}")
    print("✅ Observability demo completed!")


if __name__ == "__main__":
    demo_observability()