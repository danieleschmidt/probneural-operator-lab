"""
Comprehensive monitoring system for the ProbNeural-Operator-Lab framework.

This module provides real-time monitoring capabilities for training processes,
model performance, resource usage, and system health.
"""

import time
import threading
import psutil
import gc
from typing import Dict, Any, List, Optional, Callable, Union
from collections import deque
from datetime import datetime, timedelta
import logging

import torch
import numpy as np

from .logging_config import get_logger

logger = get_logger(__name__)


class MetricCollector:
    """Collects and aggregates metrics over time."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize metric collector.
        
        Args:
            max_history: Maximum number of historical values to keep
        """
        self.max_history = max_history
        self.metrics: Dict[str, deque] = {}
        self.timestamps: Dict[str, deque] = {}
        self._lock = threading.Lock()
    
    def record(self, metric_name: str, value: Union[float, int], timestamp: Optional[datetime] = None):
        """Record a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            timestamp: Optional timestamp (uses current time if None)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        with self._lock:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = deque(maxlen=self.max_history)
                self.timestamps[metric_name] = deque(maxlen=self.max_history)
            
            self.metrics[metric_name].append(float(value))
            self.timestamps[metric_name].append(timestamp)
    
    def get_recent(self, metric_name: str, window_seconds: int = 60) -> List[float]:
        """Get recent metric values within a time window.
        
        Args:
            metric_name: Name of the metric
            window_seconds: Time window in seconds
            
        Returns:
            List of recent metric values
        """
        if metric_name not in self.metrics:
            return []
        
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        
        with self._lock:
            values = []
            for value, timestamp in zip(self.metrics[metric_name], self.timestamps[metric_name]):
                if timestamp >= cutoff_time:
                    values.append(value)
            return values
    
    def get_statistics(self, metric_name: str, window_seconds: int = 60) -> Dict[str, float]:
        """Get statistical summary of recent metric values.
        
        Args:
            metric_name: Name of the metric
            window_seconds: Time window in seconds
            
        Returns:
            Dictionary with statistical measures
        """
        recent_values = self.get_recent(metric_name, window_seconds)
        
        if not recent_values:
            return {}
        
        return {
            'count': len(recent_values),
            'mean': np.mean(recent_values),
            'std': np.std(recent_values),
            'min': np.min(recent_values),
            'max': np.max(recent_values),
            'latest': recent_values[-1] if recent_values else 0.0
        }
    
    def get_all_metrics(self) -> List[str]:
        """Get list of all tracked metric names."""
        with self._lock:
            return list(self.metrics.keys())


class ResourceMonitor:
    """Monitors system resource usage."""
    
    def __init__(self, interval: float = 5.0):
        """Initialize resource monitor.
        
        Args:
            interval: Monitoring interval in seconds
        """
        self.interval = interval
        self.metrics = MetricCollector()
        self._running = False
        self._thread = None
        self._process = psutil.Process()
    
    def start(self):
        """Start resource monitoring."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Resource monitoring started")
    
    def stop(self):
        """Stop resource monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                self._collect_system_metrics()
                self._collect_gpu_metrics()
                time.sleep(self.interval)
            except Exception as e:
                logger.warning(f"Error in resource monitoring: {e}")
                time.sleep(self.interval)
    
    def _collect_system_metrics(self):
        """Collect system resource metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        self.metrics.record('cpu_percent', cpu_percent)
        
        # Memory usage
        memory_info = self._process.memory_info()
        self.metrics.record('memory_rss_mb', memory_info.rss / 1024 / 1024)
        self.metrics.record('memory_vms_mb', memory_info.vms / 1024 / 1024)
        
        # System memory
        system_memory = psutil.virtual_memory()
        self.metrics.record('system_memory_percent', system_memory.percent)
        self.metrics.record('system_memory_available_gb', system_memory.available / 1024**3)
        
        # Disk I/O
        try:
            io_counters = self._process.io_counters()
            self.metrics.record('disk_read_mb', io_counters.read_bytes / 1024 / 1024)
            self.metrics.record('disk_write_mb', io_counters.write_bytes / 1024 / 1024)
        except (AttributeError, psutil.AccessDenied):
            pass  # Not available on all systems
    
    def _collect_gpu_metrics(self):
        """Collect GPU metrics if CUDA is available."""
        if not torch.cuda.is_available():
            return
        
        try:
            for i in range(torch.cuda.device_count()):
                # Memory usage
                memory_allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
                max_memory = torch.cuda.max_memory_allocated(i) / 1024**3  # GB
                
                self.metrics.record(f'gpu_{i}_memory_allocated_gb', memory_allocated)
                self.metrics.record(f'gpu_{i}_memory_reserved_gb', memory_reserved)
                self.metrics.record(f'gpu_{i}_max_memory_gb', max_memory)
                
                # GPU utilization (if nvidia-ml-py is available)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    self.metrics.record(f'gpu_{i}_utilization_percent', utilization.gpu)
                    self.metrics.record(f'gpu_{i}_memory_utilization_percent', utilization.memory)
                except ImportError:
                    pass  # pynvml not available
                
        except Exception as e:
            logger.debug(f"Could not collect GPU metrics: {e}")
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """Get current resource usage summary.
        
        Returns:
            Dictionary with resource usage statistics
        """
        summary = {}
        
        for metric_name in self.metrics.get_all_metrics():
            stats = self.metrics.get_statistics(metric_name, window_seconds=60)
            if stats:
                summary[metric_name] = stats
        
        return summary


class TrainingMonitor:
    """Monitors training progress and detects issues."""
    
    def __init__(self, patience: int = 10, min_improvement: float = 1e-4):
        """Initialize training monitor.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_improvement: Minimum improvement to reset patience
        """
        self.patience = patience
        self.min_improvement = min_improvement
        self.metrics = MetricCollector()
        
        # Training state
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_start_time = None
        self.last_epoch_time = None
        
        # Issue detection
        self.loss_explosion_threshold = 1e6
        self.loss_stagnation_threshold = 1e-8
        self.gradient_explosion_threshold = 100.0
        
    def start_training(self):
        """Mark training start."""
        self.training_start_time = time.time()
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        logger.info("Training monitoring started")
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: Optional[float] = None,
                  learning_rate: Optional[float] = None, grad_norm: Optional[float] = None):
        """Log epoch results and check for issues.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss (optional)
            learning_rate: Current learning rate (optional)
            grad_norm: Gradient norm (optional)
        """
        current_time = time.time()
        
        # Record metrics
        self.metrics.record('epoch', epoch)
        self.metrics.record('train_loss', train_loss)
        
        if val_loss is not None:
            self.metrics.record('val_loss', val_loss)
        
        if learning_rate is not None:
            self.metrics.record('learning_rate', learning_rate)
        
        if grad_norm is not None:
            self.metrics.record('grad_norm', grad_norm)
        
        # Calculate epoch time
        if self.last_epoch_time is not None:
            epoch_time = current_time - self.last_epoch_time
            self.metrics.record('epoch_time', epoch_time)
        
        self.last_epoch_time = current_time
        
        # Check for improvement
        current_loss = val_loss if val_loss is not None else train_loss
        if current_loss < self.best_loss - self.min_improvement:
            self.best_loss = current_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        # Issue detection
        self._detect_issues(train_loss, val_loss, grad_norm)
    
    def _detect_issues(self, train_loss: float, val_loss: Optional[float], grad_norm: Optional[float]):
        """Detect training issues and log warnings.
        
        Args:
            train_loss: Training loss
            val_loss: Validation loss
            grad_norm: Gradient norm
        """
        issues = []
        
        # Loss explosion
        if train_loss > self.loss_explosion_threshold:
            issues.append(f"Training loss exploded: {train_loss}")
        
        if val_loss is not None and val_loss > self.loss_explosion_threshold:
            issues.append(f"Validation loss exploded: {val_loss}")
        
        # Gradient explosion
        if grad_norm is not None and grad_norm > self.gradient_explosion_threshold:
            issues.append(f"Gradient explosion detected: {grad_norm}")
        
        # Loss stagnation
        recent_train_losses = self.metrics.get_recent('train_loss', window_seconds=600)  # 10 minutes
        if len(recent_train_losses) > 5:
            loss_variance = np.var(recent_train_losses)
            if loss_variance < self.loss_stagnation_threshold:
                issues.append(f"Training loss stagnated (variance: {loss_variance})")
        
        # Overfitting
        if val_loss is not None:
            recent_val_losses = self.metrics.get_recent('val_loss', window_seconds=600)
            if len(recent_train_losses) > 5 and len(recent_val_losses) > 5:
                train_trend = np.polyfit(range(len(recent_train_losses)), recent_train_losses, 1)[0]
                val_trend = np.polyfit(range(len(recent_val_losses)), recent_val_losses, 1)[0]
                
                if train_trend < -0.01 and val_trend > 0.01:  # Train decreasing, val increasing
                    issues.append("Potential overfitting detected")
        
        # Log issues
        for issue in issues:
            logger.warning(f"Training issue detected: {issue}")
    
    def should_early_stop(self) -> bool:
        """Check if training should be stopped early.
        
        Returns:
            True if early stopping criteria is met
        """
        return self.epochs_without_improvement >= self.patience
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get training progress summary.
        
        Returns:
            Dictionary with training statistics
        """
        summary = {
            'best_loss': self.best_loss,
            'epochs_without_improvement': self.epochs_without_improvement,
            'should_early_stop': self.should_early_stop()
        }
        
        if self.training_start_time:
            summary['total_training_time'] = time.time() - self.training_start_time
        
        # Add metric statistics
        for metric_name in self.metrics.get_all_metrics():
            stats = self.metrics.get_statistics(metric_name)
            if stats:
                summary[f'{metric_name}_stats'] = stats
        
        return summary


class SystemHealthMonitor:
    """Monitors overall system health and alerts for issues."""
    
    def __init__(self, check_interval: float = 30.0):
        """Initialize system health monitor.
        
        Args:
            check_interval: Health check interval in seconds
        """
        self.check_interval = check_interval
        self.resource_monitor = ResourceMonitor()
        self.alerts: List[Dict[str, Any]] = []
        self._running = False
        self._thread = None
        
        # Alert thresholds
        self.memory_threshold = 90.0  # Percent
        self.cpu_threshold = 95.0     # Percent
        self.gpu_memory_threshold = 95.0  # Percent
        
    def start(self):
        """Start health monitoring."""
        if self._running:
            return
        
        self._running = True
        self.resource_monitor.start()
        self._thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._thread.start()
        logger.info("System health monitoring started")
    
    def stop(self):
        """Stop health monitoring."""
        self._running = False
        self.resource_monitor.stop()
        if self._thread:
            self._thread.join(timeout=1.0)
        logger.info("System health monitoring stopped")
    
    def _health_check_loop(self):
        """Main health check loop."""
        while self._running:
            try:
                self._check_system_health()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                time.sleep(self.check_interval)
    
    def _check_system_health(self):
        """Perform system health checks."""
        # Get current resource stats
        resource_summary = self.resource_monitor.get_resource_summary()
        
        # Check memory usage
        if 'system_memory_percent' in resource_summary:
            memory_percent = resource_summary['system_memory_percent'].get('latest', 0)
            if memory_percent > self.memory_threshold:
                self._create_alert(
                    'high_memory_usage',
                    f'System memory usage is {memory_percent:.1f}%',
                    severity='warning'
                )
        
        # Check CPU usage
        if 'cpu_percent' in resource_summary:
            cpu_percent = resource_summary['cpu_percent'].get('latest', 0)
            if cpu_percent > self.cpu_threshold:
                self._create_alert(
                    'high_cpu_usage',
                    f'CPU usage is {cpu_percent:.1f}%',
                    severity='warning'
                )
        
        # Check GPU memory
        for metric_name, stats in resource_summary.items():
            if 'gpu_' in metric_name and 'memory_allocated_gb' in metric_name:
                # Estimate usage percentage (rough approximation)
                allocated_gb = stats.get('latest', 0)
                if allocated_gb > 8:  # Assume 12GB GPU, alert at 8GB
                    self._create_alert(
                        'high_gpu_memory',
                        f'GPU memory usage is high: {allocated_gb:.1f} GB',
                        severity='info'
                    )
        
        # Check for memory leaks
        self._check_memory_leaks()
        
        # Clean up old alerts (keep last 100)
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    def _check_memory_leaks(self):
        """Check for potential memory leaks."""
        memory_stats = self.resource_monitor.metrics.get_statistics('memory_rss_mb', window_seconds=1800)  # 30 min
        
        if memory_stats and memory_stats['count'] > 10:
            # Check if memory usage is consistently increasing
            recent_memory = self.resource_monitor.metrics.get_recent('memory_rss_mb', window_seconds=1800)
            if len(recent_memory) > 10:
                # Simple trend detection
                x = np.arange(len(recent_memory))
                slope, _ = np.polyfit(x, recent_memory, 1)
                
                # If slope is positive and significant
                if slope > 10:  # More than 10MB increase per measurement
                    self._create_alert(
                        'potential_memory_leak',
                        f'Memory usage trending upward: {slope:.1f} MB/measurement',
                        severity='warning'
                    )
    
    def _create_alert(self, alert_type: str, message: str, severity: str = 'info'):
        """Create a system alert.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity ('info', 'warning', 'critical')
        """
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'message': message,
            'severity': severity
        }
        
        self.alerts.append(alert)
        
        # Log based on severity
        if severity == 'critical':
            logger.critical(f"SYSTEM ALERT: {message}")
        elif severity == 'warning':
            logger.warning(f"System alert: {message}")
        else:
            logger.info(f"System info: {message}")
    
    def get_recent_alerts(self, window_seconds: int = 3600) -> List[Dict[str, Any]]:
        """Get recent system alerts.
        
        Args:
            window_seconds: Time window in seconds
            
        Returns:
            List of recent alerts
        """
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        return [alert for alert in self.alerts if alert['timestamp'] >= cutoff_time]
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report.
        
        Returns:
            System health report
        """
        resource_summary = self.resource_monitor.get_resource_summary()
        recent_alerts = self.get_recent_alerts()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'resource_usage': resource_summary,
            'recent_alerts': recent_alerts,
            'alert_counts_by_severity': {
                'info': len([a for a in recent_alerts if a['severity'] == 'info']),
                'warning': len([a for a in recent_alerts if a['severity'] == 'warning']),
                'critical': len([a for a in recent_alerts if a['severity'] == 'critical'])
            }
        }


# Convenience function to create monitoring setup
def create_monitoring_suite(experiment_name: str = "default") -> Dict[str, Any]:
    """Create a complete monitoring suite.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        Dictionary with all monitoring components
    """
    return {
        'resource_monitor': ResourceMonitor(),
        'training_monitor': TrainingMonitor(),
        'health_monitor': SystemHealthMonitor(),
        'experiment_name': experiment_name
    }