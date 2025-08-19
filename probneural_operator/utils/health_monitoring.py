"""
Comprehensive health monitoring system for probabilistic neural operators.

This module provides real-time health monitoring, automatic diagnostics,
and proactive issue detection for production deployments.
"""

import time
import logging
import threading
import queue
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    status: HealthStatus
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    unit: str = ""
    description: str = ""

@dataclass
class SystemHealth:
    """Overall system health status."""
    status: HealthStatus
    metrics: Dict[str, HealthMetric]
    timestamp: float = field(default_factory=time.time)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self, 
                 check_interval: float = 60.0,
                 history_size: int = 1000,
                 auto_start: bool = True):
        """Initialize health monitor.
        
        Args:
            check_interval: Seconds between health checks
            history_size: Number of health records to keep
            auto_start: Whether to start monitoring automatically
        """
        self.check_interval = check_interval
        self.history_size = history_size
        
        # Health state
        self.current_health = SystemHealth(HealthStatus.UNKNOWN, {})
        self.health_history = []
        self.health_checks = {}
        
        # Monitoring control
        self.monitoring_active = False
        self.monitor_thread = None
        self.alert_callbacks = []
        
        # Metrics tracking
        self.metric_trends = {}
        self.alert_cooldowns = {}
        
        if auto_start:
            self.start_monitoring()
    
    def register_health_check(self, 
                            name: str, 
                            check_func: Callable[[], HealthMetric],
                            interval: Optional[float] = None):
        """Register a health check function.
        
        Args:
            name: Unique name for the health check
            check_func: Function that returns a HealthMetric
            interval: Custom interval for this check (uses default if None)
        """
        self.health_checks[name] = {
            'function': check_func,
            'interval': interval or self.check_interval,
            'last_check': 0
        }
        logger.info(f"Registered health check: {name}")
    
    def add_alert_callback(self, callback: Callable[[SystemHealth], None]):
        """Add callback for health alerts."""
        self.alert_callbacks.append(callback)
    
    def start_monitoring(self):
        """Start the health monitoring thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop the health monitoring thread."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._perform_health_check()
                time.sleep(min(self.check_interval, 10.0))  # Max 10 second intervals
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(5.0)  # Brief pause on error
    
    def _perform_health_check(self):
        """Perform all registered health checks."""
        current_time = time.time()
        metrics = {}
        
        # Run individual health checks
        for name, check_info in self.health_checks.items():
            if current_time - check_info['last_check'] >= check_info['interval']:
                try:
                    metric = check_info['function']()
                    metrics[name] = metric
                    check_info['last_check'] = current_time
                    
                    # Update trend tracking
                    self._update_metric_trend(name, metric)
                    
                except Exception as e:
                    logger.error(f"Health check '{name}' failed: {e}")
                    metrics[name] = HealthMetric(
                        name=name,
                        value=0.0,
                        status=HealthStatus.CRITICAL,
                        description=f"Check failed: {e}"
                    )
        
        # Determine overall health status
        overall_status = self._determine_overall_status(metrics)
        
        # Create health report
        health_report = SystemHealth(
            status=overall_status,
            metrics=metrics,
            timestamp=current_time
        )
        
        # Add issues and recommendations
        self._analyze_health_issues(health_report)
        
        # Update current health
        self.current_health = health_report
        
        # Add to history
        self._add_to_history(health_report)
        
        # Trigger alerts if needed
        self._check_alerts(health_report)
    
    def _determine_overall_status(self, metrics: Dict[str, HealthMetric]) -> HealthStatus:
        """Determine overall system health from individual metrics."""
        if not metrics:
            return HealthStatus.UNKNOWN
        
        statuses = [metric.status for metric in metrics.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def _analyze_health_issues(self, health_report: SystemHealth):
        """Analyze health metrics and add issues/recommendations."""
        issues = []
        recommendations = []
        
        for name, metric in health_report.metrics.items():
            if metric.status == HealthStatus.CRITICAL:
                issues.append(f"Critical issue in {name}: {metric.description}")
                recommendations.append(f"Investigate {name} immediately")
            elif metric.status == HealthStatus.WARNING:
                issues.append(f"Warning in {name}: {metric.description}")
                
                # Add trend-based recommendations
                if name in self.metric_trends:
                    trend = self._calculate_trend(name)
                    if trend == "deteriorating":
                        recommendations.append(f"Monitor {name} closely - trend is deteriorating")
        
        # System-wide analysis
        if health_report.status == HealthStatus.CRITICAL:
            recommendations.append("System requires immediate attention")
        elif health_report.status == HealthStatus.WARNING:
            recommendations.append("Consider preventive maintenance")
        
        health_report.issues = issues
        health_report.recommendations = recommendations
    
    def _update_metric_trend(self, name: str, metric: HealthMetric):
        """Update trend tracking for a metric."""
        if name not in self.metric_trends:
            self.metric_trends[name] = []
        
        self.metric_trends[name].append({
            'timestamp': metric.timestamp,
            'value': metric.value,
            'status': metric.status
        })
        
        # Keep only recent history
        cutoff_time = time.time() - 3600  # Last hour
        self.metric_trends[name] = [
            entry for entry in self.metric_trends[name]
            if entry['timestamp'] > cutoff_time
        ]
    
    def _calculate_trend(self, metric_name: str) -> str:
        """Calculate trend for a metric."""
        if metric_name not in self.metric_trends:
            return "unknown"
        
        history = self.metric_trends[metric_name]
        if len(history) < 3:
            return "insufficient_data"
        
        # Simple trend calculation based on recent values
        recent_values = [entry['value'] for entry in history[-5:]]
        
        if len(recent_values) >= 3:
            # Check if trend is increasing, decreasing, or stable
            slope = (recent_values[-1] - recent_values[0]) / len(recent_values)
            
            if slope > 0.1:
                return "improving"
            elif slope < -0.1:
                return "deteriorating"
            else:
                return "stable"
        
        return "unknown"
    
    def _add_to_history(self, health_report: SystemHealth):
        """Add health report to history."""
        self.health_history.append(health_report)
        
        # Maintain history size limit
        if len(self.health_history) > self.history_size:
            self.health_history = self.health_history[-self.history_size:]
    
    def _check_alerts(self, health_report: SystemHealth):
        """Check if alerts should be triggered."""
        current_time = time.time()
        
        # Check for status changes that warrant alerts
        should_alert = False
        alert_reason = ""
        
        if health_report.status == HealthStatus.CRITICAL:
            if "critical" not in self.alert_cooldowns or \
               current_time - self.alert_cooldowns["critical"] > 300:  # 5 min cooldown
                should_alert = True
                alert_reason = "System status is CRITICAL"
                self.alert_cooldowns["critical"] = current_time
        
        elif health_report.status == HealthStatus.WARNING:
            if "warning" not in self.alert_cooldowns or \
               current_time - self.alert_cooldowns["warning"] > 900:  # 15 min cooldown
                should_alert = True
                alert_reason = "System status is WARNING"
                self.alert_cooldowns["warning"] = current_time
        
        # Trigger alert callbacks
        if should_alert:
            logger.warning(f"Health alert: {alert_reason}")
            for callback in self.alert_callbacks:
                try:
                    callback(health_report)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
    
    def get_current_health(self) -> SystemHealth:
        """Get current health status."""
        return self.current_health
    
    def get_health_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get health summary for the specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        
        recent_health = [
            h for h in self.health_history
            if h.timestamp > cutoff_time
        ]
        
        if not recent_health:
            return {"status": "no_data", "period_hours": hours}
        
        # Calculate statistics
        status_counts = {}
        metric_stats = {}
        
        for health in recent_health:
            # Count status occurrences
            status = health.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            
            # Collect metric statistics
            for name, metric in health.metrics.items():
                if name not in metric_stats:
                    metric_stats[name] = []
                metric_stats[name].append(metric.value)
        
        # Calculate metric averages and trends
        for name, values in metric_stats.items():
            if values:
                metric_stats[name] = {
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values),
                    'trend': self._calculate_trend(name)
                }
        
        return {
            "period_hours": hours,
            "current_status": self.current_health.status.value,
            "status_distribution": status_counts,
            "metric_statistics": metric_stats,
            "total_checks": len(recent_health),
            "last_update": self.current_health.timestamp
        }
    
    def export_health_data(self, filepath: str):
        """Export health data to JSON file."""
        try:
            export_data = {
                "current_health": {
                    "status": self.current_health.status.value,
                    "timestamp": self.current_health.timestamp,
                    "metrics": {
                        name: {
                            "value": metric.value,
                            "status": metric.status.value,
                            "unit": metric.unit,
                            "description": metric.description
                        }
                        for name, metric in self.current_health.metrics.items()
                    },
                    "issues": self.current_health.issues,
                    "recommendations": self.current_health.recommendations
                },
                "summary": self.get_health_summary(),
                "export_timestamp": time.time()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Health data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export health data: {e}")

# Default health checks for common system metrics
def create_memory_health_check(warning_threshold: float = 80.0, 
                              critical_threshold: float = 95.0) -> Callable[[], HealthMetric]:
    """Create a memory usage health check."""
    
    def memory_check():
        try:
            # Try psutil first
            try:
                import psutil
                memory_percent = psutil.virtual_memory().percent
            except ImportError:
                # Fallback to basic system check
                import os
                # Simple heuristic based on available memory
                try:
                    with open('/proc/meminfo', 'r') as f:
                        lines = f.readlines()
                    
                    mem_total = int([l for l in lines if 'MemTotal' in l][0].split()[1])
                    mem_available = int([l for l in lines if 'MemAvailable' in l][0].split()[1])
                    memory_percent = ((mem_total - mem_available) / mem_total) * 100
                except:
                    memory_percent = 50.0  # Default assumption
            
            # Determine status
            if memory_percent >= critical_threshold:
                status = HealthStatus.CRITICAL
            elif memory_percent >= warning_threshold:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY
            
            return HealthMetric(
                name="memory_usage",
                value=memory_percent,
                status=status,
                threshold_warning=warning_threshold,
                threshold_critical=critical_threshold,
                unit="%",
                description=f"Memory usage at {memory_percent:.1f}%"
            )
            
        except Exception as e:
            return HealthMetric(
                name="memory_usage",
                value=0.0,
                status=HealthStatus.CRITICAL,
                description=f"Memory check failed: {e}"
            )
    
    return memory_check

def create_disk_health_check(path: str = "/", 
                           warning_threshold: float = 80.0,
                           critical_threshold: float = 95.0) -> Callable[[], HealthMetric]:
    """Create a disk usage health check."""
    
    def disk_check():
        try:
            import shutil
            total, used, free = shutil.disk_usage(path)
            disk_percent = (used / total) * 100
            
            # Determine status
            if disk_percent >= critical_threshold:
                status = HealthStatus.CRITICAL
            elif disk_percent >= warning_threshold:
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY
            
            return HealthMetric(
                name="disk_usage",
                value=disk_percent,
                status=status,
                threshold_warning=warning_threshold,
                threshold_critical=critical_threshold,
                unit="%",
                description=f"Disk usage at {disk_percent:.1f}% ({path})"
            )
            
        except Exception as e:
            return HealthMetric(
                name="disk_usage",
                value=0.0,
                status=HealthStatus.CRITICAL,
                description=f"Disk check failed: {e}"
            )
    
    return disk_check

# Global health monitor instance
global_health_monitor = HealthMonitor()

# Register default health checks
global_health_monitor.register_health_check("memory", create_memory_health_check())
global_health_monitor.register_health_check("disk", create_disk_health_check())