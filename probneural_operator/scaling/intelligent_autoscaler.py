"""
Intelligent auto-scaling system for probabilistic neural operators.

This module provides advanced auto-scaling capabilities including:
- Predictive scaling based on load patterns
- Multi-dimensional resource optimization
- Cost-aware scaling decisions
- Horizontal and vertical scaling
- Real-time load balancing
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import math
from collections import deque, defaultdict
import statistics

logger = logging.getLogger(__name__)

class ScalingDirection(Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"

class ScalingType(Enum):
    """Type of scaling."""
    HORIZONTAL = "horizontal"  # Add/remove instances
    VERTICAL = "vertical"     # Increase/decrease resources per instance

class ResourceMetric(Enum):
    """Resource metrics for scaling decisions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    GPU_UTILIZATION = "gpu_utilization"
    REQUEST_RATE = "request_rate"
    RESPONSE_TIME = "response_time"
    QUEUE_LENGTH = "queue_length"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"

@dataclass
class ScalingEvent:
    """Record of a scaling event."""
    timestamp: float
    scaling_type: ScalingType
    scaling_direction: ScalingDirection
    resource_metric: ResourceMetric
    before_value: float
    after_value: float
    instances_before: int
    instances_after: int
    reason: str
    cost_impact: float = 0.0

@dataclass
class ScalingRule:
    """Scaling rule configuration."""
    metric: ResourceMetric
    scale_up_threshold: float
    scale_down_threshold: float
    scaling_type: ScalingType
    cooldown_period: float = 300.0  # 5 minutes
    min_instances: int = 1
    max_instances: int = 100
    step_size: int = 1
    evaluation_periods: int = 3
    priority: int = 1

@dataclass
class LoadPattern:
    """Detected load pattern."""
    pattern_type: str
    confidence: float
    prediction_horizon: float
    expected_peak_time: Optional[float] = None
    expected_peak_value: Optional[float] = None
    seasonality_period: Optional[float] = None

class IntelligentAutoScaler:
    """Intelligent auto-scaling system with predictive capabilities."""
    
    def __init__(self,
                 min_instances: int = 1,
                 max_instances: int = 100,
                 target_utilization: float = 70.0,
                 prediction_enabled: bool = True,
                 cost_optimization_enabled: bool = True):
        """Initialize intelligent auto-scaler.
        
        Args:
            min_instances: Minimum number of instances
            max_instances: Maximum number of instances
            target_utilization: Target utilization percentage
            prediction_enabled: Whether to enable predictive scaling
            cost_optimization_enabled: Whether to consider costs in scaling decisions
        """
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.target_utilization = target_utilization
        self.prediction_enabled = prediction_enabled
        self.cost_optimization_enabled = cost_optimization_enabled
        
        # Current state
        self.current_instances = min_instances
        self.current_metrics = {}
        self.scaling_rules = []
        self.scaling_history = deque(maxlen=1000)
        
        # Prediction system
        self.metric_history = defaultdict(lambda: deque(maxlen=1440))  # 24 hours of minutes
        self.load_patterns = []
        self.prediction_cache = {}
        
        # Cost tracking
        self.cost_per_instance_hour = 1.0  # Default cost
        self.total_cost = 0.0
        self.cost_savings = 0.0
        
        # Control
        self.scaling_enabled = True
        self.last_scaling_time = {}
        self.lock = threading.Lock()
        
        # Initialize default scaling rules
        self._initialize_default_rules()
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
    
    def _initialize_default_rules(self):
        """Initialize default scaling rules."""
        default_rules = [
            ScalingRule(
                metric=ResourceMetric.CPU_UTILIZATION,
                scale_up_threshold=80.0,
                scale_down_threshold=30.0,
                scaling_type=ScalingType.HORIZONTAL,
                cooldown_period=300.0,
                priority=1
            ),
            ScalingRule(
                metric=ResourceMetric.MEMORY_UTILIZATION,
                scale_up_threshold=85.0,
                scale_down_threshold=40.0,
                scaling_type=ScalingType.HORIZONTAL,
                cooldown_period=300.0,
                priority=2
            ),
            ScalingRule(
                metric=ResourceMetric.RESPONSE_TIME,
                scale_up_threshold=2.0,  # 2 seconds
                scale_down_threshold=0.5,  # 0.5 seconds
                scaling_type=ScalingType.HORIZONTAL,
                cooldown_period=180.0,
                priority=3
            ),
            ScalingRule(
                metric=ResourceMetric.QUEUE_LENGTH,
                scale_up_threshold=50.0,
                scale_down_threshold=5.0,
                scaling_type=ScalingType.HORIZONTAL,
                cooldown_period=120.0,
                priority=4
            )
        ]
        
        self.scaling_rules.extend(default_rules)
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a custom scaling rule."""
        with self.lock:
            self.scaling_rules.append(rule)
            # Sort by priority
            self.scaling_rules.sort(key=lambda r: r.priority)
    
    def update_metrics(self, metrics: Dict[ResourceMetric, float]):
        """Update current resource metrics."""
        current_time = time.time()
        
        with self.lock:
            self.current_metrics = metrics.copy()
            
            # Store in history for pattern analysis
            for metric, value in metrics.items():
                self.metric_history[metric].append((current_time, value))
            
            # Trigger scaling evaluation
            if self.scaling_enabled:
                self._evaluate_scaling_decisions()
    
    def _monitoring_loop(self):
        """Main monitoring loop for pattern detection and prediction."""
        while self.monitoring_active:
            try:
                # Detect load patterns
                self._detect_load_patterns()
                
                # Update predictions
                if self.prediction_enabled:
                    self._update_predictions()
                
                # Proactive scaling based on predictions
                self._evaluate_predictive_scaling()
                
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"Auto-scaler monitoring error: {e}")
                time.sleep(60)
    
    def _evaluate_scaling_decisions(self):
        """Evaluate whether scaling is needed based on current metrics."""
        current_time = time.time()
        
        for rule in self.scaling_rules:
            if rule.metric not in self.current_metrics:
                continue
            
            # Check cooldown period
            if rule.metric in self.last_scaling_time:
                time_since_last = current_time - self.last_scaling_time[rule.metric]
                if time_since_last < rule.cooldown_period:
                    continue
            
            metric_value = self.current_metrics[rule.metric]
            
            # Evaluate scaling decision
            scaling_decision = self._make_scaling_decision(rule, metric_value)
            
            if scaling_decision != ScalingDirection.STABLE:
                self._execute_scaling(rule, scaling_decision, metric_value, "reactive")
    
    def _make_scaling_decision(self, rule: ScalingRule, metric_value: float) -> ScalingDirection:
        """Make scaling decision based on rule and metric value."""
        
        # Get recent metric values for evaluation periods
        recent_values = []
        current_time = time.time()
        
        for timestamp, value in list(self.metric_history[rule.metric])[-rule.evaluation_periods:]:
            if current_time - timestamp <= rule.evaluation_periods * 60:  # Last N minutes
                recent_values.append(value)
        
        if len(recent_values) < rule.evaluation_periods:
            return ScalingDirection.STABLE
        
        # Calculate average over evaluation periods
        avg_value = statistics.mean(recent_values)
        
        # Make decision based on thresholds
        if avg_value >= rule.scale_up_threshold:
            if self.current_instances < rule.max_instances:
                return ScalingDirection.UP
        elif avg_value <= rule.scale_down_threshold:
            if self.current_instances > rule.min_instances:
                return ScalingDirection.DOWN
        
        return ScalingDirection.STABLE
    
    def _execute_scaling(self, 
                        rule: ScalingRule, 
                        direction: ScalingDirection, 
                        metric_value: float,
                        reason: str):
        """Execute scaling action."""
        current_time = time.time()
        
        with self.lock:
            instances_before = self.current_instances
            
            if direction == ScalingDirection.UP:
                new_instances = min(
                    self.current_instances + rule.step_size,
                    rule.max_instances,
                    self.max_instances
                )
            else:  # ScalingDirection.DOWN
                new_instances = max(
                    self.current_instances - rule.step_size,
                    rule.min_instances,
                    self.min_instances
                )
            
            # Check if scaling is actually needed
            if new_instances == self.current_instances:
                return
            
            # Consider cost impact if enabled
            if self.cost_optimization_enabled:
                cost_impact = self._calculate_cost_impact(instances_before, new_instances)
                
                # Skip scaling if cost impact is too high for minor performance gain
                if direction == ScalingDirection.UP and cost_impact > 100.0:  # $100/hour threshold
                    if metric_value < rule.scale_up_threshold * 1.2:  # Not critically overloaded
                        logger.info(f"Skipping scale-up due to cost impact: ${cost_impact:.2f}/hour")
                        return
            else:
                cost_impact = 0.0
            
            # Execute scaling
            success = self._perform_scaling(instances_before, new_instances)
            
            if success:
                self.current_instances = new_instances
                self.last_scaling_time[rule.metric] = current_time
                
                # Record scaling event
                scaling_event = ScalingEvent(
                    timestamp=current_time,
                    scaling_type=rule.scaling_type,
                    scaling_direction=direction,
                    resource_metric=rule.metric,
                    before_value=metric_value,
                    after_value=metric_value,  # Will be updated after scaling takes effect
                    instances_before=instances_before,
                    instances_after=new_instances,
                    reason=reason,
                    cost_impact=cost_impact
                )
                
                self.scaling_history.append(scaling_event)
                
                # Update cost tracking
                self.total_cost += abs(cost_impact)
                if direction == ScalingDirection.DOWN:
                    self.cost_savings += abs(cost_impact)
                
                logger.info(
                    f"Scaling {direction.value}: {instances_before} -> {new_instances} instances "
                    f"(reason: {reason}, metric: {rule.metric.value}={metric_value:.2f})"
                )
    
    def _perform_scaling(self, current_instances: int, target_instances: int) -> bool:
        """Perform the actual scaling operation.
        
        This is a placeholder that should be implemented based on your infrastructure.
        For example, it might call AWS Auto Scaling, Kubernetes HPA, or custom scaling logic.
        """
        # Simulate scaling operation
        logger.info(f"Scaling from {current_instances} to {target_instances} instances")
        
        # In a real implementation, this would:
        # 1. Call infrastructure APIs to add/remove instances
        # 2. Wait for instances to be ready
        # 3. Update load balancer configuration
        # 4. Return success/failure status
        
        return True  # Assume success for simulation
    
    def _calculate_cost_impact(self, instances_before: int, instances_after: int) -> float:
        """Calculate cost impact of scaling decision."""
        instance_diff = instances_after - instances_before
        hourly_cost_impact = instance_diff * self.cost_per_instance_hour
        return hourly_cost_impact
    
    def _detect_load_patterns(self):
        """Detect patterns in load metrics for predictive scaling."""
        if not self.metric_history:
            return
        
        # Analyze patterns for each metric
        detected_patterns = []
        
        for metric, history in self.metric_history.items():
            if len(history) < 60:  # Need at least 1 hour of data
                continue
            
            values = [value for _, value in history]
            timestamps = [timestamp for timestamp, _ in history]
            
            # Detect daily patterns
            daily_pattern = self._detect_daily_pattern(timestamps, values)
            if daily_pattern:
                detected_patterns.append(daily_pattern)
            
            # Detect trend patterns
            trend_pattern = self._detect_trend_pattern(values)
            if trend_pattern:
                detected_patterns.append(trend_pattern)
            
            # Detect spike patterns
            spike_pattern = self._detect_spike_pattern(values)
            if spike_pattern:
                detected_patterns.append(spike_pattern)
        
        # Update load patterns
        with self.lock:
            self.load_patterns = detected_patterns
    
    def _detect_daily_pattern(self, timestamps: List[float], values: List[float]) -> Optional[LoadPattern]:
        """Detect daily recurring patterns."""
        if len(timestamps) < 1440:  # Less than 24 hours
            return None
        
        # Group by hour of day
        hourly_avg = defaultdict(list)
        
        for timestamp, value in zip(timestamps, values):
            hour = int((timestamp % 86400) // 3600)  # Hour of day
            hourly_avg[hour].append(value)
        
        # Calculate average for each hour
        hourly_pattern = {}
        for hour, hour_values in hourly_avg.items():
            if hour_values:
                hourly_pattern[hour] = statistics.mean(hour_values)
        
        if len(hourly_pattern) < 12:  # Not enough data
            return None
        
        # Find peak hour
        peak_hour = max(hourly_pattern.keys(), key=lambda h: hourly_pattern[h])
        peak_value = hourly_pattern[peak_hour]
        
        # Calculate confidence based on consistency
        if len(hourly_pattern) >= 20:  # Good data coverage
            values_list = list(hourly_pattern.values())
            std_dev = statistics.stdev(values_list)
            mean_value = statistics.mean(values_list)
            coefficient_of_variation = std_dev / mean_value if mean_value > 0 else 0
            confidence = max(0.1, 1.0 - coefficient_of_variation)
        else:
            confidence = 0.3
        
        return LoadPattern(
            pattern_type="daily_recurring",
            confidence=confidence,
            prediction_horizon=86400.0,  # 24 hours
            expected_peak_time=peak_hour * 3600,  # Convert to seconds
            expected_peak_value=peak_value,
            seasonality_period=86400.0
        )
    
    def _detect_trend_pattern(self, values: List[float]) -> Optional[LoadPattern]:
        """Detect trending patterns."""
        if len(values) < 30:  # Need at least 30 data points
            return None
        
        # Simple linear regression to detect trend
        n = len(values)
        x = list(range(n))
        y = values
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi * xi for xi in x)
        
        # Calculate slope
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # Calculate correlation coefficient for confidence
        mean_x = sum_x / n
        mean_y = sum_y / n
        
        numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        denom_x = sum((xi - mean_x) ** 2 for xi in x)
        denom_y = sum((yi - mean_y) ** 2 for yi in y)
        
        correlation = numerator / math.sqrt(denom_x * denom_y) if denom_x > 0 and denom_y > 0 else 0
        confidence = abs(correlation)
        
        if confidence < 0.5:  # Weak correlation
            return None
        
        pattern_type = "upward_trend" if slope > 0 else "downward_trend"
        
        return LoadPattern(
            pattern_type=pattern_type,
            confidence=confidence,
            prediction_horizon=1800.0,  # 30 minutes
            expected_peak_value=values[-1] + slope * 30 if slope > 0 else None
        )
    
    def _detect_spike_pattern(self, values: List[float]) -> Optional[LoadPattern]:
        """Detect spike patterns."""
        if len(values) < 10:
            return None
        
        # Calculate moving average and standard deviation
        window_size = min(10, len(values) // 3)
        recent_values = values[-window_size:]
        
        mean_value = statistics.mean(recent_values)
        std_dev = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
        
        # Detect if current value is significantly higher than recent average
        current_value = values[-1]
        threshold = mean_value + 2 * std_dev  # 2 standard deviations
        
        if current_value > threshold and std_dev > 0:
            confidence = min(1.0, (current_value - mean_value) / (3 * std_dev))
            
            return LoadPattern(
                pattern_type="sudden_spike",
                confidence=confidence,
                prediction_horizon=300.0,  # 5 minutes
                expected_peak_value=current_value
            )
        
        return None
    
    def _update_predictions(self):
        """Update scaling predictions based on detected patterns."""
        current_time = time.time()
        
        for pattern in self.load_patterns:
            if pattern.confidence < 0.5:
                continue
            
            prediction_key = f"{pattern.pattern_type}_{pattern.expected_peak_time}"
            
            # Generate prediction
            if pattern.pattern_type == "daily_recurring" and pattern.expected_peak_time:
                # Predict when next peak will occur
                current_hour = (current_time % 86400) // 3600
                peak_hour = pattern.expected_peak_time // 3600
                
                if peak_hour > current_hour:
                    time_to_peak = (peak_hour - current_hour) * 3600
                else:
                    time_to_peak = (24 - current_hour + peak_hour) * 3600
                
                if time_to_peak <= 3600:  # Peak within next hour
                    self.prediction_cache[prediction_key] = {
                        'predicted_time': current_time + time_to_peak,
                        'predicted_value': pattern.expected_peak_value,
                        'confidence': pattern.confidence,
                        'action': 'scale_up_proactive'
                    }
            
            elif pattern.pattern_type in ["upward_trend", "sudden_spike"]:
                # Predict continued increase
                prediction_time = current_time + 300  # 5 minutes ahead
                
                self.prediction_cache[prediction_key] = {
                    'predicted_time': prediction_time,
                    'predicted_value': pattern.expected_peak_value,
                    'confidence': pattern.confidence,
                    'action': 'scale_up_proactive'
                }
    
    def _evaluate_predictive_scaling(self):
        """Evaluate whether predictive scaling should be triggered."""
        if not self.prediction_enabled or not self.prediction_cache:
            return
        
        current_time = time.time()
        
        for prediction_key, prediction in list(self.prediction_cache.items()):
            time_to_event = prediction['predicted_time'] - current_time
            
            # Remove old predictions
            if time_to_event < -300:  # 5 minutes past
                del self.prediction_cache[prediction_key]
                continue
            
            # Trigger proactive scaling if event is imminent
            if 0 <= time_to_event <= 900 and prediction['confidence'] > 0.7:  # Next 15 minutes, high confidence
                self._trigger_predictive_scaling(prediction)
                del self.prediction_cache[prediction_key]
    
    def _trigger_predictive_scaling(self, prediction: Dict[str, Any]):
        """Trigger predictive scaling based on prediction."""
        if prediction['action'] == 'scale_up_proactive':
            # Calculate how many instances we might need
            predicted_load_ratio = prediction['predicted_value'] / self.target_utilization
            target_instances = min(
                math.ceil(self.current_instances * predicted_load_ratio),
                self.max_instances
            )
            
            if target_instances > self.current_instances:
                # Create a synthetic scaling rule for predictive scaling
                predictive_rule = ScalingRule(
                    metric=ResourceMetric.CPU_UTILIZATION,  # Use as default
                    scale_up_threshold=self.target_utilization,
                    scale_down_threshold=0,
                    scaling_type=ScalingType.HORIZONTAL,
                    step_size=target_instances - self.current_instances,
                    max_instances=self.max_instances
                )
                
                self._execute_scaling(
                    predictive_rule,
                    ScalingDirection.UP,
                    prediction['predicted_value'],
                    f"predictive (confidence: {prediction['confidence']:.2f})"
                )
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and metrics."""
        with self.lock:
            recent_events = list(self.scaling_history)[-10:]  # Last 10 events
            
            return {
                "current_instances": self.current_instances,
                "min_instances": self.min_instances,
                "max_instances": self.max_instances,
                "target_utilization": self.target_utilization,
                "scaling_enabled": self.scaling_enabled,
                "prediction_enabled": self.prediction_enabled,
                "current_metrics": dict(self.current_metrics),
                "recent_scaling_events": [
                    {
                        "timestamp": event.timestamp,
                        "type": event.scaling_type.value,
                        "direction": event.scaling_direction.value,
                        "instances_before": event.instances_before,
                        "instances_after": event.instances_after,
                        "reason": event.reason,
                        "cost_impact": event.cost_impact
                    }
                    for event in recent_events
                ],
                "detected_patterns": [
                    {
                        "type": pattern.pattern_type,
                        "confidence": pattern.confidence,
                        "prediction_horizon": pattern.prediction_horizon
                    }
                    for pattern in self.load_patterns
                ],
                "total_cost": self.total_cost,
                "cost_savings": self.cost_savings,
                "scaling_rules_count": len(self.scaling_rules)
            }
    
    def stop(self):
        """Stop the auto-scaler."""
        self.monitoring_active = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)

# Global auto-scaler instance
global_autoscaler = IntelligentAutoScaler()

# Convenience functions
def update_load_metrics(cpu: float, memory: float, response_time: float, queue_length: float = 0):
    """Update load metrics for auto-scaling decisions."""
    metrics = {
        ResourceMetric.CPU_UTILIZATION: cpu,
        ResourceMetric.MEMORY_UTILIZATION: memory,
        ResourceMetric.RESPONSE_TIME: response_time,
        ResourceMetric.QUEUE_LENGTH: queue_length
    }
    global_autoscaler.update_metrics(metrics)

def get_current_capacity():
    """Get current system capacity."""
    status = global_autoscaler.get_scaling_status()
    return status["current_instances"]