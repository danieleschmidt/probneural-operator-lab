"""Auto-scaling and load balancing for dynamic resource management."""

import torch
import torch.nn as nn
import time
import threading
import queue
import psutil
import logging
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import json
import socket
import subprocess
from enum import Enum


class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down" 
    MAINTAIN = "maintain"
    EMERGENCY_SCALE = "emergency_scale"


@dataclass
class ResourceMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    gpu_memory_percent: Dict[int, float] = field(default_factory=dict)
    gpu_utilization: Dict[int, float] = field(default_factory=dict)
    active_requests: int = 0
    request_latency_ms: float = 0.0
    throughput_rps: float = 0.0
    error_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'gpu_memory_percent': self.gpu_memory_percent,
            'gpu_utilization': self.gpu_utilization,
            'active_requests': self.active_requests,
            'request_latency_ms': self.request_latency_ms,
            'throughput_rps': self.throughput_rps,
            'error_rate': self.error_rate
        }


@dataclass 
class ScalingPolicy:
    """Configuration for auto-scaling behavior."""
    # Thresholds for scaling decisions
    cpu_scale_up_threshold: float = 80.0
    cpu_scale_down_threshold: float = 30.0
    memory_scale_up_threshold: float = 85.0
    memory_scale_down_threshold: float = 40.0
    gpu_memory_scale_up_threshold: float = 90.0
    gpu_memory_scale_down_threshold: float = 50.0
    
    # Request-based thresholds
    latency_scale_up_threshold_ms: float = 1000.0
    latency_scale_down_threshold_ms: float = 200.0
    throughput_scale_up_threshold: float = 0.8  # 80% of capacity
    error_rate_scale_up_threshold: float = 0.05  # 5% error rate
    
    # Scaling parameters
    min_instances: int = 1
    max_instances: int = 10
    scale_up_count: int = 1
    scale_down_count: int = 1
    cooldown_period_seconds: float = 300.0  # 5 minutes
    
    # Emergency scaling
    emergency_cpu_threshold: float = 95.0
    emergency_memory_threshold: float = 95.0
    emergency_scale_count: int = 3


class ResourceMonitor:
    """Monitor system resources and application metrics."""
    
    def __init__(self, 
                 monitoring_interval: float = 10.0,
                 history_size: int = 100):
        """Initialize resource monitor.
        
        Args:
            monitoring_interval: Monitoring interval in seconds
            history_size: Number of metric samples to keep in history
        """
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        
        self._metrics_history: deque = deque(maxlen=history_size)
        self._is_monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._callbacks: List[Callable[[ResourceMetrics], None]] = []
        self._lock = threading.Lock()
        
        # Request tracking
        self._active_requests = 0
        self._request_times: deque = deque(maxlen=1000)
        self._error_count = 0
        self._request_count = 0
        
    def add_callback(self, callback: Callable[[ResourceMetrics], None]):
        """Add callback to be called when new metrics are available."""
        self._callbacks.append(callback)
    
    def start_monitoring(self):
        """Start resource monitoring."""
        if self._is_monitoring:
            return
        
        self._is_monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        logging.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self._is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        logging.info("Resource monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._is_monitoring:
            try:
                metrics = self._collect_metrics()
                
                with self._lock:
                    self._metrics_history.append(metrics)
                
                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logging.error(f"Monitor callback error: {e}")
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logging.error(f"Monitoring loop error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # GPU metrics
        gpu_memory_percent = {}
        gpu_utilization = {}
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    # Memory usage
                    allocated = torch.cuda.memory_allocated(i)
                    total = torch.cuda.get_device_properties(i).total_memory
                    gpu_memory_percent[i] = (allocated / total) * 100 if total > 0 else 0
                    
                    # GPU utilization (approximated)
                    gpu_utilization[i] = min(100, gpu_memory_percent[i] * 1.2)
                    
                except Exception as e:
                    logging.warning(f"Error collecting GPU {i} metrics: {e}")
                    gpu_memory_percent[i] = 0
                    gpu_utilization[i] = 0
        
        # Request metrics
        with self._lock:
            current_time = time.time()
            
            # Calculate recent latency
            recent_times = [t for t in self._request_times 
                           if current_time - t['end_time'] < 60]  # Last minute
            avg_latency = np.mean([t['duration'] for t in recent_times]) * 1000 if recent_times else 0
            
            # Calculate throughput (requests per second)
            recent_requests = len(recent_times)
            throughput = recent_requests / 60.0 if recent_requests > 0 else 0
            
            # Calculate error rate
            error_rate = (self._error_count / max(1, self._request_count)) if self._request_count > 0 else 0
        
        return ResourceMetrics(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            gpu_memory_percent=gpu_memory_percent,
            gpu_utilization=gpu_utilization,
            active_requests=self._active_requests,
            request_latency_ms=avg_latency,
            throughput_rps=throughput,
            error_rate=error_rate
        )
    
    def record_request_start(self) -> str:
        """Record the start of a request."""
        request_id = str(time.time())
        with self._lock:
            self._active_requests += 1
            self._request_count += 1
        return request_id
    
    def record_request_end(self, request_id: str, success: bool = True):
        """Record the end of a request."""
        end_time = time.time()
        start_time = float(request_id)
        duration = end_time - start_time
        
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)
            self._request_times.append({
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration
            })
            
            if not success:
                self._error_count += 1
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get the most recent metrics."""
        with self._lock:
            return self._metrics_history[-1] if self._metrics_history else None
    
    def get_metrics_history(self, 
                           duration_seconds: Optional[float] = None) -> List[ResourceMetrics]:
        """Get metrics history."""
        with self._lock:
            if duration_seconds is None:
                return list(self._metrics_history)
            
            cutoff_time = time.time() - duration_seconds
            return [m for m in self._metrics_history if m.timestamp >= cutoff_time]


class AutoScaler:
    """Automatic scaling based on system metrics and policies."""
    
    def __init__(self, 
                 policy: ScalingPolicy,
                 resource_monitor: ResourceMonitor):
        """Initialize auto-scaler.
        
        Args:
            policy: Scaling policy configuration
            resource_monitor: Resource monitoring instance
        """
        self.policy = policy
        self.resource_monitor = resource_monitor
        
        self._current_instances = policy.min_instances
        self._last_scaling_action_time = 0.0
        self._scaling_history: List[Dict[str, Any]] = []
        self._is_active = False
        
        # Register with resource monitor
        self.resource_monitor.add_callback(self._on_metrics_update)
    
    def start(self):
        """Start auto-scaling."""
        self._is_active = True
        logging.info("Auto-scaling started")
    
    def stop(self):
        """Stop auto-scaling."""
        self._is_active = False
        logging.info("Auto-scaling stopped")
    
    def _on_metrics_update(self, metrics: ResourceMetrics):
        """Handle new metrics from resource monitor."""
        if not self._is_active:
            return
        
        try:
            scaling_decision = self._make_scaling_decision(metrics)
            
            if scaling_decision != ScalingAction.MAINTAIN:
                self._execute_scaling_action(scaling_decision, metrics)
        
        except Exception as e:
            logging.error(f"Auto-scaling error: {e}")
    
    def _make_scaling_decision(self, metrics: ResourceMetrics) -> ScalingAction:
        """Make scaling decision based on current metrics."""
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self._last_scaling_action_time < self.policy.cooldown_period_seconds:
            return ScalingAction.MAINTAIN
        
        # Check for emergency conditions
        if (metrics.cpu_percent >= self.policy.emergency_cpu_threshold or
            metrics.memory_percent >= self.policy.emergency_memory_threshold):
            return ScalingAction.EMERGENCY_SCALE
        
        # Count scale-up triggers
        scale_up_triggers = 0
        scale_down_triggers = 0
        
        # CPU-based decisions
        if metrics.cpu_percent >= self.policy.cpu_scale_up_threshold:
            scale_up_triggers += 1
        elif metrics.cpu_percent <= self.policy.cpu_scale_down_threshold:
            scale_down_triggers += 1
        
        # Memory-based decisions
        if metrics.memory_percent >= self.policy.memory_scale_up_threshold:
            scale_up_triggers += 1
        elif metrics.memory_percent <= self.policy.memory_scale_down_threshold:
            scale_down_triggers += 1
        
        # GPU memory-based decisions
        if metrics.gpu_memory_percent:
            max_gpu_memory = max(metrics.gpu_memory_percent.values())
            if max_gpu_memory >= self.policy.gpu_memory_scale_up_threshold:
                scale_up_triggers += 1
            elif max_gpu_memory <= self.policy.gpu_memory_scale_down_threshold:
                scale_down_triggers += 1
        
        # Request latency-based decisions
        if metrics.request_latency_ms >= self.policy.latency_scale_up_threshold_ms:
            scale_up_triggers += 1
        elif metrics.request_latency_ms <= self.policy.latency_scale_down_threshold_ms:
            scale_down_triggers += 1
        
        # Error rate-based decisions
        if metrics.error_rate >= self.policy.error_rate_scale_up_threshold:
            scale_up_triggers += 2  # Weight errors more heavily
        
        # Make decision based on trigger counts
        if scale_up_triggers >= 2 and self._current_instances < self.policy.max_instances:
            return ScalingAction.SCALE_UP
        elif scale_down_triggers >= 2 and self._current_instances > self.policy.min_instances:
            return ScalingAction.SCALE_DOWN
        else:
            return ScalingAction.MAINTAIN
    
    def _execute_scaling_action(self, action: ScalingAction, metrics: ResourceMetrics):
        """Execute scaling action."""
        current_time = time.time()
        
        if action == ScalingAction.SCALE_UP:
            new_instances = min(
                self._current_instances + self.policy.scale_up_count,
                self.policy.max_instances
            )
        elif action == ScalingAction.SCALE_DOWN:
            new_instances = max(
                self._current_instances - self.policy.scale_down_count,
                self.policy.min_instances
            )
        elif action == ScalingAction.EMERGENCY_SCALE:
            new_instances = min(
                self._current_instances + self.policy.emergency_scale_count,
                self.policy.max_instances
            )
        else:
            return
        
        if new_instances != self._current_instances:
            # Record scaling action
            scaling_record = {
                'timestamp': current_time,
                'action': action.value,
                'old_instances': self._current_instances,
                'new_instances': new_instances,
                'trigger_metrics': metrics.to_dict(),
                'reason': self._get_scaling_reason(action, metrics)
            }
            
            self._scaling_history.append(scaling_record)
            
            # Update state
            old_instances = self._current_instances
            self._current_instances = new_instances
            self._last_scaling_action_time = current_time
            
            # Log scaling action
            logging.info(f"Scaling action: {action.value} from {old_instances} to {new_instances} instances")
            
            # Execute scaling (this would integrate with container orchestration)
            self._perform_scaling(old_instances, new_instances, action)
    
    def _get_scaling_reason(self, action: ScalingAction, metrics: ResourceMetrics) -> str:
        """Get human-readable reason for scaling decision."""
        reasons = []
        
        if metrics.cpu_percent >= self.policy.cpu_scale_up_threshold:
            reasons.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent >= self.policy.memory_scale_up_threshold:
            reasons.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.gpu_memory_percent:
            max_gpu_memory = max(metrics.gpu_memory_percent.values())
            if max_gpu_memory >= self.policy.gpu_memory_scale_up_threshold:
                reasons.append(f"High GPU memory usage: {max_gpu_memory:.1f}%")
        
        if metrics.request_latency_ms >= self.policy.latency_scale_up_threshold_ms:
            reasons.append(f"High latency: {metrics.request_latency_ms:.1f}ms")
        
        if metrics.error_rate >= self.policy.error_rate_scale_up_threshold:
            reasons.append(f"High error rate: {metrics.error_rate:.2%}")
        
        return "; ".join(reasons) if reasons else f"Triggered by {action.value} policy"
    
    def _perform_scaling(self, old_count: int, new_count: int, action: ScalingAction):
        """Perform the actual scaling operation."""
        # This would integrate with container orchestration systems
        # For now, we just log the action
        
        if new_count > old_count:
            instances_to_add = new_count - old_count
            logging.info(f"Would scale up by {instances_to_add} instances")
            # In production: call container orchestration API to add instances
        
        elif new_count < old_count:
            instances_to_remove = old_count - new_count
            logging.info(f"Would scale down by {instances_to_remove} instances")
            # In production: call container orchestration API to remove instances
    
    def get_current_instances(self) -> int:
        """Get current number of instances."""
        return self._current_instances
    
    def get_scaling_history(self) -> List[Dict[str, Any]]:
        """Get scaling history."""
        return self._scaling_history.copy()
    
    def force_scaling(self, target_instances: int) -> bool:
        """Force scaling to specific number of instances."""
        if target_instances < self.policy.min_instances or target_instances > self.policy.max_instances:
            return False
        
        current_time = time.time()
        scaling_record = {
            'timestamp': current_time,
            'action': 'manual_override',
            'old_instances': self._current_instances,
            'new_instances': target_instances,
            'trigger_metrics': None,
            'reason': 'Manual override'
        }
        
        self._scaling_history.append(scaling_record)
        self._perform_scaling(self._current_instances, target_instances, ScalingAction.MAINTAIN)
        self._current_instances = target_instances
        self._last_scaling_action_time = current_time
        
        return True


class LoadBalancer:
    """Load balancer for distributing requests across model instances."""
    
    def __init__(self, 
                 balancing_strategy: str = "round_robin",
                 health_check_interval: float = 30.0):
        """Initialize load balancer.
        
        Args:
            balancing_strategy: Strategy for load balancing ("round_robin", "least_connections", "weighted")
            health_check_interval: Health check interval in seconds
        """
        self.balancing_strategy = balancing_strategy
        self.health_check_interval = health_check_interval
        
        self._instances: List[Dict[str, Any]] = []
        self._current_index = 0
        self._lock = threading.Lock()
        
        # Health checking
        self._health_check_thread: Optional[threading.Thread] = None
        self._is_health_checking = False
        
        # Load balancing metrics
        self._request_counts = defaultdict(int)
        self._response_times = defaultdict(list)
        self._error_counts = defaultdict(int)
    
    def register_instance(self, 
                         instance_id: str,
                         endpoint: str,
                         weight: float = 1.0,
                         metadata: Optional[Dict[str, Any]] = None):
        """Register a model instance."""
        with self._lock:
            instance = {
                'id': instance_id,
                'endpoint': endpoint,
                'weight': weight,
                'metadata': metadata or {},
                'healthy': True,
                'last_health_check': time.time(),
                'connections': 0,
                'total_requests': 0,
                'total_errors': 0,
                'avg_response_time': 0.0
            }
            
            self._instances.append(instance)
            
        logging.info(f"Registered instance {instance_id} at {endpoint}")
    
    def unregister_instance(self, instance_id: str):
        """Unregister a model instance."""
        with self._lock:
            self._instances = [inst for inst in self._instances if inst['id'] != instance_id]
        
        logging.info(f"Unregistered instance {instance_id}")
    
    def get_next_instance(self) -> Optional[Dict[str, Any]]:
        """Get next instance according to load balancing strategy."""
        with self._lock:
            healthy_instances = [inst for inst in self._instances if inst['healthy']]
            
            if not healthy_instances:
                return None
            
            if self.balancing_strategy == "round_robin":
                instance = healthy_instances[self._current_index % len(healthy_instances)]
                self._current_index = (self._current_index + 1) % len(healthy_instances)
                
            elif self.balancing_strategy == "least_connections":
                instance = min(healthy_instances, key=lambda x: x['connections'])
                
            elif self.balancing_strategy == "weighted":
                # Weighted random selection
                total_weight = sum(inst['weight'] for inst in healthy_instances)
                if total_weight == 0:
                    instance = healthy_instances[0]
                else:
                    import random
                    r = random.random() * total_weight
                    cumulative = 0
                    instance = healthy_instances[-1]  # Fallback
                    
                    for inst in healthy_instances:
                        cumulative += inst['weight']
                        if r <= cumulative:
                            instance = inst
                            break
            
            elif self.balancing_strategy == "performance":
                # Choose instance with best performance (low latency, low error rate)
                def performance_score(inst):
                    latency_score = 1.0 / max(0.001, inst['avg_response_time'])
                    error_rate = inst['total_errors'] / max(1, inst['total_requests'])
                    error_score = 1.0 - min(1.0, error_rate)
                    return latency_score * error_score
                
                instance = max(healthy_instances, key=performance_score)
                
            else:
                # Default to round robin
                instance = healthy_instances[self._current_index % len(healthy_instances)]
                self._current_index = (self._current_index + 1) % len(healthy_instances)
            
            # Update connection count
            instance['connections'] += 1
            
            return instance
    
    def record_request_completion(self, 
                                instance_id: str, 
                                response_time: float,
                                success: bool = True):
        """Record completion of a request."""
        with self._lock:
            for instance in self._instances:
                if instance['id'] == instance_id:
                    instance['connections'] = max(0, instance['connections'] - 1)
                    instance['total_requests'] += 1
                    
                    if not success:
                        instance['total_errors'] += 1
                    
                    # Update average response time (exponential moving average)
                    alpha = 0.1
                    instance['avg_response_time'] = (
                        alpha * response_time + 
                        (1 - alpha) * instance['avg_response_time']
                    )
                    break
    
    def start_health_checks(self):
        """Start health checking of instances."""
        if self._is_health_checking:
            return
        
        self._is_health_checking = True
        self._health_check_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._health_check_thread.start()
        
        logging.info("Health checking started")
    
    def stop_health_checks(self):
        """Stop health checking."""
        self._is_health_checking = False
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5)
        
        logging.info("Health checking stopped")
    
    def _health_check_loop(self):
        """Health check loop."""
        while self._is_health_checking:
            try:
                with self._lock:
                    instances_to_check = self._instances.copy()
                
                for instance in instances_to_check:
                    try:
                        healthy = self._check_instance_health(instance)
                        
                        with self._lock:
                            # Find instance in list and update health
                            for i, inst in enumerate(self._instances):
                                if inst['id'] == instance['id']:
                                    self._instances[i]['healthy'] = healthy
                                    self._instances[i]['last_health_check'] = time.time()
                                    break
                        
                        if not healthy:
                            logging.warning(f"Instance {instance['id']} failed health check")
                    
                    except Exception as e:
                        logging.error(f"Health check error for {instance['id']}: {e}")
                        
                        with self._lock:
                            for i, inst in enumerate(self._instances):
                                if inst['id'] == instance['id']:
                                    self._instances[i]['healthy'] = False
                                    break
                
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logging.error(f"Health check loop error: {e}")
                time.sleep(self.health_check_interval)
    
    def _check_instance_health(self, instance: Dict[str, Any]) -> bool:
        """Check health of a specific instance."""
        try:
            # Simple TCP connection check
            endpoint = instance['endpoint']
            
            if '://' in endpoint:
                # Parse URL
                parts = endpoint.split('://')
                if len(parts) == 2:
                    host_port = parts[1].split('/')[0]
                    if ':' in host_port:
                        host, port = host_port.split(':')
                        port = int(port)
                    else:
                        host = host_port
                        port = 80 if parts[0] == 'http' else 443
                else:
                    return False
            else:
                # Direct host:port
                if ':' in endpoint:
                    host, port = endpoint.split(':')
                    port = int(port)
                else:
                    return False
            
            # Try to connect
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            return result == 0
            
        except Exception:
            return False
    
    def get_instance_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all instances."""
        with self._lock:
            stats = []
            for instance in self._instances:
                error_rate = (instance['total_errors'] / max(1, instance['total_requests']))
                
                stats.append({
                    'id': instance['id'],
                    'endpoint': instance['endpoint'],
                    'healthy': instance['healthy'],
                    'connections': instance['connections'],
                    'total_requests': instance['total_requests'],
                    'error_rate': error_rate,
                    'avg_response_time_ms': instance['avg_response_time'] * 1000,
                    'weight': instance['weight'],
                    'last_health_check': instance['last_health_check']
                })
            
            return stats


class ElasticBatchProcessor:
    """Elastic batch processing that adapts to system load."""
    
    def __init__(self,
                 initial_batch_size: int = 32,
                 min_batch_size: int = 1,
                 max_batch_size: int = 512,
                 target_latency_ms: float = 100.0):
        """Initialize elastic batch processor.
        
        Args:
            initial_batch_size: Starting batch size
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size  
            target_latency_ms: Target processing latency
        """
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_latency_ms = target_latency_ms
        
        self._batch_queue = queue.Queue()
        self._result_queue = queue.Queue()
        self._processing_thread: Optional[threading.Thread] = None
        self._is_processing = False
        
        # Performance tracking
        self._latency_history = deque(maxlen=20)
        self._throughput_history = deque(maxlen=20)
        self._last_adjustment = time.time()
        
    def start_processing(self, process_fn: Callable[[List[Any]], List[Any]]):
        """Start batch processing.
        
        Args:
            process_fn: Function to process batches
        """
        if self._is_processing:
            return
        
        self._is_processing = True
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            args=(process_fn,),
            daemon=True
        )
        self._processing_thread.start()
        
        logging.info("Elastic batch processing started")
    
    def stop_processing(self):
        """Stop batch processing."""
        self._is_processing = False
        if self._processing_thread:
            self._processing_thread.join(timeout=10)
        
        logging.info("Elastic batch processing stopped")
    
    def submit_item(self, item: Any) -> str:
        """Submit item for batch processing.
        
        Args:
            item: Item to process
            
        Returns:
            Request ID for tracking
        """
        request_id = str(time.time()) + str(id(item))
        request_data = {
            'id': request_id,
            'item': item,
            'submit_time': time.time()
        }
        
        self._batch_queue.put(request_data)
        return request_id
    
    def get_result(self, request_id: str, timeout: float = 30.0) -> Optional[Any]:
        """Get result for a request.
        
        Args:
            request_id: Request ID
            timeout: Timeout in seconds
            
        Returns:
            Processing result or None if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result_data = self._result_queue.get(timeout=1.0)
                if result_data['id'] == request_id:
                    return result_data['result']
                else:
                    # Put back result for other requests
                    self._result_queue.put(result_data)
            except queue.Empty:
                continue
        
        return None
    
    def _processing_loop(self, process_fn: Callable[[List[Any]], List[Any]]):
        """Main processing loop."""
        while self._is_processing:
            try:
                # Collect items for batch
                batch_items = []
                batch_start_time = time.time()
                
                # Wait for first item
                try:
                    first_item = self._batch_queue.get(timeout=1.0)
                    batch_items.append(first_item)
                except queue.Empty:
                    continue
                
                # Collect additional items up to batch size
                while (len(batch_items) < self.current_batch_size and 
                       time.time() - batch_start_time < 0.1):  # 100ms collection window
                    try:
                        item = self._batch_queue.get(timeout=0.01)
                        batch_items.append(item)
                    except queue.Empty:
                        break
                
                if not batch_items:
                    continue
                
                # Process batch
                processing_start = time.time()
                
                items_to_process = [item['item'] for item in batch_items]
                results = process_fn(items_to_process)
                
                processing_end = time.time()
                processing_latency = (processing_end - processing_start) * 1000  # ms
                
                # Store results
                for batch_item, result in zip(batch_items, results):
                    result_data = {
                        'id': batch_item['id'],
                        'result': result,
                        'processing_time': processing_latency / len(batch_items),
                        'batch_size': len(batch_items)
                    }
                    self._result_queue.put(result_data)
                
                # Update performance metrics
                throughput = len(batch_items) / (processing_latency / 1000)
                self._latency_history.append(processing_latency / len(batch_items))
                self._throughput_history.append(throughput)
                
                # Adjust batch size
                self._adjust_batch_size()
                
            except Exception as e:
                logging.error(f"Batch processing error: {e}")
                time.sleep(1)
    
    def _adjust_batch_size(self):
        """Adjust batch size based on performance."""
        current_time = time.time()
        
        # Only adjust every 10 seconds
        if current_time - self._last_adjustment < 10.0:
            return
        
        if len(self._latency_history) < 5:
            return
        
        avg_latency = np.mean(list(self._latency_history)[-5:])
        
        if avg_latency > self.target_latency_ms * 1.2:  # 20% above target
            # Reduce batch size
            new_batch_size = max(
                self.min_batch_size,
                int(self.current_batch_size * 0.8)
            )
        elif avg_latency < self.target_latency_ms * 0.8:  # 20% below target
            # Increase batch size
            new_batch_size = min(
                self.max_batch_size,
                int(self.current_batch_size * 1.2)
            )
        else:
            new_batch_size = self.current_batch_size
        
        if new_batch_size != self.current_batch_size:
            logging.info(f"Adjusting batch size from {self.current_batch_size} to {new_batch_size}")
            self.current_batch_size = new_batch_size
            self._last_adjustment = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        if not self._latency_history or not self._throughput_history:
            return {}
        
        return {
            'current_batch_size': self.current_batch_size,
            'avg_latency_ms': np.mean(list(self._latency_history)),
            'avg_throughput': np.mean(list(self._throughput_history)),
            'queue_size': self._batch_queue.qsize(),
            'pending_results': self._result_queue.qsize()
        }