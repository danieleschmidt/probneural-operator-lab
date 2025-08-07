"""
Comprehensive health checks and diagnostics system for ProbNeural-Operator-Lab.

This module provides model health diagnostics, convergence monitoring, 
GPU/CPU compatibility verification, and system health assessments.
"""

import time
import warnings
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import logging

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from .validation import check_numerical_stability, validate_tensor_finite
from .exceptions import ConvergenceError, DeviceError, handle_exception

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning" 
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class DiagnosticResult:
    """Result of a diagnostic check."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    execution_time: float


class ModelHealthChecker:
    """Comprehensive model health diagnostics."""
    
    def __init__(self, model: nn.Module):
        """Initialize model health checker.
        
        Args:
            model: Neural network model to diagnose
        """
        self.model = model
        self.diagnostic_history: List[DiagnosticResult] = []
    
    def run_full_health_check(self, 
                            sample_input: Optional[torch.Tensor] = None,
                            sample_target: Optional[torch.Tensor] = None) -> Dict[str, DiagnosticResult]:
        """Run comprehensive health check on the model.
        
        Args:
            sample_input: Sample input for testing (optional)
            sample_target: Sample target for loss computation (optional)
            
        Returns:
            Dictionary of diagnostic results
        """
        results = {}
        
        # Basic model structure checks
        results["parameter_health"] = self._check_parameter_health()
        results["gradient_health"] = self._check_gradient_health()
        results["architecture_health"] = self._check_architecture_health()
        
        # Runtime checks if sample data provided
        if sample_input is not None:
            results["forward_pass_health"] = self._check_forward_pass_health(sample_input)
            
            if sample_target is not None:
                results["backward_pass_health"] = self._check_backward_pass_health(
                    sample_input, sample_target
                )
        
        # Device and memory checks
        results["device_health"] = self._check_device_health()
        results["memory_health"] = self._check_memory_health()
        
        # Store results
        self.diagnostic_history.extend(results.values())
        
        # Log summary
        self._log_health_summary(results)
        
        return results
    
    def _check_parameter_health(self) -> DiagnosticResult:
        """Check health of model parameters."""
        start_time = time.time()
        
        try:
            issues = []
            details = {}
            
            # Collect parameter statistics
            param_stats = {}
            total_params = 0
            
            for name, param in self.model.named_parameters():
                if param.data.numel() > 0:
                    param_data = param.data
                    
                    # Basic statistics
                    stats = {
                        'mean': param_data.mean().item(),
                        'std': param_data.std().item(),
                        'min': param_data.min().item(),
                        'max': param_data.max().item(),
                        'shape': list(param_data.shape),
                        'numel': param_data.numel()
                    }
                    
                    param_stats[name] = stats
                    total_params += param_data.numel()
                    
                    # Check for issues
                    if torch.any(torch.isnan(param_data)):
                        issues.append(f"NaN values in parameter {name}")
                    
                    if torch.any(torch.isinf(param_data)):
                        issues.append(f"Infinite values in parameter {name}")
                    
                    if stats['std'] < 1e-8:
                        issues.append(f"Very small variance in parameter {name} (std: {stats['std']})")
                    
                    if abs(stats['mean']) > 10:
                        issues.append(f"Large mean in parameter {name} (mean: {stats['mean']})")
            
            details['parameter_stats'] = param_stats
            details['total_parameters'] = total_params
            details['issues_found'] = len(issues)
            
            # Determine status
            if not issues:
                status = HealthStatus.HEALTHY
                message = f"All {len(param_stats)} parameter groups are healthy"
            elif len(issues) < 3:
                status = HealthStatus.WARNING
                message = f"Found {len(issues)} parameter issues"
            else:
                status = HealthStatus.CRITICAL
                message = f"Found {len(issues)} critical parameter issues"
            
            details['issues'] = issues
            
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Error checking parameters: {e}"
            details = {"error": str(e)}
        
        execution_time = time.time() - start_time
        
        return DiagnosticResult(
            name="parameter_health",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now(),
            execution_time=execution_time
        )
    
    def _check_gradient_health(self) -> DiagnosticResult:
        """Check health of gradients (if available)."""
        start_time = time.time()
        
        try:
            issues = []
            details = {}
            
            grad_stats = {}
            has_gradients = False
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    has_gradients = True
                    grad_data = param.grad.data
                    
                    stats = {
                        'mean': grad_data.mean().item(),
                        'std': grad_data.std().item(),
                        'min': grad_data.min().item(),
                        'max': grad_data.max().item(),
                        'norm': torch.norm(grad_data).item()
                    }
                    
                    grad_stats[name] = stats
                    
                    # Check for issues
                    if torch.any(torch.isnan(grad_data)):
                        issues.append(f"NaN gradients in {name}")
                    
                    if torch.any(torch.isinf(grad_data)):
                        issues.append(f"Infinite gradients in {name}")
                    
                    if stats['norm'] > 100:
                        issues.append(f"Large gradient norm in {name} (norm: {stats['norm']})")
                    
                    if stats['norm'] < 1e-8:
                        issues.append(f"Very small gradients in {name} (norm: {stats['norm']})")
            
            details['gradient_stats'] = grad_stats
            details['has_gradients'] = has_gradients
            
            if not has_gradients:
                status = HealthStatus.WARNING
                message = "No gradients available for analysis"
            elif not issues:
                status = HealthStatus.HEALTHY
                message = f"All {len(grad_stats)} gradient groups are healthy"
            else:
                status = HealthStatus.WARNING if len(issues) < 3 else HealthStatus.CRITICAL
                message = f"Found {len(issues)} gradient issues"
            
            details['issues'] = issues
            
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Error checking gradients: {e}"
            details = {"error": str(e)}
        
        execution_time = time.time() - start_time
        
        return DiagnosticResult(
            name="gradient_health",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now(),
            execution_time=execution_time
        )
    
    def _check_architecture_health(self) -> DiagnosticResult:
        """Check model architecture health."""
        start_time = time.time()
        
        try:
            issues = []
            details = {}
            
            # Count modules by type
            module_counts = {}
            total_modules = 0
            
            for name, module in self.model.named_modules():
                module_type = type(module).__name__
                module_counts[module_type] = module_counts.get(module_type, 0) + 1
                total_modules += 1
                
                # Check for known problematic patterns
                if isinstance(module, nn.Dropout) and module.p > 0.8:
                    issues.append(f"Very high dropout rate in {name}: {module.p}")
                
                if isinstance(module, nn.BatchNorm2d) and not module.track_running_stats:
                    issues.append(f"BatchNorm without running stats tracking in {name}")
            
            details['module_counts'] = module_counts
            details['total_modules'] = total_modules
            
            # Check for architectural issues
            if module_counts.get('Linear', 0) > 50:
                issues.append(f"Very deep network: {module_counts['Linear']} Linear layers")
            
            if total_modules > 1000:
                issues.append(f"Very complex architecture: {total_modules} modules")
            
            # Determine status
            if not issues:
                status = HealthStatus.HEALTHY
                message = f"Architecture with {total_modules} modules looks healthy"
            else:
                status = HealthStatus.WARNING
                message = f"Found {len(issues)} architectural concerns"
            
            details['issues'] = issues
            
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Error checking architecture: {e}"
            details = {"error": str(e)}
        
        execution_time = time.time() - start_time
        
        return DiagnosticResult(
            name="architecture_health",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now(),
            execution_time=execution_time
        )
    
    def _check_forward_pass_health(self, sample_input: torch.Tensor) -> DiagnosticResult:
        """Check forward pass health with sample input."""
        start_time = time.time()
        
        try:
            issues = []
            details = {}
            
            # Record initial state
            self.model.eval()
            
            with torch.no_grad():
                # Forward pass
                forward_start = time.time()
                output = self.model(sample_input)
                forward_time = time.time() - forward_start
                
                # Analyze output
                output_stats = {
                    'shape': list(output.shape),
                    'mean': output.mean().item(),
                    'std': output.std().item(), 
                    'min': output.min().item(),
                    'max': output.max().item()
                }
                
                details['output_stats'] = output_stats
                details['forward_time'] = forward_time
                
                # Check for issues
                if torch.any(torch.isnan(output)):
                    issues.append("NaN values in model output")
                
                if torch.any(torch.isinf(output)):
                    issues.append("Infinite values in model output")
                
                if output_stats['std'] < 1e-8:
                    issues.append(f"Output has very low variance: {output_stats['std']}")
                
                if forward_time > 1.0:  # More than 1 second for inference
                    issues.append(f"Slow forward pass: {forward_time:.2f}s")
                
                # Check activation ranges
                if abs(output_stats['mean']) > 1000:
                    issues.append(f"Large output magnitude: mean={output_stats['mean']}")
            
            # Determine status
            if not issues:
                status = HealthStatus.HEALTHY
                message = f"Forward pass completed successfully in {forward_time:.3f}s"
            else:
                status = HealthStatus.WARNING if len(issues) < 3 else HealthStatus.CRITICAL
                message = f"Forward pass completed with {len(issues)} issues"
            
            details['issues'] = issues
            
        except Exception as e:
            status = HealthStatus.CRITICAL
            message = f"Forward pass failed: {e}"
            details = {"error": str(e)}
        
        execution_time = time.time() - start_time
        
        return DiagnosticResult(
            name="forward_pass_health",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now(),
            execution_time=execution_time
        )
    
    def _check_backward_pass_health(self, sample_input: torch.Tensor, 
                                  sample_target: torch.Tensor) -> DiagnosticResult:
        """Check backward pass health."""
        start_time = time.time()
        
        try:
            issues = []
            details = {}
            
            self.model.train()
            
            # Clear gradients
            self.model.zero_grad()
            
            # Forward pass
            output = self.model(sample_input)
            
            # Compute loss
            loss_fn = nn.MSELoss()
            loss = loss_fn(output, sample_target)
            loss_value = loss.item()
            
            # Backward pass
            backward_start = time.time()
            loss.backward()
            backward_time = time.time() - backward_start
            
            # Analyze gradients
            grad_norms = {}
            total_grad_norm = 0
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad).item()
                    grad_norms[name] = grad_norm
                    total_grad_norm += grad_norm ** 2
            
            total_grad_norm = total_grad_norm ** 0.5
            
            details.update({
                'loss_value': loss_value,
                'backward_time': backward_time,
                'total_grad_norm': total_grad_norm,
                'individual_grad_norms': grad_norms
            })
            
            # Check for issues
            if np.isnan(loss_value) or np.isinf(loss_value):
                issues.append(f"Invalid loss value: {loss_value}")
            
            if total_grad_norm > 100:
                issues.append(f"Large gradient norm: {total_grad_norm}")
            
            if total_grad_norm < 1e-8:
                issues.append(f"Very small gradient norm: {total_grad_norm}")
            
            if backward_time > 5.0:
                issues.append(f"Slow backward pass: {backward_time:.2f}s")
            
            # Determine status
            if not issues:
                status = HealthStatus.HEALTHY
                message = f"Backward pass completed successfully (loss: {loss_value:.6f})"
            else:
                status = HealthStatus.WARNING if len(issues) < 2 else HealthStatus.CRITICAL
                message = f"Backward pass completed with {len(issues)} issues"
            
            details['issues'] = issues
            
        except Exception as e:
            status = HealthStatus.CRITICAL
            message = f"Backward pass failed: {e}"
            details = {"error": str(e)}
        
        execution_time = time.time() - start_time
        
        return DiagnosticResult(
            name="backward_pass_health",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now(),
            execution_time=execution_time
        )
    
    def _check_device_health(self) -> DiagnosticResult:
        """Check device compatibility and health."""
        start_time = time.time()
        
        try:
            issues = []
            details = {}
            
            # Get model device
            model_device = next(self.model.parameters()).device
            details['model_device'] = str(model_device)
            
            # CUDA availability
            details['cuda_available'] = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                details['cuda_device_count'] = torch.cuda.device_count()
                details['current_cuda_device'] = torch.cuda.current_device()
                
                # GPU memory info
                if model_device.type == 'cuda':
                    device_idx = model_device.index or 0
                    details['gpu_memory_allocated'] = torch.cuda.memory_allocated(device_idx)
                    details['gpu_memory_reserved'] = torch.cuda.memory_reserved(device_idx)
                    details['gpu_max_memory'] = torch.cuda.max_memory_allocated(device_idx)
                    
                    # Check memory usage
                    memory_usage = details['gpu_memory_allocated'] / (1024**3)  # GB
                    if memory_usage > 10:  # More than 10GB
                        issues.append(f"High GPU memory usage: {memory_usage:.1f}GB")
            else:
                if model_device.type == 'cuda':
                    issues.append("Model is on CUDA but CUDA is not available")
            
            # Check device consistency
            devices = set()
            for param in self.model.parameters():
                devices.add(param.device)
            
            if len(devices) > 1:
                issues.append(f"Model parameters on multiple devices: {devices}")
            
            details['parameter_devices'] = [str(d) for d in devices]
            
            # Determine status
            if not issues:
                status = HealthStatus.HEALTHY
                message = f"Model properly configured on {model_device}"
            else:
                status = HealthStatus.WARNING
                message = f"Found {len(issues)} device issues"
            
            details['issues'] = issues
            
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Error checking device health: {e}"
            details = {"error": str(e)}
        
        execution_time = time.time() - start_time
        
        return DiagnosticResult(
            name="device_health",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now(),
            execution_time=execution_time
        )
    
    def _check_memory_health(self) -> DiagnosticResult:
        """Check memory usage and health."""
        start_time = time.time()
        
        try:
            import psutil
            
            issues = []
            details = {}
            
            # System memory
            system_memory = psutil.virtual_memory()
            details['system_memory_total_gb'] = system_memory.total / (1024**3)
            details['system_memory_used_gb'] = system_memory.used / (1024**3)
            details['system_memory_percent'] = system_memory.percent
            
            # Process memory
            process = psutil.Process()
            process_memory = process.memory_info()
            details['process_memory_rss_gb'] = process_memory.rss / (1024**3)
            details['process_memory_vms_gb'] = process_memory.vms / (1024**3)
            
            # Model memory estimation
            model_params = sum(p.numel() * p.element_size() for p in self.model.parameters())
            details['model_memory_mb'] = model_params / (1024**2)
            
            # Check for issues
            if system_memory.percent > 90:
                issues.append(f"High system memory usage: {system_memory.percent:.1f}%")
            
            if details['process_memory_rss_gb'] > 16:  # More than 16GB
                issues.append(f"High process memory usage: {details['process_memory_rss_gb']:.1f}GB")
            
            # GPU memory (if available)
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    
                    details[f'gpu_{i}_allocated_gb'] = allocated
                    details[f'gpu_{i}_reserved_gb'] = reserved
                    
                    if allocated > 10:
                        issues.append(f"High GPU {i} memory usage: {allocated:.1f}GB")
            
            # Determine status
            if not issues:
                status = HealthStatus.HEALTHY
                message = "Memory usage is within normal ranges"
            else:
                status = HealthStatus.WARNING
                message = f"Found {len(issues)} memory concerns"
            
            details['issues'] = issues
            
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Error checking memory health: {e}"
            details = {"error": str(e)}
        
        execution_time = time.time() - start_time
        
        return DiagnosticResult(
            name="memory_health",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now(),
            execution_time=execution_time
        )
    
    def _log_health_summary(self, results: Dict[str, DiagnosticResult]) -> None:
        """Log summary of health check results."""
        total_checks = len(results)
        healthy_count = sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY)
        warning_count = sum(1 for r in results.values() if r.status == HealthStatus.WARNING)
        critical_count = sum(1 for r in results.values() if r.status == HealthStatus.CRITICAL)
        
        logger.info(
            f"Model health check completed: {healthy_count}/{total_checks} healthy, "
            f"{warning_count} warnings, {critical_count} critical issues"
        )
        
        # Log individual issues
        for result in results.values():
            if result.status != HealthStatus.HEALTHY:
                issues = result.details.get('issues', [])
                for issue in issues:
                    if result.status == HealthStatus.CRITICAL:
                        logger.error(f"{result.name}: {issue}")
                    else:
                        logger.warning(f"{result.name}: {issue}")


class ConvergenceMonitor:
    """Monitors training convergence and detects convergence issues."""
    
    def __init__(self, 
                 patience: int = 20,
                 min_improvement: float = 1e-6,
                 divergence_threshold: float = 10.0):
        """Initialize convergence monitor.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_improvement: Minimum improvement to reset patience counter
            divergence_threshold: Threshold for detecting divergence
        """
        self.patience = patience
        self.min_improvement = min_improvement
        self.divergence_threshold = divergence_threshold
        
        # Tracking variables
        self.loss_history: List[float] = []
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0
        self.converged = False
        self.diverged = False
        
    def update(self, loss: float) -> Dict[str, Any]:
        """Update monitor with new loss value.
        
        Args:
            loss: Current epoch loss
            
        Returns:
            Dictionary with convergence status
        """
        self.loss_history.append(loss)
        
        # Check for improvement
        improved = loss < self.best_loss - self.min_improvement
        if improved:
            self.best_loss = loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
        
        # Check for divergence
        if len(self.loss_history) >= 2:
            recent_increase = loss / self.loss_history[-2] if self.loss_history[-2] != 0 else 1
            if recent_increase > self.divergence_threshold:
                self.diverged = True
        
        # Check for convergence
        if self.epochs_without_improvement >= self.patience:
            self.converged = True
        
        return {
            'loss': loss,
            'best_loss': self.best_loss,
            'epochs_without_improvement': self.epochs_without_improvement,
            'converged': self.converged,
            'diverged': self.diverged,
            'improved_this_epoch': improved
        }
    
    def get_convergence_diagnostics(self) -> Dict[str, Any]:
        """Get detailed convergence diagnostics.
        
        Returns:
            Dictionary with convergence analysis
        """
        if len(self.loss_history) < 2:
            return {'status': 'insufficient_data'}
        
        losses = np.array(self.loss_history)
        
        # Trend analysis
        epochs = np.arange(len(losses))
        if len(epochs) >= 3:
            slope, _ = np.polyfit(epochs[-10:], losses[-10:], 1)  # Last 10 epochs
        else:
            slope = 0
        
        # Smoothness analysis
        if len(losses) >= 3:
            second_derivatives = np.diff(np.diff(losses))
            smoothness = np.std(second_derivatives)
        else:
            smoothness = 0
        
        # Oscillation detection
        recent_losses = losses[-20:] if len(losses) >= 20 else losses
        oscillation_amplitude = np.std(recent_losses) if len(recent_losses) > 1 else 0
        
        return {
            'total_epochs': len(losses),
            'current_loss': losses[-1],
            'best_loss': self.best_loss,
            'loss_improvement': self.best_loss - losses[-1] if len(losses) > 0 else 0,
            'trend_slope': slope,
            'smoothness': smoothness,
            'oscillation_amplitude': oscillation_amplitude,
            'converged': self.converged,
            'diverged': self.diverged,
            'epochs_without_improvement': self.epochs_without_improvement
        }


class SystemCompatibilityChecker:
    """Checks system compatibility and requirements."""
    
    def __init__(self):
        """Initialize system compatibility checker."""
        self.compatibility_results: Dict[str, DiagnosticResult] = {}
    
    def check_full_compatibility(self) -> Dict[str, DiagnosticResult]:
        """Run full system compatibility check.
        
        Returns:
            Dictionary of compatibility check results
        """
        results = {}
        
        results["python_version"] = self._check_python_version()
        results["pytorch_version"] = self._check_pytorch_version()
        results["cuda_compatibility"] = self._check_cuda_compatibility()
        results["cpu_capabilities"] = self._check_cpu_capabilities()
        results["memory_requirements"] = self._check_memory_requirements()
        results["dependency_versions"] = self._check_dependency_versions()
        
        self.compatibility_results = results
        return results
    
    def _check_python_version(self) -> DiagnosticResult:
        """Check Python version compatibility."""
        import sys
        
        start_time = time.time()
        
        try:
            python_version = sys.version_info
            version_str = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
            
            issues = []
            if python_version < (3, 9):
                issues.append(f"Python {version_str} is below minimum required version 3.9")
                status = HealthStatus.CRITICAL
            elif python_version >= (3, 12):
                issues.append(f"Python {version_str} may have compatibility issues")
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY
            
            message = f"Python {version_str}"
            details = {
                'version': version_str,
                'version_info': python_version,
                'issues': issues
            }
            
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Error checking Python version: {e}"
            details = {"error": str(e)}
        
        execution_time = time.time() - start_time
        
        return DiagnosticResult(
            name="python_version",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now(),
            execution_time=execution_time
        )
    
    def _check_pytorch_version(self) -> DiagnosticResult:
        """Check PyTorch version compatibility."""
        start_time = time.time()
        
        try:
            pytorch_version = torch.__version__
            
            issues = []
            # Check for minimum PyTorch version
            version_parts = pytorch_version.split('.')
            major, minor = int(version_parts[0]), int(version_parts[1])
            
            if major < 2:
                issues.append(f"PyTorch {pytorch_version} is below recommended version 2.0")
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY
            
            message = f"PyTorch {pytorch_version}"
            details = {
                'version': pytorch_version,
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'debug': torch.version.debug,
                'issues': issues
            }
            
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Error checking PyTorch version: {e}"
            details = {"error": str(e)}
        
        execution_time = time.time() - start_time
        
        return DiagnosticResult(
            name="pytorch_version",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now(),
            execution_time=execution_time
        )
    
    def _check_cuda_compatibility(self) -> DiagnosticResult:
        """Check CUDA compatibility."""
        start_time = time.time()
        
        try:
            issues = []
            details = {
                'cuda_available': torch.cuda.is_available(),
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
            
            if torch.cuda.is_available():
                details['cuda_version'] = torch.version.cuda
                details['cudnn_version'] = torch.backends.cudnn.version()
                
                # Get device information
                device_info = []
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    device_info.append({
                        'name': props.name,
                        'total_memory': props.total_memory,
                        'major': props.major,
                        'minor': props.minor,
                        'multi_processor_count': props.multi_processor_count
                    })
                
                details['devices'] = device_info
                
                # Check compute capability
                for i, device in enumerate(device_info):
                    if device['major'] < 6:  # Compute capability < 6.0
                        issues.append(f"GPU {i} has old compute capability {device['major']}.{device['minor']}")
                    
                    if device['total_memory'] < 4 * 1024**3:  # Less than 4GB
                        issues.append(f"GPU {i} has limited memory: {device['total_memory'] / 1024**3:.1f}GB")
                
                status = HealthStatus.HEALTHY if not issues else HealthStatus.WARNING
                message = f"CUDA {details['cuda_version']} with {details['device_count']} device(s)"
            else:
                status = HealthStatus.WARNING
                message = "CUDA not available - CPU only mode"
            
            details['issues'] = issues
            
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Error checking CUDA compatibility: {e}"
            details = {"error": str(e)}
        
        execution_time = time.time() - start_time
        
        return DiagnosticResult(
            name="cuda_compatibility",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now(),
            execution_time=execution_time
        )
    
    def _check_cpu_capabilities(self) -> DiagnosticResult:
        """Check CPU capabilities."""
        start_time = time.time()
        
        try:
            import psutil
            
            issues = []
            
            # Basic CPU info
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            details = {
                'cpu_count': cpu_count,
                'cpu_freq_max': cpu_freq.max if cpu_freq else None,
                'cpu_freq_current': cpu_freq.current if cpu_freq else None
            }
            
            # Check for minimum requirements
            if cpu_count < 4:
                issues.append(f"Low CPU count: {cpu_count} cores (recommended: 4+)")
            
            if cpu_freq and cpu_freq.max < 2000:  # Less than 2GHz
                issues.append(f"Low CPU frequency: {cpu_freq.max}MHz")
            
            # Check CPU features (if available)
            try:
                import cpuinfo
                cpu_info = cpuinfo.get_cpu_info()
                details['cpu_brand'] = cpu_info.get('brand_raw', 'Unknown')
                details['cpu_arch'] = cpu_info.get('arch', 'Unknown')
                details['cpu_flags'] = cpu_info.get('flags', [])
                
                # Check for useful instruction sets
                useful_flags = ['avx', 'avx2', 'fma', 'sse4_1', 'sse4_2']
                missing_flags = [flag for flag in useful_flags if flag not in details['cpu_flags']]
                
                if missing_flags:
                    issues.append(f"Missing CPU features: {missing_flags}")
                
            except ImportError:
                details['cpu_info_available'] = False
            
            status = HealthStatus.HEALTHY if not issues else HealthStatus.WARNING
            message = f"CPU: {cpu_count} cores"
            details['issues'] = issues
            
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Error checking CPU capabilities: {e}"
            details = {"error": str(e)}
        
        execution_time = time.time() - start_time
        
        return DiagnosticResult(
            name="cpu_capabilities",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now(),
            execution_time=execution_time
        )
    
    def _check_memory_requirements(self) -> DiagnosticResult:
        """Check memory requirements."""
        start_time = time.time()
        
        try:
            import psutil
            
            issues = []
            
            # System memory
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            details = {
                'total_memory_gb': memory.total / (1024**3),
                'available_memory_gb': memory.available / (1024**3),
                'swap_total_gb': swap.total / (1024**3),
                'memory_percent': memory.percent
            }
            
            # Check minimum requirements
            min_memory_gb = 8
            recommended_memory_gb = 16
            
            if details['total_memory_gb'] < min_memory_gb:
                issues.append(f"Insufficient memory: {details['total_memory_gb']:.1f}GB (minimum: {min_memory_gb}GB)")
                status = HealthStatus.CRITICAL
            elif details['total_memory_gb'] < recommended_memory_gb:
                issues.append(f"Low memory: {details['total_memory_gb']:.1f}GB (recommended: {recommended_memory_gb}GB)")
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY
            
            if memory.percent > 80:
                issues.append(f"High memory usage: {memory.percent:.1f}%")
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
            
            message = f"Memory: {details['total_memory_gb']:.1f}GB total, {details['available_memory_gb']:.1f}GB available"
            details['issues'] = issues
            
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Error checking memory requirements: {e}"
            details = {"error": str(e)}
        
        execution_time = time.time() - start_time
        
        return DiagnosticResult(
            name="memory_requirements",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now(),
            execution_time=execution_time
        )
    
    def _check_dependency_versions(self) -> DiagnosticResult:
        """Check versions of key dependencies."""
        start_time = time.time()
        
        try:
            issues = []
            versions = {}
            
            # Check key dependencies
            dependencies = {
                'numpy': ('numpy', '1.21.0'),
                'scipy': ('scipy', '1.7.0'),
                'matplotlib': ('matplotlib', '3.5.0'),
                'scikit-learn': ('sklearn', '1.1.0')
            }
            
            for dep_name, (module_name, min_version) in dependencies.items():
                try:
                    module = __import__(module_name)
                    version = getattr(module, '__version__', 'unknown')
                    versions[dep_name] = version
                    
                    # Simple version comparison (assumes semantic versioning)
                    if version != 'unknown':
                        version_parts = version.split('.')
                        min_parts = min_version.split('.')
                        
                        for i in range(min(len(version_parts), len(min_parts))):
                            v_part = int(version_parts[i].split('+')[0])  # Handle dev versions
                            m_part = int(min_parts[i])
                            
                            if v_part < m_part:
                                issues.append(f"{dep_name} {version} is below minimum {min_version}")
                                break
                            elif v_part > m_part:
                                break
                
                except ImportError:
                    issues.append(f"Missing dependency: {dep_name}")
                    versions[dep_name] = 'not_installed'
            
            status = HealthStatus.HEALTHY if not issues else HealthStatus.WARNING
            message = f"Checked {len(dependencies)} dependencies"
            
            details = {
                'versions': versions,
                'issues': issues
            }
            
        except Exception as e:
            status = HealthStatus.UNKNOWN
            message = f"Error checking dependency versions: {e}"
            details = {"error": str(e)}
        
        execution_time = time.time() - start_time
        
        return DiagnosticResult(
            name="dependency_versions",
            status=status,
            message=message,
            details=details,
            timestamp=datetime.now(),
            execution_time=execution_time
        )


# Utility functions
def run_comprehensive_diagnostics(model: nn.Module,
                                sample_input: Optional[torch.Tensor] = None,
                                sample_target: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """Run comprehensive diagnostics on model and system.
    
    Args:
        model: Neural network model
        sample_input: Sample input for testing
        sample_target: Sample target for testing
        
    Returns:
        Complete diagnostic report
    """
    logger.info("Starting comprehensive diagnostics...")
    
    # Model health check
    model_checker = ModelHealthChecker(model)
    model_results = model_checker.run_full_health_check(sample_input, sample_target)
    
    # System compatibility check
    system_checker = SystemCompatibilityChecker()
    system_results = system_checker.check_full_compatibility()
    
    # Combine results
    all_results = {**model_results, **system_results}
    
    # Generate summary
    total_checks = len(all_results)
    healthy = sum(1 for r in all_results.values() if r.status == HealthStatus.HEALTHY)
    warnings = sum(1 for r in all_results.values() if r.status == HealthStatus.WARNING)
    critical = sum(1 for r in all_results.values() if r.status == HealthStatus.CRITICAL)
    unknown = sum(1 for r in all_results.values() if r.status == HealthStatus.UNKNOWN)
    
    summary = {
        'total_checks': total_checks,
        'healthy': healthy,
        'warnings': warnings,
        'critical': critical,
        'unknown': unknown,
        'overall_status': 'healthy' if critical == 0 and warnings == 0 else 
                         'warning' if critical == 0 else 'critical'
    }
    
    logger.info(f"Diagnostics completed: {healthy}/{total_checks} healthy, {warnings} warnings, {critical} critical")
    
    return {
        'summary': summary,
        'detailed_results': all_results,
        'timestamp': datetime.now().isoformat()
    }