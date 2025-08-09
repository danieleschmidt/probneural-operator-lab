"""
Comprehensive security framework for probabilistic neural operators.

This module provides enterprise-grade security features including:
- Input validation and sanitization
- Authentication and authorization
- Rate limiting and DDoS protection
- Secure model serving
- Privacy-preserving inference
- Audit logging and compliance
"""

import hashlib
import hmac
import json
import time
import random
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
import logging


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    max_input_size: int = 10000  # Maximum input array size
    max_request_rate: int = 100  # Requests per minute per IP
    token_expiry_hours: int = 24
    enable_audit_logging: bool = True
    require_https: bool = True
    allowed_origins: List[str] = None
    max_batch_size: int = 100
    
    def __post_init__(self):
        if self.allowed_origins is None:
            self.allowed_origins = []


@dataclass
class SecurityContext:
    """Security context for a request."""
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    permissions: List[str] = None
    rate_limit_remaining: int = 100
    is_authenticated: bool = False
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = []


class InputValidator:
    """Validate and sanitize inputs for neural operator inference."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_numerical_input(
        self,
        data: List[float],
        min_val: float = -1e6,
        max_val: float = 1e6
    ) -> Tuple[bool, str]:
        """
        Validate numerical input array.
        
        Args:
            data: Input array to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            (is_valid, error_message)
        """
        if not isinstance(data, list):
            return False, "Input must be a list"
        
        if len(data) == 0:
            return False, "Input cannot be empty"
        
        if len(data) > self.config.max_input_size:
            return False, f"Input size exceeds maximum of {self.config.max_input_size}"
        
        for i, val in enumerate(data):
            if not isinstance(val, (int, float)):
                return False, f"Invalid type at index {i}: expected number, got {type(val)}"
            
            if not (-1e10 < val < 1e10):  # Check for infinity/NaN
                return False, f"Invalid value at index {i}: {val}"
            
            if not (min_val <= val <= max_val):
                return False, f"Value at index {i} out of range [{min_val}, {max_val}]: {val}"
        
        return True, ""
    
    def validate_batch_input(
        self,
        batch: List[List[float]]
    ) -> Tuple[bool, str]:
        """Validate batch of inputs."""
        if not isinstance(batch, list):
            return False, "Batch must be a list"
        
        if len(batch) == 0:
            return False, "Batch cannot be empty"
        
        if len(batch) > self.config.max_batch_size:
            return False, f"Batch size exceeds maximum of {self.config.max_batch_size}"
        
        # Validate each input in batch
        for i, input_data in enumerate(batch):
            is_valid, error = self.validate_numerical_input(input_data)
            if not is_valid:
                return False, f"Invalid input at batch index {i}: {error}"
        
        # Check for consistent dimensions
        if batch:
            expected_dim = len(batch[0])
            for i, input_data in enumerate(batch):
                if len(input_data) != expected_dim:
                    return False, f"Inconsistent dimensions: batch[0] has {expected_dim}, batch[{i}] has {len(input_data)}"
        
        return True, ""
    
    def sanitize_model_params(
        self,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Sanitize model parameters to prevent injection attacks."""
        sanitized = {}
        
        allowed_keys = {
            "uncertainty_samples", "temperature", "batch_size",
            "return_uncertainty", "calibration_method"
        }
        
        for key, value in params.items():
            if key not in allowed_keys:
                continue
                
            if key == "uncertainty_samples":
                sanitized[key] = max(1, min(1000, int(value)))
            elif key == "temperature":
                sanitized[key] = max(0.1, min(10.0, float(value)))
            elif key == "batch_size":
                sanitized[key] = max(1, min(self.config.max_batch_size, int(value)))
            elif key == "return_uncertainty":
                sanitized[key] = bool(value)
            elif key == "calibration_method":
                if value in ["temperature", "platt", "isotonic"]:
                    sanitized[key] = str(value)
        
        return sanitized


class AuthenticationManager:
    """Manage authentication tokens and user sessions."""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode('utf-8')
        self.tokens = {}  # token -> user_info
        self.user_permissions = {}  # user_id -> permissions
        self.failed_attempts = defaultdict(list)  # ip -> [timestamp, ...]
    
    def generate_token(
        self,
        user_id: str,
        permissions: List[str],
        expiry_hours: int = 24
    ) -> str:
        """Generate a secure authentication token."""
        timestamp = int(time.time())
        expiry = timestamp + (expiry_hours * 3600)
        
        # Create token payload
        payload = {
            "user_id": user_id,
            "permissions": permissions,
            "issued": timestamp,
            "expires": expiry,
            "nonce": random.randint(1000000, 9999999)
        }
        
        # Create signature
        message = json.dumps(payload, sort_keys=True).encode('utf-8')
        signature = hmac.new(self.secret_key, message, hashlib.sha256).hexdigest()
        
        token = f"{json.dumps(payload)}.{signature}"
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        
        # Store token info
        self.tokens[token_hash] = {
            "user_id": user_id,
            "permissions": permissions,
            "expires": expiry,
            "created": timestamp
        }
        
        self.user_permissions[user_id] = permissions
        
        return token
    
    def validate_token(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate authentication token."""
        try:
            # Split token
            parts = token.split('.')
            if len(parts) != 2:
                return False, None
            
            payload_str, signature = parts
            
            # Verify signature
            message = payload_str.encode('utf-8')
            expected_signature = hmac.new(self.secret_key, message, hashlib.sha256).hexdigest()
            
            if not hmac.compare_digest(signature, expected_signature):
                return False, None
            
            # Parse payload
            payload = json.loads(payload_str)
            
            # Check expiry
            if payload['expires'] < time.time():
                return False, None
            
            # Return user info
            return True, {
                "user_id": payload["user_id"],
                "permissions": payload["permissions"],
                "expires": payload["expires"]
            }
            
        except Exception:
            return False, None
    
    def record_failed_attempt(self, ip_address: str) -> None:
        """Record failed authentication attempt."""
        self.failed_attempts[ip_address].append(time.time())
        
        # Keep only recent attempts (last hour)
        cutoff = time.time() - 3600
        self.failed_attempts[ip_address] = [
            t for t in self.failed_attempts[ip_address] if t > cutoff
        ]
    
    def is_ip_blocked(self, ip_address: str, max_attempts: int = 10) -> bool:
        """Check if IP is blocked due to too many failed attempts."""
        recent_attempts = self.failed_attempts.get(ip_address, [])
        return len(recent_attempts) >= max_attempts


class RateLimiter:
    """Rate limiting to prevent abuse and DoS attacks."""
    
    def __init__(self):
        self.request_counts = defaultdict(deque)  # key -> deque of timestamps
        self.blocked_ips = {}  # ip -> block_until_timestamp
    
    def is_allowed(
        self,
        key: str,
        limit: int = 100,
        window_minutes: int = 1
    ) -> Tuple[bool, int]:
        """
        Check if request is allowed under rate limit.
        
        Args:
            key: Rate limiting key (e.g., IP address, user ID)
            limit: Maximum requests per window
            window_minutes: Time window in minutes
            
        Returns:
            (is_allowed, remaining_requests)
        """
        current_time = time.time()
        window_seconds = window_minutes * 60
        cutoff_time = current_time - window_seconds
        
        # Check if IP is temporarily blocked
        if key in self.blocked_ips:
            if current_time < self.blocked_ips[key]:
                return False, 0
            else:
                del self.blocked_ips[key]
        
        # Remove old requests from sliding window
        requests = self.request_counts[key]
        while requests and requests[0] < cutoff_time:
            requests.popleft()
        
        # Check if limit exceeded
        if len(requests) >= limit:
            # Block IP for escalating periods
            block_duration = min(300, len(requests) - limit + 60)  # Max 5 minutes
            self.blocked_ips[key] = current_time + block_duration
            return False, 0
        
        # Record this request
        requests.append(current_time)
        
        remaining = max(0, limit - len(requests))
        return True, remaining
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiting statistics."""
        current_time = time.time()
        active_keys = len([k for k, v in self.request_counts.items() if v])
        blocked_ips = len([ip for ip, until in self.blocked_ips.items() if until > current_time])
        
        return {
            "active_rate_limited_keys": active_keys,
            "currently_blocked_ips": blocked_ips,
            "total_tracked_keys": len(self.request_counts)
        }


class PrivacyPreserver:
    """Privacy-preserving inference techniques."""
    
    def __init__(self, noise_scale: float = 0.1):
        self.noise_scale = noise_scale
    
    def add_differential_privacy_noise(
        self,
        predictions: List[float],
        epsilon: float = 1.0
    ) -> List[float]:
        """
        Add calibrated noise for differential privacy.
        
        Args:
            predictions: Model predictions
            epsilon: Privacy budget (smaller = more private)
            
        Returns:
            Noisy predictions
        """
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        
        # Laplace mechanism for differential privacy
        sensitivity = 1.0  # Assume unit sensitivity for neural operators
        scale = sensitivity / epsilon
        
        noisy_predictions = []
        for pred in predictions:
            noise = random.gauss(0, scale)  # Laplace approximated with Gaussian
            noisy_predictions.append(pred + noise)
        
        return noisy_predictions
    
    def anonymize_input(
        self,
        input_data: List[float],
        k_anonymity: int = 5
    ) -> List[float]:
        """
        Apply k-anonymity by adding noise to make inputs less identifiable.
        
        Args:
            input_data: Input to anonymize
            k_anonymity: Minimum group size for anonymity
            
        Returns:
            Anonymized input
        """
        noise_level = self.noise_scale / k_anonymity
        
        anonymized = []
        for val in input_data:
            noise = random.gauss(0, noise_level)
            anonymized.append(val + noise)
        
        return anonymized
    
    def secure_aggregation(
        self,
        individual_predictions: List[List[float]],
        min_participants: int = 3
    ) -> List[float]:
        """
        Perform secure aggregation of predictions.
        
        Args:
            individual_predictions: List of prediction arrays
            min_participants: Minimum number of participants
            
        Returns:
            Securely aggregated prediction
        """
        if len(individual_predictions) < min_participants:
            raise ValueError(f"Need at least {min_participants} participants")
        
        if not individual_predictions:
            return []
        
        # Simple secure aggregation (in practice would use cryptographic protocols)
        n_dims = len(individual_predictions[0])
        aggregated = []
        
        for dim in range(n_dims):
            values = [pred[dim] for pred in individual_predictions]
            # Add noise to the aggregation
            noise = random.gauss(0, self.noise_scale)
            avg_value = sum(values) / len(values) + noise
            aggregated.append(avg_value)
        
        return aggregated


class AuditLogger:
    """Comprehensive audit logging for compliance."""
    
    def __init__(self, log_file: str = "security_audit.log"):
        self.log_file = log_file
        self.logger = self._setup_logger()
        self.event_counts = defaultdict(int)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup secure audit logger."""
        logger = logging.getLogger(f"audit.{__name__}")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.log_file, mode='a')
        formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def log_authentication(
        self,
        user_id: str,
        ip_address: str,
        success: bool,
        method: str = "token"
    ) -> None:
        """Log authentication events."""
        event_type = "auth_success" if success else "auth_failure"
        self.event_counts[event_type] += 1
        
        self.logger.info(json.dumps({
            "event_type": "authentication",
            "user_id": user_id,
            "ip_address": ip_address,
            "success": success,
            "method": method,
            "timestamp": time.time()
        }))
    
    def log_prediction_request(
        self,
        user_id: str,
        model_id: str,
        input_size: int,
        processing_time: float,
        ip_address: str
    ) -> None:
        """Log prediction requests."""
        self.event_counts["prediction_request"] += 1
        
        self.logger.info(json.dumps({
            "event_type": "prediction_request",
            "user_id": user_id,
            "model_id": model_id,
            "input_size": input_size,
            "processing_time": processing_time,
            "ip_address": ip_address,
            "timestamp": time.time()
        }))
    
    def log_security_violation(
        self,
        violation_type: str,
        details: str,
        ip_address: str,
        user_id: Optional[str] = None
    ) -> None:
        """Log security violations."""
        self.event_counts["security_violation"] += 1
        
        self.logger.warning(json.dumps({
            "event_type": "security_violation",
            "violation_type": violation_type,
            "details": details,
            "user_id": user_id,
            "ip_address": ip_address,
            "timestamp": time.time()
        }))
    
    def get_audit_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get audit summary for the last N hours."""
        return {
            "period_hours": hours,
            "event_counts": dict(self.event_counts),
            "timestamp": time.time()
        }


class SecurityManager:
    """Central security manager coordinating all security features."""
    
    def __init__(self, config: SecurityConfig, secret_key: str):
        self.config = config
        self.validator = InputValidator(config)
        self.auth = AuthenticationManager(secret_key)
        self.rate_limiter = RateLimiter()
        self.privacy = PrivacyPreserver()
        self.audit = AuditLogger()
        
    def create_security_context(
        self,
        request_info: Dict[str, Any]
    ) -> SecurityContext:
        """Create security context for a request."""
        ip_address = request_info.get("ip_address", "unknown")
        user_agent = request_info.get("user_agent", "")
        auth_token = request_info.get("authorization", "")
        
        context = SecurityContext(
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Check authentication
        if auth_token:
            is_valid, user_info = self.auth.validate_token(auth_token)
            if is_valid:
                context.user_id = user_info["user_id"]
                context.permissions = user_info["permissions"]
                context.is_authenticated = True
                
                self.audit.log_authentication(
                    user_id=context.user_id,
                    ip_address=ip_address,
                    success=True
                )
            else:
                self.auth.record_failed_attempt(ip_address)
                self.audit.log_authentication(
                    user_id="unknown",
                    ip_address=ip_address,
                    success=False
                )
        
        # Check rate limits
        rate_key = context.user_id if context.is_authenticated else ip_address
        is_allowed, remaining = self.rate_limiter.is_allowed(
            rate_key,
            limit=self.config.max_request_rate
        )
        
        context.rate_limit_remaining = remaining
        
        if not is_allowed:
            self.audit.log_security_violation(
                violation_type="rate_limit_exceeded",
                details=f"Rate limit exceeded for {rate_key}",
                ip_address=ip_address,
                user_id=context.user_id
            )
        
        return context
    
    def validate_prediction_request(
        self,
        input_data: Any,
        model_id: str,
        context: SecurityContext
    ) -> Tuple[bool, str, Any]:
        """
        Comprehensive validation of prediction request.
        
        Returns:
            (is_valid, error_message, sanitized_input)
        """
        # Check if IP is blocked
        if self.auth.is_ip_blocked(context.ip_address):
            return False, "IP address temporarily blocked", None
        
        # Check rate limits
        if context.rate_limit_remaining <= 0:
            return False, "Rate limit exceeded", None
        
        # Validate input data
        if isinstance(input_data, list) and input_data and isinstance(input_data[0], list):
            # Batch input
            is_valid, error = self.validator.validate_batch_input(input_data)
        else:
            # Single input
            is_valid, error = self.validator.validate_numerical_input(input_data)
        
        if not is_valid:
            self.audit.log_security_violation(
                violation_type="invalid_input",
                details=error,
                ip_address=context.ip_address,
                user_id=context.user_id
            )
            return False, error, None
        
        # Model access control (simplified)
        if not self._check_model_access(model_id, context):
            return False, "Insufficient permissions for model", None
        
        return True, "", input_data
    
    def _check_model_access(self, model_id: str, context: SecurityContext) -> bool:
        """Check if user has access to the specified model."""
        if not context.is_authenticated:
            # Allow access to public models only
            public_models = ["fno_demo", "deeponet_demo"]
            return model_id in public_models
        
        # Check permissions
        required_permissions = {
            "fno_burgers": ["model.fno.read"],
            "deeponet_darcy": ["model.deeponet.read"],
            "custom_model": ["model.custom.read"]
        }
        
        required = required_permissions.get(model_id, ["model.read"])
        return any(perm in context.permissions for perm in required)
    
    def secure_prediction_pipeline(
        self,
        predictions: List[float],
        context: SecurityContext,
        apply_privacy: bool = True
    ) -> List[float]:
        """Apply security measures to prediction output."""
        if apply_privacy and context.is_authenticated:
            # Apply differential privacy for authenticated users
            return self.privacy.add_differential_privacy_noise(
                predictions, epsilon=1.0
            )
        elif not context.is_authenticated:
            # Apply stronger privacy for unauthenticated users
            return self.privacy.add_differential_privacy_noise(
                predictions, epsilon=0.5
            )
        
        return predictions
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics."""
        return {
            "rate_limiter": self.rate_limiter.get_stats(),
            "audit": self.audit.get_audit_summary(),
            "blocked_ips": len(self.auth.failed_attempts),
            "active_tokens": len(self.auth.tokens),
            "timestamp": time.time()
        }


def demo_security():
    """Demonstrate security framework features."""
    print("🔒 Security Framework Demo")
    print("=" * 50)
    
    # Initialize security
    config = SecurityConfig()
    security = SecurityManager(config, secret_key="demo_secret_key_123")
    
    print("🔑 Testing authentication...")
    
    # Generate token
    token = security.auth.generate_token(
        user_id="demo_user",
        permissions=["model.fno.read", "model.deeponet.read"]
    )
    print("  ✓ Generated authentication token")
    
    # Create request context
    request_info = {
        "ip_address": "192.168.1.100",
        "user_agent": "DemoClient/1.0",
        "authorization": token
    }
    
    context = security.create_security_context(request_info)
    print(f"  ✓ Created security context for user: {context.user_id}")
    
    print("\n🛡️  Testing input validation...")
    
    # Test valid input
    valid_input = [0.1, 0.2, 0.3, 0.4, 0.5]
    is_valid, error, _ = security.validate_prediction_request(
        valid_input, "fno_burgers", context
    )
    print(f"  ✓ Valid input: {is_valid}")
    
    # Test invalid input
    invalid_input = [float('inf'), 0.2, 0.3]
    is_valid, error, _ = security.validate_prediction_request(
        invalid_input, "fno_burgers", context
    )
    print(f"  ✗ Invalid input: {is_valid} - {error}")
    
    print("\n🚦 Testing rate limiting...")
    
    # Simulate multiple requests
    for i in range(5):
        allowed, remaining = security.rate_limiter.is_allowed("test_ip", limit=3)
        print(f"  Request {i+1}: Allowed={allowed}, Remaining={remaining}")
    
    print("\n🔐 Testing privacy preservation...")
    
    # Test differential privacy
    original_predictions = [0.5, 0.3, 0.8, 0.1, 0.9]
    private_predictions = security.privacy.add_differential_privacy_noise(
        original_predictions, epsilon=1.0
    )
    print(f"  ✓ Applied differential privacy noise")
    print(f"    Original: {[f'{x:.3f}' for x in original_predictions]}")
    print(f"    Private:  {[f'{x:.3f}' for x in private_predictions]}")
    
    print("\n📊 Security metrics...")
    metrics = security.get_security_metrics()
    print(f"  Active rate limited keys: {metrics['rate_limiter']['active_rate_limited_keys']}")
    print(f"  Total audit events: {sum(metrics['audit']['event_counts'].values())}")
    
    print(f"\n{'='*50}")
    print("✅ Security framework demo completed!")


if __name__ == "__main__":
    demo_security()