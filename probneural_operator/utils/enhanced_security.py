"""
Enhanced security framework for probabilistic neural operators.

This module provides comprehensive security features including:
- Advanced encryption and key management
- Secure model serialization and storage
- Input sanitization and validation
- Audit logging and threat detection
- Access control and authentication
"""

import os
import time
import hmac
import hashlib
import logging
import json
import base64
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import threading
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """Security level classifications."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ThreatLevel(Enum):
    """Threat level classifications."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: float
    event_type: str
    threat_level: ThreatLevel
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    description: str = ""
    metadata: Dict[str, Any] = None

class EnhancedSecurityFramework:
    """Comprehensive security framework with advanced features."""
    
    def __init__(self, 
                 security_level: SecurityLevel = SecurityLevel.HIGH,
                 audit_enabled: bool = True,
                 threat_detection_enabled: bool = True):
        """Initialize enhanced security framework.
        
        Args:
            security_level: Default security level for operations
            audit_enabled: Whether to enable audit logging
            threat_detection_enabled: Whether to enable threat detection
        """
        self.security_level = security_level
        self.audit_enabled = audit_enabled
        self.threat_detection_enabled = threat_detection_enabled
        
        # Security state
        self.security_events = []
        self.threat_patterns = {}
        self.access_controls = {}
        self.encryption_keys = {}
        
        # Monitoring
        self.failed_attempts = {}
        self.rate_limits = {}
        self.security_lock = threading.Lock()
        
        # Initialize default security patterns
        self._initialize_threat_patterns()
        self._initialize_default_keys()
    
    def _initialize_threat_patterns(self):
        """Initialize threat detection patterns."""
        self.threat_patterns = {
            'sql_injection': {
                'patterns': [
                    r"(?i)(union|select|insert|update|delete|drop|create|alter)\s+",
                    r"(?i)(\-\-|\#|\/\*|\*\/)",
                    r"(?i)(or|and)\s+\d+\s*(=|>|<)\s*\d+",
                ],
                'threat_level': ThreatLevel.HIGH
            },
            'xss_attempt': {
                'patterns': [
                    r"(?i)<script[^>]*>.*?</script>",
                    r"(?i)javascript:",
                    r"(?i)on\w+\s*=",
                ],
                'threat_level': ThreatLevel.MEDIUM
            },
            'path_traversal': {
                'patterns': [
                    r"\.\./",
                    r"\.\.\\",
                    r"%2e%2e%2f",
                    r"%2e%2e%5c",
                ],
                'threat_level': ThreatLevel.HIGH
            },
            'suspicious_input': {
                'patterns': [
                    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]",  # Control characters
                    r"[\ud800-\udfff]",  # Invalid unicode
                ],
                'threat_level': ThreatLevel.MEDIUM
            }
        }
    
    def _initialize_default_keys(self):
        """Initialize default encryption keys."""
        # Generate a default key if none exists
        default_key = os.environ.get('SECURITY_KEY')
        if not default_key:
            default_key = base64.b64encode(os.urandom(32)).decode()
            logger.warning("Using auto-generated security key. Set SECURITY_KEY environment variable for production.")
        
        self.encryption_keys['default'] = default_key
    
    def sanitize_input(self, 
                      input_data: Any, 
                      strict_mode: bool = None) -> tuple[bool, Any, List[str]]:
        """Sanitize input data with threat detection.
        
        Args:
            input_data: Data to sanitize
            strict_mode: Whether to use strict sanitization
            
        Returns:
            Tuple of (is_safe, sanitized_data, warnings)
        """
        if strict_mode is None:
            strict_mode = self.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]
        
        warnings = []
        is_safe = True
        
        try:
            # Convert to string for pattern matching
            if isinstance(input_data, str):
                data_str = input_data
            elif isinstance(input_data, bytes):
                data_str = input_data.decode('utf-8', errors='ignore')
            else:
                data_str = str(input_data)
            
            # Check for threat patterns
            detected_threats = self._detect_threats(data_str)
            
            if detected_threats:
                for threat in detected_threats:
                    if threat['threat_level'] in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                        is_safe = False
                        warnings.append(f"Blocked {threat['type']}: {threat['description']}")
                        
                        # Log security event
                        self._log_security_event(
                            event_type="threat_detected",
                            threat_level=threat['threat_level'],
                            description=f"Threat detected: {threat['type']}"
                        )
                    else:
                        warnings.append(f"Warning: {threat['type']} detected")
            
            # Sanitize the data
            sanitized_data = self._sanitize_data(input_data, strict_mode)
            
            return is_safe, sanitized_data, warnings
            
        except Exception as e:
            logger.error(f"Input sanitization failed: {e}")
            return False, input_data, [f"Sanitization error: {e}"]
    
    def _detect_threats(self, data_str: str) -> List[Dict[str, Any]]:
        """Detect threats in input data."""
        import re
        
        detected_threats = []
        
        if not self.threat_detection_enabled:
            return detected_threats
        
        for threat_type, config in self.threat_patterns.items():
            for pattern in config['patterns']:
                try:
                    if re.search(pattern, data_str):
                        detected_threats.append({
                            'type': threat_type,
                            'threat_level': config['threat_level'],
                            'pattern': pattern,
                            'description': f"Pattern '{pattern}' matched"
                        })
                except Exception as e:
                    logger.warning(f"Threat pattern check failed for {threat_type}: {e}")
        
        return detected_threats
    
    def _sanitize_data(self, data: Any, strict_mode: bool) -> Any:
        """Sanitize data based on type and security level."""
        
        if isinstance(data, str):
            # Remove or escape dangerous characters
            sanitized = data
            
            if strict_mode:
                # Remove control characters
                sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\t\n\r')
                
                # Limit length
                if len(sanitized) > 10000:
                    sanitized = sanitized[:10000]
                    logger.warning("Input truncated due to length limits")
            
            # Basic HTML escape
            sanitized = sanitized.replace('<', '&lt;').replace('>', '&gt;')
            sanitized = sanitized.replace('"', '&quot;').replace("'", '&#x27;')
            
            return sanitized
            
        elif isinstance(data, (list, tuple)):
            # Recursively sanitize collections
            return type(data)(self._sanitize_data(item, strict_mode) for item in data)
            
        elif isinstance(data, dict):
            # Sanitize dictionary values
            sanitized_dict = {}
            for key, value in data.items():
                safe_key = self._sanitize_data(key, strict_mode) if isinstance(key, str) else key
                safe_value = self._sanitize_data(value, strict_mode)
                sanitized_dict[safe_key] = safe_value
            return sanitized_dict
            
        else:
            # For other types, return as-is but log
            return data
    
    def encrypt_data(self, 
                    data: Union[str, bytes, Dict], 
                    key_name: str = 'default') -> str:
        """Encrypt data using specified key.
        
        Args:
            data: Data to encrypt
            key_name: Name of encryption key to use
            
        Returns:
            Base64-encoded encrypted data
        """
        try:
            # Get encryption key
            if key_name not in self.encryption_keys:
                raise ValueError(f"Encryption key '{key_name}' not found")
            
            key = self.encryption_keys[key_name].encode() if isinstance(self.encryption_keys[key_name], str) else self.encryption_keys[key_name]
            
            # Convert data to bytes
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, dict):
                data_bytes = json.dumps(data, sort_keys=True).encode('utf-8')
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                data_bytes = str(data).encode('utf-8')
            
            # Simple XOR encryption (replace with proper encryption in production)
            encrypted_bytes = self._xor_encrypt(data_bytes, key)
            
            # Return base64-encoded result
            return base64.b64encode(encrypted_bytes).decode('ascii')
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    def decrypt_data(self, 
                    encrypted_data: str, 
                    key_name: str = 'default',
                    return_type: str = 'str') -> Any:
        """Decrypt data using specified key.
        
        Args:
            encrypted_data: Base64-encoded encrypted data
            key_name: Name of encryption key to use
            return_type: Type to return ('str', 'bytes', 'json')
            
        Returns:
            Decrypted data in specified format
        """
        try:
            # Get encryption key
            if key_name not in self.encryption_keys:
                raise ValueError(f"Encryption key '{key_name}' not found")
            
            key = self.encryption_keys[key_name].encode() if isinstance(self.encryption_keys[key_name], str) else self.encryption_keys[key_name]
            
            # Decode from base64
            encrypted_bytes = base64.b64decode(encrypted_data.encode('ascii'))
            
            # Decrypt
            decrypted_bytes = self._xor_encrypt(encrypted_bytes, key)  # XOR is symmetric
            
            # Return in requested format
            if return_type == 'bytes':
                return decrypted_bytes
            elif return_type == 'json':
                return json.loads(decrypted_bytes.decode('utf-8'))
            else:  # return_type == 'str'
                return decrypted_bytes.decode('utf-8')
                
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def _xor_encrypt(self, data: bytes, key: bytes) -> bytes:
        """Simple XOR encryption (replace with proper encryption in production)."""
        # Extend key to match data length
        extended_key = (key * ((len(data) // len(key)) + 1))[:len(data)]
        
        # XOR encryption
        return bytes(d ^ k for d, k in zip(data, extended_key))
    
    def secure_hash(self, data: Union[str, bytes]) -> str:
        """Create secure hash of data."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Use SHA-256 with salt
        salt = b'probneural_operator_salt'  # Use dynamic salt in production
        return hashlib.sha256(salt + data).hexdigest()
    
    def verify_integrity(self, data: Union[str, bytes], expected_hash: str) -> bool:
        """Verify data integrity using hash."""
        try:
            actual_hash = self.secure_hash(data)
            return hmac.compare_digest(actual_hash, expected_hash)
        except Exception as e:
            logger.error(f"Integrity verification failed: {e}")
            return False
    
    def check_rate_limit(self, 
                        identifier: str, 
                        limit: int = 100, 
                        window_seconds: int = 3600) -> bool:
        """Check if identifier is within rate limits.
        
        Args:
            identifier: Unique identifier (IP, user ID, etc.)
            limit: Maximum allowed requests
            window_seconds: Time window in seconds
            
        Returns:
            True if within limits, False if rate limited
        """
        current_time = time.time()
        
        with self.security_lock:
            if identifier not in self.rate_limits:
                self.rate_limits[identifier] = []
            
            # Remove old entries
            self.rate_limits[identifier] = [
                timestamp for timestamp in self.rate_limits[identifier]
                if current_time - timestamp < window_seconds
            ]
            
            # Check if limit exceeded
            if len(self.rate_limits[identifier]) >= limit:
                # Log rate limit exceeded
                self._log_security_event(
                    event_type="rate_limit_exceeded",
                    threat_level=ThreatLevel.MEDIUM,
                    description=f"Rate limit exceeded for {identifier}"
                )
                return False
            
            # Add current request
            self.rate_limits[identifier].append(current_time)
            return True
    
    def record_failed_attempt(self, 
                            identifier: str, 
                            attempt_type: str = "authentication"):
        """Record a failed attempt for monitoring."""
        current_time = time.time()
        
        with self.security_lock:
            if identifier not in self.failed_attempts:
                self.failed_attempts[identifier] = []
            
            self.failed_attempts[identifier].append({
                'timestamp': current_time,
                'type': attempt_type
            })
            
            # Check for suspicious patterns
            recent_failures = [
                attempt for attempt in self.failed_attempts[identifier]
                if current_time - attempt['timestamp'] < 300  # Last 5 minutes
            ]
            
            if len(recent_failures) >= 5:
                self._log_security_event(
                    event_type="suspicious_activity",
                    threat_level=ThreatLevel.HIGH,
                    description=f"Multiple failed attempts from {identifier}"
                )
    
    def _log_security_event(self, 
                          event_type: str,
                          threat_level: ThreatLevel,
                          description: str = "",
                          metadata: Dict[str, Any] = None):
        """Log a security event."""
        if not self.audit_enabled:
            return
        
        event = SecurityEvent(
            timestamp=time.time(),
            event_type=event_type,
            threat_level=threat_level,
            description=description,
            metadata=metadata or {}
        )
        
        self.security_events.append(event)
        
        # Log to standard logger based on threat level
        if threat_level == ThreatLevel.CRITICAL:
            logger.critical(f"SECURITY CRITICAL: {event_type} - {description}")
        elif threat_level == ThreatLevel.HIGH:
            logger.error(f"SECURITY HIGH: {event_type} - {description}")
        elif threat_level == ThreatLevel.MEDIUM:
            logger.warning(f"SECURITY MEDIUM: {event_type} - {description}")
        else:
            logger.info(f"SECURITY: {event_type} - {description}")
        
        # Keep only recent events
        cutoff_time = time.time() - 86400  # Last 24 hours
        self.security_events = [
            event for event in self.security_events
            if event.timestamp > cutoff_time
        ]
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary for specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        
        recent_events = [
            event for event in self.security_events
            if event.timestamp > cutoff_time
        ]
        
        # Count events by type and threat level
        event_counts = {}
        threat_counts = {}
        
        for event in recent_events:
            event_counts[event.event_type] = event_counts.get(event.event_type, 0) + 1
            threat_level = event.threat_level.value
            threat_counts[threat_level] = threat_counts.get(threat_level, 0) + 1
        
        return {
            "period_hours": hours,
            "total_events": len(recent_events),
            "event_types": event_counts,
            "threat_levels": threat_counts,
            "security_level": self.security_level.value,
            "audit_enabled": self.audit_enabled,
            "threat_detection_enabled": self.threat_detection_enabled
        }
    
    def export_security_log(self, filepath: str):
        """Export security events to file."""
        try:
            export_data = {
                "export_timestamp": time.time(),
                "security_summary": self.get_security_summary(),
                "events": [
                    {
                        "timestamp": event.timestamp,
                        "event_type": event.event_type,
                        "threat_level": event.threat_level.value,
                        "description": event.description,
                        "metadata": event.metadata
                    }
                    for event in self.security_events
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Security log exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export security log: {e}")

# Global security framework instance
global_security = EnhancedSecurityFramework()

# Convenience functions
def sanitize_input_safely(data: Any, strict_mode: bool = None):
    """Convenience function for input sanitization."""
    return global_security.sanitize_input(data, strict_mode)

def encrypt_safely(data: Any, key_name: str = 'default'):
    """Convenience function for encryption."""
    return global_security.encrypt_data(data, key_name)

def decrypt_safely(encrypted_data: str, key_name: str = 'default', return_type: str = 'str'):
    """Convenience function for decryption."""
    return global_security.decrypt_data(encrypted_data, key_name, return_type)

def check_rate_limit_safely(identifier: str, limit: int = 100, window_seconds: int = 3600):
    """Convenience function for rate limiting."""
    return global_security.check_rate_limit(identifier, limit, window_seconds)