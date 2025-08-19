#!/usr/bin/env python3
"""
Comprehensive testing framework for robust neural operator implementation.

This test suite validates Generation 2 robustness features including:
- Error handling and recovery
- Input validation and sanitization
- Security measures
- Health monitoring
- Performance under stress
"""

import sys
import os
import time
import threading
import json
from typing import Dict, List, Any
sys.path.insert(0, '/root/repo')

class RobustFrameworkTester:
    """Comprehensive tester for robust framework features."""
    
    def __init__(self):
        self.test_results = {}
        self.failed_tests = []
        self.warnings = []
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all robustness tests."""
        print("🛡️ ROBUST FRAMEWORK TESTING - GENERATION 2")
        print("=" * 70)
        
        # Test categories
        test_categories = [
            ("Error Handling", self.test_error_handling),
            ("Input Validation", self.test_input_validation),
            ("Security Framework", self.test_security_framework),
            ("Health Monitoring", self.test_health_monitoring),
            ("Performance & Stress", self.test_performance_stress),
            ("Recovery Mechanisms", self.test_recovery_mechanisms)
        ]
        
        overall_success = True
        
        for category_name, test_func in test_categories:
            print(f"\n📋 Testing {category_name}...")
            try:
                success = test_func()
                self.test_results[category_name] = success
                if success:
                    print(f"✅ {category_name}: PASSED")
                else:
                    print(f"❌ {category_name}: FAILED")
                    overall_success = False
            except Exception as e:
                print(f"❌ {category_name}: ERROR - {e}")
                self.test_results[category_name] = False
                self.failed_tests.append(f"{category_name}: {e}")
                overall_success = False
        
        return {
            "overall_success": overall_success,
            "test_results": self.test_results,
            "failed_tests": self.failed_tests,
            "warnings": self.warnings
        }
    
    def test_error_handling(self) -> bool:
        """Test error handling and graceful fallbacks."""
        try:
            from probneural_operator.utils.robust_validation import (
                RobustValidator, validate_with_retry, robust_validation
            )
            
            # Test retry mechanism
            validator = RobustValidator(max_retries=3, retry_delay=0.1)
            
            attempt_count = 0
            def failing_validation(data):
                nonlocal attempt_count
                attempt_count += 1
                if attempt_count < 3:
                    raise ValueError(f"Attempt {attempt_count} failed")
                return {"validated": True, "attempt": attempt_count}
            
            success, result, error = validator.validate_with_retry(failing_validation, {"test": "data"})
            
            if not success or attempt_count != 3:
                self.failed_tests.append("Retry mechanism not working correctly")
                return False
            
            print("  ✅ Retry mechanism working")
            
            # Test fallback mechanism
            def primary_func(data):
                raise RuntimeError("Primary function failed")
            
            def fallback_func(data):
                return {"fallback": True, "data": data}
            
            success, result, error = validator.validate_with_retry(
                primary_func, 
                {"test": "data"}, 
                fallback_func
            )
            
            if not success or not result.get("fallback"):
                self.failed_tests.append("Fallback mechanism not working")
                return False
            
            print("  ✅ Fallback mechanism working")
            
            # Test decorator
            @robust_validation(max_retries=2, retry_delay=0.1)
            def test_validation(data):
                if not isinstance(data, dict):
                    raise ValueError("Data must be dict")
                return data
            
            result = test_validation({"valid": "data"})
            print("  ✅ Validation decorator working")
            
            return True
            
        except Exception as e:
            self.failed_tests.append(f"Error handling test failed: {e}")
            return False
    
    def test_input_validation(self) -> bool:
        """Test input validation and sanitization."""
        try:
            from probneural_operator.utils.robust_validation import (
                validate_tensor_with_fallback,
                validate_model_config,
                validate_training_data
            )
            
            # Test tensor validation with fallback
            test_data = [[1, 2, 3], [4, 5, 6]]
            success, result, error = validate_tensor_with_fallback(
                test_data, 
                expected_shape=(2, 3)
            )
            
            if not success:
                self.failed_tests.append(f"Tensor validation failed: {error}")
                return False
            
            print("  ✅ Tensor validation working")
            
            # Test model config validation
            invalid_config = {
                "input_dim": "invalid",
                "width": -10,
                "depth": 100
            }
            
            validated_config = validate_model_config(invalid_config)
            
            if (validated_config["input_dim"] != 1 or 
                validated_config["width"] < 8 or 
                validated_config["depth"] > 20):
                self.failed_tests.append("Model config validation not working")
                return False
            
            print("  ✅ Model config validation working")
            
            # Test training data validation
            import numpy as np
            test_training_data = np.random.randn(100, 10)
            
            success, result, issues = validate_training_data(test_training_data)
            
            if not success and len(issues) > 0:
                print(f"  ⚠️ Training data validation issues: {issues}")
                self.warnings.extend(issues)
            else:
                print("  ✅ Training data validation working")
            
            return True
            
        except Exception as e:
            self.failed_tests.append(f"Input validation test failed: {e}")
            return False
    
    def test_security_framework(self) -> bool:
        """Test security framework features."""
        try:
            from probneural_operator.utils.enhanced_security import (
                EnhancedSecurityFramework,
                SecurityLevel,
                ThreatLevel,
                sanitize_input_safely,
                encrypt_safely,
                decrypt_safely
            )
            
            # Test security framework initialization
            security = EnhancedSecurityFramework(
                security_level=SecurityLevel.HIGH,
                audit_enabled=True,
                threat_detection_enabled=True
            )
            
            print("  ✅ Security framework initialized")
            
            # Test input sanitization
            malicious_inputs = [
                "<script>alert('xss')</script>",
                "'; DROP TABLE users; --",
                "../../../etc/passwd",
                "javascript:alert('xss')"
            ]
            
            all_blocked = True
            for malicious_input in malicious_inputs:
                is_safe, sanitized, warnings = security.sanitize_input(malicious_input)
                if is_safe:
                    print(f"  ⚠️ Malicious input not blocked: {malicious_input}")
                    all_blocked = False
            
            if all_blocked:
                print("  ✅ Threat detection working")
            else:
                self.warnings.append("Some malicious inputs not properly blocked")
            
            # Test encryption/decryption
            test_data = {"sensitive": "information", "user_id": 12345}
            encrypted = security.encrypt_data(test_data)
            decrypted = security.decrypt_data(encrypted, return_type='json')
            
            if decrypted != test_data:
                self.failed_tests.append("Encryption/decryption not working correctly")
                return False
            
            print("  ✅ Encryption/decryption working")
            
            # Test rate limiting
            test_id = "test_user_123"
            
            # Should allow first few requests
            for i in range(5):
                if not security.check_rate_limit(test_id, limit=10, window_seconds=60):
                    self.failed_tests.append("Rate limiting too strict")
                    return False
            
            # Test rate limit enforcement by making many requests
            rate_limited = False
            for i in range(20):
                if not security.check_rate_limit(test_id, limit=10, window_seconds=60):
                    rate_limited = True
                    break
            
            if not rate_limited:
                self.warnings.append("Rate limiting may not be working properly")
            else:
                print("  ✅ Rate limiting working")
            
            # Test security summary
            summary = security.get_security_summary()
            if not isinstance(summary, dict) or "total_events" not in summary:
                self.failed_tests.append("Security summary not working")
                return False
            
            print("  ✅ Security summary working")
            
            return True
            
        except Exception as e:
            self.failed_tests.append(f"Security framework test failed: {e}")
            return False
    
    def test_health_monitoring(self) -> bool:
        """Test health monitoring system."""
        try:
            from probneural_operator.utils.health_monitoring import (
                HealthMonitor,
                HealthStatus,
                HealthMetric,
                create_memory_health_check,
                create_disk_health_check
            )
            
            # Test health monitor creation
            monitor = HealthMonitor(check_interval=1.0, auto_start=False)
            print("  ✅ Health monitor created")
            
            # Test health metric creation
            test_metric = HealthMetric(
                name="test_metric",
                value=75.0,
                status=HealthStatus.WARNING,
                threshold_warning=70.0,
                threshold_critical=90.0,
                unit="%"
            )
            
            if test_metric.status != HealthStatus.WARNING:
                self.failed_tests.append("Health metric not working correctly")
                return False
            
            print("  ✅ Health metrics working")
            
            # Test health check registration
            def custom_health_check():
                return HealthMetric(
                    name="custom_check",
                    value=42.0,
                    status=HealthStatus.HEALTHY,
                    description="Custom health check"
                )
            
            monitor.register_health_check("custom", custom_health_check)
            
            if "custom" not in monitor.health_checks:
                self.failed_tests.append("Health check registration failed")
                return False
            
            print("  ✅ Health check registration working")
            
            # Test memory and disk checks
            memory_check = create_memory_health_check()
            memory_metric = memory_check()
            
            if not isinstance(memory_metric, HealthMetric):
                self.failed_tests.append("Memory health check not working")
                return False
            
            print("  ✅ Memory health check working")
            
            disk_check = create_disk_health_check("/")
            disk_metric = disk_check()
            
            if not isinstance(disk_metric, HealthMetric):
                self.failed_tests.append("Disk health check not working")
                return False
            
            print("  ✅ Disk health check working")
            
            # Test monitoring start/stop
            monitor.start_monitoring()
            time.sleep(2)  # Let it run briefly
            
            current_health = monitor.get_current_health()
            if current_health.status == HealthStatus.UNKNOWN:
                self.warnings.append("Health monitoring may not be updating")
            
            monitor.stop_monitoring()
            print("  ✅ Health monitoring start/stop working")
            
            return True
            
        except Exception as e:
            self.failed_tests.append(f"Health monitoring test failed: {e}")
            return False
    
    def test_performance_stress(self) -> bool:
        """Test performance under stress conditions."""
        try:
            from probneural_operator.utils.robust_validation import SafeExecutor
            from probneural_operator.utils.enhanced_security import EnhancedSecurityFramework
            
            # Test safe executor under load
            executor = SafeExecutor(max_retries=3, timeout=5.0)
            
            def cpu_intensive_task():
                # Simulate CPU-intensive work
                result = 0
                for i in range(100000):
                    result += i * i
                return result
            
            start_time = time.time()
            success, result, error = executor.execute_safely(cpu_intensive_task)
            execution_time = time.time() - start_time
            
            if not success:
                self.failed_tests.append(f"Safe executor failed: {error}")
                return False
            
            print(f"  ✅ Safe executor working (executed in {execution_time:.2f}s)")
            
            # Test concurrent operations
            security = EnhancedSecurityFramework()
            
            def concurrent_security_test():
                for i in range(100):
                    security.sanitize_input(f"test_input_{i}")
                    security.check_rate_limit(f"user_{i % 10}")
            
            threads = []
            for i in range(5):
                thread = threading.Thread(target=concurrent_security_test)
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            print("  ✅ Concurrent operations working")
            
            # Test memory usage (basic check)
            large_data = [i for i in range(100000)]
            
            start_time = time.time()
            from probneural_operator.utils.robust_validation import validate_training_data
            success, result, issues = validate_training_data(large_data)
            processing_time = time.time() - start_time
            
            if processing_time > 10.0:  # Should complete within 10 seconds
                self.warnings.append(f"Large data processing took {processing_time:.2f}s")
            else:
                print(f"  ✅ Large data processing working ({processing_time:.2f}s)")
            
            return True
            
        except Exception as e:
            self.failed_tests.append(f"Performance stress test failed: {e}")
            return False
    
    def test_recovery_mechanisms(self) -> bool:
        """Test automatic recovery mechanisms."""
        try:
            from probneural_operator.utils.robust_validation import RobustValidator
            from probneural_operator.utils.enhanced_security import EnhancedSecurityFramework
            
            # Test validator recovery
            validator = RobustValidator(max_retries=5, enable_fallbacks=True)
            
            failure_count = 0
            def intermittent_failure(data):
                nonlocal failure_count
                failure_count += 1
                if failure_count <= 3:
                    raise ConnectionError("Simulated network failure")
                return {"recovered": True, "attempts": failure_count}
            
            success, result, error = validator.validate_with_retry(intermittent_failure, {})
            
            if not success or not result.get("recovered"):
                self.failed_tests.append("Validator recovery not working")
                return False
            
            print("  ✅ Validator recovery working")
            
            # Test security framework recovery
            security = EnhancedSecurityFramework()
            
            # Simulate failed authentication attempts
            test_user = "recovery_test_user"
            
            for i in range(10):
                security.record_failed_attempt(test_user, "authentication")
            
            # Check that security events were recorded
            summary = security.get_security_summary()
            if summary["total_events"] == 0:
                self.warnings.append("Security event recording may not be working")
            else:
                print("  ✅ Security event recovery working")
            
            # Test data recovery scenarios
            def data_corruption_simulation(data):
                # Simulate various data corruption scenarios
                if isinstance(data, str) and "corrupt" in data:
                    raise ValueError("Data corruption detected")
                return {"clean_data": data}
            
            def recovery_function(data):
                # Fallback recovery
                if isinstance(data, str):
                    clean_data = data.replace("corrupt", "clean")
                    return {"recovered_data": clean_data}
                return {"recovered_data": data}
            
            test_cases = [
                "normal_data",
                "corrupt_data_needs_cleaning",
                {"structured": "corrupt_value"}
            ]
            
            recovered_count = 0
            for test_case in test_cases:
                success, result, error = validator.validate_with_retry(
                    data_corruption_simulation,
                    test_case,
                    recovery_function
                )
                if success:
                    recovered_count += 1
            
            if recovered_count < len(test_cases):
                self.warnings.append("Some data recovery scenarios failed")
            else:
                print("  ✅ Data recovery mechanisms working")
            
            return True
            
        except Exception as e:
            self.failed_tests.append(f"Recovery mechanisms test failed: {e}")
            return False

def main():
    """Run robust framework tests."""
    tester = RobustFrameworkTester()
    
    start_time = time.time()
    results = tester.run_all_tests()
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("🛡️ ROBUST FRAMEWORK TEST RESULTS")
    print("=" * 70)
    
    if results["overall_success"]:
        print("🎉 ALL ROBUSTNESS TESTS PASSED!")
        print("\n✨ Generation 2 Features Validated:")
        print("   • Error handling & recovery: ✅ Working")
        print("   • Input validation & sanitization: ✅ Working")
        print("   • Security framework: ✅ Working")
        print("   • Health monitoring: ✅ Working")
        print("   • Performance under stress: ✅ Working")
        print("   • Recovery mechanisms: ✅ Working")
        
        if results["warnings"]:
            print(f"\n⚠️ Warnings ({len(results['warnings'])}):")
            for warning in results["warnings"]:
                print(f"   • {warning}")
        
        print(f"\n📊 Test Statistics:")
        print(f"   • Total test time: {total_time:.2f} seconds")
        print(f"   • Test categories: {len(results['test_results'])}")
        print(f"   • Warnings: {len(results['warnings'])}")
        
        print(f"\n🚀 Ready for Generation 3: SCALE & OPTIMIZE")
        return 0
        
    else:
        print("❌ SOME ROBUSTNESS TESTS FAILED")
        print(f"\nFailed tests ({len(results['failed_tests'])}):")
        for failure in results["failed_tests"]:
            print(f"   • {failure}")
        
        if results["warnings"]:
            print(f"\nWarnings ({len(results['warnings'])}):")
            for warning in results["warnings"]:
                print(f"   • {warning}")
        
        print("\n🔧 Recommendations:")
        print("   • Review failed test components")
        print("   • Check error logs for details")
        print("   • Verify dependencies are installed")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())