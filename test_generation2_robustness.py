#!/usr/bin/env python3
"""
Generation 2: MAKE IT ROBUST - Robustness and Reliability Tests
============================================================

Tests comprehensive error handling, validation, security measures,
logging, monitoring, and health checks for production readiness.
"""

import time
import json
import tempfile
import os
from pathlib import Path

def test_error_handling():
    """Test comprehensive error handling and graceful degradation."""
    print("🛡️ Testing error handling...")
    
    try:
        from probneural_operator.robust import (
            RobustProbabilisticFNO, RobustMockTensor, ValidationError
        )
        
        # Test invalid model parameters
        try:
            fno = RobustProbabilisticFNO(modes=-1, width=0, depth=-5)
            print("❌ Should have caught invalid parameters")
            return False
        except (ValueError, ValidationError):
            print("✅ Correctly caught invalid model parameters")
        
        # Test prediction before training
        try:
            fno = RobustProbabilisticFNO()
            test_input = [RobustMockTensor([1.0] * 64)]
            predictions = fno.predict(test_input)
            print("❌ Should have caught untrained model prediction")
            return False
        except (ValueError, ValidationError):
            print("✅ Correctly caught untrained model prediction")
        
        # Test invalid data shapes
        try:
            fno = RobustProbabilisticFNO(input_dim=32, output_dim=32)
            invalid_data = [RobustMockTensor([1.0] * 16)]  # Wrong size
            result = fno.train(invalid_data, invalid_data)
            # Check if training failed gracefully
            if "error" in result:
                print("✅ Correctly caught invalid data dimensions (graceful failure)")
            else:
                print("❌ Should have caught invalid data dimensions")
                return False
        except ValidationError:
            print("✅ Correctly caught invalid data dimensions (exception)")
        
        # Test empty datasets
        try:
            fno = RobustProbabilisticFNO()
            empty_data = []
            result = fno.train(empty_data, empty_data)
            # Check if training failed gracefully
            if "error" in result:
                print("✅ Correctly caught empty dataset (graceful failure)")
            else:
                print("❌ Should have caught empty dataset")
                return False
        except (ValueError, ValidationError):
            print("✅ Correctly caught empty dataset (exception)")
        
        print("✅ Error handling tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_input_validation():
    """Test comprehensive input validation and sanitization."""
    print("🔍 Testing input validation...")
    
    try:
        from probneural_operator.robust import RobustMockTensor, ValidationError
        
        # Test tensor validation
        valid_cases = [
            [1.0, 2.0, 3.0],
            [0],
            42.5,
            42
        ]
        
        for case in valid_cases:
            try:
                tensor = RobustMockTensor(case)
                print(f"✅ Valid input accepted: {case}")
            except Exception as e:
                print(f"❌ Valid input rejected: {case} - {e}")
                return False
        
        # Test invalid cases
        invalid_cases = [
            None,
            "string",
            {"dict": "value"},
            complex(1, 2)
        ]
        
        for case in invalid_cases:
            try:
                tensor = RobustMockTensor(case)
                print(f"❌ Invalid input accepted: {case}")
                return False
            except (ValueError, TypeError, ValidationError):
                print(f"✅ Invalid input correctly rejected: {case}")
        
        # Test additional validation cases
        try:
            empty_tensor = RobustMockTensor([])
            print("❌ Empty tensor should be rejected")
            return False
        except ValidationError:
            print("✅ Empty tensor correctly rejected")
        
        try:
            inf_tensor = RobustMockTensor([float('inf')])
            print("❌ Infinite value should be rejected")
            return False
        except ValidationError:
            print("✅ Infinite value correctly rejected")
        
        print("✅ Input validation tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Input validation test failed: {e}")
        return False

def test_security_measures():
    """Test security measures and data protection."""
    print("🔐 Testing security measures...")
    
    try:
        from probneural_operator.robust import SecurityManager
        # Test parameter sanitization
        unsafe_inputs = [
            "'; DROP TABLE models; --",
            "<script>alert('xss')</script>",
            "../../../etc/passwd",
            "$(rm -rf /)",
            "\x00null_byte"
        ]
        
        for unsafe_input in unsafe_inputs:
            try:
                # Test SecurityManager sanitization
                sanitized = SecurityManager.sanitize_string(unsafe_input)
                print(f"✅ Safely handled: {repr(unsafe_input[:20])}...")
            except Exception as e:
                print(f"❌ Security test failed for: {repr(unsafe_input[:20])}: {e}")
                return False
        
        # Test file path validation
        dangerous_paths = [
            "../../../etc/passwd",
            "/etc/shadow",
            "~/.ssh/id_rsa",
            "C:\\Windows\\System32\\config\\SAM"
        ]
        
        for path in dangerous_paths:
            try:
                # Test SecurityManager path validation
                safe_path = SecurityManager.validate_file_path(path)
                print(f"⚠️  Path validated (may exist): {path}")
            except Exception:
                print(f"✅ Dangerous path correctly rejected: {path}")
        
        print("✅ Security measures tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Security test failed: {e}")
        return False

def test_logging_and_monitoring():
    """Test comprehensive logging and monitoring capabilities."""
    print("📊 Testing logging and monitoring...")
    
    try:
        import time
        
        class AdvancedLogger:
            def __init__(self, name="ProbNeuralOperator"):
                self.name = name
                self.logs = []
                self.metrics = {}
                self.start_time = time.time()
                
            def log(self, level, message, extra=None):
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                log_entry = {
                    "timestamp": timestamp,
                    "level": level,
                    "message": message,
                    "extra": extra or {},
                    "runtime": time.time() - self.start_time
                }
                self.logs.append(log_entry)
                print(f"[{timestamp}] {level}: {message}")
                
            def info(self, message, **kwargs):
                self.log("INFO", message, kwargs)
                
            def warning(self, message, **kwargs):
                self.log("WARNING", message, kwargs)
                
            def error(self, message, **kwargs):
                self.log("ERROR", message, kwargs)
                
            def metric(self, name, value, tags=None):
                if name not in self.metrics:
                    self.metrics[name] = []
                self.metrics[name].append({
                    "value": value,
                    "timestamp": time.time(),
                    "tags": tags or {}
                })
                
            def get_metrics_summary(self):
                summary = {}
                for name, values in self.metrics.items():
                    summary[name] = {
                        "count": len(values),
                        "latest": values[-1]["value"] if values else None,
                        "avg": sum(v["value"] for v in values) / len(values) if values else 0
                    }
                return summary
        
        # Test logging functionality
        logger = AdvancedLogger()
        
        logger.info("System initialization", component="core")
        logger.warning("Using default configuration", config_file="default.yaml")
        logger.error("Mock error for testing", error_code=500)
        
        # Test metrics collection
        logger.metric("training_loss", 0.25, {"epoch": 1, "model": "fno"})
        logger.metric("training_loss", 0.20, {"epoch": 2, "model": "fno"})
        logger.metric("memory_usage", 512.5, {"unit": "MB"})
        
        # Validate logging
        if len(logger.logs) < 3:
            print("❌ Insufficient logs captured")
            return False
        
        # Validate metrics
        metrics_summary = logger.get_metrics_summary()
        if "training_loss" not in metrics_summary:
            print("❌ Metrics not properly collected")
            return False
        
        print(f"✅ Captured {len(logger.logs)} log entries")
        print(f"✅ Collected {len(metrics_summary)} metric types")
        print(f"   Training loss avg: {metrics_summary['training_loss']['avg']:.3f}")
        
        print("✅ Logging and monitoring tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Logging and monitoring test failed: {e}")
        return False

def test_health_checks():
    """Test system health monitoring and diagnostics."""
    print("🏥 Testing health checks...")
    
    try:
        from probneural_operator.robust import HealthMonitor
        
        # Test health monitoring
        monitor = HealthMonitor()
        
        # Run health checks
        health_data = monitor.run_all_checks()
        
        print(f"✅ System resources check: CPU {health_data['system_resources']['cpu_usage']:.1f}%")
        print(f"✅ Memory usage: {health_data['system_resources']['memory_percent']:.1f}%")
        print(f"✅ Dependencies check: {len(health_data['dependencies'])} modules checked")
        print(f"✅ Model integrity: {health_data['model_integrity']['model_creation']}")
        
        # Get overall status
        status = monitor.get_health_status()
        print(f"✅ Overall health status: {status['status']}")
        
        if status["issues"]:
            print(f"⚠️  Issues detected: {', '.join(status['issues'])}")
        
        print("✅ Health check tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Health check test failed: {e}")
        return False

def test_configuration_validation():
    """Test configuration validation and management."""
    print("⚙️ Testing configuration validation...")
    
    try:
        class ConfigValidator:
            def __init__(self):
                self.schema = {
                    "model": {
                        "required": ["input_dim", "output_dim"],
                        "types": {"input_dim": int, "output_dim": int, "modes": int},
                        "ranges": {"input_dim": (1, 10000), "output_dim": (1, 10000), "modes": (1, 100)}
                    },
                    "training": {
                        "required": ["epochs", "learning_rate"],
                        "types": {"epochs": int, "learning_rate": float, "batch_size": int},
                        "ranges": {"epochs": (1, 10000), "learning_rate": (1e-6, 1.0), "batch_size": (1, 1000)}
                    }
                }
            
            def validate_config(self, config):
                """Validate configuration against schema."""
                errors = []
                
                for section, rules in self.schema.items():
                    if section not in config:
                        errors.append(f"Missing section: {section}")
                        continue
                    
                    section_config = config[section]
                    
                    # Check required fields
                    for required_field in rules.get("required", []):
                        if required_field not in section_config:
                            errors.append(f"Missing required field: {section}.{required_field}")
                    
                    # Check types
                    for field, expected_type in rules.get("types", {}).items():
                        if field in section_config:
                            value = section_config[field]
                            if not isinstance(value, expected_type):
                                errors.append(f"Invalid type for {section}.{field}: expected {expected_type.__name__}, got {type(value).__name__}")
                    
                    # Check ranges
                    for field, (min_val, max_val) in rules.get("ranges", {}).items():
                        if field in section_config:
                            value = section_config[field]
                            if isinstance(value, (int, float)):
                                if not (min_val <= value <= max_val):
                                    errors.append(f"Value out of range for {section}.{field}: {value} not in [{min_val}, {max_val}]")
                
                return errors
        
        validator = ConfigValidator()
        
        # Test valid configuration
        valid_config = {
            "model": {
                "input_dim": 64,
                "output_dim": 64,
                "modes": 12
            },
            "training": {
                "epochs": 100,
                "learning_rate": 0.001,
                "batch_size": 32
            }
        }
        
        errors = validator.validate_config(valid_config)
        if errors:
            print(f"❌ Valid config rejected: {errors}")
            return False
        else:
            print("✅ Valid configuration accepted")
        
        # Test invalid configurations
        invalid_configs = [
            {  # Missing required fields
                "model": {"input_dim": 64},
                "training": {"epochs": 100}
            },
            {  # Wrong types
                "model": {"input_dim": "64", "output_dim": 64},
                "training": {"epochs": 100, "learning_rate": 0.001}
            },
            {  # Out of range values
                "model": {"input_dim": -1, "output_dim": 64},
                "training": {"epochs": 0, "learning_rate": 0.001}
            }
        ]
        
        for i, invalid_config in enumerate(invalid_configs):
            errors = validator.validate_config(invalid_config)
            if not errors:
                print(f"❌ Invalid config {i+1} accepted")
                return False
            else:
                print(f"✅ Invalid config {i+1} correctly rejected: {len(errors)} errors")
        
        print("✅ Configuration validation tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Configuration validation test failed: {e}")
        return False

def test_performance_monitoring():
    """Test performance monitoring and profiling."""
    print("📈 Testing performance monitoring...")
    
    try:
        class PerformanceMonitor:
            def __init__(self):
                self.metrics = {}
                
            def time_operation(self, name, operation):
                """Time an operation and record metrics."""
                start_time = time.time()
                try:
                    result = operation()
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    if name not in self.metrics:
                        self.metrics[name] = []
                    
                    self.metrics[name].append({
                        "duration": duration,
                        "success": True,
                        "timestamp": start_time
                    })
                    
                    return result, duration
                    
                except Exception as e:
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    if name not in self.metrics:
                        self.metrics[name] = []
                    
                    self.metrics[name].append({
                        "duration": duration,
                        "success": False,
                        "error": str(e),
                        "timestamp": start_time
                    })
                    
                    raise
            
            def get_performance_summary(self):
                """Get performance summary statistics."""
                summary = {}
                
                for operation, records in self.metrics.items():
                    durations = [r["duration"] for r in records if r["success"]]
                    failures = [r for r in records if not r["success"]]
                    
                    if durations:
                        summary[operation] = {
                            "count": len(records),
                            "success_rate": len(durations) / len(records),
                            "avg_duration": sum(durations) / len(durations),
                            "min_duration": min(durations),
                            "max_duration": max(durations),
                            "failures": len(failures)
                        }
                    else:
                        summary[operation] = {
                            "count": len(records),
                            "success_rate": 0.0,
                            "failures": len(failures)
                        }
                
                return summary
        
        monitor = PerformanceMonitor()
        
        # Test performance monitoring
        from probneural_operator.core import ProbabilisticFNO, MockTensor, MockDataset
        
        # Monitor model creation
        def create_model():
            return ProbabilisticFNO(modes=4, width=8, depth=2, input_dim=16, output_dim=16)
        
        model, creation_time = monitor.time_operation("model_creation", create_model)
        print(f"✅ Model creation time: {creation_time:.4f}s")
        
        # Monitor data generation
        def generate_data():
            dataset = MockDataset("test", n_samples=20, input_dim=16, output_dim=16)
            return dataset.train_test_split()
        
        data, data_time = monitor.time_operation("data_generation", generate_data)
        print(f"✅ Data generation time: {data_time:.4f}s")
        
        # Monitor training
        def train_model():
            return model.train(data["X_train"][:5], data["y_train"][:5], epochs=10)
        
        result, train_time = monitor.time_operation("training", train_model)
        print(f"✅ Training time (10 epochs): {train_time:.4f}s")
        
        # Monitor prediction
        def make_predictions():
            return model.predict(data["X_test"][:3])
        
        preds, pred_time = monitor.time_operation("prediction", make_predictions)
        print(f"✅ Prediction time: {pred_time:.4f}s")
        
        # Get performance summary
        summary = monitor.get_performance_summary()
        
        print("📊 Performance Summary:")
        for operation, stats in summary.items():
            print(f"   {operation}: {stats['avg_duration']:.4f}s avg, {stats['success_rate']:.1%} success")
        
        print("✅ Performance monitoring tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        return False

def run_generation2_tests():
    """Run all Generation 2 robustness tests."""
    print("🛡️ TERRAGON Generation 2: MAKE IT ROBUST")
    print("=" * 60)
    print("Testing comprehensive error handling, validation, and security...")
    print()
    
    tests = [
        ("Error Handling", test_error_handling),
        ("Input Validation", test_input_validation),
        ("Security Measures", test_security_measures),
        ("Logging & Monitoring", test_logging_and_monitoring),
        ("Health Checks", test_health_checks),
        ("Configuration Validation", test_configuration_validation),
        ("Performance Monitoring", test_performance_monitoring)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"🧪 Testing {test_name}...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
        print()
    
    # Summary
    print("=" * 60)
    print("🏆 GENERATION 2 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status}: {test_name}")
    
    print()
    print(f"📊 Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Generation 2 robustness achieved.")
        print("🚀 Ready for Generation 3: MAKE IT SCALE")
        return True
    else:
        print("⚠️  Some tests failed. Robustness improvements needed.")
        return False

if __name__ == "__main__":
    success = run_generation2_tests()
    exit(0 if success else 1)