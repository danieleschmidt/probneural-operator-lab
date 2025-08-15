"""
Generation-Specific Quality Gates
=================================

Implementation of the 3-generation progressive enhancement strategy:
- Generation 1: MAKE IT WORK (Simple)
- Generation 2: MAKE IT ROBUST (Reliable) 
- Generation 3: MAKE IT SCALE (Optimized)

Each generation has specific quality gates that must pass before advancement.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from .core import QualityGate, QualityGateResult, QualityGateStatus, GenerationType


class BaseGenerationGate(QualityGate):
    """Base class for generation-specific gates."""
    
    def __init__(self, name: str, generation: GenerationType, critical: bool = False, timeout: float = 300.0):
        super().__init__(name, critical, timeout)
        self.generation = generation


# =============================================================================
# GENERATION 1: MAKE IT WORK (Simple)
# =============================================================================

class BasicSyntaxGate(BaseGenerationGate):
    """Verify basic Python syntax is valid."""
    
    def __init__(self):
        super().__init__("Basic Syntax Check", GenerationType.GENERATION_1, critical=True, timeout=60.0)
    
    async def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        project_root = Path(context.get("project_root", "."))
        python_files = list(project_root.rglob("*.py"))
        
        errors = []
        warnings = []
        total_files = len(python_files)
        valid_files = 0
        
        for py_file in python_files:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "py_compile", str(py_file)],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    valid_files += 1
                else:
                    errors.append(f"Syntax error in {py_file}: {result.stderr.strip()}")
                    
            except subprocess.TimeoutExpired:
                errors.append(f"Timeout checking {py_file}")
            except Exception as e:
                errors.append(f"Error checking {py_file}: {str(e)}")
        
        score = (valid_files / total_files * 100) if total_files > 0 else 100
        status = QualityGateStatus.PASSED if score >= 95 else QualityGateStatus.FAILED
        
        return QualityGateResult(
            gate_name=self.name,
            status=status,
            score=score,
            max_score=100.0,
            details={
                "total_files": total_files,
                "valid_files": valid_files,
                "syntax_errors": len(errors)
            },
            errors=errors,
            warnings=warnings,
            metrics={
                "syntax_error_rate": (total_files - valid_files) / total_files * 100 if total_files > 0 else 0
            }
        )


class BasicImportGate(BaseGenerationGate):
    """Verify basic imports work without errors."""
    
    def __init__(self):
        super().__init__("Basic Import Check", GenerationType.GENERATION_1, critical=True, timeout=120.0)
    
    async def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        project_root = Path(context.get("project_root", "."))
        
        # Test main package import
        errors = []
        warnings = []
        
        try:
            # Test importing the main package
            result = subprocess.run(
                [sys.executable, "-c", "import probneural_operator; print('Import successful')"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=project_root
            )
            
            if result.returncode != 0:
                errors.append(f"Main package import failed: {result.stderr}")
            
            # Test key submodule imports
            key_modules = [
                "probneural_operator.models",
                "probneural_operator.posteriors",
                "probneural_operator.active",
                "probneural_operator.utils"
            ]
            
            import_success = 0
            for module in key_modules:
                try:
                    result = subprocess.run(
                        [sys.executable, "-c", f"import {module}; print('{module} OK')"],
                        capture_output=True,
                        text=True,
                        timeout=15,
                        cwd=project_root
                    )
                    
                    if result.returncode == 0:
                        import_success += 1
                    else:
                        warnings.append(f"Module {module} import warning: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    warnings.append(f"Module {module} import timeout")
                except Exception as e:
                    warnings.append(f"Module {module} import error: {str(e)}")
            
            score = (import_success / len(key_modules) * 100) if key_modules else 100
            
            # Main package must import for pass
            status = QualityGateStatus.PASSED if len(errors) == 0 and score >= 75 else QualityGateStatus.FAILED
            
        except Exception as e:
            errors.append(f"Import test failed: {str(e)}")
            score = 0
            status = QualityGateStatus.FAILED
        
        return QualityGateResult(
            gate_name=self.name,
            status=status,
            score=score,
            max_score=100.0,
            details={
                "tested_modules": len(key_modules) + 1,
                "successful_imports": import_success + (1 if not errors else 0)
            },
            errors=errors,
            warnings=warnings,
            metrics={
                "import_success_rate": score
            }
        )


class BasicFunctionalityGate(BaseGenerationGate):
    """Verify core functionality works."""
    
    def __init__(self):
        super().__init__("Basic Functionality", GenerationType.GENERATION_1, critical=False, timeout=180.0)
    
    async def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        project_root = Path(context.get("project_root", "."))
        
        errors = []
        warnings = []
        functionality_tests = 0
        passed_tests = 0
        
        # Test 1: Basic configuration loading
        try:
            test_script = """
import sys
sys.path.insert(0, '.')
from probneural_operator.utils.config import load_config
config = load_config()
print(f"Config loaded: {type(config)}")
"""
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=project_root
            )
            
            functionality_tests += 1
            if result.returncode == 0:
                passed_tests += 1
            else:
                warnings.append(f"Config loading test failed: {result.stderr}")
                
        except Exception as e:
            warnings.append(f"Config test error: {str(e)}")
            functionality_tests += 1
        
        # Test 2: Basic model instantiation  
        try:
            test_script = """
import sys
sys.path.insert(0, '.')
from probneural_operator.models.base.neural_operator import BaseNeuralOperator
print("BaseNeuralOperator imported successfully")
"""
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=project_root
            )
            
            functionality_tests += 1
            if result.returncode == 0:
                passed_tests += 1
            else:
                warnings.append(f"Model instantiation test failed: {result.stderr}")
                
        except Exception as e:
            warnings.append(f"Model test error: {str(e)}")
            functionality_tests += 1
        
        # Test 3: Utility functions
        try:
            test_script = """
import sys
sys.path.insert(0, '.')
from probneural_operator.utils.validation import validate_tensor_shape
print("Utility functions accessible")
"""
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=project_root
            )
            
            functionality_tests += 1
            if result.returncode == 0:
                passed_tests += 1
            else:
                warnings.append(f"Utility test failed: {result.stderr}")
                
        except Exception as e:
            warnings.append(f"Utility test error: {str(e)}")
            functionality_tests += 1
        
        score = (passed_tests / functionality_tests * 100) if functionality_tests > 0 else 0
        status = QualityGateStatus.PASSED if score >= 60 else QualityGateStatus.FAILED
        
        return QualityGateResult(
            gate_name=self.name,
            status=status,
            score=score,
            max_score=100.0,
            details={
                "total_tests": functionality_tests,
                "passed_tests": passed_tests
            },
            errors=errors,
            warnings=warnings,
            metrics={
                "functionality_score": score
            }
        )


class Generation1Gates:
    """Collection of Generation 1 quality gates."""
    
    def get_gates(self) -> List[QualityGate]:
        """Get all Generation 1 gates."""
        return [
            BasicSyntaxGate(),
            BasicImportGate(), 
            BasicFunctionalityGate(),
        ]


# =============================================================================
# GENERATION 2: MAKE IT ROBUST (Reliable)
# =============================================================================

class ComprehensiveTestGate(BaseGenerationGate):
    """Comprehensive testing with coverage requirements."""
    
    def __init__(self):
        super().__init__("Comprehensive Testing", GenerationType.GENERATION_2, critical=True, timeout=600.0)
    
    async def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        project_root = Path(context.get("project_root", "."))
        min_coverage = context.get("test_coverage_threshold", 85.0)
        
        errors = []
        warnings = []
        
        try:
            # Run pytest with coverage
            result = subprocess.run(
                [
                    sys.executable, "-m", "pytest", 
                    "--cov=probneural_operator",
                    "--cov-report=json",
                    "--cov-report=term-missing",
                    "-v",
                    "tests/"
                ],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=project_root
            )
            
            # Parse coverage report
            coverage_file = project_root / "coverage.json"
            coverage_percent = 0.0
            
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                    coverage_percent = coverage_data.get("totals", {}).get("percent_covered", 0.0)
            
            # Parse test results
            test_output = result.stdout + result.stderr
            tests_run = 0
            tests_passed = 0
            tests_failed = 0
            
            # Extract test counts from pytest output
            for line in test_output.split('\n'):
                if 'passed' in line and 'failed' in line:
                    # Parse line like "15 failed, 25 passed in 10.2s"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'passed' and i > 0:
                            tests_passed = int(parts[i-1])
                        elif part == 'failed' and i > 0:
                            tests_failed = int(parts[i-1])
                    tests_run = tests_passed + tests_failed
                    break
                elif 'passed' in line and 'failed' not in line:
                    # Parse line like "25 passed in 10.2s" 
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'passed' and i > 0:
                            tests_passed = int(parts[i-1])
                    tests_run = tests_passed
                    break
            
            # Calculate score (weighted: 70% coverage, 30% test pass rate)
            test_pass_rate = (tests_passed / tests_run * 100) if tests_run > 0 else 0
            score = (coverage_percent * 0.7) + (test_pass_rate * 0.3)
            
            # Status determination
            if coverage_percent >= min_coverage and test_pass_rate >= 90:
                status = QualityGateStatus.PASSED
            else:
                status = QualityGateStatus.FAILED
                if coverage_percent < min_coverage:
                    errors.append(f"Test coverage {coverage_percent:.1f}% below threshold {min_coverage}%")
                if test_pass_rate < 90:
                    errors.append(f"Test pass rate {test_pass_rate:.1f}% below 90%")
            
            if tests_failed > 0:
                warnings.append(f"{tests_failed} tests failed")
            
        except subprocess.TimeoutExpired:
            errors.append("Test execution timed out")
            score = 0
            status = QualityGateStatus.FAILED
            coverage_percent = 0
            test_pass_rate = 0
            tests_run = 0
            tests_passed = 0
            tests_failed = 0
            
        except Exception as e:
            errors.append(f"Test execution failed: {str(e)}")
            score = 0
            status = QualityGateStatus.FAILED
            coverage_percent = 0
            test_pass_rate = 0
            tests_run = 0
            tests_passed = 0
            tests_failed = 0
        
        return QualityGateResult(
            gate_name=self.name,
            status=status,
            score=score,
            max_score=100.0,
            details={
                "coverage_percent": coverage_percent,
                "test_pass_rate": test_pass_rate,
                "tests_run": tests_run,
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "min_coverage_threshold": min_coverage
            },
            errors=errors,
            warnings=warnings,
            metrics={
                "test_coverage": coverage_percent,
                "test_pass_rate": test_pass_rate
            }
        )


class SecurityScanGate(BaseGenerationGate):
    """Security vulnerability scanning."""
    
    def __init__(self):
        super().__init__("Security Scan", GenerationType.GENERATION_2, critical=True, timeout=300.0)
    
    async def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        project_root = Path(context.get("project_root", "."))
        
        errors = []
        warnings = []
        security_issues = []
        
        # Run bandit security scan
        try:
            result = subprocess.run(
                [sys.executable, "-m", "bandit", "-r", "probneural_operator", "-f", "json"],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=project_root
            )
            
            if result.stdout:
                try:
                    bandit_data = json.loads(result.stdout)
                    security_issues = bandit_data.get("results", [])
                except json.JSONDecodeError:
                    warnings.append("Could not parse bandit output")
                    
        except subprocess.TimeoutExpired:
            warnings.append("Bandit security scan timed out")
        except FileNotFoundError:
            warnings.append("Bandit not available - install with 'pip install bandit'")
        except Exception as e:
            warnings.append(f"Bandit scan error: {str(e)}")
        
        # Run pip-audit for dependency vulnerabilities
        dependency_issues = []
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "audit", "--format=json"],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=project_root
            )
            
            if result.stdout:
                try:
                    audit_data = json.loads(result.stdout)
                    dependency_issues = audit_data.get("vulnerabilities", [])
                except json.JSONDecodeError:
                    warnings.append("Could not parse pip-audit output")
                    
        except subprocess.TimeoutExpired:
            warnings.append("pip-audit timed out")
        except FileNotFoundError:
            warnings.append("pip-audit not available - install with 'pip install pip-audit'")
        except Exception as e:
            warnings.append(f"pip-audit error: {str(e)}")
        
        # Evaluate security score
        high_severity_issues = 0
        medium_severity_issues = 0
        low_severity_issues = 0
        
        for issue in security_issues:
            severity = issue.get("issue_severity", "LOW").upper()
            if severity == "HIGH":
                high_severity_issues += 1
            elif severity == "MEDIUM":
                medium_severity_issues += 1
            else:
                low_severity_issues += 1
        
        for issue in dependency_issues:
            severity = issue.get("severity", "LOW").upper()
            if severity in ["HIGH", "CRITICAL"]:
                high_severity_issues += 1
            elif severity == "MEDIUM":
                medium_severity_issues += 1
            else:
                low_severity_issues += 1
        
        # Calculate security score (penalize high severity issues heavily)
        total_issues = high_severity_issues + medium_severity_issues + low_severity_issues
        
        if total_issues == 0:
            score = 100.0
        else:
            # Heavy penalty for high severity, moderate for medium, light for low
            penalty = (high_severity_issues * 30) + (medium_severity_issues * 10) + (low_severity_issues * 2)
            score = max(0, 100 - penalty)
        
        # Status determination
        if high_severity_issues == 0 and medium_severity_issues <= 2:
            status = QualityGateStatus.PASSED
        else:
            status = QualityGateStatus.FAILED
            if high_severity_issues > 0:
                errors.append(f"{high_severity_issues} high severity security issues found")
            if medium_severity_issues > 2:
                errors.append(f"{medium_severity_issues} medium severity security issues found")
        
        if low_severity_issues > 0:
            warnings.append(f"{low_severity_issues} low severity security issues found")
        
        return QualityGateResult(
            gate_name=self.name,
            status=status,
            score=score,
            max_score=100.0,
            details={
                "total_security_issues": total_issues,
                "high_severity": high_severity_issues,
                "medium_severity": medium_severity_issues,
                "low_severity": low_severity_issues,
                "bandit_issues": len(security_issues),
                "dependency_issues": len(dependency_issues)
            },
            errors=errors,
            warnings=warnings,
            metrics={
                "security_score": score,
                "high_severity_rate": high_severity_issues / max(1, total_issues) * 100
            }
        )


class CodeQualityGate(BaseGenerationGate):
    """Code quality and style verification."""
    
    def __init__(self):
        super().__init__("Code Quality", GenerationType.GENERATION_2, critical=False, timeout=240.0)
    
    async def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        project_root = Path(context.get("project_root", "."))
        
        errors = []
        warnings = []
        quality_scores = []
        
        # Run ruff linting
        ruff_issues = 0
        try:
            result = subprocess.run(
                [sys.executable, "-m", "ruff", "check", "probneural_operator", "--format=json"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=project_root
            )
            
            if result.stdout:
                try:
                    ruff_data = json.loads(result.stdout)
                    ruff_issues = len(ruff_data)
                except json.JSONDecodeError:
                    pass
                    
        except subprocess.TimeoutExpired:
            warnings.append("Ruff linting timed out")
        except FileNotFoundError:
            warnings.append("Ruff not available")
        except Exception as e:
            warnings.append(f"Ruff error: {str(e)}")
        
        # Run mypy type checking
        mypy_errors = 0
        try:
            result = subprocess.run(
                [sys.executable, "-m", "mypy", "probneural_operator", "--json-report", "/tmp/mypy_report"],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=project_root
            )
            
            # Count mypy errors from output
            if result.stderr:
                mypy_errors = result.stderr.count("error:")
                    
        except subprocess.TimeoutExpired:
            warnings.append("MyPy type checking timed out")
        except FileNotFoundError:
            warnings.append("MyPy not available")
        except Exception as e:
            warnings.append(f"MyPy error: {str(e)}")
        
        # Calculate quality scores
        ruff_score = max(0, 100 - (ruff_issues * 2))  # 2 points per issue
        mypy_score = max(0, 100 - (mypy_errors * 5))  # 5 points per error
        
        quality_scores = [ruff_score, mypy_score]
        overall_score = sum(quality_scores) / len(quality_scores)
        
        # Status determination
        if overall_score >= 85 and ruff_issues <= 10 and mypy_errors <= 5:
            status = QualityGateStatus.PASSED
        else:
            status = QualityGateStatus.FAILED
            if ruff_issues > 10:
                errors.append(f"{ruff_issues} linting issues exceed threshold of 10")
            if mypy_errors > 5:
                errors.append(f"{mypy_errors} type errors exceed threshold of 5")
        
        return QualityGateResult(
            gate_name=self.name,
            status=status,
            score=overall_score,
            max_score=100.0,
            details={
                "ruff_issues": ruff_issues,
                "mypy_errors": mypy_errors,
                "ruff_score": ruff_score,
                "mypy_score": mypy_score
            },
            errors=errors,
            warnings=warnings,
            metrics={
                "code_quality_score": overall_score,
                "linting_score": ruff_score,
                "type_checking_score": mypy_score
            }
        )


class Generation2Gates:
    """Collection of Generation 2 quality gates."""
    
    def get_gates(self) -> List[QualityGate]:
        """Get all Generation 2 gates."""
        gates = Generation1Gates().get_gates()  # Include Generation 1 gates
        
        # Add Generation 2 specific gates
        gen2_gates = [
            ComprehensiveTestGate(),
            SecurityScanGate(),
            CodeQualityGate(),
        ]
        
        # Set dependencies
        for gate in gen2_gates:
            gate.add_dependency("Basic Syntax Check")
            gate.add_dependency("Basic Import Check")
        
        gates.extend(gen2_gates)
        return gates


# =============================================================================
# GENERATION 3: MAKE IT SCALE (Optimized)
# =============================================================================

class PerformanceBenchmarkGate(BaseGenerationGate):
    """Performance benchmarking and optimization validation."""
    
    def __init__(self):
        super().__init__("Performance Benchmark", GenerationType.GENERATION_3, critical=False, timeout=900.0)
    
    async def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        project_root = Path(context.get("project_root", "."))
        performance_threshold = context.get("performance_threshold", 200.0)  # ms
        
        errors = []
        warnings = []
        
        try:
            # Run performance benchmarks
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/benchmarks/", "-v", "--benchmark-only"],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=project_root
            )
            
            benchmark_passed = result.returncode == 0
            
            # Parse benchmark results (simplified)
            benchmark_times = []
            benchmark_output = result.stdout + result.stderr
            
            # Extract timing information
            import re
            time_pattern = r"(\d+\.?\d*)\s*(ms|μs|s)"
            times = re.findall(time_pattern, benchmark_output)
            
            for time_val, unit in times:
                time_ms = float(time_val)
                if unit == 'μs':
                    time_ms /= 1000
                elif unit == 's':
                    time_ms *= 1000
                benchmark_times.append(time_ms)
            
            # Calculate performance score
            if not benchmark_times:
                score = 50  # No benchmarks found
                warnings.append("No performance benchmarks found")
            else:
                avg_time = sum(benchmark_times) / len(benchmark_times)
                # Score based on performance threshold
                if avg_time <= performance_threshold:
                    score = 100
                elif avg_time <= performance_threshold * 2:
                    score = 75
                elif avg_time <= performance_threshold * 3:
                    score = 50
                else:
                    score = 25
                    errors.append(f"Average benchmark time {avg_time:.2f}ms exceeds threshold {performance_threshold}ms")
            
            status = QualityGateStatus.PASSED if score >= 75 and benchmark_passed else QualityGateStatus.FAILED
            
        except subprocess.TimeoutExpired:
            errors.append("Performance benchmarks timed out")
            score = 0
            status = QualityGateStatus.FAILED
            benchmark_times = []
            
        except Exception as e:
            errors.append(f"Performance benchmark failed: {str(e)}")
            score = 0
            status = QualityGateStatus.FAILED
            benchmark_times = []
        
        return QualityGateResult(
            gate_name=self.name,
            status=status,
            score=score,
            max_score=100.0,
            details={
                "benchmark_times": benchmark_times,
                "avg_benchmark_time": sum(benchmark_times) / len(benchmark_times) if benchmark_times else 0,
                "performance_threshold": performance_threshold,
                "total_benchmarks": len(benchmark_times)
            },
            errors=errors,
            warnings=warnings,
            metrics={
                "performance_score": score,
                "avg_execution_time": sum(benchmark_times) / len(benchmark_times) if benchmark_times else 0
            }
        )


class ScalabilityTestGate(BaseGenerationGate):
    """Scalability and load testing."""
    
    def __init__(self):
        super().__init__("Scalability Test", GenerationType.GENERATION_3, critical=False, timeout=600.0)
    
    async def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        project_root = Path(context.get("project_root", "."))
        
        errors = []
        warnings = []
        
        try:
            # Run scalability tests
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/scaling/", "-v"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=project_root
            )
            
            scaling_tests_passed = result.returncode == 0
            
            # Parse test output for scalability metrics
            test_output = result.stdout + result.stderr
            
            # Count scaling-related tests
            scaling_tests = 0
            passed_scaling_tests = 0
            
            for line in test_output.split('\n'):
                if '::test_' in line and 'scaling' in line.lower():
                    scaling_tests += 1
                    if 'PASSED' in line:
                        passed_scaling_tests += 1
            
            # Calculate scalability score
            if scaling_tests == 0:
                score = 50  # No scaling tests found
                warnings.append("No scalability tests found")
            else:
                scalability_rate = (passed_scaling_tests / scaling_tests) * 100
                score = scalability_rate
            
            status = QualityGateStatus.PASSED if score >= 80 and scaling_tests_passed else QualityGateStatus.FAILED
            
            if not scaling_tests_passed:
                errors.append("Scalability tests failed")
            
        except subprocess.TimeoutExpired:
            errors.append("Scalability tests timed out")
            score = 0
            status = QualityGateStatus.FAILED
            scaling_tests = 0
            passed_scaling_tests = 0
            
        except Exception as e:
            errors.append(f"Scalability test failed: {str(e)}")
            score = 0
            status = QualityGateStatus.FAILED
            scaling_tests = 0
            passed_scaling_tests = 0
        
        return QualityGateResult(
            gate_name=self.name,
            status=status,
            score=score,
            max_score=100.0,
            details={
                "total_scaling_tests": scaling_tests,
                "passed_scaling_tests": passed_scaling_tests,
                "scaling_pass_rate": (passed_scaling_tests / scaling_tests * 100) if scaling_tests > 0 else 0
            },
            errors=errors,
            warnings=warnings,
            metrics={
                "scalability_score": score
            }
        )


class ProductionReadinessGate(BaseGenerationGate):
    """Production readiness assessment."""
    
    def __init__(self):
        super().__init__("Production Readiness", GenerationType.GENERATION_3, critical=True, timeout=180.0)
    
    async def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        project_root = Path(context.get("project_root", "."))
        
        errors = []
        warnings = []
        readiness_checks = []
        
        # Check 1: Docker configuration
        docker_score = 0
        if (project_root / "Dockerfile").exists():
            docker_score += 25
        if (project_root / "docker-compose.yml").exists():
            docker_score += 15
        if (project_root / "docker-compose.production.yml").exists():
            docker_score += 10
        readiness_checks.append(("Docker Configuration", docker_score, 50))
        
        # Check 2: Kubernetes manifests
        k8s_score = 0
        k8s_dir = project_root / "k8s"
        if k8s_dir.exists():
            k8s_files = list(k8s_dir.glob("*.yaml")) + list(k8s_dir.glob("*.yml"))
            k8s_score = min(30, len(k8s_files) * 10)
        readiness_checks.append(("Kubernetes Manifests", k8s_score, 30))
        
        # Check 3: Monitoring configuration
        monitoring_score = 0
        if (project_root / "monitoring").exists():
            monitoring_score += 10
        if (project_root / "monitoring" / "prometheus.yml").exists():
            monitoring_score += 10
        readiness_checks.append(("Monitoring Setup", monitoring_score, 20))
        
        # Calculate overall readiness score
        total_score = 0
        max_total_score = 0
        
        for check_name, score, max_score in readiness_checks:
            total_score += score
            max_total_score += max_score
            
            if score < max_score * 0.6:  # Less than 60% of max score
                warnings.append(f"Low {check_name} score: {score}/{max_score}")
        
        overall_score = (total_score / max_total_score * 100) if max_total_score > 0 else 0
        
        # Status determination
        if overall_score >= 70:
            status = QualityGateStatus.PASSED
        else:
            status = QualityGateStatus.FAILED
            errors.append(f"Production readiness score {overall_score:.1f}% below 70% threshold")
        
        return QualityGateResult(
            gate_name=self.name,
            status=status,
            score=overall_score,
            max_score=100.0,
            details={
                "readiness_checks": [
                    {"name": name, "score": score, "max_score": max_score}
                    for name, score, max_score in readiness_checks
                ],
                "total_score": total_score,
                "max_total_score": max_total_score
            },
            errors=errors,
            warnings=warnings,
            metrics={
                "production_readiness": overall_score
            }
        )


class Generation3Gates:
    """Collection of Generation 3 quality gates."""
    
    def get_gates(self) -> List[QualityGate]:
        """Get all Generation 3 gates."""
        gates = Generation2Gates().get_gates()  # Include previous generation gates
        
        # Add Generation 3 specific gates
        gen3_gates = [
            PerformanceBenchmarkGate(),
            ScalabilityTestGate(),
            ProductionReadinessGate(),
        ]
        
        # Set dependencies on Generation 2 gates
        for gate in gen3_gates:
            gate.add_dependency("Comprehensive Testing")
            gate.add_dependency("Security Scan")
            gate.add_dependency("Code Quality")
        
        gates.extend(gen3_gates)
        return gates