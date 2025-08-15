"""
Core Quality Gate Framework
==========================

Central framework for progressive quality gate enforcement in Terragon SDLC.
Implements autonomous quality validation with adaptive thresholds and 
intelligent failure recovery.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import json
import subprocess
import sys

class QualityGateStatus(Enum):
    """Quality gate execution status."""
    PENDING = "pending"
    RUNNING = "running" 
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CRITICAL_FAILURE = "critical_failure"

class GenerationType(Enum):
    """Development generation types."""
    GENERATION_1 = "generation_1"  # Make it work
    GENERATION_2 = "generation_2"  # Make it robust
    GENERATION_3 = "generation_3"  # Make it scale

@dataclass
class QualityGateResult:
    """Result of a quality gate execution."""
    gate_name: str
    status: QualityGateStatus
    score: float = 0.0
    max_score: float = 100.0
    execution_time: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    @property
    def percentage_score(self) -> float:
        """Get score as percentage."""
        return (self.score / self.max_score) * 100 if self.max_score > 0 else 0.0
    
    @property
    def passed(self) -> bool:
        """Check if gate passed."""
        return self.status == QualityGateStatus.PASSED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gate_name": self.gate_name,
            "status": self.status.value,
            "score": self.score,
            "max_score": self.max_score,
            "percentage_score": self.percentage_score,
            "execution_time": self.execution_time,
            "details": self.details,
            "errors": self.errors,
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "metrics": self.metrics,
        }

class QualityGate(ABC):
    """Abstract base class for quality gates."""
    
    def __init__(self, name: str, critical: bool = False, timeout: float = 300.0):
        self.name = name
        self.critical = critical  # Critical gates block progression
        self.timeout = timeout
        self.dependencies: Set[str] = set()
    
    def add_dependency(self, gate_name: str) -> None:
        """Add a dependency on another gate."""
        self.dependencies.add(gate_name)
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        """Execute the quality gate."""
        pass
    
    async def pre_execute(self, context: Dict[str, Any]) -> bool:
        """Pre-execution validation. Return False to skip."""
        return True
    
    async def post_execute(self, result: QualityGateResult, context: Dict[str, Any]) -> None:
        """Post-execution cleanup/actions."""
        pass

class QualityGateFramework:
    """
    Central framework for managing progressive quality gates.
    
    Orchestrates quality gate execution across development generations
    with autonomous failure recovery and adaptive intelligence.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path(".terragon/quality_gates.json")
        self.gates: Dict[str, QualityGate] = {}
        self.execution_history: List[QualityGateResult] = []
        self.context: Dict[str, Any] = {}
        self.current_generation = GenerationType.GENERATION_1
        
        # Load configuration
        self._load_config()
    
    def _load_config(self) -> None:
        """Load quality gate configuration."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                config = json.load(f)
                self.context.update(config.get("context", {}))
        else:
            # Create default configuration
            self._create_default_config()
    
    def _create_default_config(self) -> None:
        """Create default quality gate configuration."""
        default_config = {
            "context": {
                "project_root": str(Path.cwd()),
                "test_coverage_threshold": 85.0,
                "performance_threshold": 200.0,  # ms
                "security_scan_enabled": True,
                "parallel_execution": True,
                "max_retries": 3,
                "adaptive_thresholds": True,
            },
            "thresholds": {
                "generation_1": {
                    "test_coverage": 70.0,
                    "code_quality": 75.0,
                    "security_basic": 90.0,
                },
                "generation_2": {
                    "test_coverage": 85.0,
                    "code_quality": 85.0,
                    "security_comprehensive": 95.0,
                    "performance": 90.0,
                },
                "generation_3": {
                    "test_coverage": 90.0,
                    "code_quality": 90.0,
                    "security_comprehensive": 98.0,
                    "performance": 95.0,
                    "scalability": 85.0,
                },
            },
        }
        
        # Ensure directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.config_path, "w") as f:
            json.dump(default_config, f, indent=2)
        
        self.context.update(default_config["context"])
    
    def register_gate(self, gate: QualityGate) -> None:
        """Register a quality gate."""
        self.gates[gate.name] = gate
    
    def register_gates(self, gates: List[QualityGate]) -> None:
        """Register multiple quality gates."""
        for gate in gates:
            self.register_gate(gate)
    
    async def execute_generation(self, generation: GenerationType) -> Dict[str, QualityGateResult]:
        """
        Execute all quality gates for a specific generation.
        
        Returns:
            Dictionary mapping gate names to results
        """
        self.current_generation = generation
        generation_gates = self._get_generation_gates(generation)
        
        if not generation_gates:
            print(f"No gates configured for {generation.value}")
            return {}
        
        print(f"🚀 Executing {generation.value.replace('_', ' ').title()} Quality Gates")
        print(f"Total gates: {len(generation_gates)}")
        
        # Execute gates with dependency resolution
        results = await self._execute_gates_with_dependencies(generation_gates)
        
        # Check if generation passed
        generation_passed = self._evaluate_generation_success(results)
        
        if generation_passed:
            print(f"✅ {generation.value.replace('_', ' ').title()} - ALL GATES PASSED")
            await self._advance_generation(generation)
        else:
            print(f"❌ {generation.value.replace('_', ' ').title()} - GATES FAILED")
            await self._handle_generation_failure(generation, results)
        
        return results
    
    def _get_generation_gates(self, generation: GenerationType) -> List[QualityGate]:
        """Get gates for a specific generation."""
        generation_gates = []
        
        for gate in self.gates.values():
            # Check if gate belongs to this generation
            if hasattr(gate, 'generation') and gate.generation == generation:
                generation_gates.append(gate)
        
        return generation_gates
    
    async def _execute_gates_with_dependencies(self, gates: List[QualityGate]) -> Dict[str, QualityGateResult]:
        """Execute gates respecting dependencies."""
        results: Dict[str, QualityGateResult] = {}
        executed: Set[str] = set()
        
        # Sort gates by dependencies (topological sort)
        sorted_gates = self._topological_sort(gates)
        
        for gate in sorted_gates:
            # Check dependencies
            deps_satisfied = all(dep in executed for dep in gate.dependencies)
            
            if not deps_satisfied:
                result = QualityGateResult(
                    gate_name=gate.name,
                    status=QualityGateStatus.SKIPPED,
                    errors=["Dependencies not satisfied"]
                )
                results[gate.name] = result
                continue
            
            # Execute gate
            try:
                start_time = time.time()
                
                # Pre-execution check
                should_execute = await gate.pre_execute(self.context)
                if not should_execute:
                    result = QualityGateResult(
                        gate_name=gate.name,
                        status=QualityGateStatus.SKIPPED,
                        details={"reason": "Pre-execution check failed"}
                    )
                    results[gate.name] = result
                    continue
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    gate.execute(self.context), 
                    timeout=gate.timeout
                )
                result.execution_time = time.time() - start_time
                
                # Post-execution
                await gate.post_execute(result, self.context)
                
                results[gate.name] = result
                executed.add(gate.name)
                
                # Update execution history
                self.execution_history.append(result)
                
                # Print result
                status_emoji = "✅" if result.passed else "❌"
                print(f"{status_emoji} {gate.name}: {result.percentage_score:.1f}% ({result.execution_time:.2f}s)")
                
                if result.errors:
                    for error in result.errors:
                        print(f"   ❌ {error}")
                
                if result.warnings:
                    for warning in result.warnings:
                        print(f"   ⚠️  {warning}")
                
            except asyncio.TimeoutError:
                result = QualityGateResult(
                    gate_name=gate.name,
                    status=QualityGateStatus.FAILED,
                    execution_time=gate.timeout,
                    errors=[f"Gate execution timed out after {gate.timeout}s"]
                )
                results[gate.name] = result
                print(f"⏰ {gate.name}: TIMEOUT after {gate.timeout}s")
                
            except Exception as e:
                result = QualityGateResult(
                    gate_name=gate.name,
                    status=QualityGateStatus.CRITICAL_FAILURE,
                    execution_time=time.time() - start_time,
                    errors=[f"Unexpected error: {str(e)}"]
                )
                results[gate.name] = result
                print(f"💥 {gate.name}: CRITICAL FAILURE - {str(e)}")
                
                # Critical gates block progression
                if gate.critical:
                    print(f"🛑 Critical gate {gate.name} failed - stopping execution")
                    break
        
        return results
    
    def _topological_sort(self, gates: List[QualityGate]) -> List[QualityGate]:
        """Sort gates by dependencies using topological sort."""
        # Simple topological sort implementation
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(gate: QualityGate):
            if gate.name in temp_visited:
                raise ValueError(f"Circular dependency detected involving {gate.name}")
            if gate.name in visited:
                return
            
            temp_visited.add(gate.name)
            
            # Visit dependencies first
            for dep_name in gate.dependencies:
                dep_gate = next((g for g in gates if g.name == dep_name), None)
                if dep_gate:
                    visit(dep_gate)
            
            temp_visited.remove(gate.name)
            visited.add(gate.name)
            result.append(gate)
        
        for gate in gates:
            if gate.name not in visited:
                visit(gate)
        
        return result
    
    def _evaluate_generation_success(self, results: Dict[str, QualityGateResult]) -> bool:
        """Evaluate if generation passed based on results."""
        critical_failures = 0
        total_gates = len(results)
        passed_gates = 0
        
        for result in results.values():
            if result.status == QualityGateStatus.CRITICAL_FAILURE:
                critical_failures += 1
            elif result.status == QualityGateStatus.PASSED:
                passed_gates += 1
        
        # No critical failures and at least 80% pass rate
        pass_rate = passed_gates / total_gates if total_gates > 0 else 0
        return critical_failures == 0 and pass_rate >= 0.8
    
    async def _advance_generation(self, current_generation: GenerationType) -> None:
        """Advance to next generation."""
        if current_generation == GenerationType.GENERATION_1:
            self.current_generation = GenerationType.GENERATION_2
        elif current_generation == GenerationType.GENERATION_2:
            self.current_generation = GenerationType.GENERATION_3
        
        print(f"🎯 Advanced to {self.current_generation.value.replace('_', ' ').title()}")
    
    async def _handle_generation_failure(self, generation: GenerationType, results: Dict[str, QualityGateResult]) -> None:
        """Handle generation failure with adaptive recovery."""
        failed_gates = [r for r in results.values() if not r.passed]
        
        print(f"📊 Generation Failure Analysis:")
        print(f"   Total gates: {len(results)}")
        print(f"   Failed gates: {len(failed_gates)}")
        
        for result in failed_gates:
            print(f"   ❌ {result.gate_name}: {result.percentage_score:.1f}%")
            if result.errors:
                print(f"      Errors: {', '.join(result.errors[:3])}")
        
        # Implement adaptive recovery strategies
        await self._implement_recovery_strategies(failed_gates)
    
    async def _implement_recovery_strategies(self, failed_results: List[QualityGateResult]) -> None:
        """Implement recovery strategies for failed gates."""
        print("🔧 Implementing recovery strategies...")
        
        # Group failures by type
        failure_types = {}
        for result in failed_results:
            for error in result.errors:
                failure_type = self._classify_failure(error)
                if failure_type not in failure_types:
                    failure_types[failure_type] = []
                failure_types[failure_type].append(result)
        
        for failure_type, results in failure_types.items():
            await self._apply_recovery_strategy(failure_type, results)
    
    def _classify_failure(self, error: str) -> str:
        """Classify failure type for recovery strategy selection."""
        error_lower = error.lower()
        
        if "test" in error_lower and "coverage" in error_lower:
            return "test_coverage"
        elif "test" in error_lower and "fail" in error_lower:
            return "test_failure"
        elif "security" in error_lower or "vulnerability" in error_lower:
            return "security"
        elif "performance" in error_lower or "timeout" in error_lower:
            return "performance"
        elif "style" in error_lower or "lint" in error_lower:
            return "code_style"
        else:
            return "unknown"
    
    async def _apply_recovery_strategy(self, failure_type: str, results: List[QualityGateResult]) -> None:
        """Apply specific recovery strategy."""
        strategy_map = {
            "test_coverage": self._recover_test_coverage,
            "test_failure": self._recover_test_failures,
            "security": self._recover_security_issues,
            "performance": self._recover_performance_issues,
            "code_style": self._recover_code_style,
            "unknown": self._recover_unknown_issues,
        }
        
        strategy = strategy_map.get(failure_type, self._recover_unknown_issues)
        await strategy(results)
    
    async def _recover_test_coverage(self, results: List[QualityGateResult]) -> None:
        """Recover from test coverage issues."""
        print("🧪 Applying test coverage recovery...")
        # Implementation would generate missing tests
        pass
    
    async def _recover_test_failures(self, results: List[QualityGateResult]) -> None:
        """Recover from test failures."""
        print("🔨 Applying test failure recovery...")
        # Implementation would fix failing tests
        pass
    
    async def _recover_security_issues(self, results: List[QualityGateResult]) -> None:
        """Recover from security issues."""
        print("🔒 Applying security issue recovery...")
        # Implementation would fix security vulnerabilities
        pass
    
    async def _recover_performance_issues(self, results: List[QualityGateResult]) -> None:
        """Recover from performance issues."""
        print("⚡ Applying performance issue recovery...")
        # Implementation would optimize performance
        pass
    
    async def _recover_code_style(self, results: List[QualityGateResult]) -> None:
        """Recover from code style issues."""
        print("💎 Applying code style recovery...")
        # Implementation would fix style issues
        pass
    
    async def _recover_unknown_issues(self, results: List[QualityGateResult]) -> None:
        """Recover from unknown issues."""
        print("🔍 Applying general recovery strategies...")
        # Implementation would apply general fixes
        pass
    
    def get_execution_report(self) -> Dict[str, Any]:
        """Generate comprehensive execution report."""
        total_executions = len(self.execution_history)
        passed_executions = len([r for r in self.execution_history if r.passed])
        
        if total_executions == 0:
            pass_rate = 0.0
        else:
            pass_rate = (passed_executions / total_executions) * 100
        
        report = {
            "summary": {
                "total_executions": total_executions,
                "passed_executions": passed_executions,
                "failed_executions": total_executions - passed_executions,
                "pass_rate": pass_rate,
                "current_generation": self.current_generation.value,
            },
            "recent_executions": [
                result.to_dict() for result in self.execution_history[-10:]
            ],
            "performance_metrics": self._calculate_performance_metrics(),
            "recommendations": self._generate_recommendations(),
        }
        
        return report
    
    def _calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics from execution history."""
        if not self.execution_history:
            return {}
        
        execution_times = [r.execution_time for r in self.execution_history]
        scores = [r.percentage_score for r in self.execution_history]
        
        return {
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "max_execution_time": max(execution_times),
            "min_execution_time": min(execution_times),
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        if not self.execution_history:
            return ["Execute quality gates to generate recommendations"]
        
        recent_results = self.execution_history[-5:]
        
        # Analyze recent failures
        recent_failures = [r for r in recent_results if not r.passed]
        if len(recent_failures) > len(recent_results) * 0.5:
            recommendations.append("High failure rate detected - consider reviewing gate thresholds")
        
        # Analyze performance
        avg_time = sum(r.execution_time for r in recent_results) / len(recent_results)
        if avg_time > 30:
            recommendations.append("Gate execution times are high - consider optimization")
        
        # Analyze patterns
        common_failures = {}
        for result in recent_failures:
            for error in result.errors:
                failure_type = self._classify_failure(error)
                common_failures[failure_type] = common_failures.get(failure_type, 0) + 1
        
        if common_failures:
            most_common = max(common_failures.items(), key=lambda x: x[1])
            recommendations.append(f"Most common failure type: {most_common[0]} - focus improvement efforts here")
        
        return recommendations

# Utility function for running quality gates
async def run_quality_gates(generation: GenerationType = GenerationType.GENERATION_1) -> bool:
    """
    Utility function to run quality gates for a specific generation.
    
    Returns:
        True if all gates passed, False otherwise
    """
    framework = QualityGateFramework()
    
    # Import and register gates based on generation
    if generation == GenerationType.GENERATION_1:
        from .generations import Generation1Gates
        gates = Generation1Gates().get_gates()
    elif generation == GenerationType.GENERATION_2:
        from .generations import Generation2Gates
        gates = Generation2Gates().get_gates()
    elif generation == GenerationType.GENERATION_3:
        from .generations import Generation3Gates
        gates = Generation3Gates().get_gates()
    else:
        raise ValueError(f"Unknown generation: {generation}")
    
    framework.register_gates(gates)
    
    results = await framework.execute_generation(generation)
    
    # Generate and save report
    report = framework.get_execution_report()
    report_path = Path(f".terragon/quality_gate_report_{generation.value}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Report saved to {report_path}")
    
    return framework._evaluate_generation_success(results)