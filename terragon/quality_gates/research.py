"""
Research-Specific Quality Gates
==============================

Advanced quality gates for research projects with focus on:
- Reproducible results and statistical significance
- Baseline comparisons and novel algorithm validation
- Publication-ready code and documentation
- Academic scrutiny and peer review readiness
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

from .core import QualityGate, QualityGateResult, QualityGateStatus

try:
    import numpy as np
except ImportError:
    # Fallback for basic statistics without numpy
    class MockNumpy:
        @staticmethod
        def std(values):
            if not values:
                return 0.0
            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            return variance ** 0.5
        
        @staticmethod
        def var(values):
            if not values:
                return 0.0
            mean = sum(values) / len(values)
            return sum((x - mean) ** 2 for x in values) / len(values)
        
        @staticmethod
        def corrcoef(x, y):
            if len(x) != len(y) or len(x) < 2:
                return [[1.0, 0.0], [0.0, 1.0]]
            
            mean_x = sum(x) / len(x)
            mean_y = sum(y) / len(y)
            
            numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x)))
            sum_sq_x = sum((x[i] - mean_x) ** 2 for i in range(len(x)))
            sum_sq_y = sum((y[i] - mean_y) ** 2 for i in range(len(y)))
            
            denominator = (sum_sq_x * sum_sq_y) ** 0.5
            
            if denominator == 0:
                correlation = 0.0
            else:
                correlation = numerator / denominator
            
            return [[1.0, correlation], [correlation, 1.0]]
    
    np = MockNumpy()


class ResearchReproducibilityGate(QualityGate):
    """Validate reproducible research results."""
    
    def __init__(self):
        super().__init__("Research Reproducibility", critical=True, timeout=600.0)
    
    async def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        project_root = Path(context.get("project_root", "."))
        
        errors = []
        warnings = []
        reproducibility_score = 0
        
        # Check 1: Random seed management
        seed_management_score = 0
        try:
            result = subprocess.run(
                [sys.executable, "-c", 
                 "import probneural_operator; "
                 "from probneural_operator.utils.config import set_random_seed; "
                 "print('Seed management available')"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=project_root
            )
            
            if result.returncode == 0:
                seed_management_score = 25
            else:
                warnings.append("Random seed management not properly implemented")
                
        except Exception as e:
            warnings.append(f"Could not verify seed management: {str(e)}")
        
        # Check 2: Experiment configuration files
        config_score = 0
        config_dir = project_root / "configs"
        if config_dir.exists():
            config_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.json"))
            if len(config_files) >= 2:
                config_score = 25
            elif len(config_files) >= 1:
                config_score = 15
                warnings.append("Limited experiment configurations found")
        else:
            warnings.append("No experiment configuration directory found")
        
        # Check 3: Reproducibility tests
        repro_test_score = 0
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/research/", "-k", "reproducibility", "-v"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=project_root
            )
            
            if result.returncode == 0 and "PASSED" in result.stdout:
                repro_test_score = 25
            else:
                warnings.append("Reproducibility tests not passing")
                
        except Exception as e:
            warnings.append(f"Could not run reproducibility tests: {str(e)}")
        
        # Check 4: Documentation of experimental setup
        docs_score = 0
        experiment_docs = [
            project_root / "docs" / "experiments.md",
            project_root / "docs" / "EXPERIMENTS.md", 
            project_root / "EXPERIMENTS.md"
        ]
        
        for doc_path in experiment_docs:
            if doc_path.exists():
                docs_score = 25
                break
        else:
            warnings.append("No experiment documentation found")
        
        reproducibility_score = seed_management_score + config_score + repro_test_score + docs_score
        
        status = QualityGateStatus.PASSED if reproducibility_score >= 75 else QualityGateStatus.FAILED
        
        if reproducibility_score < 75:
            errors.append(f"Reproducibility score {reproducibility_score}% below 75% threshold")
        
        return QualityGateResult(
            gate_name=self.name,
            status=status,
            score=reproducibility_score,
            max_score=100.0,
            details={
                "seed_management_score": seed_management_score,
                "config_score": config_score,
                "repro_test_score": repro_test_score,
                "docs_score": docs_score,
                "config_files_found": len(list((project_root / "configs").glob("*"))) if (project_root / "configs").exists() else 0
            },
            errors=errors,
            warnings=warnings,
            metrics={
                "reproducibility_score": reproducibility_score
            }
        )


class StatisticalValidationGate(QualityGate):
    """Validate statistical significance of research results."""
    
    def __init__(self):
        super().__init__("Statistical Validation", critical=True, timeout=900.0)
    
    async def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        project_root = Path(context.get("project_root", "."))
        
        errors = []
        warnings = []
        
        # Run statistical validation tests
        statistical_score = 0
        
        try:
            # Check for statistical test implementations
            result = subprocess.run(
                [sys.executable, "-c",
                 "from probneural_operator.benchmarks.research_validation import validate_statistical_significance; "
                 "print('Statistical validation available')"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=project_root
            )
            
            if result.returncode == 0:
                statistical_score += 30
            else:
                warnings.append("Statistical validation framework not found")
            
            # Run statistical significance tests
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/research/", "-k", "statistical", "-v"],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=project_root
            )
            
            if result.returncode == 0:
                # Parse test results for p-values and effect sizes
                test_output = result.stdout + result.stderr
                
                # Look for p-value reporting
                if "p-value" in test_output.lower() or "p_value" in test_output.lower():
                    statistical_score += 35
                else:
                    warnings.append("No p-value reporting found in tests")
                
                # Look for effect size reporting
                if "effect size" in test_output.lower() or "cohen" in test_output.lower():
                    statistical_score += 20
                else:
                    warnings.append("No effect size reporting found")
                
                # Look for confidence intervals
                if "confidence interval" in test_output.lower() or "ci" in test_output.lower():
                    statistical_score += 15
                else:
                    warnings.append("No confidence interval reporting found")
            else:
                warnings.append("Statistical tests failed to run")
                
        except Exception as e:
            errors.append(f"Statistical validation failed: {str(e)}")
        
        status = QualityGateStatus.PASSED if statistical_score >= 70 else QualityGateStatus.FAILED
        
        if statistical_score < 70:
            errors.append(f"Statistical validation score {statistical_score}% below 70% threshold")
        
        return QualityGateResult(
            gate_name=self.name,
            status=status,
            score=statistical_score,
            max_score=100.0,
            details={
                "has_statistical_framework": statistical_score >= 30,
                "reports_p_values": "p-value" in str(statistical_score),
                "reports_effect_sizes": statistical_score >= 50,
                "reports_confidence_intervals": statistical_score >= 70
            },
            errors=errors,
            warnings=warnings,
            metrics={
                "statistical_validation_score": statistical_score
            }
        )


class BaselineComparisonGate(QualityGate):
    """Validate baseline comparisons and benchmarking."""
    
    def __init__(self):
        super().__init__("Baseline Comparison", critical=True, timeout=1200.0)
    
    async def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        project_root = Path(context.get("project_root", "."))
        
        errors = []
        warnings = []
        baseline_score = 0
        
        # Check 1: Baseline implementations
        baseline_implementations = 0
        try:
            result = subprocess.run(
                [sys.executable, "-c",
                 "from probneural_operator.benchmarks.research_benchmarks import get_baseline_models; "
                 "baselines = get_baseline_models(); "
                 f"print(f'Found {{len(baselines)}} baseline models')"],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=project_root
            )
            
            if result.returncode == 0:
                # Extract number of baselines from output
                output = result.stdout.strip()
                if "Found" in output:
                    try:
                        baseline_implementations = int(output.split()[1])
                    except (IndexError, ValueError):
                        pass
                
                if baseline_implementations >= 3:
                    baseline_score += 40
                elif baseline_implementations >= 2:
                    baseline_score += 25
                    warnings.append("Limited baseline implementations found")
                elif baseline_implementations >= 1:
                    baseline_score += 15
                    warnings.append("Very limited baseline implementations")
                else:
                    warnings.append("No baseline implementations found")
            else:
                warnings.append("Baseline framework not available")
                
        except Exception as e:
            warnings.append(f"Could not check baseline implementations: {str(e)}")
        
        # Check 2: Comparative benchmarks
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/benchmarks/", "-k", "comparison", "-v"],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=project_root
            )
            
            if result.returncode == 0:
                benchmark_output = result.stdout + result.stderr
                
                # Count successful comparative tests
                comparison_tests = benchmark_output.count("PASSED")
                if comparison_tests >= 5:
                    baseline_score += 30
                elif comparison_tests >= 3:
                    baseline_score += 20
                    warnings.append("Limited comparative benchmarks")
                elif comparison_tests >= 1:
                    baseline_score += 10
                    warnings.append("Very limited comparative benchmarks")
                else:
                    warnings.append("No comparative benchmarks found")
            else:
                warnings.append("Comparative benchmarks failed")
                
        except Exception as e:
            warnings.append(f"Could not run comparative benchmarks: {str(e)}")
        
        # Check 3: Performance metrics reporting
        metrics_file = project_root / "benchmark_results" / "benchmark_results.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    metrics_data = json.load(f)
                    
                # Check for baseline comparisons in metrics
                if "baseline_comparison" in metrics_data or "comparisons" in metrics_data:
                    baseline_score += 30
                else:
                    baseline_score += 15
                    warnings.append("Metrics file exists but lacks baseline comparisons")
                    
            except Exception as e:
                warnings.append(f"Could not parse metrics file: {str(e)}")
        else:
            warnings.append("No benchmark results file found")
        
        status = QualityGateStatus.PASSED if baseline_score >= 75 else QualityGateStatus.FAILED
        
        if baseline_score < 75:
            errors.append(f"Baseline comparison score {baseline_score}% below 75% threshold")
        
        return QualityGateResult(
            gate_name=self.name,
            status=status,
            score=baseline_score,
            max_score=100.0,
            details={
                "baseline_implementations": baseline_implementations,
                "has_metrics_file": metrics_file.exists(),
                "baseline_comparison_available": baseline_score >= 30
            },
            errors=errors,
            warnings=warnings,
            metrics={
                "baseline_score": baseline_score
            }
        )


class PublicationReadinessGate(QualityGate):
    """Validate publication readiness for academic peer review."""
    
    def __init__(self):
        super().__init__("Publication Readiness", critical=False, timeout=300.0)
    
    async def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        project_root = Path(context.get("project_root", "."))
        
        errors = []
        warnings = []
        publication_score = 0
        
        # Check 1: Mathematical documentation
        math_docs_score = 0
        docs_dir = project_root / "docs"
        if docs_dir.exists():
            math_files = list(docs_dir.glob("*math*")) + list(docs_dir.glob("*formula*"))
            if math_files:
                math_docs_score = 20
            else:
                warnings.append("No mathematical documentation found")
        
        # Check 2: Algorithm descriptions
        algo_docs_score = 0
        algorithm_files = [
            docs_dir / "algorithms.md",
            docs_dir / "ALGORITHMS.md",
            project_root / "ALGORITHMS.md"
        ]
        
        for algo_file in algorithm_files:
            if algo_file.exists():
                algo_docs_score = 20
                break
        else:
            warnings.append("No algorithm documentation found")
        
        # Check 3: Example notebooks
        notebook_score = 0
        notebook_dirs = [
            project_root / "notebooks",
            project_root / "examples", 
            project_root / "tutorials"
        ]
        
        notebook_count = 0
        for notebook_dir in notebook_dirs:
            if notebook_dir.exists():
                notebooks = list(notebook_dir.glob("*.ipynb"))
                notebook_count += len(notebooks)
        
        if notebook_count >= 3:
            notebook_score = 20
        elif notebook_count >= 1:
            notebook_score = 10
            warnings.append("Limited example notebooks found")
        else:
            warnings.append("No example notebooks found")
        
        # Check 4: Research contributions documentation
        contrib_score = 0
        contrib_files = [
            docs_dir / "RESEARCH_CONTRIBUTIONS.md",
            project_root / "RESEARCH_CONTRIBUTIONS.md",
            docs_dir / "contributions.md"
        ]
        
        for contrib_file in contrib_files:
            if contrib_file.exists():
                contrib_score = 20
                break
        else:
            warnings.append("No research contributions documentation found")
        
        # Check 5: Citation and references
        citation_score = 0
        citation_files = [
            project_root / "CITATION.cff",
            project_root / "CITATION.bib",
            docs_dir / "citation.md"
        ]
        
        for citation_file in citation_files:
            if citation_file.exists():
                citation_score = 20
                break
        else:
            warnings.append("No citation information found")
        
        publication_score = math_docs_score + algo_docs_score + notebook_score + contrib_score + citation_score
        
        status = QualityGateStatus.PASSED if publication_score >= 60 else QualityGateStatus.FAILED
        
        if publication_score < 60:
            errors.append(f"Publication readiness score {publication_score}% below 60% threshold")
        
        return QualityGateResult(
            gate_name=self.name,
            status=status,
            score=publication_score,
            max_score=100.0,
            details={
                "math_docs_score": math_docs_score,
                "algo_docs_score": algo_docs_score,
                "notebook_score": notebook_score,
                "contrib_score": contrib_score,
                "citation_score": citation_score,
                "notebook_count": notebook_count
            },
            errors=errors,
            warnings=warnings,
            metrics={
                "publication_readiness": publication_score
            }
        )


class NoveltyValidationGate(QualityGate):
    """Validate novel algorithmic contributions."""
    
    def __init__(self):
        super().__init__("Novelty Validation", critical=False, timeout=600.0)
    
    async def execute(self, context: Dict[str, Any]) -> QualityGateResult:
        project_root = Path(context.get("project_root", "."))
        
        errors = []
        warnings = []
        novelty_score = 0
        
        # Check 1: Novel algorithm implementations
        try:
            result = subprocess.run(
                [sys.executable, "-c",
                 "from probneural_operator.posteriors.adaptive_uncertainty import AdaptiveUncertaintyPosterior; "
                 "from probneural_operator.posteriors.laplace.hierarchical_laplace import HierarchicalLaplace; "
                 "print('Novel algorithms found')"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=project_root
            )
            
            if result.returncode == 0:
                novelty_score += 30
            else:
                warnings.append("Novel algorithm implementations not found")
                
        except Exception as e:
            warnings.append(f"Could not verify novel algorithms: {str(e)}")
        
        # Check 2: Theoretical validation tests
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/research/", "-k", "theoretical", "-v"],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=project_root
            )
            
            if result.returncode == 0 and "PASSED" in result.stdout:
                novelty_score += 25
            else:
                warnings.append("Theoretical validation tests not passing")
                
        except Exception as e:
            warnings.append(f"Could not run theoretical tests: {str(e)}")
        
        # Check 3: Performance improvements over baselines
        metrics_file = project_root / "benchmark_results" / "benchmark_results.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    metrics_data = json.load(f)
                    
                # Look for improvement metrics
                improvements_found = False
                for key in metrics_data.keys():
                    if "improvement" in key.lower() or "better" in key.lower():
                        improvements_found = True
                        break
                
                if improvements_found:
                    novelty_score += 25
                else:
                    warnings.append("No performance improvements documented")
                    
            except Exception as e:
                warnings.append(f"Could not parse benchmark results: {str(e)}")
        
        # Check 4: Methodological contributions
        method_docs = [
            project_root / "docs" / "methodology.md",
            project_root / "docs" / "METHODOLOGY.md",
            project_root / "METHODOLOGY.md"
        ]
        
        for method_file in method_docs:
            if method_file.exists():
                novelty_score += 20
                break
        else:
            warnings.append("No methodology documentation found")
        
        status = QualityGateStatus.PASSED if novelty_score >= 70 else QualityGateStatus.FAILED
        
        if novelty_score < 70:
            errors.append(f"Novelty validation score {novelty_score}% below 70% threshold")
        
        return QualityGateResult(
            gate_name=self.name,
            status=status,
            score=novelty_score,
            max_score=100.0,
            details={
                "has_novel_algorithms": novelty_score >= 30,
                "has_theoretical_validation": novelty_score >= 55,
                "has_performance_improvements": novelty_score >= 80,
                "has_methodology_docs": novelty_score >= 90
            },
            errors=errors,
            warnings=warnings,
            metrics={
                "novelty_score": novelty_score
            }
        )


class ResearchQualityGates:
    """Collection of research-specific quality gates."""
    
    def get_gates(self) -> List[QualityGate]:
        """Get all research quality gates."""
        gates = [
            ResearchReproducibilityGate(),
            StatisticalValidationGate(),
            BaselineComparisonGate(),
            PublicationReadinessGate(),
            NoveltyValidationGate(),
        ]
        
        # Set dependencies
        statistical_gate = next(g for g in gates if g.name == "Statistical Validation")
        baseline_gate = next(g for g in gates if g.name == "Baseline Comparison")
        publication_gate = next(g for g in gates if g.name == "Publication Readiness")
        novelty_gate = next(g for g in gates if g.name == "Novelty Validation")
        
        # Statistical validation depends on reproducibility
        statistical_gate.add_dependency("Research Reproducibility")
        
        # Baseline comparison depends on statistical validation
        baseline_gate.add_dependency("Statistical Validation")
        
        # Publication readiness depends on baseline comparison
        publication_gate.add_dependency("Baseline Comparison")
        
        # Novelty validation depends on baseline comparison
        novelty_gate.add_dependency("Baseline Comparison")
        
        return gates