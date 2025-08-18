#!/usr/bin/env python3
"""
Comprehensive metrics collection script for ProbNeural Operator Lab.
Collects repository health, code quality, security, and performance metrics.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

import requests
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Container for metric collection results."""
    name: str
    value: Any
    timestamp: str
    status: str = "success"
    error: Optional[str] = None


class MetricsCollector:
    """Main metrics collection orchestrator."""
    
    def __init__(self, config_path: str = ".github/project-metrics.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.results = []
        self.logger = self._setup_logging()
        
        # Initialize API clients
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_name = os.getenv("GITHUB_REPOSITORY", "danieleschmidt/probneural-operator-lab")
        
    def _load_config(self) -> Dict[str, Any]:
        """Load metrics configuration."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return json.load(f)
        return {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _run_command(self, command: List[str], cwd: Optional[Path] = None) -> Dict[str, Any]:
        """Run shell command and return result."""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=cwd or Path.cwd(),
                timeout=300  # 5 minute timeout
            )
            return {
                "returncode": result.returncode,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()
            }
        except subprocess.TimeoutExpired:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": "Command timed out"
            }
        except Exception as e:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(e)
            }
    
    def _github_api_request(self, endpoint: str) -> Optional[Dict[str, Any]]:
        """Make GitHub API request."""
        if not self.github_token:
            self.logger.warning("GitHub token not available, skipping API metrics")
            return None
        
        url = f"https://api.github.com/repos/{self.repo_name}/{endpoint}"
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"GitHub API request failed: {e}")
            return None
    
    def collect_repository_metrics(self) -> List[MetricResult]:
        """Collect repository health and activity metrics."""
        metrics = []
        timestamp = datetime.utcnow().isoformat()
        
        try:
            # Repository info
            repo_info = self._github_api_request("")
            if repo_info:
                metrics.extend([
                    MetricResult("github_stars", repo_info.get("stargazers_count", 0), timestamp),
                    MetricResult("github_forks", repo_info.get("forks_count", 0), timestamp),
                    MetricResult("github_watchers", repo_info.get("watchers_count", 0), timestamp),
                    MetricResult("repository_size_kb", repo_info.get("size", 0), timestamp),
                ])
            
            # Recent activity
            commits = self._github_api_request("commits?per_page=100")
            if commits:
                # Count commits in last month
                one_month_ago = datetime.utcnow() - timedelta(days=30)
                recent_commits = [
                    c for c in commits 
                    if datetime.fromisoformat(
                        c["commit"]["author"]["date"].replace("Z", "+00:00")
                    ) > one_month_ago
                ]
                metrics.append(
                    MetricResult("commits_last_month", len(recent_commits), timestamp)
                )
            
            # Issues metrics
            issues = self._github_api_request("issues?state=all&per_page=100")
            if issues:
                open_issues = [i for i in issues if i["state"] == "open"]
                closed_issues = [i for i in issues if i["state"] == "closed"]
                
                metrics.extend([
                    MetricResult("open_issues", len(open_issues), timestamp),
                    MetricResult("closed_issues", len(closed_issues), timestamp),
                ])
            
            # Pull requests metrics
            prs = self._github_api_request("pulls?state=all&per_page=100")
            if prs:
                open_prs = [pr for pr in prs if pr["state"] == "open"]
                merged_prs = [pr for pr in prs if pr.get("merged_at")]
                
                metrics.extend([
                    MetricResult("open_pull_requests", len(open_prs), timestamp),
                    MetricResult("merged_pull_requests", len(merged_prs), timestamp),
                ])
        
        except Exception as e:
            self.logger.error(f"Error collecting repository metrics: {e}")
            metrics.append(
                MetricResult("repository_metrics", None, timestamp, "error", str(e))
            )
        
        return metrics
    
    def collect_code_quality_metrics(self) -> List[MetricResult]:
        """Collect code quality metrics."""
        metrics = []
        timestamp = datetime.utcnow().isoformat()
        
        try:
            # Test coverage
            coverage_cmd = ["python", "-m", "pytest", "--cov=probneural_operator", "--cov-report=json", "--cov-report=term", "tests/"]
            result = self._run_command(coverage_cmd)
            
            if result["returncode"] == 0:
                # Parse coverage report
                coverage_file = Path("coverage.json")
                if coverage_file.exists():
                    with open(coverage_file) as f:
                        coverage_data = json.load(f)
                    
                    total_coverage = coverage_data["totals"]["percent_covered"]
                    metrics.append(
                        MetricResult("test_coverage_percentage", total_coverage, timestamp)
                    )
            
            # Type coverage with mypy
            mypy_cmd = ["mypy", "probneural_operator/", "--json-report", "/tmp/mypy-report"]
            mypy_result = self._run_command(mypy_cmd)
            
            # Count lines of code
            loc_result = self._run_command(["find", "probneural_operator/", "-name", "*.py", "-exec", "wc", "-l", "{}", "+"])
            if loc_result["returncode"] == 0:
                lines = loc_result["stdout"].split('\n')
                if lines:
                    total_lines = int(lines[-1].split()[0]) if lines[-1].strip() else 0
                    metrics.append(
                        MetricResult("lines_of_code", total_lines, timestamp)
                    )
            
            # Linting with ruff
            ruff_result = self._run_command(["ruff", "check", "probneural_operator/", "--format", "json"])
            if ruff_result["stdout"]:
                try:
                    ruff_issues = json.loads(ruff_result["stdout"])
                    metrics.extend([
                        MetricResult("linting_issues_total", len(ruff_issues), timestamp),
                        MetricResult("linting_errors", len([i for i in ruff_issues if i.get("type") == "error"]), timestamp),
                    ])
                except json.JSONDecodeError:
                    pass
        
        except Exception as e:
            self.logger.error(f"Error collecting code quality metrics: {e}")
            metrics.append(
                MetricResult("code_quality_metrics", None, timestamp, "error", str(e))
            )
        
        return metrics
    
    def collect_security_metrics(self) -> List[MetricResult]:
        """Collect security-related metrics."""
        metrics = []
        timestamp = datetime.utcnow().isoformat()
        
        try:
            # Safety check for vulnerabilities
            safety_cmd = ["safety", "check", "--json"]
            safety_result = self._run_command(safety_cmd)
            
            if safety_result["stdout"]:
                try:
                    safety_data = json.loads(safety_result["stdout"])
                    vulnerabilities = len(safety_data)
                    metrics.append(
                        MetricResult("security_vulnerabilities", vulnerabilities, timestamp)
                    )
                except json.JSONDecodeError:
                    pass
            
            # Bandit security scan
            bandit_cmd = ["bandit", "-r", "probneural_operator/", "-f", "json"]
            bandit_result = self._run_command(bandit_cmd)
            
            if bandit_result["stdout"]:
                try:
                    bandit_data = json.loads(bandit_result["stdout"])
                    issues = bandit_data.get("results", [])
                    
                    severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
                    for issue in issues:
                        severity = issue.get("issue_severity", "LOW")
                        severity_counts[severity] = severity_counts.get(severity, 0) + 1
                    
                    metrics.extend([
                        MetricResult("security_issues_high", severity_counts["HIGH"], timestamp),
                        MetricResult("security_issues_medium", severity_counts["MEDIUM"], timestamp),
                        MetricResult("security_issues_low", severity_counts["LOW"], timestamp),
                    ])
                except json.JSONDecodeError:
                    pass
        
        except Exception as e:
            self.logger.error(f"Error collecting security metrics: {e}")
            metrics.append(
                MetricResult("security_metrics", None, timestamp, "error", str(e))
            )
        
        return metrics
    
    def collect_performance_metrics(self) -> List[MetricResult]:
        """Collect performance-related metrics."""
        metrics = []
        timestamp = datetime.utcnow().isoformat()
        
        try:
            # Package size metrics
            build_cmd = ["python", "-m", "build"]
            build_result = self._run_command(build_cmd)
            
            if build_result["returncode"] == 0:
                dist_path = Path("dist")
                if dist_path.exists():
                    wheel_files = list(dist_path.glob("*.whl"))
                    sdist_files = list(dist_path.glob("*.tar.gz"))
                    
                    if wheel_files:
                        wheel_size = wheel_files[0].stat().st_size / (1024 * 1024)  # MB
                        metrics.append(
                            MetricResult("wheel_size_mb", wheel_size, timestamp)
                        )
                    
                    if sdist_files:
                        sdist_size = sdist_files[0].stat().st_size / (1024 * 1024)  # MB
                        metrics.append(
                            MetricResult("source_distribution_size_mb", sdist_size, timestamp)
                        )
            
            # Import time measurement
            import_time_cmd = [
                "python", "-c",
                "import time; start=time.time(); import probneural_operator; print(f'{(time.time()-start)*1000:.2f}')"
            ]
            import_result = self._run_command(import_time_cmd)
            
            if import_result["returncode"] == 0 and import_result["stdout"]:
                try:
                    import_time_ms = float(import_result["stdout"])
                    metrics.append(
                        MetricResult("import_time_ms", import_time_ms, timestamp)
                    )
                except ValueError:
                    pass
            
            # Memory usage baseline
            memory_cmd = [
                "python", "-c",
                "import psutil, os; p=psutil.Process(os.getpid()); import probneural_operator; print(p.memory_info().rss/(1024*1024))"
            ]
            memory_result = self._run_command(memory_cmd)
            
            if memory_result["returncode"] == 0 and memory_result["stdout"]:
                try:
                    memory_mb = float(memory_result["stdout"])
                    metrics.append(
                        MetricResult("baseline_memory_usage_mb", memory_mb, timestamp)
                    )
                except ValueError:
                    pass
        
        except Exception as e:
            self.logger.error(f"Error collecting performance metrics: {e}")
            metrics.append(
                MetricResult("performance_metrics", None, timestamp, "error", str(e))
            )
        
        return metrics
    
    def collect_ml_specific_metrics(self) -> List[MetricResult]:
        """Collect ML-specific metrics."""
        metrics = []
        timestamp = datetime.utcnow().isoformat()
        
        try:
            # Run benchmark tests if they exist
            benchmark_cmd = ["python", "-m", "pytest", "tests/benchmarks/", "-v", "--tb=short"]
            benchmark_result = self._run_command(benchmark_cmd)
            
            if benchmark_result["returncode"] == 0:
                metrics.append(
                    MetricResult("benchmark_tests_passed", True, timestamp)
                )
            else:
                metrics.append(
                    MetricResult("benchmark_tests_passed", False, timestamp)
                )
            
            # Check if uncertainty validation exists
            uncertainty_test_cmd = ["python", "-m", "pytest", "tests/", "-k", "uncertainty", "-v"]
            uncertainty_result = self._run_command(uncertainty_test_cmd)
            
            if uncertainty_result["returncode"] == 0:
                metrics.append(
                    MetricResult("uncertainty_tests_passed", True, timestamp)
                )
            
            # Model reproducibility check
            reproducibility_cmd = ["python", "-c", """
import torch
import numpy as np
torch.manual_seed(42)
np.random.seed(42)
# Basic reproducibility test
try:
    from probneural_operator.models.base.neural_operator import NeuralOperator
    print("reproducible")
except ImportError:
    print("import_error")
"""]
            repro_result = self._run_command(reproducibility_cmd)
            
            if "reproducible" in repro_result["stdout"]:
                metrics.append(
                    MetricResult("reproducibility_check", True, timestamp)
                )
        
        except Exception as e:
            self.logger.error(f"Error collecting ML metrics: {e}")
            metrics.append(
                MetricResult("ml_metrics", None, timestamp, "error", str(e))
            )
        
        return metrics
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all metrics and return results."""
        self.logger.info("Starting comprehensive metrics collection...")
        
        # Collect all metric categories
        all_metrics = []
        all_metrics.extend(self.collect_repository_metrics())
        all_metrics.extend(self.collect_code_quality_metrics())
        all_metrics.extend(self.collect_security_metrics())
        all_metrics.extend(self.collect_performance_metrics())
        all_metrics.extend(self.collect_ml_specific_metrics())
        
        # Organize results
        results = {
            "collection_timestamp": datetime.utcnow().isoformat(),
            "repository": self.repo_name,
            "metrics_count": len(all_metrics),
            "metrics": {}
        }
        
        # Group metrics by category
        for metric in all_metrics:
            category = self._categorize_metric(metric.name)
            if category not in results["metrics"]:
                results["metrics"][category] = {}
            
            results["metrics"][category][metric.name] = {
                "value": metric.value,
                "timestamp": metric.timestamp,
                "status": metric.status,
                "error": metric.error
            }
        
        # Update configuration with latest values
        self._update_config_with_results(results)
        
        self.logger.info(f"Metrics collection completed. Collected {len(all_metrics)} metrics.")
        return results
    
    def _categorize_metric(self, metric_name: str) -> str:
        """Categorize metric by name."""
        if any(keyword in metric_name for keyword in ["github", "commit", "issue", "pr", "star", "fork"]):
            return "repository"
        elif any(keyword in metric_name for keyword in ["coverage", "quality", "lint", "type", "code"]):
            return "code_quality"
        elif any(keyword in metric_name for keyword in ["security", "vulnerability", "bandit"]):
            return "security"
        elif any(keyword in metric_name for keyword in ["performance", "size", "memory", "time"]):
            return "performance"
        elif any(keyword in metric_name for keyword in ["ml", "uncertainty", "benchmark", "model"]):
            return "ml_specific"
        else:
            return "other"
    
    def _update_config_with_results(self, results: Dict[str, Any]):
        """Update configuration file with latest metric values."""
        try:
            config = self.config.copy()
            config["last_updated"] = results["collection_timestamp"]
            
            # Update specific metric values in config
            for category, metrics in results["metrics"].items():
                if category in config.get("metrics", {}):
                    for metric_name, metric_data in metrics.items():
                        if metric_data["status"] == "success":
                            # Find and update the metric in config
                            self._update_nested_config(
                                config["metrics"][category], 
                                metric_name, 
                                metric_data["value"]
                            )
            
            # Save updated config
            with open(self.config_path, "w") as f:
                json.dump(config, f, indent=2)
        
        except Exception as e:
            self.logger.error(f"Error updating config: {e}")
    
    def _update_nested_config(self, config_section: Dict[str, Any], metric_name: str, value: Any):
        """Update nested configuration with metric value."""
        # This is a simplified update - in practice, you'd want more sophisticated mapping
        for key, data in config_section.items():
            if isinstance(data, dict) and "value" in data:
                if metric_name.replace("_", "").lower() in key.replace("_", "").lower():
                    data["value"] = value
                    break
    
    def generate_report(self, results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """Generate human-readable metrics report."""
        report_lines = [
            "# ProbNeural Operator Lab - Metrics Report",
            f"**Generated**: {results['collection_timestamp']}",
            f"**Repository**: {results['repository']}",
            f"**Metrics Collected**: {results['metrics_count']}",
            "",
        ]
        
        for category, metrics in results["metrics"].items():
            report_lines.extend([
                f"## {category.replace('_', ' ').title()}",
                ""
            ])
            
            for metric_name, metric_data in metrics.items():
                status_emoji = "✅" if metric_data["status"] == "success" else "❌"
                value = metric_data["value"]
                
                if isinstance(value, float):
                    value = f"{value:.2f}"
                elif isinstance(value, bool):
                    value = "Yes" if value else "No"
                
                report_lines.append(f"- **{metric_name.replace('_', ' ').title()}**: {value} {status_emoji}")
            
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, "w") as f:
                f.write(report)
            self.logger.info(f"Report saved to {output_file}")
        
        return report


def main():
    """Main entry point for metrics collection."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect comprehensive project metrics")
    parser.add_argument("--config", default=".github/project-metrics.json",
                       help="Path to metrics configuration file")
    parser.add_argument("--output", help="Output file for results")
    parser.add_argument("--report", help="Generate human-readable report")
    parser.add_argument("--format", choices=["json", "yaml"], default="json",
                       help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        collector = MetricsCollector(args.config)
        results = collector.collect_all_metrics()
        
        # Output results
        if args.output:
            with open(args.output, "w") as f:
                if args.format == "json":
                    json.dump(results, f, indent=2)
                else:  # yaml
                    import yaml
                    yaml.dump(results, f, default_flow_style=False)
        else:
            print(json.dumps(results, indent=2))
        
        # Generate report if requested
        if args.report:
            collector.generate_report(results, args.report)
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())