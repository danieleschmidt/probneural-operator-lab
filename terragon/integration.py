"""
Terragon SDLC Integration Module
===============================

Integration layer between Terragon Quality Gates and existing Terragon SDLC infrastructure.
Provides seamless integration with Terragon configuration, value metrics, and reporting systems.
"""

import json
from datetime import datetime

try:
    import yaml
except ImportError:
    # Fallback YAML parser for basic functionality
    class MockYaml:
        @staticmethod
        def safe_load(stream):
            # Very basic YAML parsing for configuration files
            if hasattr(stream, 'read'):
                content = stream.read()
            else:
                content = stream
            
            # Handle empty or None content
            if not content or content.strip() == '':
                return {}
            
            # Basic parsing - this is a simple fallback
            result = {}
            lines = content.strip().split('\n')
            current_section = result
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if ':' in line and not line.startswith(' '):
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if value == '':
                        current_section[key] = {}
                    elif value.lower() in ['true', 'false']:
                        current_section[key] = value.lower() == 'true'
                    elif value.isdigit():
                        current_section[key] = int(value)
                    else:
                        current_section[key] = value.strip('"\'')
            
            return result
        
        @staticmethod
        def dump(data, stream=None, **kwargs):
            def dict_to_yaml(d, indent=0):
                lines = []
                for key, value in d.items():
                    if isinstance(value, dict):
                        lines.append('  ' * indent + f"{key}:")
                        lines.extend(dict_to_yaml(value, indent + 1))
                    else:
                        lines.append('  ' * indent + f"{key}: {value}")
                return lines
            
            yaml_content = '\n'.join(dict_to_yaml(data))
            
            if stream:
                stream.write(yaml_content)
            else:
                return yaml_content
    
    yaml = MockYaml()
from pathlib import Path
from typing import Any, Dict, List, Optional

from .quality_gates.core import QualityGateFramework, QualityGateResult
from .quality_gates.monitoring import ContinuousQualityMonitor
from .quality_gates.adaptive import AdaptiveQualityController


class TeragonSDLCIntegration:
    """
    Integration layer for Terragon SDLC systems.
    
    Provides integration with:
    - Terragon configuration system
    - Value metrics tracking
    - Autonomous discovery and prioritization
    - GitHub Actions integration
    - Reporting and notifications
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.terragon_dir = self.project_root / ".terragon"
        
        # Initialize components
        self.quality_framework = QualityGateFramework()
        self.monitor = ContinuousQualityMonitor()
        self.adaptive_controller = AdaptiveQualityController()
        
        # Load Terragon configuration
        self.terragon_config = self._load_terragon_config()
        self.value_metrics = self._load_value_metrics()
        
        # Integration state
        self.integration_history: List[Dict[str, Any]] = []
    
    def _load_terragon_config(self) -> Dict[str, Any]:
        """Load Terragon configuration."""
        config_path = self.terragon_dir / "config.yaml"
        
        if config_path.exists():
            try:
                with open(config_path) as f:
                    return yaml.safe_load(f)
            except Exception as e:
                print(f"Warning: Could not load Terragon config: {e}")
        
        return {}
    
    def _load_value_metrics(self) -> Dict[str, Any]:
        """Load Terragon value metrics."""
        metrics_path = self.terragon_dir / "value-metrics.json"
        
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load value metrics: {e}")
        
        return {}
    
    def update_value_metrics(self, quality_results: Dict[str, QualityGateResult]) -> None:
        """Update Terragon value metrics with quality gate results."""
        # Calculate overall quality score
        if quality_results:
            total_score = sum(result.percentage_score for result in quality_results.values())
            avg_score = total_score / len(quality_results)
        else:
            avg_score = 0.0
        
        # Update value metrics
        current_metrics = self.value_metrics.copy()
        
        quality_metrics = {
            "overall_quality_score": avg_score,
            "quality_gate_results": {
                name: {
                    "score": result.percentage_score,
                    "status": result.status.value,
                    "execution_time": result.execution_time
                }
                for name, result in quality_results.items()
            },
            "last_quality_assessment": datetime.now().isoformat(),
            "quality_trend": self._calculate_quality_trend(avg_score),
        }
        
        # Merge with existing metrics
        current_metrics["quality"] = quality_metrics
        
        # Calculate value impact
        value_impact = self._calculate_value_impact(quality_results)
        current_metrics["value_impact"] = value_impact
        
        # Save updated metrics
        metrics_path = self.terragon_dir / "value-metrics.json"
        try:
            with open(metrics_path, 'w') as f:
                json.dump(current_metrics, f, indent=2)
            
            self.value_metrics = current_metrics
            
        except Exception as e:
            print(f"Warning: Could not save value metrics: {e}")
    
    def _calculate_quality_trend(self, current_score: float) -> str:
        """Calculate quality trend based on historical data."""
        # Get previous score from metrics
        previous_score = self.value_metrics.get("quality", {}).get("overall_quality_score", current_score)
        
        diff = current_score - previous_score
        
        if abs(diff) < 2:
            return "stable"
        elif diff > 0:
            return "improving"
        else:
            return "declining"
    
    def _calculate_value_impact(self, quality_results: Dict[str, QualityGateResult]) -> Dict[str, Any]:
        """Calculate business value impact of quality improvements."""
        # Map quality improvements to business value
        value_mapping = {
            "test_coverage": {
                "category": "risk_reduction",
                "multiplier": 0.8,
                "description": "Reduces production defect risk"
            },
            "security": {
                "category": "compliance",
                "multiplier": 1.5,
                "description": "Improves security posture and compliance"
            },
            "performance": {
                "category": "user_experience",
                "multiplier": 1.2,
                "description": "Enhances user experience and scalability"
            },
            "code_quality": {
                "category": "maintainability",
                "multiplier": 0.9,
                "description": "Improves development velocity and maintainability"
            }
        }
        
        total_value_score = 0.0
        value_breakdown = {}
        
        for gate_name, result in quality_results.items():
            # Map gate to value category
            for category, mapping in value_mapping.items():
                if category.lower() in gate_name.lower():
                    category_value = result.percentage_score * mapping["multiplier"]
                    total_value_score += category_value
                    
                    value_breakdown[mapping["category"]] = {
                        "score": category_value,
                        "description": mapping["description"],
                        "gate": gate_name
                    }
                    break
        
        return {
            "total_value_score": total_value_score,
            "value_breakdown": value_breakdown,
            "value_trend": "positive" if total_value_score > 80 else "needs_improvement",
            "calculated_at": datetime.now().isoformat()
        }
    
    def integrate_with_github_actions(self) -> Dict[str, Any]:
        """Generate GitHub Actions integration configuration."""
        workflow_config = {
            "name": "Terragon Quality Gates",
            "on": {
                "push": {"branches": ["main", "develop"]},
                "pull_request": {"branches": ["main"]},
                "schedule": [{"cron": "0 2 * * *"}]  # Daily at 2 AM
            },
            "jobs": {
                "quality-gates": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v4"
                        },
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "3.11"}
                        },
                        {
                            "name": "Install dependencies",
                            "run": "pip install -e .[dev,test]"
                        },
                        {
                            "name": "Run Quality Gates - Generation 1",
                            "run": "python terragon/quality_gates/quality_gates.py gen1"
                        },
                        {
                            "name": "Run Quality Gates - Generation 2", 
                            "run": "python terragon/quality_gates/quality_gates.py gen2",
                            "if": "success()"
                        },
                        {
                            "name": "Run Quality Gates - Generation 3",
                            "run": "python terragon/quality_gates/quality_gates.py gen3",
                            "if": "success()"
                        },
                        {
                            "name": "Run Research Quality Gates",
                            "run": "python terragon/quality_gates/quality_gates.py research",
                            "if": "success()"
                        },
                        {
                            "name": "Upload Quality Reports",
                            "uses": "actions/upload-artifact@v3",
                            "with": {
                                "name": "quality-reports",
                                "path": ".terragon/reports/"
                            },
                            "if": "always()"
                        }
                    ]
                }
            }
        }
        
        # Save workflow file
        workflows_dir = self.project_root / ".github" / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)
        
        workflow_path = workflows_dir / "terragon-quality-gates.yml"
        
        try:
            with open(workflow_path, 'w') as f:
                yaml.dump(workflow_config, f, default_flow_style=False)
            
            return {
                "status": "success",
                "workflow_path": str(workflow_path),
                "message": "GitHub Actions workflow created successfully"
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to create GitHub Actions workflow: {e}"
            }
    
    def generate_terragon_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive Terragon SDLC summary report."""
        # Get quality data
        quality_report = self.quality_framework.get_execution_report()
        monitoring_status = self.monitor.get_current_status()
        adaptation_report = self.adaptive_controller.generate_adaptation_report()
        
        # Calculate SDLC maturity score
        maturity_score = self._calculate_sdlc_maturity()
        
        summary_report = {
            "terragon_sdlc_summary": {
                "generated_at": datetime.now().isoformat(),
                "project_root": str(self.project_root),
                "maturity_level": self._get_maturity_level(maturity_score),
                "maturity_score": maturity_score,
            },
            "quality_assessment": {
                "overall_score": quality_report.get("summary", {}).get("pass_rate", 0),
                "total_executions": quality_report.get("summary", {}).get("total_executions", 0),
                "current_generation": quality_report.get("summary", {}).get("current_generation", "unknown"),
                "monitoring_active": monitoring_status.get("monitoring_active", False),
                "adaptation_enabled": adaptation_report.get("summary", {}).get("adaptation_enabled", False),
            },
            "value_metrics": self.value_metrics,
            "terragon_config": {
                "maturity_level": self.terragon_config.get("metadata", {}).get("maturity_level", "unknown"),
                "primary_language": self.terragon_config.get("metadata", {}).get("primary_language", "unknown"),
                "domain": self.terragon_config.get("metadata", {}).get("domain", "unknown"),
            },
            "integration_status": {
                "quality_gates_integrated": True,
                "monitoring_integrated": monitoring_status.get("monitoring_active", False),
                "adaptive_control_integrated": True,
                "github_actions_available": (self.project_root / ".github" / "workflows").exists(),
                "value_metrics_tracking": bool(self.value_metrics),
            },
            "recommendations": self._generate_integration_recommendations(),
            "next_steps": self._generate_next_steps(),
        }
        
        return summary_report
    
    def _calculate_sdlc_maturity(self) -> float:
        """Calculate overall SDLC maturity score."""
        maturity_factors = {
            "quality_gates": 25.0,  # Quality gates implementation
            "monitoring": 20.0,     # Continuous monitoring
            "adaptation": 15.0,     # Adaptive intelligence
            "automation": 20.0,     # CI/CD automation
            "value_tracking": 10.0, # Value metrics tracking
            "integration": 10.0,    # System integration
        }
        
        scores = {}
        
        # Quality gates score
        quality_report = self.quality_framework.get_execution_report()
        scores["quality_gates"] = min(100.0, quality_report.get("summary", {}).get("pass_rate", 0))
        
        # Monitoring score
        monitoring_status = self.monitor.get_current_status()
        scores["monitoring"] = 100.0 if monitoring_status.get("monitoring_active") else 50.0
        
        # Adaptation score
        adaptation_report = self.adaptive_controller.generate_adaptation_report()
        adaptation_enabled = adaptation_report.get("summary", {}).get("adaptation_enabled", False)
        executions_learned = adaptation_report.get("summary", {}).get("executions_learned", 0)
        scores["adaptation"] = (100.0 if adaptation_enabled else 0.0) + min(50.0, executions_learned)
        
        # Automation score
        github_workflows = (self.project_root / ".github" / "workflows").exists()
        ci_config = (self.project_root / ".github" / "workflows" / "ci.yml").exists()
        scores["automation"] = (50.0 if github_workflows else 0.0) + (50.0 if ci_config else 0.0)
        
        # Value tracking score
        scores["value_tracking"] = 100.0 if self.value_metrics else 0.0
        
        # Integration score
        terragon_config_exists = (self.terragon_dir / "config.yaml").exists()
        quality_config_exists = (self.terragon_dir / "quality_gates.json").exists()
        scores["integration"] = (50.0 if terragon_config_exists else 0.0) + (50.0 if quality_config_exists else 0.0)
        
        # Calculate weighted average
        total_score = 0.0
        total_weight = 0.0
        
        for factor, weight in maturity_factors.items():
            if factor in scores:
                total_score += scores[factor] * weight / 100.0
                total_weight += weight
        
        return (total_score / total_weight * 100.0) if total_weight > 0 else 0.0
    
    def _get_maturity_level(self, score: float) -> str:
        """Get maturity level description."""
        if score >= 90:
            return "advanced"
        elif score >= 75:
            return "maturing"
        elif score >= 50:
            return "developing"
        else:
            return "nascent"
    
    def _generate_integration_recommendations(self) -> List[str]:
        """Generate integration recommendations."""
        recommendations = []
        
        # Check GitHub integration
        if not (self.project_root / ".github" / "workflows").exists():
            recommendations.append("Set up GitHub Actions workflows for automated quality gates")
        
        # Check monitoring
        monitoring_status = self.monitor.get_current_status()
        if not monitoring_status.get("monitoring_active"):
            recommendations.append("Enable continuous quality monitoring for real-time feedback")
        
        # Check value metrics
        if not self.value_metrics:
            recommendations.append("Initialize value metrics tracking for business impact measurement")
        
        # Check adaptive learning
        adaptation_report = self.adaptive_controller.generate_adaptation_report()
        if adaptation_report.get("summary", {}).get("executions_learned", 0) < 10:
            recommendations.append("Run more quality gate executions to improve adaptive learning")
        
        return recommendations
    
    def _generate_next_steps(self) -> List[str]:
        """Generate next steps for SDLC improvement."""
        next_steps = []
        
        maturity_score = self._calculate_sdlc_maturity()
        
        if maturity_score < 50:
            next_steps.extend([
                "Complete basic quality gate implementation",
                "Set up continuous integration workflows",
                "Establish baseline quality metrics"
            ])
        elif maturity_score < 75:
            next_steps.extend([
                "Enable continuous quality monitoring",
                "Implement adaptive quality thresholds",
                "Set up automated reporting and notifications"
            ])
        else:
            next_steps.extend([
                "Optimize quality gate performance",
                "Implement advanced analytics and predictions",
                "Share best practices with other teams"
            ])
        
        return next_steps
    
    def save_integration_report(self) -> str:
        """Save comprehensive integration report."""
        report = self.generate_terragon_summary_report()
        
        # Save to Terragon reports directory
        reports_dir = self.terragon_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"terragon_integration_report_{timestamp}.json"
        
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            # Also save as latest
            latest_path = reports_dir / "terragon_integration_latest.json"
            with open(latest_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            return str(report_path)
            
        except Exception as e:
            raise Exception(f"Failed to save integration report: {e}")


# Utility functions for easy integration
def setup_terragon_integration(project_root: Optional[Path] = None) -> TeragonSDLCIntegration:
    """Set up Terragon SDLC integration."""
    integration = TeragonSDLCIntegration(project_root)
    
    # Create GitHub Actions workflow
    workflow_result = integration.integrate_with_github_actions()
    
    if workflow_result["status"] == "success":
        print(f"✅ GitHub Actions workflow created: {workflow_result['workflow_path']}")
    else:
        print(f"⚠️  GitHub Actions setup failed: {workflow_result['message']}")
    
    return integration


def generate_integration_summary() -> Dict[str, Any]:
    """Generate integration summary for current project."""
    integration = TeragonSDLCIntegration()
    return integration.generate_terragon_summary_report()


if __name__ == "__main__":
    # CLI usage
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "setup":
        integration = setup_terragon_integration()
        report_path = integration.save_integration_report()
        print(f"📄 Integration report saved: {report_path}")
    else:
        summary = generate_integration_summary()
        print(json.dumps(summary, indent=2))