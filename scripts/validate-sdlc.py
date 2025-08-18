#!/usr/bin/env python3
"""
Comprehensive SDLC validation script for ProbNeural Operator Lab.
Validates all aspects of the Terragon SDLC implementation.
"""

import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging


@dataclass
class ValidationResult:
    """Container for validation results."""
    category: str
    check_name: str
    status: str  # "pass", "fail", "warning", "skip"
    message: str
    details: Optional[str] = None
    fix_suggestion: Optional[str] = None


class SDLCValidator:
    """Comprehensive SDLC validation orchestrator."""
    
    def __init__(self, verbose: bool = False):
        self.results: List[ValidationResult] = []
        self.repo_root = Path.cwd()
        self.verbose = verbose
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        level = logging.DEBUG if self.verbose else logging.INFO
        logger.setLevel(level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _run_command(self, command: List[str], cwd: Optional[Path] = None, timeout: int = 60) -> Tuple[int, str, str]:
        """Run shell command and return result."""
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=cwd or self.repo_root,
                timeout=timeout
            )
            return result.returncode, result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)
    
    def _file_exists(self, path: str) -> bool:
        """Check if file exists."""
        return (self.repo_root / path).exists()
    
    def _directory_exists(self, path: str) -> bool:
        """Check if directory exists."""
        return (self.repo_root / path).is_dir()
    
    def _add_result(self, category: str, check_name: str, status: str, 
                   message: str, details: str = None, fix_suggestion: str = None):
        """Add validation result."""
        self.results.append(ValidationResult(
            category=category,
            check_name=check_name,
            status=status,
            message=message,
            details=details,
            fix_suggestion=fix_suggestion
        ))
    
    def validate_project_foundation(self):
        """Validate project foundation and documentation."""
        self.logger.info("Validating project foundation...")
        
        # Essential files
        essential_files = {
            "README.md": "Project overview and quick start guide",
            "LICENSE": "Project license file", 
            "CONTRIBUTING.md": "Contribution guidelines",
            "CODE_OF_CONDUCT.md": "Community code of conduct",
            "SECURITY.md": "Security policy and vulnerability reporting",
            "CHANGELOG.md": "Version history and release notes",
            "PROJECT_CHARTER.md": "Project charter and scope",
            "CODEOWNERS": "Code ownership assignments"
        }
        
        for file_path, description in essential_files.items():
            if self._file_exists(file_path):
                self._add_result("foundation", f"{file_path}_exists", "pass", 
                               f"{description} exists")
            else:
                self._add_result("foundation", f"{file_path}_exists", "fail",
                               f"{description} missing",
                               fix_suggestion=f"Create {file_path} file")
        
        # Documentation structure
        doc_dirs = ["docs/", "docs/adr/", "docs/guides/", "docs/workflows/"]
        for doc_dir in doc_dirs:
            if self._directory_exists(doc_dir):
                self._add_result("foundation", f"{doc_dir}_exists", "pass",
                               f"Documentation directory {doc_dir} exists")
            else:
                self._add_result("foundation", f"{doc_dir}_exists", "warning",
                               f"Documentation directory {doc_dir} missing",
                               fix_suggestion=f"Create directory: mkdir -p {doc_dir}")
        
        # Architecture documentation
        if self._file_exists("docs/ARCHITECTURE.md"):
            self._add_result("foundation", "architecture_docs", "pass",
                           "Architecture documentation exists")
        else:
            self._add_result("foundation", "architecture_docs", "fail",
                           "Architecture documentation missing",
                           fix_suggestion="Create docs/ARCHITECTURE.md")
    
    def validate_development_environment(self):
        """Validate development environment and tooling."""
        self.logger.info("Validating development environment...")
        
        # Configuration files
        config_files = {
            "pyproject.toml": "Python project configuration",
            ".editorconfig": "Editor configuration",
            ".gitignore": "Git ignore patterns",
            ".pre-commit-config.yaml": "Pre-commit hooks configuration",
            ".env.example": "Environment variables template"
        }
        
        for file_path, description in config_files.items():
            if self._file_exists(file_path):
                self._add_result("dev_env", f"{file_path}_exists", "pass",
                               f"{description} exists")
            else:
                self._add_result("dev_env", f"{file_path}_exists", "fail",
                               f"{description} missing")
        
        # VSCode configuration
        vscode_files = [".vscode/settings.json", ".vscode/extensions.json", ".vscode/tasks.json"]
        for vscode_file in vscode_files:
            if self._file_exists(vscode_file):
                self._add_result("dev_env", f"vscode_config", "pass",
                               f"VSCode configuration exists")
                break
        else:
            self._add_result("dev_env", "vscode_config", "warning",
                           "VSCode configuration missing")
        
        # DevContainer
        if self._file_exists(".devcontainer/devcontainer.json"):
            self._add_result("dev_env", "devcontainer", "pass",
                           "DevContainer configuration exists")
        else:
            self._add_result("dev_env", "devcontainer", "warning",
                           "DevContainer configuration missing")
        
        # Setup scripts
        setup_scripts = ["scripts/setup-dev.sh"]
        for script in setup_scripts:
            if self._file_exists(script):
                # Check if executable
                script_path = self.repo_root / script
                if os.access(script_path, os.X_OK):
                    self._add_result("dev_env", f"setup_script", "pass",
                                   f"Setup script {script} exists and is executable")
                else:
                    self._add_result("dev_env", f"setup_script", "warning",
                                   f"Setup script {script} exists but not executable",
                                   fix_suggestion=f"chmod +x {script}")
            else:
                self._add_result("dev_env", f"setup_script", "fail",
                               f"Setup script {script} missing")
    
    def validate_testing_infrastructure(self):
        """Validate testing infrastructure."""
        self.logger.info("Validating testing infrastructure...")
        
        # Test directories
        test_dirs = ["tests/", "tests/unit/", "tests/integration/", "tests/fixtures/"]
        for test_dir in test_dirs:
            if self._directory_exists(test_dir):
                self._add_result("testing", f"{test_dir}_exists", "pass",
                               f"Test directory {test_dir} exists")
            else:
                self._add_result("testing", f"{test_dir}_exists", "fail",
                               f"Test directory {test_dir} missing")
        
        # Test configuration
        if self._file_exists("tests/conftest.py"):
            self._add_result("testing", "conftest_exists", "pass",
                           "Test configuration (conftest.py) exists")
        else:
            self._add_result("testing", "conftest_exists", "fail",
                           "Test configuration (conftest.py) missing")
        
        # Test utilities
        test_utils = ["tests/utils/", "tests/fixtures/"]
        for util_dir in test_utils:
            if self._directory_exists(util_dir):
                self._add_result("testing", f"test_utils", "pass",
                               f"Test utilities directory {util_dir} exists")
                break
        else:
            self._add_result("testing", "test_utils", "warning",
                           "Test utilities directories missing")
        
        # Check if tests can run
        returncode, stdout, stderr = self._run_command(["python", "-m", "pytest", "--version"])
        if returncode == 0:
            self._add_result("testing", "pytest_available", "pass",
                           "pytest is available")
        else:
            self._add_result("testing", "pytest_available", "fail",
                           "pytest is not available",
                           fix_suggestion="pip install pytest")
    
    def validate_build_containerization(self):
        """Validate build and containerization setup."""
        self.logger.info("Validating build and containerization...")
        
        # Docker files
        docker_files = ["Dockerfile", "docker-compose.yml", ".dockerignore"]
        for docker_file in docker_files:
            if self._file_exists(docker_file):
                self._add_result("build", f"{docker_file}_exists", "pass",
                               f"Docker file {docker_file} exists")
            else:
                self._add_result("build", f"{docker_file}_exists", "fail",
                               f"Docker file {docker_file} missing")
        
        # Build configuration
        if self._file_exists("Makefile"):
            self._add_result("build", "makefile_exists", "pass",
                           "Makefile exists")
        else:
            self._add_result("build", "makefile_exists", "warning",
                           "Makefile missing")
        
        # Build scripts
        build_scripts = ["scripts/build.sh", "scripts/release.sh"]
        for script in build_scripts:
            if self._file_exists(script):
                self._add_result("build", f"build_script", "pass",
                               f"Build script {script} exists")
            else:
                self._add_result("build", f"build_script", "warning",
                               f"Build script {script} missing")
        
        # Security configuration
        if self._file_exists(".security.yml"):
            self._add_result("build", "security_config", "pass",
                           "Security configuration exists")
        else:
            self._add_result("build", "security_config", "warning",
                           "Security configuration missing")
        
        # SBOM generation
        if self._file_exists("scripts/generate-sbom.sh"):
            self._add_result("build", "sbom_script", "pass",
                           "SBOM generation script exists")
        else:
            self._add_result("build", "sbom_script", "warning",
                           "SBOM generation script missing")
    
    def validate_monitoring_observability(self):
        """Validate monitoring and observability setup."""
        self.logger.info("Validating monitoring and observability...")
        
        # Monitoring directory
        if self._directory_exists("monitoring/"):
            self._add_result("monitoring", "monitoring_dir", "pass",
                           "Monitoring directory exists")
        else:
            self._add_result("monitoring", "monitoring_dir", "fail",
                           "Monitoring directory missing")
            return
        
        # Monitoring configuration files
        monitoring_configs = [
            "monitoring/prometheus.yml",
            "monitoring/alertmanager.yml", 
            "monitoring/grafana-dashboards.json",
            "monitoring/logging-config.yml"
        ]
        
        for config in monitoring_configs:
            if self._file_exists(config):
                self._add_result("monitoring", f"config_{Path(config).name}", "pass",
                               f"Configuration {config} exists")
            else:
                self._add_result("monitoring", f"config_{Path(config).name}", "warning",
                               f"Configuration {config} missing")
        
        # Monitoring scripts
        if self._file_exists("scripts/setup-monitoring.sh"):
            self._add_result("monitoring", "setup_script", "pass",
                           "Monitoring setup script exists")
        else:
            self._add_result("monitoring", "setup_script", "warning",
                           "Monitoring setup script missing")
    
    def validate_workflow_templates(self):
        """Validate CI/CD workflow templates."""
        self.logger.info("Validating workflow templates...")
        
        # Workflow templates directory
        if self._directory_exists("templates/github/workflows/"):
            self._add_result("workflows", "template_dir", "pass",
                           "Workflow templates directory exists")
        else:
            self._add_result("workflows", "template_dir", "fail",
                           "Workflow templates directory missing")
            return
        
        # Required workflow templates
        required_workflows = [
            "templates/github/workflows/ci.yml",
            "templates/github/workflows/release.yml",
            "templates/github/workflows/security-scan.yml"
        ]
        
        for workflow in required_workflows:
            if self._file_exists(workflow):
                self._add_result("workflows", f"template_{Path(workflow).stem}", "pass",
                               f"Workflow template {Path(workflow).name} exists")
            else:
                self._add_result("workflows", f"template_{Path(workflow).stem}", "fail",
                               f"Workflow template {Path(workflow).name} missing")
        
        # GitHub configuration templates
        github_configs = [
            "templates/github/dependabot.yml",
            "templates/github/codeql.yml"
        ]
        
        for config in github_configs:
            if self._file_exists(config):
                self._add_result("workflows", f"config_{Path(config).stem}", "pass",
                               f"GitHub configuration {Path(config).name} exists")
            else:
                self._add_result("workflows", f"config_{Path(config).stem}", "warning",
                               f"GitHub configuration {Path(config).name} missing")
        
        # Issue and PR templates
        template_dirs = ["templates/github/ISSUE_TEMPLATE/"]
        for template_dir in template_dirs:
            if self._directory_exists(template_dir):
                self._add_result("workflows", "issue_templates", "pass",
                               "Issue templates directory exists")
            else:
                self._add_result("workflows", "issue_templates", "warning",
                               "Issue templates directory missing")
        
        # Workflow setup documentation
        if self._file_exists("docs/workflows/WORKFLOW_SETUP_GUIDE.md"):
            self._add_result("workflows", "setup_docs", "pass",
                           "Workflow setup documentation exists")
        else:
            self._add_result("workflows", "setup_docs", "fail",
                           "Workflow setup documentation missing")
    
    def validate_metrics_automation(self):
        """Validate metrics and automation setup."""
        self.logger.info("Validating metrics and automation...")
        
        # Metrics configuration
        if self._file_exists(".github/project-metrics.json"):
            self._add_result("metrics", "metrics_config", "pass",
                           "Project metrics configuration exists")
            
            # Validate JSON structure
            try:
                with open(self.repo_root / ".github/project-metrics.json") as f:
                    metrics_data = json.load(f)
                    
                required_sections = ["project_name", "version", "metrics"]
                for section in required_sections:
                    if section in metrics_data:
                        self._add_result("metrics", f"metrics_{section}", "pass",
                                       f"Metrics section {section} exists")
                    else:
                        self._add_result("metrics", f"metrics_{section}", "warning",
                                       f"Metrics section {section} missing")
            except Exception as e:
                self._add_result("metrics", "metrics_json_valid", "fail",
                               f"Invalid metrics JSON: {e}")
        else:
            self._add_result("metrics", "metrics_config", "fail",
                           "Project metrics configuration missing")
        
        # Automation scripts
        automation_scripts = [
            "scripts/collect-metrics.py",
            "scripts/update-dependencies.sh",
            "scripts/generate-dashboard.py"
        ]
        
        for script in automation_scripts:
            if self._file_exists(script):
                script_path = self.repo_root / script
                if os.access(script_path, os.X_OK):
                    self._add_result("metrics", f"script_{Path(script).stem}", "pass",
                                   f"Automation script {script} exists and is executable")
                else:
                    self._add_result("metrics", f"script_{Path(script).stem}", "warning",
                                   f"Automation script {script} exists but not executable")
            else:
                self._add_result("metrics", f"script_{Path(script).stem}", "fail",
                               f"Automation script {script} missing")
    
    def validate_integration_completeness(self):
        """Validate overall integration and completeness."""
        self.logger.info("Validating integration completeness...")
        
        # Setup documentation
        if self._file_exists("SETUP_REQUIRED.md"):
            self._add_result("integration", "setup_docs", "pass",
                           "Setup documentation exists")
        else:
            self._add_result("integration", "setup_docs", "fail",
                           "Setup documentation missing")
        
        # Python package structure
        if self._directory_exists("probneural_operator/"):
            self._add_result("integration", "package_structure", "pass",
                           "Main package directory exists")
        else:
            self._add_result("integration", "package_structure", "fail",
                           "Main package directory missing")
        
        # Version consistency check
        try:
            with open(self.repo_root / "pyproject.toml") as f:
                content = f.read()
                if 'version = ' in content:
                    self._add_result("integration", "version_in_pyproject", "pass",
                                   "Version defined in pyproject.toml")
                else:
                    self._add_result("integration", "version_in_pyproject", "warning",
                                   "Version not found in pyproject.toml")
        except Exception as e:
            self._add_result("integration", "version_check", "fail",
                           f"Error checking version: {e}")
        
        # Git repository validation
        returncode, stdout, stderr = self._run_command(["git", "status"])
        if returncode == 0:
            self._add_result("integration", "git_repo", "pass",
                           "Git repository is properly initialized")
        else:
            self._add_result("integration", "git_repo", "fail",
                           "Git repository issues detected")
        
        # Python import test
        returncode, stdout, stderr = self._run_command([
            "python", "-c", "import probneural_operator; print('Import successful')"
        ])
        if returncode == 0:
            self._add_result("integration", "import_test", "pass",
                           "Package imports successfully")
        else:
            self._add_result("integration", "import_test", "warning",
                           f"Package import issues: {stderr}")
    
    def run_all_validations(self) -> List[ValidationResult]:
        """Run all validation checks."""
        self.logger.info("Starting comprehensive SDLC validation...")
        
        validation_methods = [
            self.validate_project_foundation,
            self.validate_development_environment,
            self.validate_testing_infrastructure,
            self.validate_build_containerization,
            self.validate_monitoring_observability,
            self.validate_workflow_templates,
            self.validate_metrics_automation,
            self.validate_integration_completeness
        ]
        
        for method in validation_methods:
            try:
                method()
            except Exception as e:
                self.logger.error(f"Error in {method.__name__}: {e}")
                self._add_result("error", method.__name__, "fail",
                               f"Validation method failed: {e}")
        
        self.logger.info(f"Validation completed. {len(self.results)} checks performed.")
        return self.results
    
    def generate_report(self) -> str:
        """Generate validation report."""
        if not self.results:
            return "No validation results available."
        
        # Group results by category
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)
        
        # Count statistics
        total_checks = len(self.results)
        passed = len([r for r in self.results if r.status == "pass"])
        failed = len([r for r in self.results if r.status == "fail"])
        warnings = len([r for r in self.results if r.status == "warning"])
        skipped = len([r for r in self.results if r.status == "skip"])
        
        # Generate report
        report_lines = [
            "# Terragon SDLC Validation Report",
            f"**Generated**: {os.popen('date').read().strip()}",
            f"**Repository**: {os.getcwd()}",
            "",
            "## Summary",
            f"- **Total Checks**: {total_checks}",
            f"- **Passed**: {passed} ✅",
            f"- **Failed**: {failed} ❌",
            f"- **Warnings**: {warnings} ⚠️",
            f"- **Skipped**: {skipped} ⏭️",
            "",
            f"**Overall Status**: {'✅ PASS' if failed == 0 else '❌ ISSUES FOUND'}",
            ""
        ]
        
        # Add details by category
        for category, results in categories.items():
            report_lines.extend([
                f"## {category.replace('_', ' ').title()}",
                ""
            ])
            
            for result in results:
                status_emoji = {
                    "pass": "✅",
                    "fail": "❌", 
                    "warning": "⚠️",
                    "skip": "⏭️"
                }.get(result.status, "❓")
                
                report_lines.append(f"### {result.check_name} {status_emoji}")
                report_lines.append(f"**Status**: {result.status.upper()}")
                report_lines.append(f"**Message**: {result.message}")
                
                if result.details:
                    report_lines.append(f"**Details**: {result.details}")
                
                if result.fix_suggestion:
                    report_lines.append(f"**Fix**: {result.fix_suggestion}")
                
                report_lines.append("")
        
        # Add recommendations
        if failed > 0:
            report_lines.extend([
                "## Recommended Actions",
                ""
            ])
            
            critical_failures = [r for r in self.results if r.status == "fail"]
            for i, failure in enumerate(critical_failures[:5], 1):  # Top 5 failures
                report_lines.append(f"{i}. **{failure.check_name}**: {failure.message}")
                if failure.fix_suggestion:
                    report_lines.append(f"   - *Fix*: {failure.fix_suggestion}")
                report_lines.append("")
        
        return "\n".join(report_lines)


def main():
    """Main entry point for SDLC validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Terragon SDLC implementation")
    parser.add_argument("--output", help="Output file for validation report")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown",
                       help="Output format")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose logging")
    parser.add_argument("--category", help="Run specific validation category only")
    
    args = parser.parse_args()
    
    try:
        validator = SDLCValidator(verbose=args.verbose)
        results = validator.run_all_validations()
        
        # Filter by category if specified
        if args.category:
            results = [r for r in results if r.category == args.category]
        
        # Generate output
        if args.format == "json":
            output = json.dumps([
                {
                    "category": r.category,
                    "check_name": r.check_name,
                    "status": r.status,
                    "message": r.message,
                    "details": r.details,
                    "fix_suggestion": r.fix_suggestion
                }
                for r in results
            ], indent=2)
        else:
            output = validator.generate_report()
        
        # Write to file or stdout
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"✅ Validation report saved to {args.output}")
        else:
            print(output)
        
        # Exit with error code if there are failures
        failed_count = len([r for r in results if r.status == "fail"])
        return 1 if failed_count > 0 else 0
    
    except Exception as e:
        print(f"❌ Error running validation: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())