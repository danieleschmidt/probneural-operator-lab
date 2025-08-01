# Automated Dependency Management Setup

Comprehensive dependency management automation for secure, up-to-date dependencies in ProbNeural-Operator-Lab.

## Overview

This document provides complete automation for dependency management including vulnerability scanning, automated updates, compatibility testing, and security patching.

## Dependency Management Strategy

### 1. Current Dependency Structure

The repository uses modern Python packaging with `pyproject.toml`:

```toml
[project]
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "matplotlib>=3.5.0",
    "scikit-learn>=1.1.0",
    "tqdm>=4.64.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
    # ... other dev dependencies
]
```

### 2. Dependency Security Levels

**Critical Dependencies** (Core functionality):
- torch, numpy, scipy - Require careful testing before updates
- Manual review required for major version changes

**Standard Dependencies** (Features):
- matplotlib, scikit-learn, tqdm - Automated minor/patch updates
- Major versions require review

**Development Dependencies** (Tooling):
- pytest, black, ruff, mypy - Automated updates with testing
- Low risk for automated updates

## Automated Dependency Management Tools

### 1. Dependabot Configuration

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "08:00"
      timezone: "UTC"
    
    # Limit concurrent PRs
    open-pull-requests-limit: 5
    
    # Grouping strategies
    groups:
      # Group dev dependencies together
      dev-dependencies:
        patterns:
          - "pytest*"
          - "black"
          - "ruff"
          - "mypy"
          - "isort"
          - "pre-commit"
        
      # Group ML/scientific dependencies  
      ml-dependencies:
        patterns:
          - "torch*"
          - "numpy"
          - "scipy"
          - "scikit-learn"
        update-types:
          - "minor"
          - "patch"
    
    # Auto-merge configuration
    allow:
      - dependency-type: "direct:production"
        update-type: "security"
      - dependency-type: "direct:development"
        update-type: "version-update:semver-patch"
      - dependency-type: "indirect"
        update-type: "security"
    
    # Review settings
    reviewers:
      - "maintainer-team"
    assignees:
      - "security-team"
    
    # Labels for organization
    labels:
      - "dependencies"
      - "automated"
    
    # Commit message customization
    commit-message:
      prefix: "deps"
      prefix-development: "deps-dev"
      include: "scope"
    
    # Target branch
    target-branch: "main"
    
    # Vendor files to ignore
    ignore:
      - dependency-name: "*"
        versions: ["< 1.0"]  # Ignore pre-1.0 versions
  
  # GitHub Actions dependencies
  - package-ecosystem: "github-actions"
    directory: "/.github/workflows"
    schedule:
      interval: "monthly"
      day: "first-monday"
      time: "08:00"
    
    commit-message:
      prefix: "ci"
      include: "scope"
    
    labels:
      - "github-actions"
      - "dependencies"

  # Docker dependencies (when added)
  - package-ecosystem: "docker"
    directory: "/docker"
    schedule:
      interval: "weekly"
    
    commit-message:
      prefix: "docker"
    
    labels:
      - "docker"
      - "dependencies"
```

### 2. Renovate Configuration (Alternative to Dependabot)

Create `renovate.json` for more advanced dependency management:

```json
{
  "$schema": "https://docs.renovatebot.com/renovate-schema.json",
  "extends": [
    "config:base",
    "schedule:weeklyNonOfficeHours",
    ":dependencyDashboard",
    ":semanticCommits",
    ":separatePatchReleases"
  ],
  "timezone": "UTC",
  "schedule": ["before 6am on monday"],
  
  "packageRules": [
    {
      "groupName": "ML/Scientific packages",
      "matchPackagePatterns": ["torch", "numpy", "scipy", "scikit-learn"],
      "schedule": ["before 6am on first day of month"],
      "automerge": false,
      "reviewersFromCodeowners": true,
      "labels": ["ml-dependencies", "requires-review"]
    },
    {
      "groupName": "Development tools",
      "matchPackagePatterns": ["pytest", "black", "ruff", "mypy", "isort"],
      "automerge": true,
      "automergeType": "pr",
      "matchUpdateTypes": ["patch", "minor"],
      "labels": ["dev-dependencies", "auto-merge"]
    },
    {
      "groupName": "Security updates",
      "matchDatasources": ["pypi"],
      "matchUpdateTypes": ["patch"],
      "isVulnerabilityAlert": true,
      "automerge": true,
      "schedule": ["at any time"],
      "labels": ["security", "auto-merge"],
      "prPriority": 10
    }
  ],
  
  "vulnerabilityAlerts": {
    "enabled": true,
    "schedule": ["at any time"]
  },
  
  "lockFileMaintenance": {
    "enabled": false
  },
  
  "prConcurrentLimit": 5,
  "prHourlyLimit": 2,
  
  "commitMessagePrefix": "deps:",
  "commitMessageTopic": "{{depName}}",
  "commitMessageExtra": "to {{newVersion}}",
  
  "prTitle": "deps: update {{depName}} to {{newVersion}}",
  "prBodyTemplate": "This PR updates {{depName}} from {{currentVersion}} to {{newVersion}}.\n\n{{{changelog}}}\n\n---\n\n**Automated by Renovate**",
  
  "assignees": ["security-team"],
  "reviewers": ["maintainer-team"]
}
```

### 3. Custom Dependency Management Scripts

Create `scripts/dependency-manager.py`:

```python
#!/usr/bin/env python3
"""
Advanced dependency management automation.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
import requests
import semver
import toml

class DependencyManager:
    def __init__(self, pyproject_path: str = "pyproject.toml"):
        self.pyproject_path = Path(pyproject_path)
        self.pyproject_data = self.load_pyproject()
        
    def load_pyproject(self) -> dict:
        """Load pyproject.toml configuration."""
        if not self.pyproject_path.exists():
            raise FileNotFoundError(f"pyproject.toml not found at {self.pyproject_path}")
        
        return toml.load(self.pyproject_path)
    
    def get_current_dependencies(self) -> Dict[str, str]:
        """Extract current dependency versions."""
        deps = {}
        
        # Main dependencies
        for dep in self.pyproject_data['project']['dependencies']:
            name, version = self.parse_dependency(dep)
            deps[name] = version
            
        # Optional dependencies
        for group, group_deps in self.pyproject_data['project']['optional-dependencies'].items():
            for dep in group_deps:
                name, version = self.parse_dependency(dep)
                deps[f"{name}[{group}]"] = version
        
        return deps
    
    def parse_dependency(self, dep_string: str) -> tuple:
        """Parse dependency string into name and version constraint."""
        if '>=' in dep_string:
            name, version = dep_string.split('>=')
            return name.strip(), f">={version.strip()}"
        elif '==' in dep_string:
            name, version = dep_string.split('==')
            return name.strip(), f"=={version.strip()}"
        else:
            return dep_string.strip(), ""
    
    def check_vulnerabilities(self) -> List[dict]:
        """Check for known vulnerabilities using pip-audit."""
        try:
            result = subprocess.run(
                ['pip-audit', '--format', 'json', '--desc'],
                capture_output=True,
                text=True,
                check=True
            )
            
            data = json.loads(result.stdout)
            return data.get('vulnerabilities', [])
            
        except subprocess.CalledProcessError as e:
            print(f"Error running pip-audit: {e}")
            return []
        except json.JSONDecodeError:
            print("Error parsing pip-audit output")
            return []
    
    def get_latest_versions(self, packages: List[str]) -> Dict[str, str]:
        """Get latest versions from PyPI."""
        latest_versions = {}
        
        for package in packages:
            try:
                response = requests.get(f"https://pypi.org/pypi/{package}/json")
                if response.status_code == 200:
                    data = response.json()
                    latest_versions[package] = data['info']['version']
                else:
                    print(f"Could not fetch version for {package}")
                    
            except requests.RequestException as e:
                print(f"Error fetching {package}: {e}")
        
        return latest_versions
    
    def check_compatibility(self, package: str, version: str) -> bool:
        """Check if package version is compatible with Python version."""
        try:
            response = requests.get(f"https://pypi.org/pypi/{package}/{version}/json")
            if response.status_code == 200:
                data = response.json()
                classifiers = data['info']['classifiers']
                
                # Check Python version compatibility
                python_versions = [c for c in classifiers if c.startswith('Programming Language :: Python ::')]
                
                # Simple compatibility check (can be enhanced)
                return any('3.9' in v or '3.10' in v or '3.11' in v or '3.12' in v 
                          for v in python_versions)
            
        except requests.RequestException:
            return False
        
        return True
    
    def generate_update_plan(self) -> Dict[str, dict]:
        """Generate dependency update plan."""
        current_deps = self.get_current_dependencies()
        vulnerabilities = self.check_vulnerabilities()
        
        # Extract package names without version constraints
        package_names = [name.split('[')[0] for name in current_deps.keys()]
        latest_versions = self.get_latest_versions(package_names)
        
        update_plan = {}
        
        for package, current_version in current_deps.items():
            package_name = package.split('[')[0]  # Remove optional group notation
            
            if package_name in latest_versions:
                latest = latest_versions[package_name]
                
                # Check if update is needed
                needs_update = self.needs_update(current_version, latest)
                
                # Check for vulnerabilities
                has_vulnerability = any(
                    vuln['package'] == package_name for vuln in vulnerabilities
                )
                
                update_plan[package] = {
                    'current_version': current_version,
                    'latest_version': latest,
                    'needs_update': needs_update,
                    'has_vulnerability': has_vulnerability,
                    'update_priority': self.calculate_priority(
                        package_name, needs_update, has_vulnerability
                    ),
                    'compatibility_checked': self.check_compatibility(package_name, latest)
                }
        
        return update_plan
    
    def needs_update(self, current: str, latest: str) -> bool:
        """Determine if package needs update."""
        if not current or not latest:
            return False
            
        # Extract version number from constraint
        if '>=' in current:
            current_version = current.replace('>=', '').strip()
        elif '==' in current:
            current_version = current.replace('==', '').strip()
        else:
            return True  # No version constraint, update available
        
        try:
            return semver.compare(current_version, latest) < 0
        except ValueError:
            # Handle non-semver versions
            return current_version != latest
    
    def calculate_priority(self, package: str, needs_update: bool, has_vulnerability: bool) -> str:
        """Calculate update priority."""
        if has_vulnerability:
            return "CRITICAL"
        
        # Define critical packages
        critical_packages = ['torch', 'numpy', 'scipy']
        
        if package in critical_packages:
            return "HIGH" if needs_update else "LOW"
        
        # Development dependencies
        dev_packages = ['pytest', 'black', 'ruff', 'mypy', 'isort']
        if package in dev_packages:
            return "MEDIUM" if needs_update else "LOW"
        
        return "MEDIUM" if needs_update else "LOW"
    
    def create_update_pr_description(self, update_plan: Dict[str, dict]) -> str:
        """Create PR description for dependency updates."""
        critical_updates = [p for p, info in update_plan.items() 
                           if info['update_priority'] == 'CRITICAL']
        high_updates = [p for p, info in update_plan.items() 
                       if info['update_priority'] == 'HIGH']
        
        description = "## Automated Dependency Updates\n\n"
        
        if critical_updates:
            description += "### 🚨 Critical Security Updates\n"
            for package in critical_updates:
                info = update_plan[package]
                description += f"- **{package}**: {info['current_version']} → {info['latest_version']} (SECURITY)\n"
            description += "\n"
        
        if high_updates:
            description += "### ⬆️ High Priority Updates\n"
            for package in high_updates:
                info = update_plan[package]
                description += f"- **{package}**: {info['current_version']} → {info['latest_version']}\n"
            description += "\n"
        
        description += "### 🧪 Testing\n"
        description += "- [ ] All tests pass\n"
        description += "- [ ] Security scan passes\n"
        description += "- [ ] Compatibility verified\n"
        description += "- [ ] Performance benchmarks stable\n\n"
        
        description += "### 📋 Review Checklist\n"
        description += "- [ ] Breaking changes reviewed\n"
        description += "- [ ] Documentation updated if needed\n"
        description += "- [ ] Changelog updated\n"
        
        return description

def main():
    """Main dependency management workflow."""
    manager = DependencyManager()
    
    # Generate update plan
    update_plan = manager.generate_update_plan()
    
    # Filter for packages that need updates or have vulnerabilities
    updates_needed = {
        package: info for package, info in update_plan.items()
        if info['needs_update'] or info['has_vulnerability']
    }
    
    if not updates_needed:
        print("✅ All dependencies are up to date and secure!")
        return
    
    print("📦 Dependency Update Report")
    print("=" * 50)
    
    for package, info in updates_needed.items():
        status_icon = "🚨" if info['has_vulnerability'] else "⬆️"
        print(f"{status_icon} {package}")
        print(f"   Current: {info['current_version']}")
        print(f"   Latest:  {info['latest_version']}")
        print(f"   Priority: {info['update_priority']}")
        print()
    
    # Generate PR description
    pr_description = manager.create_update_pr_description(updates_needed)
    
    # Save PR description to file
    with open('dependency-update-pr.md', 'w') as f:
        f.write(pr_description)
    
    print("📝 PR description saved to: dependency-update-pr.md")
    
    # Save full report
    with open('dependency-report.json', 'w') as f:
        json.dump(update_plan, f, indent=2)
    
    print("📊 Full report saved to: dependency-report.json")

if __name__ == '__main__':
    main()
```

### 4. Enhanced Makefile Targets

Add to existing `Makefile`:

```makefile
# Dependency management targets
deps-check:		## Check for dependency updates and vulnerabilities
	@echo "Checking dependencies..."
	python scripts/dependency-manager.py
	pip-audit --desc

deps-update:		## Update all dependencies to latest versions
	@echo "Updating dependencies..."
	pip install --upgrade pip
	pip install --upgrade -e ".[dev,docs,test]"

deps-audit:		## Run comprehensive dependency security audit
	@echo "Running dependency security audit..."
	pip-audit --format json --output pip-audit-report.json
	safety check --json --output safety-report.json
	@echo "Audit complete. Check *-report.json files."

deps-install:		## Install dependencies from lock file
	pip install -e ".[dev]"

deps-lock:		## Generate dependency lock file
	pip freeze > requirements-lock.txt

deps-clean:		## Clean dependency cache
	pip cache purge
	rm -rf .pip_cache

deps-outdated:		## Show outdated dependencies
	pip list --outdated --format=json > outdated-deps.json
	@echo "Outdated dependencies saved to outdated-deps.json"

deps-tree:		## Show dependency tree
	pip install pipdeptree
	pipdeptree --json > dependency-tree.json
	pipdeptree

deps-conflicts:		## Check for dependency conflicts
	pip check

deps-security-baseline:	## Create security baseline for dependencies
	pip-audit --format json --output deps-security-baseline.json
	@echo "Security baseline created: deps-security-baseline.json"
```

### 5. Dependency Testing Strategy

Create `scripts/test-dependencies.py`:

```python
#!/usr/bin/env python3
"""
Test dependency updates for compatibility and performance.
"""

import subprocess
import sys
import time
import json
from pathlib import Path

class DependencyTester:
    def __init__(self):
        self.test_results = {}
    
    def run_tests(self) -> bool:
        """Run test suite and return success status."""
        try:
            result = subprocess.run(['pytest', '-v', '--tb=short'], 
                                  capture_output=True, text=True)
            
            self.test_results['unit_tests'] = {
                'passed': result.returncode == 0,
                'output': result.stdout,
                'errors': result.stderr
            }
            
            return result.returncode == 0
            
        except subprocess.CalledProcessError:
            return False
    
    def run_performance_benchmarks(self) -> bool:
        """Run performance benchmarks to detect regressions."""
        try:
            # Run pytest-benchmark if available
            result = subprocess.run([
                'pytest', 'tests/benchmarks/', '--benchmark-json=benchmark-results.json'
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and Path('benchmark-results.json').exists():
                with open('benchmark-results.json') as f:
                    benchmark_data = json.load(f)
                
                self.test_results['benchmarks'] = benchmark_data
                return True
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Benchmarks not available or failed
            pass
        
        return False
    
    def run_import_tests(self) -> bool:
        """Test that all imports work correctly."""
        imports_to_test = [
            'probneural_operator',
            'probneural_operator.models',
            'probneural_operator.posteriors',
            'probneural_operator.active',
            'probneural_operator.calibration'
        ]
        
        failed_imports = []
        
        for module in imports_to_test:
            try:
                subprocess.run([
                    sys.executable, '-c', f'import {module}'
                ], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                failed_imports.append(module)
        
        self.test_results['imports'] = {
            'failed': failed_imports,
            'passed': len(failed_imports) == 0
        }
        
        return len(failed_imports) == 0
    
    def run_security_tests(self) -> bool:
        """Run security tests after dependency updates."""
        try:
            # Run bandit security linter
            result = subprocess.run([
                'bandit', '-r', 'probneural_operator', '-f', 'json'
            ], capture_output=True, text=True)
            
            # Bandit returns non-zero for security issues, but we want to capture results
            if result.stdout:
                security_data = json.loads(result.stdout)
                high_severity_issues = [
                    issue for issue in security_data.get('results', [])
                    if issue.get('issue_severity') == 'HIGH'
                ]
                
                self.test_results['security'] = {
                    'high_severity_issues': len(high_severity_issues),
                    'passed': len(high_severity_issues) == 0
                }
                
                return len(high_severity_issues) == 0
            
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            pass
        
        return False
    
    def run_comprehensive_test_suite(self) -> dict:
        """Run all tests and return comprehensive results."""
        print("🧪 Running comprehensive dependency tests...")
        
        start_time = time.time()
        
        # Run all test categories
        tests = [
            ('Import Tests', self.run_import_tests),
            ('Unit Tests', self.run_tests),
            ('Security Tests', self.run_security_tests),
            ('Performance Benchmarks', self.run_performance_benchmarks)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            print(f"  Running {test_name}...")
            try:
                results[test_name] = test_func()
            except Exception as e:
                print(f"    ❌ {test_name} failed with exception: {e}")
                results[test_name] = False
            
            status = "✅ PASSED" if results[test_name] else "❌ FAILED"
            print(f"    {status}")
        
        duration = time.time() - start_time
        
        # Overall success
        overall_success = all(results.values())
        
        final_results = {
            'overall_success': overall_success,
            'duration_seconds': duration,
            'individual_results': results,
            'detailed_results': self.test_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        }
        
        # Save results
        with open('dependency-test-results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n📊 Test Results Summary:")
        print(f"   Overall: {'✅ PASSED' if overall_success else '❌ FAILED'}")
        print(f"   Duration: {duration:.2f} seconds")
        print(f"   Results saved to: dependency-test-results.json")
        
        return final_results

if __name__ == '__main__':
    tester = DependencyTester()
    results = tester.run_comprehensive_test_suite()
    
    # Exit with appropriate code
    sys.exit(0 if results['overall_success'] else 1)
```

## Integration Checklist

- [ ] Choose between Dependabot and Renovate (or use both)
- [ ] Configure dependency update automation
- [ ] Set up vulnerability scanning
- [ ] Create dependency management scripts
- [ ] Add Makefile targets for dependency operations
- [ ] Implement dependency testing framework
- [ ] Configure security policies for dependencies
- [ ] Set up automated PR creation for updates
- [ ] Configure review requirements for critical dependencies
- [ ] Test entire dependency management workflow
- [ ] Train team on dependency management processes
- [ ] Create documentation for dependency policies
- [ ] Set up monitoring for security vulnerabilities
- [ ] Create rollback procedures for failed updates

## Monitoring and Maintenance

### Daily
- Review security vulnerability alerts
- Check automated PR status

### Weekly  
- Review and merge low-risk dependency updates
- Analyze dependency test results

### Monthly
- Review dependency update patterns
- Update dependency management configurations
- Audit security scanning results
- Review and update critical dependency policies

This comprehensive dependency management system ensures secure, up-to-date dependencies while maintaining stability and compatibility.