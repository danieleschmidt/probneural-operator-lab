# Security Scanning Automation Setup

Comprehensive security scanning automation for ProbNeural-Operator-Lab repository.

## Overview

This document outlines the security scanning tools and configurations needed to maintain high security posture for the Python ML research repository.

## Security Scanning Tools

### 1. Dependency Vulnerability Scanning

#### pip-audit
Primary tool for Python dependency vulnerability scanning.

**Installation:**
```bash
pip install pip-audit
```

**Usage:**
```bash
# Basic scan
pip-audit

# Detailed output with descriptions
pip-audit --desc

# JSON output for automation
pip-audit --format json --output audit-results.json

# Scan specific requirements file
pip-audit -r requirements.txt
```

**Configuration file:** `.pip-audit.toml`
```toml
[tool.pip-audit]
# Ignore specific vulnerabilities (use with caution)
ignore-vuln = []

# Skip specific packages
skip-vuln = []

# Output format
format = "json"

# Index URL for private repositories
index-url = "https://pypi.org/simple"
```

#### Safety
Alternative dependency scanner with commercial database.

**Installation:**
```bash
pip install safety
```

**Usage:**
```bash
# Scan installed packages
safety check

# Scan requirements file
safety check -r requirements.txt

# JSON output
safety check --json --output safety-report.json

# Full report with remediation advice
safety check --full-report
```

### 2. Static Application Security Testing (SAST)

#### Bandit
Python-specific security linter for common security issues.

**Installation:**
```bash
pip install bandit
```

**Configuration file:** `.bandit`
```ini
[bandit]
# Test IDs to skip (comma-separated)
skips = B101,B601

# Test IDs to run
tests = B201,B301,B302,B303,B304,B305,B306,B307,B308,B309,B310,B311,B312,B313,B314,B315,B316,B317,B318,B319,B320,B321,B322,B323,B324,B325,B501,B502,B503,B504,B505,B506,B507,B601,B602,B603,B604,B605,B606,B607,B608,B609,B610,B611,B701,B702,B703

# Paths to exclude
exclude_dirs = /tests,/docs

# Report format
format = json
```

**Usage:**
```bash
# Scan entire codebase
bandit -r probneural_operator

# Generate JSON report
bandit -r probneural_operator -f json -o bandit-report.json

# Scan with confidence levels
bandit -r probneural_operator -i  # Skip low severity issues

# Baseline scan (for comparing future scans)
bandit -r probneural_operator -f json -o baseline.json
bandit -r probneural_operator -f json -b baseline.json
```

#### Semgrep
Advanced static analysis with community and pro rules.

**Installation:**
```bash
pip install semgrep
```

**Configuration file:** `.semgrep.yml`
```yaml
rules:
  - id: python-security-audit
    patterns:
      - pattern-either:
          - pattern: eval(...)
          - pattern: exec(...)
          - pattern: os.system(...)
          - pattern: subprocess.call($SHELL, shell=True, ...)
    message: "Potentially dangerous function call detected"
    languages: [python]
    severity: WARNING
    
  - id: hardcoded-secrets
    patterns:
      - pattern-regex: '(password|secret|key|token)\s*=\s*["\'][^"\']{8,}["\']'
    message: "Possible hardcoded secret detected"
    languages: [python]
    severity: ERROR
```

**Usage:**
```bash
# Run with security rules
semgrep --config=p/security-audit probneural_operator/

# Run with Python-specific rules
semgrep --config=p/python probneural_operator/

# Run with secrets detection
semgrep --config=p/secrets .

# Custom rules
semgrep --config=.semgrep.yml probneural_operator/

# JSON output
semgrep --config=p/security-audit --json --output=semgrep-results.json probneural_operator/
```

### 3. Secrets Detection

#### Gitleaks
Detect secrets and credentials in Git repositories.

**Installation:**
```bash
# Using binary release
wget https://github.com/gitleaks/gitleaks/releases/latest/download/gitleaks_linux_x64.tar.gz
tar -xzf gitleaks_linux_x64.tar.gz
sudo mv gitleaks /usr/local/bin/
```

**Configuration file:** `.gitleaks.toml`
```toml
title = "ProbNeural Operator Lab Gitleaks Config"

[[rules]]
id = "python-env-files"
description = "Python environment files"
regex = '''(?i)(api_key|secret_key|private_key|token)\s*=\s*['"']?[a-zA-Z0-9_\-]{10,}['"']?'''
path = '''\.env$|\.env\..*$|config\.py$'''

[[rules]]
id = "generic-api-key"
description = "Generic API Key"
regex = '''(?i)api[_\-]?key\s*[:=]\s*['"']?[a-zA-Z0-9_\-]{10,}['"']?'''

[[rules]]
id = "jwt-token"
description = "JWT Token"
regex = '''eyJ[A-Za-z0-9_/+-]*\.eyJ[A-Za-z0-9_/+-]*\.[A-Za-z0-9_/+-]*'''

[allowlist]
paths = [
  "docs/",
  "tests/fixtures/",
  ".gitleaks.toml"
]

commits = []
```

**Usage:**
```bash
# Scan entire repository
gitleaks detect --source . --verbose

# Scan with custom config
gitleaks detect --config .gitleaks.toml --source .

# Scan specific branch
gitleaks detect --source . --log-opts="--branch=main"

# Generate report
gitleaks detect --source . --report-path gitleaks-report.json --report-format json
```

### 4. Container Security (Future Enhancement)

#### Trivy
When Docker is added, use Trivy for container scanning.

```bash
# Install trivy
curl -sfL https://raw.githubusercontent.com/aquasecurity/trivy/main/contrib/install.sh | sh -s -- -b /usr/local/bin

# Scan filesystem
trivy fs .

# Scan container image (when available)
trivy image probneural-operator:latest
```

## Automated Security Scanning Pipeline

### Pre-commit Hooks Integration

Add to `.pre-commit-config.yaml`:

```yaml
repos:
  # ... existing hooks ...
  
  - repo: https://github.com/PyCQA/bandit
    rev: '1.7.5'
    hooks:
      - id: bandit
        args: ['-c', '.bandit']
  
  - repo: https://github.com/gitguardian/ggshield
    rev: v1.25.0
    hooks:
      - id: ggshield
        language: python
        stages: [commit]
```

### Makefile Integration

Add to existing `Makefile`:

```makefile
security:		## Run all security scans
	@echo "Running security scans..."
	pip-audit --desc
	bandit -r probneural_operator -f json -o bandit-report.json
	safety check --json --output safety-report.json
	semgrep --config=p/security-audit probneural_operator/ --json --output semgrep-report.json
	gitleaks detect --source . --report-path gitleaks-report.json --report-format json
	@echo "Security scan complete. Check *-report.json files for details."

security-baseline:	## Create security baseline
	@echo "Creating security baseline..."
	bandit -r probneural_operator -f json -o security-baseline.json
	pip-audit --format json --output dependency-baseline.json

security-diff:		## Compare against security baseline
	@echo "Comparing against baseline..."
	bandit -r probneural_operator -f json -b security-baseline.json
	# Custom script to compare dependency changes would go here

fix-security:		## Auto-fix security issues where possible
	@echo "Auto-fixing security issues..."
	semgrep --config=p/security-audit probneural_operator/ --autofix
	# Additional auto-fix commands
```

## Security Configuration Files

### 1. Security Policy (`.github/SECURITY.md`)

```markdown
# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

Please report security vulnerabilities to: security@example.com

We will respond within 48 hours and provide updates every 5 business days.

## Security Scanning

This project uses automated security scanning:
- Dependency vulnerability scanning (pip-audit, Safety)
- Static analysis security testing (Bandit, Semgrep)
- Secrets detection (Gitleaks)
- Automated security updates (Dependabot)

## Security Best Practices

- All dependencies are scanned for known vulnerabilities
- Code is analyzed for security anti-patterns
- Secrets are never committed to the repository
- Security patches are applied automatically when possible
```

### 2. Dependabot Configuration (`.github/dependabot.yml`)

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
      time: "08:00"
    open-pull-requests-limit: 5
    reviewers:
      - "security-team"
    labels:
      - "dependencies"
      - "security"
    commit-message:
      prefix: "security"
      include: "scope"
  
  - package-ecosystem: "github-actions"
    directory: "/.github/workflows"
    schedule:
      interval: "monthly"
    commit-message:
      prefix: "ci"
      include: "scope"
```

## Security Monitoring and Alerting

### 1. Security Metrics Tracking

Create `scripts/security-metrics.py`:

```python
#!/usr/bin/env python3
"""
Security metrics collection script for continuous monitoring.
"""

import json
import subprocess
import datetime
from pathlib import Path

def run_security_scan():
    """Run all security tools and collect metrics."""
    results = {
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'scans': {}
    }
    
    # pip-audit scan
    try:
        result = subprocess.run(['pip-audit', '--format', 'json'], 
                              capture_output=True, text=True)
        results['scans']['pip_audit'] = json.loads(result.stdout)
    except Exception as e:
        results['scans']['pip_audit'] = {'error': str(e)}
    
    # Bandit scan
    try:
        result = subprocess.run(['bandit', '-r', 'probneural_operator', '-f', 'json'], 
                              capture_output=True, text=True)
        results['scans']['bandit'] = json.loads(result.stdout)
    except Exception as e:
        results['scans']['bandit'] = {'error': str(e)}
    
    return results

def calculate_security_score(scan_results):
    """Calculate overall security score."""
    score = 100
    
    # Deduct points for vulnerabilities
    if 'pip_audit' in scan_results['scans']:
        vulns = scan_results['scans']['pip_audit'].get('vulnerabilities', [])
        score -= len(vulns) * 10
    
    if 'bandit' in scan_results['scans']:
        issues = scan_results['scans']['bandit'].get('results', [])
        high_severity = sum(1 for issue in issues if issue.get('issue_severity') == 'HIGH')
        medium_severity = sum(1 for issue in issues if issue.get('issue_severity') == 'MEDIUM')
        score -= high_severity * 15 + medium_severity * 5
    
    return max(0, score)

if __name__ == '__main__':
    results = run_security_scan()
    score = calculate_security_score(results)
    
    # Save results
    output_file = Path('security-metrics.json')
    with open(output_file, 'w') as f:
        json.dump({**results, 'security_score': score}, f, indent=2)
    
    print(f"Security score: {score}/100")
    if score < 80:
        print("WARNING: Security score below threshold!")
        exit(1)
```

### 2. Security Dashboard

Create `scripts/security-dashboard.py`:

```python
#!/usr/bin/env python3
"""
Generate security dashboard from scan results.
"""

import json
from pathlib import Path
import datetime

def generate_security_report():
    """Generate HTML security report."""
    
    # Load security metrics
    metrics_file = Path('security-metrics.json')
    if not metrics_file.exists():
        return "No security metrics found. Run security scan first."
    
    with open(metrics_file) as f:
        data = json.load(f)
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Security Dashboard - ProbNeural Operator Lab</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .score { font-size: 2em; color: {score_color}; }
            .section { margin: 20px 0; }
            .vulnerability { background: #ffebee; padding: 10px; margin: 5px 0; }
            .clean { background: #e8f5e8; padding: 10px; margin: 5px 0; }
        </style>
    </head>
    <body>
        <h1>Security Dashboard</h1>
        <p>Generated: {timestamp}</p>
        
        <div class="section">
            <h2>Overall Security Score</h2>
            <div class="score">{score}/100</div>
        </div>
        
        <div class="section">
            <h2>Vulnerability Summary</h2>
            {vulnerability_summary}
        </div>
        
        <div class="section">
            <h2>Recommendations</h2>
            {recommendations}
        </div>
    </body>
    </html>
    """
    
    score = data.get('security_score', 0)
    score_color = 'green' if score >= 80 else 'orange' if score >= 60 else 'red'
    
    # Generate sections
    vuln_summary = generate_vulnerability_summary(data['scans'])
    recommendations = generate_recommendations(data['scans'])
    
    return html_template.format(
        timestamp=data['timestamp'],
        score=score,
        score_color=score_color,
        vulnerability_summary=vuln_summary,
        recommendations=recommendations
    )

def generate_vulnerability_summary(scans):
    """Generate vulnerability summary HTML."""
    html = ""
    
    # pip-audit vulnerabilities
    if 'pip_audit' in scans and 'vulnerabilities' in scans['pip_audit']:
        vulns = scans['pip_audit']['vulnerabilities']
        if vulns:
            html += f"<div class='vulnerability'>Found {len(vulns)} dependency vulnerabilities</div>"
        else:
            html += "<div class='clean'>No dependency vulnerabilities found</div>"
    
    # Bandit issues
    if 'bandit' in scans and 'results' in scans['bandit']:
        issues = scans['bandit']['results']
        if issues:
            html += f"<div class='vulnerability'>Found {len(issues)} code security issues</div>"
        else:
            html += "<div class='clean'>No code security issues found</div>"
    
    return html

def generate_recommendations(scans):
    """Generate recommendations HTML."""
    recommendations = []
    
    # Check for high-priority items
    if 'pip_audit' in scans and scans['pip_audit'].get('vulnerabilities'):
        recommendations.append("Update vulnerable dependencies immediately")
    
    if 'bandit' in scans and scans['bandit'].get('results'):
        high_severity = [r for r in scans['bandit']['results'] 
                        if r.get('issue_severity') == 'HIGH']
        if high_severity:
            recommendations.append("Fix high-severity code security issues")
    
    if not recommendations:
        recommendations.append("Security posture is good. Continue regular scanning.")
    
    return "<ul>" + "".join(f"<li>{rec}</li>" for rec in recommendations) + "</ul>"

if __name__ == '__main__':
    report = generate_security_report()
    with open('security-dashboard.html', 'w') as f:
        f.write(report)
    print("Security dashboard generated: security-dashboard.html")
```

## Integration Checklist

- [ ] Install all security scanning tools
- [ ] Configure tool-specific configuration files
- [ ] Add security scanning to pre-commit hooks
- [ ] Update Makefile with security targets
- [ ] Set up automated GitHub Actions workflows
- [ ] Configure Dependabot for automated updates
- [ ] Create security policy documentation
- [ ] Implement security metrics collection
- [ ] Set up security dashboard generation
- [ ] Test all security scanning tools
- [ ] Configure security alerting (email/Slack)
- [ ] Train team on security tool usage
- [ ] Establish security review process
- [ ] Create security incident response plan

## Maintenance

- Review security scan results daily
- Update security tools monthly
- Audit security configuration quarterly
- Conduct security training annually
- Review and update security policies as needed

This comprehensive security scanning setup provides automated detection of vulnerabilities, secrets, and security anti-patterns throughout the development lifecycle.