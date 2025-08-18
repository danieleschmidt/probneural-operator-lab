# Manual Setup Required

This document outlines the manual setup steps required by repository administrators due to GitHub App permission limitations. The Terragon SDLC implementation has created comprehensive templates and documentation, but some steps require manual action.

## Overview

The Terragon SDLC implementation has been completed with checkpointed commits, providing:
- ✅ Project foundation and documentation
- ✅ Development environment and tooling  
- ✅ Comprehensive testing infrastructure
- ✅ Build and containerization setup
- ✅ Monitoring and observability configuration
- ✅ CI/CD workflow templates and documentation
- ✅ Metrics tracking and automation scripts
- ✅ Integration configuration templates

## Required Manual Actions

### 1. GitHub Workflows Setup (HIGH PRIORITY)

**Templates Location**: `templates/github/workflows/`

**Action Required**: Copy workflow files to `.github/workflows/`

```bash
# Create workflows directory
mkdir -p .github/workflows

# Copy all workflow templates
cp templates/github/workflows/*.yml .github/workflows/

# Copy other GitHub configurations
cp templates/github/dependabot.yml .github/
cp templates/github/codeql.yml .github/
```

**Workflows to Enable**:
- `ci.yml` - Continuous Integration
- `release.yml` - Automated Releases  
- `security-scan.yml` - Security Scanning
- `dependency-update.yml` - Dependency Updates
- `deploy.yml` - Production Deployment

### 2. Repository Secrets Configuration

**Required Secrets** (Settings > Secrets and variables > Actions):

```
# Essential Secrets
PYPI_API_TOKEN          # For package publishing
GITHUB_TOKEN           # Usually provided automatically
DOCKER_USERNAME        # For container registry
DOCKER_PASSWORD        # For container registry

# Optional but Recommended
CODECOV_TOKEN          # Code coverage reporting
SLACK_WEBHOOK_URL      # Team notifications  
SECURITY_EMAIL         # Security notifications
```

### 3. Branch Protection Rules

**Action Required**: Enable branch protection for `main` branch

**Settings** (Settings > Branches > Add rule):
- Branch name pattern: `main`
- ✅ Require a pull request before merging
- ✅ Require approvals: 1
- ✅ Dismiss stale reviews when new commits are pushed
- ✅ Require review from code owners
- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ Restrict pushes that create files
- ✅ Include administrators

**Required Status Checks**:
- `Code Quality`
- `Test Suite (ubuntu-latest, 3.11)`
- `Security Scan`
- `Build and Test Installation`

### 4. Issue and PR Templates

**Templates Location**: `templates/github/ISSUE_TEMPLATE/`

**Action Required**: Copy templates to `.github/`

```bash
# Copy issue templates
mkdir -p .github/ISSUE_TEMPLATE
cp templates/github/ISSUE_TEMPLATE/* .github/ISSUE_TEMPLATE/

# Copy PR template
cp templates/github/pull_request_template.md .github/
```

### 5. Repository Settings Configuration

**General Settings**:
- ✅ Enable Issues
- ✅ Enable Projects
- ✅ Enable Wiki (if desired)
- ✅ Enable Discussions (recommended for community)

**Code Security and Analysis**:
- ✅ Enable Dependabot alerts
- ✅ Enable Dependabot security updates
- ✅ Enable Dependabot version updates
- ✅ Enable Secret scanning
- ✅ Enable Code scanning (CodeQL)

### 6. Monitoring Setup

**Prerequisites**: Ensure Docker is available

**Action Required**: Initialize monitoring stack

```bash
# Setup monitoring services
cd monitoring
chmod +x setup-monitoring.sh
./setup-monitoring.sh

# Start monitoring stack
./start-monitoring.sh
```

**Access URLs** (after setup):
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin123)
- Alertmanager: http://localhost:9093

### 7. Development Environment

**Action Required**: Setup local development environment

```bash
# Run development setup
chmod +x scripts/setup-dev.sh
./scripts/setup-dev.sh

# Verify setup
python -c "import probneural_operator; print('✅ Setup successful')"
```

## Integration Verification

### Step 1: Verify Repository Structure

Run the structure verification script:

```bash
python scripts/collect-metrics.py --verbose
```

Expected output should show no critical issues.

### Step 2: Test Build Pipeline

```bash
# Test local build
./scripts/build.sh --type development --no-push

# Test release process (dry run)
./scripts/release.sh --dry-run --type patch
```

### Step 3: Verify Security Configuration

```bash
# Run security scans
safety check
bandit -r probneural_operator/

# Generate SBOM
./scripts/generate-sbom.sh
```

### Step 4: Test Monitoring

```bash
# Health check monitoring stack
cd monitoring
./health-check.sh
```

### Step 5: Generate Dashboard

```bash
# Create project dashboard
python scripts/generate-dashboard.py --output dashboard.html
```

## Automated Maintenance

Once setup is complete, these processes will run automatically:

### Daily
- Security vulnerability scanning
- Dependency update checks
- Metrics collection

### Weekly  
- Dependency updates (via Dependabot)
- Performance benchmarks
- Code quality reports

### Monthly
- Comprehensive security audit
- Documentation updates
- Metrics dashboard refresh

## Troubleshooting

### Common Issues

**1. Workflow Permissions**
```
Error: Resource not accessible by integration
```
**Solution**: Check repository permissions and ensure secrets are properly configured.

**2. Docker Build Failures**
```
Error: Cannot connect to Docker daemon
```
**Solution**: Ensure Docker is running and user has appropriate permissions.

**3. Security Scan Failures**
```
Error: Safety check failed
```
**Solution**: Review and address security vulnerabilities, or add exceptions to `.security.yml`.

### Getting Help

1. **Documentation**: Check `docs/` directory for comprehensive guides
2. **Issues**: Create GitHub issue with `question` label
3. **Discussions**: Use GitHub Discussions for community help

## Success Criteria

The SDLC implementation is successfully configured when:

- ✅ All workflows are enabled and passing
- ✅ Branch protection rules are active
- ✅ Security scanning is operational
- ✅ Monitoring stack is running
- ✅ Automated dependency updates are working
- ✅ Documentation is accessible and current
- ✅ Development environment setup works smoothly

## Next Steps

After completing manual setup:

1. **Test the full pipeline** with a small change
2. **Configure team notifications** (Slack/email)
3. **Setup production deployment** environments
4. **Train team members** on new processes
5. **Schedule regular reviews** of metrics and health scores

## Support

For implementation questions or issues:
- **Repository**: Create issue with `setup` label
- **Urgent**: Contact repository administrators directly
- **Documentation**: Submit PR to improve this guide

---

**Note**: This setup document was generated by the Terragon SDLC automation. All templates and configurations have been created and are ready for activation.