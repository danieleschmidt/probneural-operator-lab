# GitHub Workflows Setup Guide

This guide provides step-by-step instructions for setting up GitHub Actions workflows for the ProbNeural Operator Lab project.

## Overview

The project includes comprehensive CI/CD workflows that handle:

- **Continuous Integration**: Code quality, testing, security scanning
- **Dependency Management**: Automated dependency updates with Dependabot
- **Release Automation**: Semantic versioning and automated releases
- **Security Scanning**: Vulnerability assessment and SBOM generation
- **Documentation**: Automated docs building and deployment

## Required Repository Setup

### 1. Repository Secrets

Configure these secrets in your GitHub repository settings (`Settings > Secrets and variables > Actions`):

#### Required Secrets
```
PYPI_API_TOKEN          # PyPI token for package publishing
CODECOV_TOKEN          # Codecov integration token (optional)
DOCKER_USERNAME        # Docker Hub username
DOCKER_PASSWORD        # Docker Hub password/token
```

#### Optional Secrets (for enhanced functionality)
```
SLACK_WEBHOOK_URL      # Slack notifications
GPG_PRIVATE_KEY        # For signed commits
GPG_PASSPHRASE         # GPG key passphrase
SECURITY_EMAIL         # Security notifications email
```

### 2. Branch Protection Rules

Set up branch protection for `main` branch:

1. Go to `Settings > Branches`
2. Add rule for `main` branch
3. Configure these settings:
   - ✅ Require a pull request before merging
   - ✅ Require approvals: 1
   - ✅ Dismiss stale PR approvals when new commits are pushed
   - ✅ Require review from code owners
   - ✅ Require status checks to pass before merging
   - ✅ Require branches to be up to date before merging
   - ✅ Require conversation resolution before merging
   - ✅ Include administrators

#### Required Status Checks
Enable these status checks:
- `Code Quality`
- `Test Suite (ubuntu-latest, 3.11)`
- `Security Scan`
- `Build and Test Installation`
- `Documentation`

### 3. Dependabot Configuration

The repository includes Dependabot configuration in `.github/dependabot.yml`. Ensure Dependabot is enabled:

1. Go to `Settings > Security & analysis`
2. Enable "Dependabot alerts"
3. Enable "Dependabot security updates"
4. Enable "Dependabot version updates"

## Workflow Files Setup

### Step 1: Create GitHub Workflows Directory

```bash
mkdir -p .github/workflows
```

### Step 2: Copy Workflow Templates

Copy the workflow templates from `templates/github/workflows/` to `.github/workflows/`:

```bash
# Copy main CI workflow
cp templates/github/workflows/ci.yml .github/workflows/

# Copy dependency update workflow
cp templates/github/workflows/dependency-update.yml .github/workflows/

# Copy release workflow
cp templates/github/workflows/release.yml .github/workflows/

# Copy security scanning workflow
cp templates/github/workflows/security-scan.yml .github/workflows/
```

### Step 3: Copy Issue and PR Templates

```bash
# Create GitHub templates directory
mkdir -p .github/ISSUE_TEMPLATE

# Copy templates
cp templates/github/ISSUE_TEMPLATE/* .github/ISSUE_TEMPLATE/
cp templates/github/pull_request_template.md .github/
```

### Step 4: Copy Dependabot Configuration

```bash
cp templates/github/dependabot.yml .github/
```

## Workflow Configuration

### CI Workflow (`ci.yml`)

The main CI workflow runs on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches
- Manual workflow dispatch

#### Jobs Overview:
1. **Code Quality**: Formatting, linting, type checking
2. **Test Suite**: Cross-platform testing (Ubuntu, Windows, macOS)
3. **GPU Tests**: GPU-specific tests (requires self-hosted runner)
4. **Security Scan**: Vulnerability scanning
5. **Documentation**: Docs building and link checking
6. **Build**: Package building and installation testing
7. **Performance**: Benchmark testing

#### Customization Options:

**Matrix Testing**:
```yaml
strategy:
  matrix:
    os: [ubuntu-latest, windows-latest, macos-latest]
    python-version: ['3.9', '3.10', '3.11', '3.12']
```

**GPU Testing** (requires self-hosted runner):
```yaml
runs-on: [self-hosted, gpu]
```

### Release Workflow (`release.yml`)

Automated releases trigger when:
- Tags matching `v*` pattern are pushed
- Manual workflow dispatch

#### Features:
- Semantic versioning validation
- Automated changelog generation
- PyPI package publishing
- Docker image building and pushing
- GitHub release creation with artifacts
- SBOM generation

### Security Scanning (`security-scan.yml`)

Comprehensive security scanning including:
- Dependency vulnerability scanning (Safety, pip-audit)
- Static code analysis (Bandit, Semgrep)
- Container image scanning (Trivy)
- SBOM generation
- CodeQL analysis

### Dependency Updates (`dependency-update.yml`)

Automated dependency management:
- Dependabot integration
- Automated testing of dependency updates
- Security-focused updates
- Configurable update schedules

## Advanced Configuration

### Self-Hosted Runners

For GPU testing and performance benchmarks, set up self-hosted runners:

1. **Create Self-Hosted Runner**:
   - Go to `Settings > Actions > Runners`
   - Click "New self-hosted runner"
   - Follow setup instructions

2. **GPU Runner Requirements**:
   ```bash
   # Install NVIDIA drivers
   sudo apt update
   sudo apt install nvidia-driver-470
   
   # Install CUDA toolkit
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
   sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
   sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
   sudo apt update
   sudo apt install cuda
   
   # Install Docker with NVIDIA support
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
   sudo apt update
   sudo apt install nvidia-docker2
   sudo systemctl restart docker
   ```

3. **Runner Labels**:
   - Add label `gpu` to GPU-enabled runners
   - Add label `performance` to high-performance runners

### Notification Setup

#### Slack Integration

1. Create Slack webhook URL
2. Add `SLACK_WEBHOOK_URL` secret
3. Workflows will send notifications for:
   - Failed CI runs
   - Security vulnerabilities
   - Successful releases

#### Email Notifications

Configure email notifications in workflow files:
```yaml
- name: Send email notification
  if: failure()
  uses: dawidd6/action-send-mail@v3
  with:
    server_address: smtp.gmail.com
    server_port: 587
    username: ${{ secrets.EMAIL_USERNAME }}
    password: ${{ secrets.EMAIL_PASSWORD }}
    subject: CI Failed - ${{ github.repository }}
    body: |
      CI failed for commit ${{ github.sha }}
      
      View details: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}
    to: ${{ secrets.NOTIFICATION_EMAIL }}
```

## Environment-Specific Configurations

### Development Environment

For development branches:
- Run full test suite
- Skip performance benchmarks
- Enable verbose logging

### Staging Environment

For staging deployments:
- Run integration tests
- Performance regression testing
- Security scanning

### Production Environment

For production releases:
- Comprehensive testing
- Security validation
- SBOM generation
- Automated deployment

## Monitoring and Observability

### Workflow Metrics

Track these metrics:
- Build success rate
- Test execution time
- Coverage trends
- Security scan results
- Dependency update frequency

### Integration with External Tools

#### Codecov
```yaml
- name: Upload coverage to Codecov
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
    flags: unittests
    name: codecov-umbrella
```

#### SonarCloud
```yaml
- name: SonarCloud Scan
  uses: SonarSource/sonarcloud-github-action@master
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
```

## Troubleshooting

### Common Issues

#### 1. Permission Errors
**Error**: "Resource not accessible by integration"
**Solution**: Check repository permissions and token scopes

#### 2. Test Timeouts
**Error**: Tests timing out
**Solution**: Increase timeout or optimize tests
```yaml
timeout-minutes: 30
```

#### 3. Cache Issues
**Error**: Cache restore failures
**Solution**: Clear GitHub Actions cache or update cache keys

#### 4. Secret Access
**Error**: Secret not found
**Solution**: Verify secret name and access permissions

### Debugging Workflows

Enable debug logging:
```yaml
env:
  ACTIONS_STEP_DEBUG: true
  ACTIONS_RUNNER_DEBUG: true
```

## Maintenance

### Regular Tasks

1. **Monthly**:
   - Review workflow performance
   - Update action versions
   - Check security scan results

2. **Quarterly**:
   - Update Python versions in matrix
   - Review and update dependencies
   - Audit security configurations

3. **Annually**:
   - Review overall workflow strategy
   - Update CI/CD best practices
   - Security audit of workflows

### Workflow Updates

When updating workflows:
1. Test changes in a feature branch
2. Validate with limited matrix first
3. Monitor first runs carefully
4. Document changes in changelog

## Security Best Practices

### Secrets Management
- Use repository secrets for sensitive data
- Rotate secrets regularly
- Limit secret access scope
- Use environment-specific secrets

### Action Security
- Pin action versions to specific commits
- Review action permissions
- Use trusted action publishers
- Regular security audits

### Code Scanning
- Enable CodeQL analysis
- Configure security scanning
- Monitor vulnerability alerts
- Automated security updates

## Support

For workflow issues:
1. Check GitHub Actions documentation
2. Review workflow run logs
3. Consult community forums
4. Open issue in repository

---

**Note**: This setup guide assumes GitHub App permissions allow workflow creation. If permissions are restricted, provide this guide to repository administrators for manual setup.