# GitHub Actions CI/CD Setup Guide

This document provides the complete GitHub Actions workflow configurations needed for the ProbNeural-Operator-Lab repository.

## Workflow Overview

The repository requires the following automated workflows:

1. **CI Pipeline** - Code quality, testing, and validation
2. **Security Scanning** - Vulnerability detection and dependency auditing  
3. **Dependency Management** - Automated dependency updates
4. **Release Automation** - Automated releases and changelog generation
5. **Documentation** - Auto-generated API documentation

## Required Workflow Files

Create these files in `.github/workflows/` directory:

### 1. CI Pipeline (`ci.yml`)

```yaml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements*.txt', 'pyproject.toml') }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev,test]"
    
    - name: Lint with ruff
      run: ruff check probneural_operator tests
    
    - name: Format check with black
      run: black --check probneural_operator tests
    
    - name: Import sorting check with isort
      run: isort --check probneural_operator tests
    
    - name: Type check with mypy
      run: mypy probneural_operator
    
    - name: Test with pytest
      run: |
        pytest --cov=probneural_operator --cov-report=xml --cov-report=term-missing
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  build:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Build package
      run: |
        python -m pip install --upgrade pip build
        python -m build
    
    - name: Upload build artifacts
      uses: actions/upload-artifact@v3
      with:
        name: dist
        path: dist/
```

### 2. Security Scanning (`security.yml`)

```yaml
name: Security Scanning

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM UTC

jobs:
  security-scan:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pip-audit bandit safety
        pip install -e .
    
    - name: Run pip-audit for dependency vulnerabilities
      run: pip-audit --desc --format json --output pip-audit-report.json
      continue-on-error: true
    
    - name: Run Bandit security linter
      run: |
        bandit -r probneural_operator -f json -o bandit-report.json
      continue-on-error: true
    
    - name: Run Safety for known security vulnerabilities
      run: |
        safety check --json --output safety-report.json
      continue-on-error: true
    
    - name: Upload security reports
      uses: actions/upload-artifact@v3
      with:
        name: security-reports
        path: |
          pip-audit-report.json
          bandit-report.json
          safety-report.json

  semgrep:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Run Semgrep
      uses: returntocorp/semgrep-action@v1
      with:
        config: >-
          p/security-audit
          p/secrets
          p/python
      env:
        SEMGREP_APP_TOKEN: ${{ secrets.SEMGREP_APP_TOKEN }}

  gitleaks:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Run Gitleaks
      uses: gitleaks/gitleaks-action@v2
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

### 3. Dependency Management (`dependencies.yml`)

```yaml
name: Dependency Management

on:
  schedule:
    - cron: '0 8 * * MON'  # Weekly on Monday at 8 AM UTC
  workflow_dispatch:  # Manual trigger

jobs:
  update-dependencies:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install pip-tools
      run: python -m pip install pip-tools
    
    - name: Update requirements
      run: |
        # This would update pinned versions if using requirements.txt
        # Since we use pyproject.toml, this is for reference
        echo "Dependencies managed via pyproject.toml"
    
    - name: Run security audit
      run: |
        python -m pip install pip-audit
        pip-audit --desc
    
    - name: Create Pull Request
      uses: peter-evans/create-pull-request@v5
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
        commit-message: 'chore: update dependencies'
        title: 'chore: automated dependency updates'
        body: |
          ## Automated Dependency Updates
          
          This PR contains automated dependency updates.
          
          - Security audit passed
          - All tests should pass before merging
          
          Review the changes carefully before merging.
        branch: automated/dependency-updates
        delete-branch: true
```

### 4. Release Automation (`release.yml`)

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  create-release:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip build twine
        pip install -e ".[dev]"
    
    - name: Run tests
      run: pytest
    
    - name: Build package
      run: python -m build
    
    - name: Generate changelog
      id: changelog
      run: |
        # Install auto-changelog if not present
        pip install auto-changelog
        auto-changelog --template changelog.md.j2 --output CHANGELOG_CURRENT.md
        echo "changelog<<EOF" >> $GITHUB_OUTPUT
        cat CHANGELOG_CURRENT.md >> $GITHUB_OUTPUT
        echo "EOF" >> $GITHUB_OUTPUT
    
    - name: Create GitHub Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: Release ${{ github.ref }}
        body: ${{ steps.changelog.outputs.changelog }}
        draft: false
        prerelease: false
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
      run: twine upload dist/*
```

### 5. Documentation (`docs.yml`)

```yaml
name: Documentation

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build-docs:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[docs]"
    
    - name: Build documentation
      run: |
        cd docs
        make html
    
    - name: Upload documentation artifacts
      uses: actions/upload-artifact@v3
      with:
        name: documentation
        path: docs/_build/html/
    
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: docs/_build/html/
```

## Required Secrets

Set up these secrets in GitHub repository settings:

1. **PYPI_API_TOKEN** - For automated PyPI publishing
2. **SEMGREP_APP_TOKEN** - For Semgrep security scanning (optional)
3. **CODECOV_TOKEN** - For code coverage reporting (optional)

## Branch Protection Rules

Configure these branch protection rules for `main` branch:

```yaml
Required status checks:
  - test (3.9)
  - test (3.10) 
  - test (3.11)
  - test (3.12)
  - build
  - security-scan
  - semgrep
  - gitleaks

Additional settings:
  - Require branches to be up to date before merging
  - Require linear history
  - Include administrators in restrictions
  - Allow force pushes: false
  - Allow deletions: false
```

## Setup Checklist

- [ ] Create `.github/workflows/` directory
- [ ] Add all 5 workflow files
- [ ] Configure required secrets
- [ ] Set up branch protection rules
- [ ] Enable GitHub Pages for documentation
- [ ] Configure Dependabot (optional, alternative to custom dependency workflow)
- [ ] Test workflows with a test PR

## Integration with Existing Tools

These workflows integrate with the existing project structure:

- Uses existing `pyproject.toml` configuration
- Leverages current tooling (ruff, black, isort, mypy, pytest)
- Respects existing `Makefile` commands
- Builds on current pre-commit hooks

## Monitoring and Maintenance

- Review workflow runs weekly
- Update action versions quarterly
- Monitor security scan results daily
- Adjust Python version matrix as needed
- Update branch protection rules when adding new workflows

## Troubleshooting

Common issues and solutions:

1. **Test failures** - Check Python version compatibility
2. **Security scan false positives** - Configure ignore lists
3. **Build failures** - Verify dependencies in pyproject.toml
4. **Release failures** - Check PyPI token and package name availability

This comprehensive CI/CD setup provides automated quality gates, security scanning, and release management for the developing repository maturity level.