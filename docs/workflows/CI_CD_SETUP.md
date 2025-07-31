# CI/CD Workflow Setup

## Required GitHub Actions Workflows

### 1. CI Workflow (`.github/workflows/ci.yml`)

**Triggers**: Push to main, pull requests
**Purpose**: Run tests, linting, and type checking

**Jobs needed:**
- **test**: Run pytest on multiple Python versions (3.9, 3.10, 3.11, 3.12)
- **lint**: Run ruff, black --check, isort --check
- **type-check**: Run mypy
- **coverage**: Generate and upload coverage reports

**Key steps:**
```yaml
# Example structure (create manually)
- uses: actions/checkout@v4
- uses: actions/setup-python@v4
  with:
    python-version: ${{ matrix.python-version }}
- run: pip install -e ".[dev]"
- run: make test
- run: make lint
- run: make type-check
```

### 2. Release Workflow (`.github/workflows/release.yml`)

**Triggers**: Tagged releases (v*)
**Purpose**: Build and publish to PyPI

**Jobs needed:**
- **build**: Create source and wheel distributions
- **publish**: Upload to PyPI using trusted publishing

### 3. Documentation Workflow (`.github/workflows/docs.yml`)

**Triggers**: Push to main
**Purpose**: Build and deploy documentation

**Requirements:**
- Sphinx configuration
- GitHub Pages deployment
- API documentation generation

## Security Considerations

**Required secrets:**
- `PYPI_API_TOKEN` (for PyPI publishing)

**Branch protection rules:**
- Require status checks to pass
- Require branches to be up to date
- Require review from code owners

## Manual Setup Steps

1. Enable GitHub Actions in repository settings
2. Configure branch protection rules for main branch
3. Set up PyPI trusted publishing
4. Enable GitHub Pages for documentation
5. Add required secrets to repository