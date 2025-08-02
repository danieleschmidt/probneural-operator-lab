# Contributing to ProbNeural Operator Lab

Thank you for your interest in contributing to the ProbNeural Operator Lab! This document provides guidelines and information for contributors.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [How to Contribute](#how-to-contribute)
4. [Development Workflow](#development-workflow)
5. [Coding Standards](#coding-standards)
6. [Testing Guidelines](#testing-guidelines)
7. [Documentation Guidelines](#documentation-guidelines)
8. [Pull Request Process](#pull-request-process)
9. [Issue Guidelines](#issue-guidelines)
10. [Community](#community)

## Code of Conduct

This project adheres to a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## Getting Started

### Prerequisites

- **Python**: 3.9 or higher
- **Git**: For version control
- **PyTorch**: Will be installed with dependencies
- **CUDA** (optional): For GPU acceleration

### Development Environment Setup

1. **Fork and Clone the Repository**
   ```bash
   # Fork the repository on GitHub, then clone your fork
   git clone https://github.com/YOUR-USERNAME/probneural-operator-lab.git
   cd probneural-operator-lab
   
   # Add upstream remote
   git remote add upstream https://github.com/danieleschmidt/probneural-operator-lab.git
   ```

2. **Set Up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install in development mode
   pip install -e ".[dev,test,docs]"
   
   # Or use make command
   make setup-dev
   
   # Install pre-commit hooks
   pre-commit install
   ```

3. **Verify Installation**
   ```bash
   # Run tests to verify everything works
   pytest tests/unit/ -v
   
   # Check code quality
   pre-commit run --all-files
   ```

### Development Tools

We recommend using:
- **IDE**: VS Code with Python extension
- **Linting**: flake8, black, isort (configured in pre-commit)
- **Type Checking**: mypy
- **Testing**: pytest with coverage

## How to Contribute

### Types of Contributions

We welcome various types of contributions:

1. **Bug Reports**: Help us identify and fix issues
2. **Feature Requests**: Propose new functionality
3. **Code Contributions**: Implement features, fix bugs, improve performance
4. **Documentation**: Improve docs, add examples, write tutorials
5. **Testing**: Add tests, improve test coverage
6. **Reviews**: Review pull requests from other contributors

### Contribution Areas

- **Neural Operators**: FNO, DeepONet, GNO implementations
- **Uncertainty Quantification**: Linearized Laplace, ensemble methods
- **Active Learning**: Acquisition functions, sampling strategies
- **Performance Optimization**: CUDA kernels, memory efficiency
- **Data Processing**: Loaders, preprocessors, augmentations
- **Visualization**: Plotting utilities, interactive dashboards
- **Infrastructure**: CI/CD, Docker, monitoring

## Development Workflow

### 1. Planning Your Contribution

Before starting work:

1. **Check Existing Issues**: Look for related issues or discussions
2. **Create an Issue**: If one doesn't exist, create an issue to discuss your idea
3. **Get Feedback**: Wait for maintainer feedback before starting large changes
4. **Assign Yourself**: Comment on the issue to indicate you're working on it

### 2. Development Process

```bash
# 1. Update your local main branch
git checkout main
git pull upstream main

# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Make your changes
# Edit files, add features, fix bugs...

# 4. Write/update tests
# Add tests for new functionality

# 5. Update documentation
# Update docstrings, README, docs/ if needed

# 6. Run tests and checks
make test
make lint
make format

# 7. Commit your changes
git add .
git commit -m "feat: add feature description"

# 8. Push to your fork
git push origin feature/your-feature-name

# 9. Create a pull request
# Use GitHub interface to create PR
```

### 3. Keeping Your Branch Updated

```bash
# Fetch latest changes from upstream
git fetch upstream

# Rebase your branch on latest main
git checkout feature/your-feature-name
git rebase upstream/main

# Push updated branch (force push after rebase)
git push --force-with-lease origin feature/your-feature-name
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line Length**: 88 characters (Black default)
- **Import Sorting**: isort with profile "black"
- **Docstring Style**: Google/NumPy style
- **Type Hints**: Required for public APIs

### Code Formatting

Automated formatting with pre-commit hooks:

```bash
# Format code
make format
# or
black .
isort .

# Check formatting
make lint
# or
black --check .
isort --check-only .
flake8 .
mypy probneural_operator/
```

### Naming Conventions

- **Classes**: PascalCase (`NeuralOperator`)
- **Functions/Variables**: snake_case (`compute_loss`)
- **Constants**: UPPER_SNAKE_CASE (`DEFAULT_BATCH_SIZE`)
- **Private**: Leading underscore (`_internal_function`)
- **Files**: snake_case (`fourier_layer.py`)

### Import Organization

```python
# Standard library imports
import os
import sys
from typing import Dict, List, Optional

# Third-party imports
import torch
import numpy as np
import matplotlib.pyplot as plt

# Local imports
from probneural_operator.models import FNO
from probneural_operator.utils import load_data
```

### Documentation Standards

Every public function/class should have docstrings:

```python
def compute_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """Compute loss between predictions and targets.
    
    Args:
        predictions: Model predictions of shape (batch_size, ...)
        targets: Ground truth targets of shape (batch_size, ...)
        reduction: Reduction method ('mean', 'sum', 'none')
    
    Returns:
        Loss tensor. Shape depends on reduction parameter.
    
    Raises:
        ValueError: If predictions and targets have different shapes.
    
    Example:
        >>> predictions = torch.randn(32, 10)
        >>> targets = torch.randn(32, 10)
        >>> loss = compute_loss(predictions, targets)
        >>> print(loss.item())
        0.5234
    """
```

## Testing Guidelines

### Test Organization

```
tests/
├── unit/           # Unit tests (fast, isolated)
├── integration/    # Integration tests (components together)
├── benchmarks/     # Performance benchmarks
├── conftest.py     # Shared test fixtures
└── utils.py        # Test utilities
```

### Writing Tests

1. **Test Structure**
   ```python
   def test_function_name():
       # Arrange
       input_data = create_test_data()
       
       # Act
       result = function_under_test(input_data)
       
       # Assert
       assert result.shape == expected_shape
       assert torch.allclose(result, expected_output)
   ```

2. **Use Fixtures**
   ```python
   @pytest.fixture
   def sample_data():
       return torch.randn(32, 10)
   
   def test_model_forward(sample_data):
       model = MyModel()
       output = model(sample_data)
       assert output.shape[0] == 32
   ```

3. **Parameterized Tests**
   ```python
   @pytest.mark.parametrize("batch_size,input_dim", [
       (1, 10),
       (16, 20),
       (32, 50),
   ])
   def test_model_shapes(batch_size, input_dim):
       model = MyModel(input_dim)
       x = torch.randn(batch_size, input_dim)
       output = model(x)
       assert output.shape == (batch_size, model.output_dim)
   ```

### Test Coverage

- **Minimum**: 80% overall coverage
- **Critical Paths**: 95%+ coverage for core algorithms
- **New Code**: 90%+ coverage required

```bash
# Run tests with coverage
make test-coverage
# or
pytest --cov=probneural_operator --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Documentation Guidelines

### Types of Documentation

1. **API Documentation**: Auto-generated from docstrings
2. **User Guides**: Step-by-step tutorials
3. **Developer Docs**: Architecture, contributing guidelines
4. **Examples**: Jupyter notebooks, scripts

### Writing Documentation

1. **Clear and Concise**: Use simple language
2. **Code Examples**: Include working examples
3. **Visual Aids**: Use diagrams, plots when helpful
4. **Keep Updated**: Update docs with code changes

### Building Documentation

```bash
# Build documentation locally
make docs
# or
cd docs/
make html

# View documentation
open _build/html/index.html

# Check for broken links
make linkcheck
```

## Pull Request Process

### Before Submitting

1. **Self-Review**: Review your own code thoroughly
2. **Test Locally**: Ensure all tests pass
3. **Update Documentation**: Update relevant docs
4. **Clean History**: Squash/reorder commits if needed

### PR Guidelines

1. **Title**: Use conventional commit format
   - `feat: add new feature`
   - `fix: resolve bug in component`
   - `docs: update API documentation`

2. **Description**: 
   - Explain what and why
   - Reference related issues
   - Include testing instructions

3. **Size**: Keep PRs focused and reasonably sized
   - Prefer multiple small PRs over one large PR
   - Aim for < 500 lines changed when possible

4. **Draft PRs**: Use draft PRs for work-in-progress

### Review Process

1. **Automated Checks**: All CI checks must pass
2. **Code Review**: At least one maintainer review required
3. **Testing**: Reviewer may test changes locally
4. **Documentation**: Reviewer checks doc updates
5. **Approval**: PR approved by maintainer

### After Merge

1. **Cleanup**: Delete feature branch
2. **Update Issues**: Close related issues
3. **Release Notes**: May be included in next release

## Issue Guidelines

### Creating Issues

1. **Search First**: Check for existing similar issues
2. **Use Templates**: Fill out provided issue templates
3. **Be Specific**: Provide detailed information
4. **Minimal Example**: Include reproducible example when applicable

### Issue Labels

- **Type**: `bug`, `feature`, `enhancement`, `documentation`
- **Priority**: `critical`, `high`, `medium`, `low`
- **Status**: `needs-triage`, `in-progress`, `blocked`
- **Area**: `neural-operators`, `uncertainty`, `active-learning`

## Community

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Pull Requests**: Code review, collaboration

### Getting Help

1. **Documentation**: Check docs first
2. **Search Issues**: Look for similar problems
3. **Create Issue**: Use question template
4. **Be Patient**: Maintainers volunteer their time

### Helping Others

- **Answer Questions**: Help in discussions
- **Review PRs**: Provide constructive feedback
- **Triage Issues**: Help categorize new issues
- **Write Documentation**: Improve docs based on common questions

## Recognition

Contributors are recognized in:
- **CONTRIBUTORS.md**: List of all contributors
- **Release Notes**: Major contributions highlighted
- **Git History**: All commits attributed to authors

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

## Quick Reference

### Common Commands

```bash
# Development setup
git clone <your-fork>
make setup-dev
pre-commit install

# Before committing
make test
make lint
make format

# Code formatting
make format
# or
black .
isort .

# Type checking
make type-check
# or
mypy probneural_operator/

# Documentation
make docs
```

### Getting Unstuck

If you're stuck or have questions:

1. Check existing documentation
2. Search GitHub issues
3. Create a question issue
4. Reach out to maintainers

Remember: **No question is too basic!** We're here to help and welcome contributions from developers of all experience levels.

Thank you for contributing to ProbNeural Operator Lab! 🎉