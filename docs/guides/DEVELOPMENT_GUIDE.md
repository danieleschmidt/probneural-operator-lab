# Development Guide

## Getting Started

### Prerequisites

- Python 3.9+
- Git
- Docker (optional but recommended)

### Development Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/danieleschmidt/probneural-operator-lab.git
   cd probneural-operator-lab
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### Development Workflow

#### Code Standards

We maintain high code quality through:

- **Formatting**: Black (line length 88)
- **Import sorting**: isort
- **Linting**: Ruff with strict settings
- **Type checking**: MyPy for static type analysis
- **Documentation**: Comprehensive docstrings

#### Before Committing

Always run these commands before committing:

```bash
# Format code
black .
isort .

# Lint code
ruff check .

# Type check
mypy probneural_operator/

# Run tests
pytest tests/ -v
```

#### Git Workflow

1. **Create feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make changes with descriptive commits**:
   ```bash
   git add .
   git commit -m "feat: add uncertainty calibration for FNO models"
   ```

3. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

## Package Structure

```
probneural_operator/
├── __init__.py              # Package initialization
├── models/                  # Neural operator implementations
│   ├── base/               # Base classes and interfaces
│   ├── deeponet/           # DeepONet implementations
│   ├── fno/               # Fourier Neural Operators
│   └── multifidelity/     # Multi-fidelity models
├── posteriors/             # Uncertainty quantification
│   ├── base/              # Base posterior classes
│   ├── laplace/           # Laplace approximations
│   └── adaptive_uncertainty.py
├── active/                 # Active learning components
│   ├── acquisition.py      # Acquisition functions
│   └── learner.py         # Active learning strategies
├── calibration/           # Uncertainty calibration
├── data/                  # Data handling utilities
├── scaling/               # Performance and scaling
└── utils/                 # Shared utilities
```

## Testing Guidelines

### Test Structure

```
tests/
├── unit/                  # Unit tests (fast, isolated)
├── integration/           # Integration tests (slower)
├── benchmarks/           # Performance benchmarks
└── conftest.py           # Shared test fixtures
```

### Writing Tests

1. **Unit tests** for individual functions/classes:
   ```python
   def test_laplace_approximation_basic():
       model = create_test_model()
       laplace = LinearizedLaplace(model)
       # Test specific functionality
   ```

2. **Integration tests** for component interactions:
   ```python
   def test_end_to_end_training():
       # Test complete training pipeline
   ```

3. **Property-based tests** using Hypothesis:
   ```python
   from hypothesis import given, strategies as st
   
   @given(st.floats(min_value=0.1, max_value=10.0))
   def test_uncertainty_scaling(scale_factor):
       # Test uncertainty behavior under scaling
   ```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test files
pytest tests/unit/test_models.py

# Run with coverage
pytest --cov=probneural_operator --cov-report=html

# Run only fast tests
pytest -m "not slow"

# Run tests in parallel
pytest -n auto
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def predict_with_uncertainty(
    self, 
    x: torch.Tensor, 
    return_std: bool = True
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Predict with uncertainty quantification.
    
    Args:
        x: Input tensor of shape (batch_size, ...)
        return_std: Whether to return uncertainty estimates
        
    Returns:
        Tuple of (predictions, uncertainties). Uncertainties is None 
        if return_std=False.
        
    Raises:
        ValueError: If input tensor has wrong shape
        
    Example:
        >>> model = ProbabilisticFNO()
        >>> mean, std = model.predict_with_uncertainty(x)
    """
```

### Building Documentation

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build docs
cd docs/
make html

# View locally
open _build/html/index.html
```

## Performance Considerations

### Memory Management

- Use `torch.no_grad()` for inference
- Clear GPU cache with `torch.cuda.empty_cache()`
- Profile memory usage with `torch.profiler`

### GPU Optimization

- Vectorize operations when possible
- Use mixed precision training
- Optimize data loading with multiple workers

### Benchmarking

Always benchmark performance changes:

```bash
python benchmarks/uncertainty_benchmark.py --compare-baseline
```

## Common Development Tasks

### Adding a New Neural Operator

1. Create base class in `models/base/`
2. Implement specific operator in appropriate subdirectory
3. Add uncertainty quantification wrapper
4. Write comprehensive tests
5. Add examples and documentation

### Adding New Uncertainty Method

1. Create base class in `posteriors/base/`
2. Implement specific method
3. Add calibration support
4. Write benchmarking comparisons
5. Update documentation

### Debugging Tips

1. **Enable debug logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Use PyTorch debugging tools**:
   ```python
   torch.autograd.set_detect_anomaly(True)
   ```

3. **Profile performance**:
   ```python
   with torch.profiler.profile() as prof:
       # Your code here
   print(prof.key_averages().table())
   ```

## Release Process

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md** with new features
3. **Run full test suite**: `pytest tests/ --slow`
4. **Build package**: `python -m build`
5. **Create release tag**: `git tag v0.2.0`
6. **Push to PyPI**: `twine upload dist/*`

## Getting Help

- **Documentation**: Check existing docs first
- **Issues**: Search GitHub issues for similar problems
- **Discussions**: Use GitHub Discussions for questions
- **Code Review**: All changes require review from maintainers

## Contributing Guidelines

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for detailed contribution guidelines.

Remember: Quality over quantity. We prefer well-tested, documented contributions over quick fixes.