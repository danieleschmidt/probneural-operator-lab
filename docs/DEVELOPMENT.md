# Development Guide

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/probneural-operator-lab.git
   cd probneural-operator-lab
   ```

2. Install in development mode:
   ```bash
   make install-dev
   ```

## Project Structure

```
probneural_operator/
├── models/          # Neural operator implementations
├── posteriors/      # Uncertainty quantification methods
├── active/          # Active learning strategies
├── calibration/     # Uncertainty calibration
├── applications/    # Domain-specific applications
└── benchmarks/      # Evaluation utilities
```

## Development Commands

```bash
make test           # Run tests
make test-cov       # Run tests with coverage
make format         # Format code
make lint           # Check code style
make type-check     # Type checking
make docs           # Build documentation
make clean          # Clean build artifacts
```

## Testing

Tests are located in `tests/` directory:
- Unit tests: `tests/unit/`
- Integration tests: `tests/integration/`
- Run specific tests: `pytest tests/unit/test_models.py`

## Code Standards

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write docstrings in Google style
- Maintain test coverage above 90%

## Pre-commit Hooks

Pre-commit hooks automatically run on each commit:
- Code formatting with Black
- Import sorting with isort
- Linting with Ruff
- Type checking with MyPy

## Debugging

For debugging neural operators:
1. Use smaller model sizes
2. Enable gradient tracking
3. Check tensor shapes carefully
4. Validate input data ranges