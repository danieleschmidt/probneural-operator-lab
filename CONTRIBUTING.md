# Contributing to ProbNeural-Operator-Lab

We welcome contributions! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/probneural-operator-lab.git
   cd probneural-operator-lab
   ```

3. Set up development environment:
   ```bash
   make install-dev
   ```

## Development Workflow

### Code Style
- **Black** for formatting
- **Ruff** for linting  
- **MyPy** for type checking

```bash
make format     # Format code
make lint       # Check style
make type-check # Type checking
```

### Testing
```bash
make test       # Run tests
make test-cov   # Run with coverage
```

## Pull Request Process

1. Create descriptive commits
2. Write tests for new features
3. Update documentation
4. Ensure all checks pass
5. Open PR with clear description

### PR Checklist
- [ ] Tests added/updated and passing
- [ ] Code formatted with `make format`
- [ ] Linting passes with `make lint`
- [ ] Documentation updated if needed

## Questions?

Open an issue for bugs or feature requests.

Thank you for contributing!