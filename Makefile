.PHONY: help install install-dev test test-quick test-cov test-integration benchmark lint format type-check security-check clean clean-all docs docs-serve setup-dev docker-build docker-run pre-commit-all build upload dev-setup

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color
BOLD := \033[1m

help:			## Show this help
	@echo "$(BOLD)ProbNeural Operator Lab - Development Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-30s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Quick start:$(NC) make dev-setup && make test"

# =============================================================================
# Installation and Setup
# =============================================================================

install:		## Install package
	pip install -e .

install-dev:		## Install package with development dependencies
	pip install -e ".[dev]"
	pre-commit install
	pre-commit install --hook-type commit-msg

dev-setup:		## Complete development environment setup
	@echo "$(GREEN)Setting up development environment...$(NC)"
	python -m pip install --upgrade pip setuptools wheel
	pip install -e ".[dev]"
	pre-commit install
	pre-commit install --hook-type commit-msg
	mkdir -p data/{raw,processed,external} models/{checkpoints,pretrained} logs experiments results
	@echo "$(GREEN)Development environment ready!$(NC)"

setup-env:		## Create .env file from template
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN)Created .env file from template$(NC)"; \
	else \
		echo "$(YELLOW).env file already exists$(NC)"; \
	fi

# =============================================================================
# Testing
# =============================================================================

test:			## Run all tests
	pytest tests/ --cov=probneural_operator --cov-report=html --cov-report=term-missing -v

test-quick:		## Run quick tests (skip slow ones)
	pytest tests/ -m "not slow" --tb=short

test-cov:		## Run tests with detailed coverage
	pytest tests/ --cov=probneural_operator --cov-report=html --cov-report=term-missing --cov-report=xml

test-integration:	## Run integration tests only
	pytest tests/integration/ -v

test-unit:		## Run unit tests only
	pytest tests/unit/ -v

test-watch:		## Run tests in watch mode
	pytest-watch tests/ -- --tb=short

benchmark:		## Run performance benchmarks
	pytest benchmarks/ --benchmark-only --benchmark-sort=mean

# =============================================================================
# Code Quality
# =============================================================================

lint:			## Run all linting checks
	@echo "$(GREEN)Running linting checks...$(NC)"
	ruff check probneural_operator tests scripts
	black --check probneural_operator tests scripts
	isort --check probneural_operator tests scripts

format:			## Format all code
	@echo "$(GREEN)Formatting code...$(NC)"
	black probneural_operator tests scripts
	isort probneural_operator tests scripts
	ruff check --fix probneural_operator tests scripts

type-check:		## Run type checking
	@echo "$(GREEN)Running type checks...$(NC)"
	mypy probneural_operator

pre-commit-all:		## Run pre-commit on all files
	pre-commit run --all-files

security-check:		## Run security checks
	@echo "$(GREEN)Running security checks...$(NC)"
	bandit -r probneural_operator/
	safety check
	pip-audit

# =============================================================================
# Cleaning
# =============================================================================

clean:			## Clean build artifacts
	@echo "$(GREEN)Cleaning build artifacts...$(NC)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

clean-all:		## Clean everything including data and models
	make clean
	rm -rf data/processed/*
	rm -rf models/checkpoints/*
	rm -rf logs/*
	rm -rf experiments/*
	rm -rf results/*

clean-cache:		## Clean Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# =============================================================================
# Documentation
# =============================================================================

docs:			## Build documentation
	@echo "$(GREEN)Building documentation...$(NC)"
	sphinx-build -b html docs/ docs/_build/html

docs-serve:		## Serve documentation locally
	@echo "$(GREEN)Serving documentation at http://localhost:8080$(NC)"
	python -m http.server 8080 --directory docs/_build/html

docs-clean:		## Clean documentation build
	rm -rf docs/_build/

docs-linkcheck:		## Check documentation links
	sphinx-build -b linkcheck docs/ docs/_build/linkcheck

# =============================================================================
# Development Tools
# =============================================================================

jupyter:		## Start Jupyter Lab
	@echo "$(GREEN)Starting Jupyter Lab at http://localhost:8888$(NC)"
	jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser

tensorboard:		## Start TensorBoard
	@echo "$(GREEN)Starting TensorBoard at http://localhost:6006$(NC)"
	tensorboard --logdir=logs/tensorboard --port=6006

api-dev:		## Start development API server
	@echo "$(GREEN)Starting development API server at http://localhost:8000$(NC)"
	uvicorn probneural_operator.api.main:app --host 0.0.0.0 --port 8000 --reload

profile:		## Run profiling
	python -m cProfile -o profile.stats scripts/profile_example.py
	python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# =============================================================================
# Docker
# =============================================================================

docker-build:		## Build Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker build -t probneural-operator-lab:latest .

docker-run:		## Run Docker container
	@echo "$(GREEN)Running Docker container...$(NC)"
	docker run -it --rm --gpus all -v $(PWD):/workspace probneural-operator-lab:latest

docker-dev:		## Run Docker container for development
	docker run -it --rm --gpus all -v $(PWD):/workspace -p 8888:8888 -p 6006:6006 -p 8000:8000 probneural-operator-lab:latest bash

# =============================================================================
# Data Management
# =============================================================================

download-data:		## Download sample datasets
	@echo "$(GREEN)Downloading sample datasets...$(NC)"
	python scripts/download_datasets.py

setup-data:		## Setup data directories and symlinks
	mkdir -p data/{raw,processed,external,cache}
	mkdir -p models/{checkpoints,pretrained}
	mkdir -p logs/{tensorboard,wandb}
	mkdir -p experiments results

# =============================================================================
# Build and Release
# =============================================================================

build:			## Build package
	@echo "$(GREEN)Building package...$(NC)"
	python -m build

build-wheel:		## Build wheel only
	python -m build --wheel

build-sdist:		## Build source distribution only
	python -m build --sdist

check-dist:		## Check distribution packages
	python -m twine check dist/*

upload-test:		## Upload to TestPyPI
	python -m twine upload --repository testpypi dist/*

upload:			## Upload to PyPI (requires TWINE_USERNAME and TWINE_PASSWORD)
	@echo "$(RED)Uploading to PyPI - make sure you're ready!$(NC)"
	python -m twine upload dist/*

# =============================================================================
# Git and Release Management
# =============================================================================

git-hooks:		## Install git hooks
	pre-commit install
	pre-commit install --hook-type commit-msg

bump-patch:		## Bump patch version
	bump2version patch

bump-minor:		## Bump minor version
	bump2version minor

bump-major:		## Bump major version
	bump2version major

# =============================================================================
# CI/CD Support
# =============================================================================

ci-test:		## Run tests as in CI
	pytest tests/ --cov=probneural_operator --cov-report=xml --junitxml=test-results.xml

ci-lint:		## Run linting as in CI
	ruff check probneural_operator tests scripts --output-format=github
	black --check probneural_operator tests scripts
	isort --check probneural_operator tests scripts

ci-security:		## Run security checks as in CI
	bandit -r probneural_operator/ -f json -o bandit-report.json
	safety check --json --output safety-report.json

# =============================================================================
# Monitoring and Health
# =============================================================================

health-check:		## Run health checks
	@echo "$(GREEN)Running health checks...$(NC)"
	python -c "import probneural_operator; print('✓ Package imports successfully')"
	python -c "import torch; print(f'✓ PyTorch {torch.__version__} available')"
	python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"

memory-profile:		## Profile memory usage
	python -m memory_profiler scripts/memory_test.py

gpu-info:		## Show GPU information
	nvidia-smi

# =============================================================================
# Development Utilities
# =============================================================================

fix-permissions:	## Fix file permissions
	find . -type f -name "*.py" -exec chmod 644 {} \;
	find . -type f -name "*.sh" -exec chmod 755 {} \;
	find . -type d -exec chmod 755 {} \;

count-lines:		## Count lines of code
	find probneural_operator -name "*.py" | xargs wc -l | tail -1

todo:			## Find TODO items in code
	@echo "$(YELLOW)TODO items:$(NC)"
	grep -r "TODO\|FIXME\|XXX\|HACK" probneural_operator/ || echo "No TODO items found"

dependencies:		## Show dependency tree
	pip-tree

outdated:		## Show outdated dependencies
	pip list --outdated

# =============================================================================
# Debugging
# =============================================================================

debug-env:		## Show environment information
	@echo "$(GREEN)Environment Information:$(NC)"
	@echo "Python: $(shell python --version)"
	@echo "Pip: $(shell pip --version)"
	@echo "Working Directory: $(shell pwd)"
	@echo "Virtual Environment: $(VIRTUAL_ENV)"
	@echo "CUDA_VISIBLE_DEVICES: $(CUDA_VISIBLE_DEVICES)"

debug-imports:		## Test critical imports
	@echo "$(GREEN)Testing critical imports:$(NC)"
	python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
	python -c "import scipy; print(f'SciPy: {scipy.__version__}')"
	python -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"

# =============================================================================
# Help Categories
# =============================================================================

help-dev:		## Show development-specific help
	@echo "$(BOLD)Development Commands:$(NC)"
	@echo "  make dev-setup     - Complete development setup"
	@echo "  make test-quick    - Quick test run"
	@echo "  make format        - Format all code"
	@echo "  make jupyter       - Start Jupyter Lab"
	@echo "  make clean         - Clean build artifacts"

help-ci:		## Show CI/CD commands
	@echo "$(BOLD)CI/CD Commands:$(NC)"
	@echo "  make ci-test       - Run tests as in CI"
	@echo "  make ci-lint       - Run linting as in CI"
	@echo "  make ci-security   - Run security checks as in CI"

help-docker:		## Show Docker commands
	@echo "$(BOLD)Docker Commands:$(NC)"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-run    - Run Docker container"
	@echo "  make docker-dev    - Run development container"