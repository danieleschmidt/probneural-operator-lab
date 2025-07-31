.PHONY: help install install-dev test test-cov lint format type-check clean docs build upload

help:			## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install:		## Install package
	pip install -e .

install-dev:		## Install package with development dependencies
	pip install -e ".[dev]"
	pre-commit install

test:			## Run tests
	pytest

test-cov:		## Run tests with coverage
	pytest --cov=probneural_operator --cov-report=html --cov-report=term

lint:			## Run linting
	ruff check probneural_operator tests
	black --check probneural_operator tests
	isort --check probneural_operator tests

format:			## Format code
	black probneural_operator tests
	isort probneural_operator tests
	ruff check --fix probneural_operator tests

type-check:		## Run type checking
	mypy probneural_operator

clean:			## Clean build artifacts
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

docs:			## Build documentation
	cd docs && make html

build:			## Build package
	python -m build

upload:			## Upload to PyPI (requires TWINE_USERNAME and TWINE_PASSWORD)
	python -m twine upload dist/*