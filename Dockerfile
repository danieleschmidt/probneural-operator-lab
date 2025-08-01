# Multi-stage Dockerfile for ProbNeural-Operator-Lab
# Optimized for development, testing, and production use

# Base Python image with scientific computing optimizations
FROM python:3.11-slim-bullseye as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r probneural && useradd -r -g probneural -d /app -s /sbin/nologin probneural

# Set working directory
WORKDIR /app

# Development stage
FROM base as development

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY pyproject.toml requirements-dev.txt ./
RUN pip install --no-cache-dir -e ".[dev,test,docs]"

# Copy source code
COPY --chown=probneural:probneural . .

# Install package in development mode
RUN pip install -e .

# Switch to non-root user
USER probneural

# Expose port for development server
EXPOSE 8000

# Default command for development
CMD ["python", "-m", "pytest", "--cov=probneural_operator"]

# Testing stage
FROM base as testing

# Copy requirements and install test dependencies
COPY pyproject.toml requirements-dev.txt ./
RUN pip install --no-cache-dir -e ".[test]"

# Copy source code
COPY --chown=probneural:probneural . .

# Install package
RUN pip install -e .

# Switch to non-root user
USER probneural

# Run tests by default
CMD ["python", "-m", "pytest", "-v", "--cov=probneural_operator", "--cov-report=xml"]

# Production stage
FROM base as production

# Install only production dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

# Copy only necessary files
COPY --chown=probneural:probneural probneural_operator/ ./probneural_operator/
COPY --chown=probneural:probneural README.md LICENSE ./

# Switch to non-root user
USER probneural

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import probneural_operator; print('OK')" || exit 1

# Default command
CMD ["python", "-c", "import probneural_operator; print('ProbNeural Operator Lab is ready!')"]

# Documentation stage
FROM base as docs

# Install documentation dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[docs]"

# Copy source code and docs
COPY --chown=probneural:probneural . .

# Build documentation
RUN cd docs && make html

# Switch to non-root user
USER probneural

# Serve documentation
EXPOSE 8080
CMD ["python", "-m", "http.server", "8080", "--directory", "docs/_build/html"]