# Multi-stage Dockerfile for ProbNeural-Operator-Lab
# Optimized for development, testing, and production use

ARG PYTHON_VERSION=3.11
ARG PYTORCH_VERSION=2.0.0
ARG CUDA_VERSION=11.7

# Base Python image with scientific computing optimizations
FROM python:${PYTHON_VERSION}-slim-bullseye as base

# Set environment variables for reproducible builds
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC

# Install system dependencies and security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    ca-certificates \
    gnupg \
    lsb-release \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r -g 1000 probneural && \
    useradd -r -u 1000 -g probneural -d /app -s /bin/bash -c "ProbNeural User" probneural

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs /app/results && \
    chown -R probneural:probneural /app

# Development stage
FROM base as development

# Install additional development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    nano \
    htop \
    tmux \
    tree \
    jq \
    sqlite3 \
    graphviz \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy dependency files first for better layer caching
COPY pyproject.toml requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install package dependencies
RUN pip install --no-cache-dir -e ".[dev,test,docs]"

# Copy source code
COPY --chown=probneural:probneural . .

# Install package in development mode
RUN pip install --no-deps -e .

# Create additional development directories
RUN mkdir -p /app/.vscode /app/notebooks /app/experiments && \
    chown -R probneural:probneural /app

# Switch to non-root user
USER probneural

# Set development environment variables
ENV PYTHONPATH=/app \
    DEVELOPMENT=true \
    LOG_LEVEL=DEBUG

# Expose ports for development servers
EXPOSE 8000 8888 6006

# Development health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import probneural_operator; print('Development environment ready')" || exit 1

# Default command for development
CMD ["make", "dev-setup"]

# Testing stage
FROM base as testing

# Install testing tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy dependency files
COPY pyproject.toml requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install test dependencies
RUN pip install --no-cache-dir -e ".[test]"

# Copy source code
COPY --chown=probneural:probneural . .

# Install package
RUN pip install --no-deps -e .

# Create test output directories
RUN mkdir -p /app/test-results /app/coverage && \
    chown -R probneural:probneural /app/test-results /app/coverage

# Switch to non-root user
USER probneural

# Set testing environment variables
ENV PYTHONPATH=/app \
    TESTING=true \
    COVERAGE_FILE=/app/coverage/.coverage

# Testing health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import probneural_operator; print('Testing environment ready')" || exit 1

# Run tests by default
CMD ["python", "-m", "pytest", "-v", "--cov=probneural_operator", "--cov-report=xml:/app/coverage/coverage.xml", "--junitxml=/app/test-results/junit.xml"]

# Production stage
FROM base as production

# Install minimal system dependencies for production
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy only requirements for production
COPY pyproject.toml ./

# Install only production dependencies
RUN pip install --no-cache-dir . && \
    pip cache purge

# Copy only necessary files for production
COPY --chown=probneural:probneural probneural_operator/ ./probneural_operator/
COPY --chown=probneural:probneural README.md LICENSE ./

# Remove unnecessary files and packages to minimize image size
RUN apt-get purge -y --auto-remove build-essential && \
    rm -rf /tmp/* /var/tmp/* /root/.cache

# Switch to non-root user
USER probneural

# Set production environment variables
ENV PYTHONPATH=/app \
    PRODUCTION=true \
    LOG_LEVEL=INFO

# Production health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import probneural_operator; print('Production environment ready')" || exit 1

# Default command
CMD ["python", "-c", "import probneural_operator; print('ProbNeural Operator Lab is ready!')"]

# Documentation stage
FROM base as docs

# Install documentation build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    make \
    pandoc \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy dependency files
COPY pyproject.toml ./

# Install documentation dependencies
RUN pip install --no-cache-dir -e ".[docs]"

# Copy source code and docs
COPY --chown=probneural:probneural . .

# Build documentation
RUN make docs

# Switch to non-root user
USER probneural

# Set documentation environment variables
ENV PYTHONPATH=/app

# Documentation health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080 || exit 1

# Serve documentation
EXPOSE 8080
CMD ["python", "-m", "http.server", "8080", "--directory", "docs/_build/html"]

# CUDA-enabled stage for GPU workloads
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04 as cuda

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    CUDA_VISIBLE_DEVICES=0

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symlink for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

# Create non-root user
RUN groupadd -r -g 1000 probneural && \
    useradd -r -u 1000 -g probneural -d /app -s /bin/bash -c "ProbNeural User" probneural

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/logs /app/results && \
    chown -R probneural:probneural /app

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy dependency files
COPY pyproject.toml requirements-dev.txt ./

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Install other dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Copy source code
COPY --chown=probneural:probneural . .

# Install package
RUN pip install --no-deps -e .

# Switch to non-root user
USER probneural

# Set CUDA environment variables
ENV PYTHONPATH=/app \
    TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6" \
    FORCE_CUDA=1

# CUDA health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(); print('CUDA environment ready')" || exit 1

# Expose ports
EXPOSE 8000 8888 6006

# Default command
CMD ["python", "-c", "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"]