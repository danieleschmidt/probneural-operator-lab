#!/bin/bash

# Development environment setup script for ProbNeural Operator Lab
set -e

echo "🚀 Setting up ProbNeural Operator Lab development environment..."

# Update package lists
echo "📦 Updating package lists..."
apt-get update

# Install system dependencies
echo "🔧 Installing system dependencies..."
apt-get install -y \
    build-essential \
    cmake \
    git-lfs \
    htop \
    vim \
    curl \
    wget \
    unzip \
    tree \
    graphviz \
    libgraphviz-dev \
    pkg-config

# Initialize Git LFS
echo "📁 Initializing Git LFS..."
git lfs install

# Upgrade pip and install build tools
echo "🐍 Upgrading pip and installing build tools..."
python -m pip install --upgrade pip setuptools wheel

# Install development dependencies
echo "📚 Installing Python dependencies..."
if [ -f "requirements-dev.txt" ]; then
    pip install -r requirements-dev.txt
fi

# Install package in development mode
echo "⚙️ Installing package in development mode..."
pip install -e ".[dev]"

# Install pre-commit hooks
echo "🪝 Installing pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create common directories
echo "📂 Creating common directories..."
mkdir -p data/{raw,processed,external}
mkdir -p models/{checkpoints,pretrained}
mkdir -p logs
mkdir -p experiments
mkdir -p results

# Set up Jupyter Lab extensions
echo "🔬 Setting up Jupyter Lab..."
jupyter lab build --dev-build=False --minimize=False

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env 2>/dev/null || echo "# Environment variables" > .env
fi

# Display useful information
echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Quick start commands:"
echo "  make test          - Run tests"
echo "  make lint          - Run linting"
echo "  make format        - Format code"
echo "  jupyter lab        - Start Jupyter Lab"
echo "  make docs          - Build documentation"
echo ""
echo "📖 Documentation: docs/guides/development/setup.md"
echo "🐛 Issues: https://github.com/yourusername/probneural-operator-lab/issues"
echo ""
echo "Happy coding! 🎉"