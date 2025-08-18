#!/bin/bash

# Development Environment Setup Script for ProbNeural Operator Lab
# This script sets up the complete development environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

echo_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

echo_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

echo_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo_error "Please run this script from the project root directory"
    exit 1
fi

echo_info "Setting up ProbNeural Operator Lab development environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo_info "Creating virtual environment..."
    python3 -m venv venv
    echo_success "Virtual environment created"
else
    echo_warning "Virtual environment already exists"
fi

# Activate virtual environment
echo_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo_info "Upgrading pip..."
pip install --upgrade pip

# Install development dependencies
echo_info "Installing development dependencies..."
pip install -e ".[dev,test,docs]"
echo_success "Dependencies installed"

# Install pre-commit hooks
echo_info "Setting up pre-commit hooks..."
pre-commit install
echo_success "Pre-commit hooks installed"

# Create necessary directories
echo_info "Creating project directories..."
mkdir -p data
mkdir -p checkpoints
mkdir -p outputs
mkdir -p cache
mkdir -p logs/tensorboard
echo_success "Project directories created"

# Copy environment template if .env doesn't exist
if [ ! -f ".env" ]; then
    echo_info "Creating .env file from template..."
    cp .env.example .env
    echo_success ".env file created"
    echo_warning "Please review and update .env file with your configuration"
else
    echo_warning ".env file already exists"
fi

# Run initial code quality checks
echo_info "Running initial code quality checks..."

echo_info "Formatting code with Black..."
black .

echo_info "Sorting imports with isort..."
isort .

echo_info "Linting with Ruff..."
ruff check . --fix || true

echo_info "Type checking with MyPy..."
mypy probneural_operator/ || echo_warning "Type checking found issues (not blocking)"

# Run basic tests
echo_info "Running basic tests..."
pytest tests/unit/test_import.py -v || echo_warning "Some basic tests failed"

# Check CUDA availability
echo_info "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')" || echo_warning "Could not check CUDA availability"

echo_success "Development environment setup complete!"

echo ""
echo_info "Next steps:"
echo "1. Review and update .env file with your configuration"
echo "2. Activate virtual environment: source venv/bin/activate"
echo "3. Run tests: pytest tests/ -v"
echo "4. Start development server: python examples/production_server.py"
echo "5. Open VS Code with: code ."

echo ""
echo_info "Useful commands:"
echo "• Run tests with coverage: pytest tests/ --cov=probneural_operator --cov-report=html"
echo "• Format code: black ."
echo "• Sort imports: isort ."
echo "• Lint code: ruff check ."
echo "• Type check: mypy probneural_operator/"
echo "• Build docs: make docs"
echo "• Clean cache: find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true"

echo ""
echo_success "Happy coding! 🚀"