#!/bin/bash

# Build script for ProbNeural Operator Lab
# Supports multiple build targets and configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Default values
BUILD_TYPE="development"
TAG="latest"
PUSH=false
NO_CACHE=false
PLATFORM="linux/amd64"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --push)
            PUSH=true
            shift
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -t, --type TYPE      Build type: development, production, testing, docs (default: development)"
            echo "  --tag TAG            Docker tag (default: latest)"
            echo "  --push               Push image to registry"
            echo "  --no-cache           Build without cache"
            echo "  --platform PLATFORM  Target platform (default: linux/amd64)"
            echo "  -h, --help           Show this help"
            exit 0
            ;;
        *)
            echo_error "Unknown option $1"
            exit 1
            ;;
    esac
done

# Validate build type
case $BUILD_TYPE in
    development|production|testing|docs)
        ;;
    *)
        echo_error "Invalid build type: $BUILD_TYPE"
        echo "Valid types: development, production, testing, docs"
        exit 1
        ;;
esac

# Get project info
PROJECT_NAME="probneural-operator-lab"
VERSION=$(python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])" 2>/dev/null || echo "dev")

echo_info "Building ProbNeural Operator Lab"
echo_info "Build type: $BUILD_TYPE"
echo_info "Version: $VERSION"
echo_info "Tag: $TAG"
echo_info "Platform: $PLATFORM"

# Pre-build checks
echo_info "Running pre-build checks..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo_error "Docker is not running or not accessible"
    exit 1
fi

# Check if pyproject.toml exists
if [ ! -f "pyproject.toml" ]; then
    echo_error "pyproject.toml not found. Run this script from the project root."
    exit 1
fi

# Security scan of dependencies (if available)
if command -v safety >/dev/null 2>&1; then
    echo_info "Running security scan..."
    safety check --json || echo_warning "Security scan found issues"
else
    echo_warning "Safety not installed, skipping security scan"
fi

# Build arguments
DOCKER_ARGS="--platform $PLATFORM"

if [ "$NO_CACHE" = true ]; then
    DOCKER_ARGS="$DOCKER_ARGS --no-cache"
fi

# Set image name
IMAGE_NAME="$PROJECT_NAME:$TAG-$BUILD_TYPE"

echo_info "Building Docker image: $IMAGE_NAME"

# Build the Docker image
docker build \
    $DOCKER_ARGS \
    --target $BUILD_TYPE \
    --build-arg VERSION=$VERSION \
    --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
    --build-arg VCS_REF=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown") \
    --label "org.opencontainers.image.title=$PROJECT_NAME" \
    --label "org.opencontainers.image.version=$VERSION" \
    --label "org.opencontainers.image.created=$(date -u +'%Y-%m-%dT%H:%M:%SZ')" \
    --label "org.opencontainers.image.revision=$(git rev-parse HEAD 2>/dev/null || echo "unknown")" \
    --label "org.opencontainers.image.source=https://github.com/danieleschmidt/probneural-operator-lab" \
    --tag $IMAGE_NAME \
    .

if [ $? -eq 0 ]; then
    echo_success "Build completed: $IMAGE_NAME"
else
    echo_error "Build failed"
    exit 1
fi

# Test the built image
echo_info "Testing built image..."
case $BUILD_TYPE in
    development)
        docker run --rm $IMAGE_NAME python -c "import probneural_operator; print('Import successful')"
        ;;
    production)
        docker run --rm $IMAGE_NAME python -c "import probneural_operator; print('Production image ready')"
        ;;
    testing)
        docker run --rm $IMAGE_NAME python -c "import pytest; print('Test dependencies available')"
        ;;
    docs)
        docker run --rm $IMAGE_NAME ls docs/_build/html/index.html
        ;;
esac

if [ $? -eq 0 ]; then
    echo_success "Image test passed"
else
    echo_error "Image test failed"
    exit 1
fi

# Security scan of image (if Trivy is available)
if command -v trivy >/dev/null 2>&1; then
    echo_info "Running container security scan..."
    trivy image --exit-code 0 --severity HIGH,CRITICAL $IMAGE_NAME || echo_warning "Security scan found issues"
else
    echo_warning "Trivy not installed, skipping container security scan"
fi

# Push to registry if requested
if [ "$PUSH" = true ]; then
    echo_info "Pushing image to registry..."
    docker push $IMAGE_NAME
    
    if [ $? -eq 0 ]; then
        echo_success "Image pushed: $IMAGE_NAME"
    else
        echo_error "Failed to push image"
        exit 1
    fi
fi

# Generate SBOM (Software Bill of Materials) if syft is available
if command -v syft >/dev/null 2>&1; then
    echo_info "Generating Software Bill of Materials..."
    syft packages $IMAGE_NAME -o spdx-json > sbom-$BUILD_TYPE.spdx.json
    echo_success "SBOM generated: sbom-$BUILD_TYPE.spdx.json"
else
    echo_warning "Syft not installed, skipping SBOM generation"
fi

# Show image info
echo_info "Image details:"
docker images $IMAGE_NAME --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

echo_success "Build process completed successfully!"

# Additional recommendations
echo ""
echo_info "Next steps:"
case $BUILD_TYPE in
    development)
        echo "• Run container: docker run -it --rm -p 8000:8000 $IMAGE_NAME"
        echo "• Mount code for development: docker run -it --rm -v \$(pwd):/app $IMAGE_NAME"
        ;;
    production)
        echo "• Deploy container: docker run -d -p 8000:8000 $IMAGE_NAME"
        echo "• Configure environment variables for production"
        ;;
    testing)
        echo "• Run tests: docker run --rm $IMAGE_NAME"
        echo "• Mount test results: docker run --rm -v \$(pwd)/test-results:/app/test-results $IMAGE_NAME"
        ;;
    docs)
        echo "• Serve docs: docker run -d -p 8080:8080 $IMAGE_NAME"
        echo "• Access documentation at http://localhost:8080"
        ;;
esac