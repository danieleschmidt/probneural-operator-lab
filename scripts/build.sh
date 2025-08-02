#!/bin/bash

# Build script for ProbNeural Operator Lab
# Supports multiple build targets and configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BUILD_TARGET="development"
PYTHON_VERSION="3.11"
PYTORCH_VERSION="2.0.0"
CUDA_VERSION="11.7"
PUSH_TO_REGISTRY=false
REGISTRY=""
TAG_SUFFIX=""
CACHE_FROM=""
BUILD_ARGS=""

# Print usage information
print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --target TARGET       Build target (development|testing|production|docs|cuda)"
    echo "  -p, --python VERSION      Python version (default: 3.11)"
    echo "  --pytorch VERSION         PyTorch version (default: 2.0.0)"
    echo "  --cuda VERSION            CUDA version (default: 11.7)"
    echo "  --push                    Push to registry after build"
    echo "  --registry REGISTRY       Registry to push to"
    echo "  --tag-suffix SUFFIX       Suffix to add to image tag"
    echo "  --cache-from IMAGE        Cache from image"
    echo "  --build-arg ARG=VALUE     Additional build arguments"
    echo "  --no-cache                Build without cache"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --target production --push --registry myregistry.com"
    echo "  $0 --target cuda --cuda 11.8"
    echo "  $0 --target development --no-cache"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            BUILD_TARGET="$2"
            shift 2
            ;;
        -p|--python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --pytorch)
            PYTORCH_VERSION="$2"
            shift 2
            ;;
        --cuda)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --push)
            PUSH_TO_REGISTRY=true
            shift
            ;;
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --tag-suffix)
            TAG_SUFFIX="$2"
            shift 2
            ;;
        --cache-from)
            CACHE_FROM="$2"
            shift 2
            ;;
        --build-arg)
            BUILD_ARGS="$BUILD_ARGS --build-arg $2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate build target
case $BUILD_TARGET in
    development|testing|production|docs|cuda)
        ;;
    *)
        echo -e "${RED}Error: Invalid build target '$BUILD_TARGET'${NC}"
        echo "Valid targets: development, testing, production, docs, cuda"
        exit 1
        ;;
esac

# Set image name and tag
IMAGE_NAME="probneural-operator-lab"
if [[ -n "$REGISTRY" ]]; then
    IMAGE_NAME="$REGISTRY/$IMAGE_NAME"
fi

TAG="${BUILD_TARGET}-py${PYTHON_VERSION}"
if [[ "$BUILD_TARGET" == "cuda" ]]; then
    TAG="${TAG}-cuda${CUDA_VERSION}"
fi
if [[ -n "$TAG_SUFFIX" ]]; then
    TAG="${TAG}-${TAG_SUFFIX}"
fi

FULL_IMAGE_NAME="${IMAGE_NAME}:${TAG}"

echo -e "${BLUE}Building ProbNeural Operator Lab${NC}"
echo -e "${BLUE}=================================${NC}"
echo "Target: $BUILD_TARGET"
echo "Python version: $PYTHON_VERSION"
echo "PyTorch version: $PYTORCH_VERSION"
echo "CUDA version: $CUDA_VERSION"
echo "Image: $FULL_IMAGE_NAME"
echo ""

# Build Docker command
DOCKER_CMD="docker build"

# Add cache options
if [[ -n "$CACHE_FROM" ]]; then
    DOCKER_CMD="$DOCKER_CMD --cache-from $CACHE_FROM"
fi

if [[ -n "$NO_CACHE" ]]; then
    DOCKER_CMD="$DOCKER_CMD $NO_CACHE"
fi

# Add build arguments
DOCKER_CMD="$DOCKER_CMD --build-arg PYTHON_VERSION=$PYTHON_VERSION"
DOCKER_CMD="$DOCKER_CMD --build-arg PYTORCH_VERSION=$PYTORCH_VERSION"
DOCKER_CMD="$DOCKER_CMD --build-arg CUDA_VERSION=$CUDA_VERSION"

if [[ -n "$BUILD_ARGS" ]]; then
    DOCKER_CMD="$DOCKER_CMD $BUILD_ARGS"
fi

# Add target and tags
DOCKER_CMD="$DOCKER_CMD --target $BUILD_TARGET"
DOCKER_CMD="$DOCKER_CMD --tag $FULL_IMAGE_NAME"

# Add latest tag for production builds
if [[ "$BUILD_TARGET" == "production" && -z "$TAG_SUFFIX" ]]; then
    LATEST_TAG="${IMAGE_NAME}:latest"
    DOCKER_CMD="$DOCKER_CMD --tag $LATEST_TAG"
    echo "Also tagging as: $LATEST_TAG"
fi

# Add context
DOCKER_CMD="$DOCKER_CMD ."

echo -e "${YELLOW}Building Docker image...${NC}"
echo "Command: $DOCKER_CMD"
echo ""

# Execute build
if eval "$DOCKER_CMD"; then
    echo -e "${GREEN}✅ Build successful!${NC}"
else
    echo -e "${RED}❌ Build failed!${NC}"
    exit 1
fi

# Push to registry if requested
if [[ "$PUSH_TO_REGISTRY" == "true" ]]; then
    if [[ -z "$REGISTRY" ]]; then
        echo -e "${RED}Error: --registry must be specified when using --push${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Pushing to registry...${NC}"
    
    if docker push "$FULL_IMAGE_NAME"; then
        echo -e "${GREEN}✅ Push successful!${NC}"
        
        # Also push latest tag if it exists
        if [[ "$BUILD_TARGET" == "production" && -z "$TAG_SUFFIX" ]]; then
            docker push "$LATEST_TAG"
            echo -e "${GREEN}✅ Latest tag push successful!${NC}"
        fi
    else
        echo -e "${RED}❌ Push failed!${NC}"
        exit 1
    fi
fi

# Display image information
echo ""
echo -e "${GREEN}Build Summary${NC}"
echo "============="
docker images "$IMAGE_NAME" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"

echo ""
echo -e "${GREEN}🎉 Build completed successfully!${NC}"
echo ""
echo "To run the container:"
case $BUILD_TARGET in
    development)
        echo "  docker run -it --rm -p 8000:8000 -p 8888:8888 -v \$(pwd):/app $FULL_IMAGE_NAME"
        ;;
    testing)
        echo "  docker run --rm $FULL_IMAGE_NAME"
        ;;
    production)
        echo "  docker run -d --name probneural-prod $FULL_IMAGE_NAME"
        ;;
    docs)
        echo "  docker run -d -p 8080:8080 --name probneural-docs $FULL_IMAGE_NAME"
        ;;
    cuda)
        echo "  docker run -it --rm --gpus all -p 8000:8000 -v \$(pwd):/app $FULL_IMAGE_NAME"
        ;;
esac

echo ""
echo "Or use docker-compose:"
echo "  docker-compose --profile $BUILD_TARGET up"