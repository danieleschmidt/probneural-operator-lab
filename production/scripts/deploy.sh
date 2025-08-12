#!/bin/bash

# Production deployment script for ProbNeural Operator Lab
set -euo pipefail

# Configuration
ENVIRONMENT="${ENVIRONMENT:-production}"
AWS_REGION="${AWS_REGION:-us-west-2}"
CLUSTER_NAME="${CLUSTER_NAME:-probneural-cluster}"
ECR_REGISTRY="${ECR_REGISTRY:-}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[ERROR] $1${NC}" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS] $1${NC}"
}

warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    local missing_tools=()
    
    if ! command -v aws &> /dev/null; then
        missing_tools+=("aws")
    fi
    
    if ! command -v kubectl &> /dev/null; then
        missing_tools+=("kubectl")
    fi
    
    if ! command -v terraform &> /dev/null; then
        missing_tools+=("terraform")
    fi
    
    if ! command -v docker &> /dev/null; then
        missing_tools+=("docker")
    fi
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
    
    success "All prerequisites satisfied"
}

# Deploy infrastructure
deploy_infrastructure() {
    log "Deploying infrastructure with Terraform..."
    
    cd terraform/
    
    # Initialize Terraform
    terraform init -upgrade
    
    # Plan deployment
    terraform plan -var="environment=$ENVIRONMENT" -var="aws_region=$AWS_REGION" -out=tfplan
    
    # Apply deployment
    terraform apply tfplan
    
    # Update kubeconfig
    aws eks --region "$AWS_REGION" update-kubeconfig --name "$CLUSTER_NAME"
    
    cd ..
    success "Infrastructure deployed successfully"
}

# Build and push Docker image
build_and_push_image() {
    log "Building and pushing Docker image..."
    
    if [ -z "$ECR_REGISTRY" ]; then
        warning "ECR_REGISTRY not set, skipping image push"
        return
    fi
    
    # Login to ECR
    aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"
    
    # Build image
    docker build -f docker/Dockerfile.prod -t "probneural-operator:$IMAGE_TAG" ..
    
    # Tag for ECR
    docker tag "probneural-operator:$IMAGE_TAG" "$ECR_REGISTRY/probneural-operator:$IMAGE_TAG"
    
    # Push to ECR
    docker push "$ECR_REGISTRY/probneural-operator:$IMAGE_TAG"
    
    success "Docker image built and pushed"
}

# Deploy to Kubernetes
deploy_to_kubernetes() {
    log "Deploying to Kubernetes..."
    
    # Apply namespace
    kubectl apply -f - <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: probneural-operator
  labels:
    name: probneural-operator
EOF
    
    # Apply NVIDIA device plugin for GPU support
    kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml
    
    # Update image in deployment if ECR registry is set
    if [ -n "$ECR_REGISTRY" ]; then
        sed -i.bak "s|probneural-operator:latest|$ECR_REGISTRY/probneural-operator:$IMAGE_TAG|g" k8s/deployment.yaml
    fi
    
    # Apply Kubernetes manifests
    kubectl apply -f k8s/ -n probneural-operator
    
    # Wait for deployment to be ready
    kubectl rollout status deployment/probneural-operator -n probneural-operator --timeout=600s
    
    success "Kubernetes deployment completed"
}

# Setup monitoring
setup_monitoring() {
    log "Setting up monitoring stack..."
    
    # Add Prometheus Helm repository
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    # Install Prometheus
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --values monitoring/prometheus-values.yaml \
        --wait
    
    # Install Grafana dashboards
    kubectl apply -f monitoring/grafana-dashboards/ -n monitoring
    
    success "Monitoring stack deployed"
}

# Setup logging
setup_logging() {
    log "Setting up logging stack..."
    
    # Add Elastic Helm repository
    helm repo add elastic https://helm.elastic.co
    helm repo update
    
    # Install Elasticsearch
    helm upgrade --install elasticsearch elastic/elasticsearch \
        --namespace logging \
        --create-namespace \
        --values logging/elasticsearch-values.yaml \
        --wait
    
    # Install Kibana
    helm upgrade --install kibana elastic/kibana \
        --namespace logging \
        --values logging/kibana-values.yaml \
        --wait
    
    # Install Fluent Bit
    helm upgrade --install fluent-bit fluent/fluent-bit \
        --namespace logging \
        --values logging/fluentbit-values.yaml \
        --wait
    
    success "Logging stack deployed"
}

# Run health checks
health_check() {
    log "Running health checks..."
    
    # Check if pods are running
    kubectl get pods -n probneural-operator
    
    # Check services
    kubectl get services -n probneural-operator
    
    # Test application endpoint
    local app_endpoint
    app_endpoint=$(kubectl get service probneural-operator-service -n probneural-operator -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')
    
    if [ -n "$app_endpoint" ]; then
        log "Testing application endpoint: $app_endpoint"
        if curl -f -s "http://$app_endpoint/health" > /dev/null; then
            success "Health check passed"
        else
            warning "Health check failed - application may still be starting"
        fi
    else
        warning "Load balancer endpoint not yet available"
    fi
}

# Cleanup function
cleanup() {
    log "Cleaning up temporary files..."
    rm -f terraform/tfplan
    if [ -f k8s/deployment.yaml.bak ]; then
        mv k8s/deployment.yaml.bak k8s/deployment.yaml
    fi
}

# Main deployment function
main() {
    log "Starting ProbNeural Operator Lab deployment..."
    log "Environment: $ENVIRONMENT"
    log "AWS Region: $AWS_REGION"
    log "Cluster Name: $CLUSTER_NAME"
    
    # Trap cleanup
    trap cleanup EXIT
    
    check_prerequisites
    
    case "${1:-all}" in
        infra|infrastructure)
            deploy_infrastructure
            ;;
        build)
            build_and_push_image
            ;;
        k8s|kubernetes)
            deploy_to_kubernetes
            ;;
        monitoring)
            setup_monitoring
            ;;
        logging)
            setup_logging
            ;;
        health)
            health_check
            ;;
        all)
            deploy_infrastructure
            build_and_push_image
            deploy_to_kubernetes
            setup_monitoring
            setup_logging
            health_check
            ;;
        *)
            echo "Usage: $0 {infra|build|k8s|monitoring|logging|health|all}"
            echo ""
            echo "Commands:"
            echo "  infra      - Deploy infrastructure with Terraform"
            echo "  build      - Build and push Docker image"
            echo "  k8s        - Deploy to Kubernetes"
            echo "  monitoring - Setup monitoring stack"
            echo "  logging    - Setup logging stack"
            echo "  health     - Run health checks"
            echo "  all        - Run all deployment steps (default)"
            exit 1
            ;;
    esac
    
    success "Deployment completed successfully!"
}

# Run main function with all arguments
main "$@"