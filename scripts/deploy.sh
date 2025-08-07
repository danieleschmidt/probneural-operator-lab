#!/bin/bash

set -euo pipefail

# ProbNeural-Operator-Lab Deployment Script
# Supports Docker, Kubernetes, and cloud deployments

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Default values
ENVIRONMENT="development"
PLATFORM="docker"
NAMESPACE="probneural-operator"
IMAGE_TAG="latest"
WORKERS=4
GPU_ENABLED=false
MONITORING=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

show_help() {
    cat << EOF
ProbNeural-Operator-Lab Deployment Script

Usage: $0 [OPTIONS]

OPTIONS:
    -e, --environment ENV    Environment (development|production) [default: development]
    -p, --platform PLATFORM Platform (docker|kubernetes|aws|gcp|azure) [default: docker]
    -n, --namespace NS       Kubernetes namespace [default: probneural-operator]
    -t, --tag TAG           Docker image tag [default: latest]
    -w, --workers N         Number of workers [default: 4]
    -g, --gpu               Enable GPU support
    -m, --monitoring        Enable monitoring stack
    -h, --help             Show this help message

EXAMPLES:
    # Development deployment with Docker
    $0 -e development -p docker

    # Production deployment on Kubernetes with GPU and monitoring
    $0 -e production -p kubernetes -g -m

    # Deploy to AWS with custom image tag
    $0 -e production -p aws -t v1.2.3

EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -p|--platform)
                PLATFORM="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -t|--tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            -w|--workers)
                WORKERS="$2"
                shift 2
                ;;
            -g|--gpu)
                GPU_ENABLED=true
                shift
                ;;
            -m|--monitoring)
                MONITORING=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                error "Unknown option: $1. Use -h for help."
                ;;
        esac
    done
}

check_dependencies() {
    log "Checking dependencies..."
    
    case $PLATFORM in
        docker)
            command -v docker >/dev/null 2>&1 || error "Docker is required but not installed"
            command -v docker-compose >/dev/null 2>&1 || error "Docker Compose is required but not installed"
            ;;
        kubernetes)
            command -v kubectl >/dev/null 2>&1 || error "kubectl is required but not installed"
            command -v helm >/dev/null 2>&1 || warn "Helm is recommended for Kubernetes deployments"
            ;;
        aws)
            command -v aws >/dev/null 2>&1 || error "AWS CLI is required but not installed"
            command -v eksctl >/dev/null 2>&1 || warn "eksctl is recommended for AWS EKS deployments"
            ;;
        gcp)
            command -v gcloud >/dev/null 2>&1 || error "Google Cloud SDK is required but not installed"
            ;;
        azure)
            command -v az >/dev/null 2>&1 || error "Azure CLI is required but not installed"
            ;;
    esac
}

build_image() {
    log "Building Docker image..."
    
    local dockerfile="Dockerfile"
    if [[ "$ENVIRONMENT" == "production" ]]; then
        dockerfile="Dockerfile.production"
    fi
    
    if [[ ! -f "$dockerfile" ]]; then
        dockerfile="Dockerfile"
    fi
    
    local build_args=""
    if [[ "$GPU_ENABLED" == true ]]; then
        build_args="--build-arg GPU_SUPPORT=true"
    fi
    
    docker build $build_args -t probneural-operator:$IMAGE_TAG -f $dockerfile .
    
    log "Docker image built: probneural-operator:$IMAGE_TAG"
}

deploy_docker() {
    log "Deploying with Docker Compose..."
    
    local compose_file="docker-compose.yml"
    if [[ "$ENVIRONMENT" == "production" ]]; then
        compose_file="docker-compose.production.yml"
    fi
    
    # Create environment file
    cat > .env << EOF
ENVIRONMENT=$ENVIRONMENT
WORKERS=$WORKERS
GPU_ENABLED=$GPU_ENABLED
IMAGE_TAG=$IMAGE_TAG
EOF
    
    # Deploy with compose
    docker-compose -f $compose_file up -d
    
    log "Docker deployment complete!"
    log "API available at: http://localhost:8000"
    if [[ "$MONITORING" == true ]]; then
        log "Grafana available at: http://localhost:3000"
        log "Prometheus available at: http://localhost:9090"
    fi
}

deploy_kubernetes() {
    log "Deploying to Kubernetes..."
    
    # Create namespace
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Update image tag in deployment
    sed "s|probneural-operator:latest|probneural-operator:$IMAGE_TAG|g" k8s-deployment.yaml > k8s-deployment-tmp.yaml
    
    # Apply Kubernetes manifests
    kubectl apply -f k8s-deployment-tmp.yaml -n $NAMESPACE
    
    # Clean up temporary file
    rm -f k8s-deployment-tmp.yaml
    
    # Wait for deployment to be ready
    kubectl wait --for=condition=available --timeout=300s deployment/probneural-api -n $NAMESPACE
    
    log "Kubernetes deployment complete!"
    
    # Get service info
    local service_ip=$(kubectl get service probneural-api-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
    if [[ -n "$service_ip" ]]; then
        log "API available at: http://$service_ip"
    else
        log "Use 'kubectl port-forward service/probneural-api-service 8000:80 -n $NAMESPACE' to access the API"
    fi
}

deploy_aws() {
    log "Deploying to AWS..."
    
    # Check if EKS cluster exists
    local cluster_name="probneural-cluster"
    if ! aws eks describe-cluster --name $cluster_name >/dev/null 2>&1; then
        warn "EKS cluster '$cluster_name' not found. Creating cluster..."
        eksctl create cluster --name $cluster_name --nodes 3 --node-type m5.large --nodes-min 1 --nodes-max 6 --with-oidc --ssh-access --ssh-public-key ~/.ssh/id_rsa.pub --managed
    fi
    
    # Update kubeconfig
    aws eks update-kubeconfig --name $cluster_name
    
    # Push image to ECR
    local account_id=$(aws sts get-caller-identity --query Account --output text)
    local region=$(aws configure get region)
    local ecr_repo="$account_id.dkr.ecr.$region.amazonaws.com/probneural-operator"
    
    # Create ECR repository if it doesn't exist
    aws ecr create-repository --repository-name probneural-operator --region $region >/dev/null 2>&1 || true
    
    # Push image
    aws ecr get-login-password --region $region | docker login --username AWS --password-stdin $ecr_repo
    docker tag probneural-operator:$IMAGE_TAG $ecr_repo:$IMAGE_TAG
    docker push $ecr_repo:$IMAGE_TAG
    
    # Update deployment with ECR image
    sed "s|probneural-operator:latest|$ecr_repo:$IMAGE_TAG|g" k8s-deployment.yaml > k8s-deployment-aws.yaml
    
    # Deploy to EKS
    kubectl apply -f k8s-deployment-aws.yaml
    
    # Set up load balancer
    kubectl apply -f - << EOF
apiVersion: v1
kind: Service
metadata:
  name: probneural-lb
  namespace: $NAMESPACE
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: nlb
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: probneural-api
EOF
    
    # Clean up
    rm -f k8s-deployment-aws.yaml
    
    log "AWS deployment complete!"
    log "Load balancer provisioning... This may take a few minutes."
}

deploy_gcp() {
    log "Deploying to Google Cloud Platform..."
    
    local project_id=$(gcloud config get-value project)
    local cluster_name="probneural-cluster"
    local zone="us-central1-a"
    
    # Check if GKE cluster exists
    if ! gcloud container clusters describe $cluster_name --zone $zone >/dev/null 2>&1; then
        warn "GKE cluster '$cluster_name' not found. Creating cluster..."
        gcloud container clusters create $cluster_name \
            --zone $zone \
            --num-nodes 3 \
            --machine-type n1-standard-4 \
            --enable-autoscaling \
            --min-nodes 1 \
            --max-nodes 6 \
            --enable-autorepair \
            --enable-autoupgrade
    fi
    
    # Get cluster credentials
    gcloud container clusters get-credentials $cluster_name --zone $zone
    
    # Push image to GCR
    local gcr_repo="gcr.io/$project_id/probneural-operator"
    docker tag probneural-operator:$IMAGE_TAG $gcr_repo:$IMAGE_TAG
    docker push $gcr_repo:$IMAGE_TAG
    
    # Update deployment with GCR image
    sed "s|probneural-operator:latest|$gcr_repo:$IMAGE_TAG|g" k8s-deployment.yaml > k8s-deployment-gcp.yaml
    
    # Deploy to GKE
    kubectl apply -f k8s-deployment-gcp.yaml
    
    # Clean up
    rm -f k8s-deployment-gcp.yaml
    
    log "GCP deployment complete!"
}

deploy_azure() {
    log "Deploying to Microsoft Azure..."
    
    local resource_group="probneural-rg"
    local cluster_name="probneural-cluster"
    local location="eastus"
    
    # Create resource group
    az group create --name $resource_group --location $location >/dev/null 2>&1 || true
    
    # Check if AKS cluster exists
    if ! az aks show --name $cluster_name --resource-group $resource_group >/dev/null 2>&1; then
        warn "AKS cluster '$cluster_name' not found. Creating cluster..."
        az aks create \
            --resource-group $resource_group \
            --name $cluster_name \
            --node-count 3 \
            --node-vm-size Standard_D2s_v3 \
            --enable-cluster-autoscaler \
            --min-count 1 \
            --max-count 6 \
            --generate-ssh-keys
    fi
    
    # Get cluster credentials
    az aks get-credentials --resource-group $resource_group --name $cluster_name
    
    # Push image to ACR
    local acr_name="probneuralacr"
    az acr create --resource-group $resource_group --name $acr_name --sku Basic >/dev/null 2>&1 || true
    az acr login --name $acr_name
    
    local acr_repo="$acr_name.azurecr.io/probneural-operator"
    docker tag probneural-operator:$IMAGE_TAG $acr_repo:$IMAGE_TAG
    docker push $acr_repo:$IMAGE_TAG
    
    # Update deployment with ACR image
    sed "s|probneural-operator:latest|$acr_repo:$IMAGE_TAG|g" k8s-deployment.yaml > k8s-deployment-azure.yaml
    
    # Deploy to AKS
    kubectl apply -f k8s-deployment-azure.yaml
    
    # Clean up
    rm -f k8s-deployment-azure.yaml
    
    log "Azure deployment complete!"
}

run_health_checks() {
    log "Running health checks..."
    
    case $PLATFORM in
        docker)
            local endpoint="http://localhost:8000/health"
            ;;
        *)
            log "Health check configuration varies by platform. Check your service endpoints manually."
            return
            ;;
    esac
    
    # Wait for service to be ready
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -s -f $endpoint >/dev/null 2>&1; then
            log "✓ Health check passed!"
            return
        fi
        
        warn "Health check failed (attempt $attempt/$max_attempts). Retrying in 10s..."
        sleep 10
        ((attempt++))
    done
    
    error "Health check failed after $max_attempts attempts"
}

cleanup_on_exit() {
    if [[ -f ".env" ]]; then
        rm -f .env
    fi
}

main() {
    trap cleanup_on_exit EXIT
    
    log "🚀 Starting ProbNeural-Operator-Lab deployment..."
    log "Environment: $ENVIRONMENT"
    log "Platform: $PLATFORM"
    log "Image Tag: $IMAGE_TAG"
    
    check_dependencies
    build_image
    
    case $PLATFORM in
        docker)
            deploy_docker
            ;;
        kubernetes)
            deploy_kubernetes
            ;;
        aws)
            deploy_aws
            ;;
        gcp)
            deploy_gcp
            ;;
        azure)
            deploy_azure
            ;;
        *)
            error "Unsupported platform: $PLATFORM"
            ;;
    esac
    
    if [[ "$PLATFORM" == "docker" ]]; then
        run_health_checks
    fi
    
    log "🎉 Deployment completed successfully!"
    log "Platform: $PLATFORM"
    log "Environment: $ENVIRONMENT"
    log "Namespace: $NAMESPACE"
    log "Image: probneural-operator:$IMAGE_TAG"
}

# Parse command line arguments
parse_args "$@"

# Run main deployment
main