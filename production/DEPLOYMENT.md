# ProbNeural Operator Lab - Production Deployment Guide

This guide describes how to deploy the ProbNeural Operator Lab framework to production environments using modern cloud-native technologies.

## Architecture Overview

The production deployment includes:

- **Kubernetes (EKS)**: Container orchestration with GPU support
- **Terraform**: Infrastructure as Code for AWS resources
- **Docker**: Containerized applications
- **Prometheus + Grafana**: Monitoring and visualization
- **ELK Stack**: Centralized logging
- **S3 + EFS**: Persistent storage solutions

## Prerequisites

### Required Tools

- [AWS CLI](https://aws.amazon.com/cli/) (v2.0+)
- [kubectl](https://kubernetes.io/docs/tasks/tools/) (v1.28+)
- [Terraform](https://www.terraform.io/) (v1.0+)
- [Docker](https://www.docker.com/) (v20.0+)
- [Helm](https://helm.sh/) (v3.0+)

### AWS Setup

1. Configure AWS credentials:
```bash
aws configure
```

2. Create S3 bucket for Terraform state:
```bash
aws s3 mb s3://probneural-terraform-state --region us-west-2
```

3. Create ECR repository for Docker images:
```bash
aws ecr create-repository --repository-name probneural-operator --region us-west-2
```

## Quick Start

### One-Click Deployment

Deploy everything with a single command:

```bash
cd production/
export ECR_REGISTRY="123456789012.dkr.ecr.us-west-2.amazonaws.com"
export IMAGE_TAG="v1.0.0"
./scripts/deploy.sh all
```

### Step-by-Step Deployment

1. **Deploy Infrastructure**:
```bash
./scripts/deploy.sh infra
```

2. **Build and Push Images**:
```bash
export ECR_REGISTRY="123456789012.dkr.ecr.us-west-2.amazonaws.com"
./scripts/deploy.sh build
```

3. **Deploy to Kubernetes**:
```bash
./scripts/deploy.sh k8s
```

4. **Setup Monitoring**:
```bash
./scripts/deploy.sh monitoring
```

5. **Setup Logging**:
```bash
./scripts/deploy.sh logging
```

6. **Run Health Checks**:
```bash
./scripts/deploy.sh health
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Deployment environment | `production` |
| `AWS_REGION` | AWS region | `us-west-2` |
| `CLUSTER_NAME` | EKS cluster name | `probneural-cluster` |
| `ECR_REGISTRY` | ECR registry URL | - |
| `IMAGE_TAG` | Docker image tag | `latest` |

### Terraform Variables

Edit `terraform/variables.tf` to customize:

- Instance types (GPU/CPU nodes)
- Node scaling configuration
- Storage settings
- Network configuration

### Kubernetes Configuration

Modify `k8s/deployment.yaml` for:

- Resource limits and requests
- Environment variables
- Volume mounts
- Service configuration

## Monitoring and Observability

### Prometheus Metrics

Access Prometheus at: `http://prometheus.your-domain.com:9090`

Key metrics monitored:
- Model training progress and loss
- GPU utilization and memory
- System resource usage
- Application performance

### Grafana Dashboards

Access Grafana at: `http://grafana.your-domain.com:3000`

Pre-configured dashboards:
- ProbNeural Operator Overview
- GPU Monitoring
- Kubernetes Cluster Health
- Training Job Performance

### Logging

Access Kibana at: `http://kibana.your-domain.com:5601`

Log sources:
- Application logs
- Kubernetes system logs
- Training job logs
- Infrastructure logs

## Storage

### Model Storage (S3)

- **Bucket**: Auto-created by Terraform
- **Versioning**: Enabled
- **Encryption**: AES-256
- **Lifecycle**: 30-day retention

### Shared Storage (EFS)

- **Performance**: General Purpose
- **Throughput**: Provisioned (100 MiB/s)
- **Mount**: Available to all pods

## Security

### IAM Roles and Policies

- **EKS Cluster Role**: Manages EKS control plane
- **Node Group Role**: Manages worker nodes
- **Service Account Roles**: Pod-level permissions

### Network Security

- **VPC**: Isolated network with public/private subnets
- **Security Groups**: Restrictive ingress/egress rules
- **NAT Gateway**: Outbound internet access for private subnets

### Secrets Management

Store sensitive data in Kubernetes secrets:

```bash
kubectl create secret generic probneural-secrets \
  --from-literal=database-password=your-password \
  --namespace probneural-operator
```

## Scaling

### Horizontal Pod Autoscaling (HPA)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: probneural-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: probneural-operator
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: nvidia.com/gpu
      target:
        type: Utilization
        averageUtilization: 80
```

### Cluster Autoscaling

EKS cluster autoscaling is configured in Terraform with:
- **GPU Nodes**: 1-5 instances (p3.2xlarge)
- **CPU Nodes**: 1-10 instances (c5.2xlarge)

## Backup and Disaster Recovery

### Automated Backups

- **S3 Versioning**: Automatic model version history
- **EFS Backups**: Daily snapshots
- **Database Backups**: Automated PostgreSQL backups

### Recovery Procedures

1. **Infrastructure Recovery**:
```bash
cd terraform/
terraform apply
```

2. **Application Recovery**:
```bash
./scripts/deploy.sh k8s
```

3. **Data Recovery**:
```bash
aws s3 sync s3://backup-bucket s3://probneural-model-storage
```

## Troubleshooting

### Common Issues

1. **Pod Stuck in Pending**:
```bash
kubectl describe pod <pod-name> -n probneural-operator
```
Check for resource constraints or node selector issues.

2. **GPU Not Available**:
```bash
kubectl get nodes -l accelerator=nvidia-tesla-k80
```
Ensure GPU nodes are running and NVIDIA device plugin is deployed.

3. **Storage Issues**:
```bash
kubectl get pv,pvc -n probneural-operator
```
Check persistent volume claims and mount points.

### Logs and Debugging

```bash
# Application logs
kubectl logs deployment/probneural-operator -n probneural-operator -f

# System logs
kubectl logs -n kube-system -l app=nvidia-device-plugin-daemonset

# Events
kubectl get events -n probneural-operator --sort-by='.lastTimestamp'
```

## Cost Optimization

### Resource Right-Sizing

Monitor resource usage and adjust:
- Pod resource requests/limits
- Node instance types
- Storage provisioning

### Spot Instances

Use spot instances for non-critical workloads:
```hcl
capacity_type = "SPOT"
```

### Auto-Shutdown

Implement auto-shutdown for development environments:
```bash
kubectl create cronjob auto-shutdown \
  --image=kubectl:latest \
  --schedule="0 18 * * 1-5" \
  -- kubectl scale deployment probneural-operator --replicas=0
```

## Updates and Rollbacks

### Rolling Updates

```bash
kubectl set image deployment/probneural-operator \
  probneural-operator=new-image:tag \
  -n probneural-operator
```

### Rollbacks

```bash
kubectl rollout undo deployment/probneural-operator -n probneural-operator
```

### Zero-Downtime Deployment

The deployment uses rolling updates with health checks to ensure zero-downtime deployments.

## Support

For deployment issues:
1. Check the troubleshooting section
2. Review application logs
3. Consult monitoring dashboards
4. Contact the development team

---

This production deployment provides a robust, scalable, and secure environment for running the ProbNeural Operator Lab framework at scale.