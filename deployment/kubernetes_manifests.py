"""
Kubernetes deployment manifests generator for probabilistic neural operators.

This module generates production-ready Kubernetes manifests for deploying
the ProbNeural-Operator-Lab at scale with proper resource management,
monitoring, and high availability.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class DeploymentConfig:
    """Configuration for Kubernetes deployment."""
    namespace: str = "probneural-operator"
    app_name: str = "probneural-operator"
    image: str = "probneural-operator:latest"
    replicas: int = 3
    
    # Resource limits
    cpu_request: str = "1000m"
    cpu_limit: str = "4000m"
    memory_request: str = "2Gi"
    memory_limit: str = "8Gi"
    
    # GPU settings
    gpu_enabled: bool = False
    gpu_limit: int = 1
    gpu_type: str = "nvidia.com/gpu"
    
    # Networking
    service_port: int = 8000
    metrics_port: int = 9090
    target_port: int = 8000
    
    # Storage
    storage_enabled: bool = True
    storage_size: str = "10Gi"
    storage_class: str = "fast-ssd"
    
    # Monitoring
    prometheus_enabled: bool = True
    grafana_enabled: bool = True
    
    # Security
    security_context_enabled: bool = True
    run_as_user: int = 1000
    run_as_group: int = 1000
    
    # Environment
    log_level: str = "INFO"
    workers: int = 4
    max_batch_size: int = 64


class KubernetesManifestGenerator:
    """Generate Kubernetes manifests for production deployment."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def generate_namespace(self) -> Dict[str, Any]:
        """Generate namespace manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {
                "name": self.config.namespace,
                "labels": {
                    "name": self.config.namespace,
                    "app": self.config.app_name
                }
            }
        }
    
    def generate_configmap(self) -> Dict[str, Any]:
        """Generate ConfigMap for application configuration."""
        return {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{self.config.app_name}-config",
                "namespace": self.config.namespace
            },
            "data": {
                "LOG_LEVEL": self.config.log_level,
                "WORKERS": str(self.config.workers),
                "MAX_BATCH_SIZE": str(self.config.max_batch_size),
                "PROMETHEUS_ENABLED": "true" if self.config.prometheus_enabled else "false",
                "GPU_ENABLED": "true" if self.config.gpu_enabled else "false"
            }
        }
    
    def generate_secret(self) -> Dict[str, Any]:
        """Generate Secret for sensitive configuration."""
        return {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {
                "name": f"{self.config.app_name}-secret",
                "namespace": self.config.namespace
            },
            "type": "Opaque",
            "data": {
                # Base64 encoded values (in practice, these would be real secrets)
                "JWT_SECRET": "ZGVtb19qd3Rfc2VjcmV0XzEyMw==",  # demo_jwt_secret_123
                "API_KEY": "ZGVtb19hcGlfa2V5XzQ1Ng==",        # demo_api_key_456
                "DB_PASSWORD": "ZGVtb19wYXNzd29yZA=="           # demo_password
            }
        }
    
    def generate_deployment(self) -> Dict[str, Any]:
        """Generate Deployment manifest."""
        container_spec = {
            "name": self.config.app_name,
            "image": self.config.image,
            "imagePullPolicy": "Always",
            "ports": [
                {"containerPort": self.config.target_port, "name": "http"},
                {"containerPort": self.config.metrics_port, "name": "metrics"}
            ],
            "envFrom": [
                {"configMapRef": {"name": f"{self.config.app_name}-config"}},
                {"secretRef": {"name": f"{self.config.app_name}-secret"}}
            ],
            "resources": {
                "requests": {
                    "cpu": self.config.cpu_request,
                    "memory": self.config.memory_request
                },
                "limits": {
                    "cpu": self.config.cpu_limit,
                    "memory": self.config.memory_limit
                }
            },
            "livenessProbe": {
                "httpGet": {
                    "path": "/health",
                    "port": self.config.target_port
                },
                "initialDelaySeconds": 60,
                "periodSeconds": 30,
                "timeoutSeconds": 10,
                "failureThreshold": 3
            },
            "readinessProbe": {
                "httpGet": {
                    "path": "/ready",
                    "port": self.config.target_port
                },
                "initialDelaySeconds": 30,
                "periodSeconds": 10,
                "timeoutSeconds": 5,
                "failureThreshold": 3
            },
            "volumeMounts": []
        }
        
        # Add GPU resources if enabled
        if self.config.gpu_enabled:
            container_spec["resources"]["limits"][self.config.gpu_type] = self.config.gpu_limit
        
        # Add storage volume mount if enabled
        if self.config.storage_enabled:
            container_spec["volumeMounts"].append({
                "name": "model-storage",
                "mountPath": "/app/models"
            })
        
        pod_spec = {
            "containers": [container_spec],
            "volumes": []
        }
        
        # Add security context
        if self.config.security_context_enabled:
            pod_spec["securityContext"] = {
                "runAsUser": self.config.run_as_user,
                "runAsGroup": self.config.run_as_group,
                "fsGroup": self.config.run_as_group
            }
        
        # Add storage volume if enabled
        if self.config.storage_enabled:
            pod_spec["volumes"].append({
                "name": "model-storage",
                "persistentVolumeClaim": {
                    "claimName": f"{self.config.app_name}-storage"
                }
            })
        
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.config.app_name,
                "namespace": self.config.namespace,
                "labels": {
                    "app": self.config.app_name,
                    "version": "v1"
                }
            },
            "spec": {
                "replicas": self.config.replicas,
                "selector": {
                    "matchLabels": {
                        "app": self.config.app_name
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": self.config.app_name,
                            "version": "v1"
                        },
                        "annotations": {
                            "prometheus.io/scrape": "true",
                            "prometheus.io/port": str(self.config.metrics_port),
                            "prometheus.io/path": "/metrics"
                        }
                    },
                    "spec": pod_spec
                },
                "strategy": {
                    "type": "RollingUpdate",
                    "rollingUpdate": {
                        "maxUnavailable": 1,
                        "maxSurge": 1
                    }
                }
            }
        }
    
    def generate_service(self) -> Dict[str, Any]:
        """Generate Service manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": self.config.app_name,
                "namespace": self.config.namespace,
                "labels": {
                    "app": self.config.app_name
                }
            },
            "spec": {
                "selector": {
                    "app": self.config.app_name
                },
                "ports": [
                    {
                        "name": "http",
                        "port": self.config.service_port,
                        "targetPort": self.config.target_port,
                        "protocol": "TCP"
                    },
                    {
                        "name": "metrics",
                        "port": self.config.metrics_port,
                        "targetPort": self.config.metrics_port,
                        "protocol": "TCP"
                    }
                ],
                "type": "ClusterIP"
            }
        }
    
    def generate_hpa(self) -> Dict[str, Any]:
        """Generate HorizontalPodAutoscaler manifest."""
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{self.config.app_name}-hpa",
                "namespace": self.config.namespace
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": self.config.app_name
                },
                "minReplicas": max(1, self.config.replicas // 2),
                "maxReplicas": self.config.replicas * 3,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 70
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 80
                            }
                        }
                    }
                ],
                "behavior": {
                    "scaleUp": {
                        "stabilizationWindowSeconds": 60,
                        "policies": [
                            {
                                "type": "Percent",
                                "value": 50,
                                "periodSeconds": 60
                            }
                        ]
                    },
                    "scaleDown": {
                        "stabilizationWindowSeconds": 300,
                        "policies": [
                            {
                                "type": "Percent",
                                "value": 25,
                                "periodSeconds": 60
                            }
                        ]
                    }
                }
            }
        }
    
    def generate_pvc(self) -> Dict[str, Any]:
        """Generate PersistentVolumeClaim manifest."""
        return {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": f"{self.config.app_name}-storage",
                "namespace": self.config.namespace
            },
            "spec": {
                "accessModes": ["ReadWriteOnce"],
                "storageClassName": self.config.storage_class,
                "resources": {
                    "requests": {
                        "storage": self.config.storage_size
                    }
                }
            }
        }
    
    def generate_network_policy(self) -> Dict[str, Any]:
        """Generate NetworkPolicy for security."""
        return {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": f"{self.config.app_name}-netpol",
                "namespace": self.config.namespace
            },
            "spec": {
                "podSelector": {
                    "matchLabels": {
                        "app": self.config.app_name
                    }
                },
                "policyTypes": ["Ingress", "Egress"],
                "ingress": [
                    {
                        "from": [
                            {
                                "namespaceSelector": {
                                    "matchLabels": {
                                        "name": "monitoring"
                                    }
                                }
                            }
                        ],
                        "ports": [
                            {"protocol": "TCP", "port": self.config.metrics_port}
                        ]
                    },
                    {
                        "from": [],
                        "ports": [
                            {"protocol": "TCP", "port": self.config.target_port}
                        ]
                    }
                ],
                "egress": [
                    {
                        "to": [],
                        "ports": [
                            {"protocol": "TCP", "port": 443},  # HTTPS
                            {"protocol": "TCP", "port": 53},   # DNS
                            {"protocol": "UDP", "port": 53}    # DNS
                        ]
                    }
                ]
            }
        }
    
    def generate_ingress(self, hostname: str = "probneural.example.com") -> Dict[str, Any]:
        """Generate Ingress manifest."""
        return {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": f"{self.config.app_name}-ingress",
                "namespace": self.config.namespace,
                "annotations": {
                    "kubernetes.io/ingress.class": "nginx",
                    "cert-manager.io/cluster-issuer": "letsencrypt-prod",
                    "nginx.ingress.kubernetes.io/rate-limit": "100",
                    "nginx.ingress.kubernetes.io/rate-limit-window": "1m"
                }
            },
            "spec": {
                "tls": [
                    {
                        "hosts": [hostname],
                        "secretName": f"{self.config.app_name}-tls"
                    }
                ],
                "rules": [
                    {
                        "host": hostname,
                        "http": {
                            "paths": [
                                {
                                    "path": "/",
                                    "pathType": "Prefix",
                                    "backend": {
                                        "service": {
                                            "name": self.config.app_name,
                                            "port": {
                                                "number": self.config.service_port
                                            }
                                        }
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        }
    
    def generate_service_monitor(self) -> Dict[str, Any]:
        """Generate ServiceMonitor for Prometheus scraping."""
        return {
            "apiVersion": "monitoring.coreos.com/v1",
            "kind": "ServiceMonitor",
            "metadata": {
                "name": f"{self.config.app_name}-monitor",
                "namespace": self.config.namespace,
                "labels": {
                    "app": self.config.app_name,
                    "prometheus": "kube-prometheus"
                }
            },
            "spec": {
                "selector": {
                    "matchLabels": {
                        "app": self.config.app_name
                    }
                },
                "endpoints": [
                    {
                        "port": "metrics",
                        "interval": "30s",
                        "path": "/metrics"
                    }
                ]
            }
        }
    
    def generate_all_manifests(self, hostname: Optional[str] = None) -> Dict[str, Any]:
        """Generate all Kubernetes manifests."""
        manifests = {
            "namespace": self.generate_namespace(),
            "configmap": self.generate_configmap(),
            "secret": self.generate_secret(),
            "deployment": self.generate_deployment(),
            "service": self.generate_service(),
            "hpa": self.generate_hpa(),
            "network_policy": self.generate_network_policy()
        }
        
        if self.config.storage_enabled:
            manifests["pvc"] = self.generate_pvc()
        
        if hostname:
            manifests["ingress"] = self.generate_ingress(hostname)
        
        if self.config.prometheus_enabled:
            manifests["service_monitor"] = self.generate_service_monitor()
        
        return manifests


class HelmChartGenerator:
    """Generate Helm chart for easier deployment management."""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def generate_chart_yaml(self) -> Dict[str, Any]:
        """Generate Chart.yaml."""
        return {
            "apiVersion": "v2",
            "name": "probneural-operator",
            "description": "A Helm chart for ProbNeural-Operator-Lab",
            "type": "application",
            "version": "0.1.0",
            "appVersion": "1.0.0",
            "keywords": [
                "machine-learning",
                "neural-operators", 
                "uncertainty-quantification",
                "scientific-computing"
            ],
            "maintainers": [
                {
                    "name": "ProbNeural Team",
                    "email": "team@probneural.example.com"
                }
            ]
        }
    
    def generate_values_yaml(self) -> Dict[str, Any]:
        """Generate default values.yaml."""
        return {
            "replicaCount": self.config.replicas,
            "image": {
                "repository": self.config.image.split(":")[0],
                "tag": self.config.image.split(":")[-1] if ":" in self.config.image else "latest",
                "pullPolicy": "Always"
            },
            "service": {
                "type": "ClusterIP",
                "port": self.config.service_port,
                "targetPort": self.config.target_port,
                "metricsPort": self.config.metrics_port
            },
            "ingress": {
                "enabled": False,
                "className": "nginx",
                "annotations": {},
                "hosts": [
                    {
                        "host": "probneural.local",
                        "paths": [
                            {
                                "path": "/",
                                "pathType": "Prefix"
                            }
                        ]
                    }
                ],
                "tls": []
            },
            "resources": {
                "requests": {
                    "cpu": self.config.cpu_request,
                    "memory": self.config.memory_request
                },
                "limits": {
                    "cpu": self.config.cpu_limit,
                    "memory": self.config.memory_limit
                }
            },
            "gpu": {
                "enabled": self.config.gpu_enabled,
                "limit": self.config.gpu_limit,
                "type": self.config.gpu_type
            },
            "autoscaling": {
                "enabled": True,
                "minReplicas": max(1, self.config.replicas // 2),
                "maxReplicas": self.config.replicas * 3,
                "targetCPUUtilizationPercentage": 70,
                "targetMemoryUtilizationPercentage": 80
            },
            "storage": {
                "enabled": self.config.storage_enabled,
                "size": self.config.storage_size,
                "storageClass": self.config.storage_class
            },
            "monitoring": {
                "prometheus": {
                    "enabled": self.config.prometheus_enabled,
                    "port": self.config.metrics_port
                }
            },
            "security": {
                "networkPolicy": {
                    "enabled": True
                },
                "securityContext": {
                    "enabled": self.config.security_context_enabled,
                    "runAsUser": self.config.run_as_user,
                    "runAsGroup": self.config.run_as_group
                }
            },
            "config": {
                "logLevel": self.config.log_level,
                "workers": self.config.workers,
                "maxBatchSize": self.config.max_batch_size
            }
        }


def generate_deployment_package():
    """Generate complete deployment package."""
    print("📦 Generating Kubernetes Deployment Package")
    print("=" * 60)
    
    # Configuration options
    configs = {
        "production": DeploymentConfig(
            replicas=5,
            cpu_request="2000m",
            cpu_limit="8000m",
            memory_request="4Gi",
            memory_limit="16Gi",
            gpu_enabled=True,
            gpu_limit=1,
            storage_size="50Gi",
            workers=8,
            max_batch_size=128
        ),
        "staging": DeploymentConfig(
            replicas=2,
            cpu_request="1000m",
            cpu_limit="4000m",
            memory_request="2Gi",
            memory_limit="8Gi",
            gpu_enabled=False,
            storage_size="20Gi"
        ),
        "development": DeploymentConfig(
            replicas=1,
            cpu_request="500m",
            cpu_limit="2000m",
            memory_request="1Gi",
            memory_limit="4Gi",
            gpu_enabled=False,
            storage_size="10Gi"
        )
    }
    
    deployment_package = {}
    
    for env_name, config in configs.items():
        print(f"\n🏗️  Generating {env_name} manifests...")
        
        generator = KubernetesManifestGenerator(config)
        manifests = generator.generate_all_manifests(
            hostname=f"probneural-{env_name}.example.com"
        )
        
        print(f"  ✓ Generated {len(manifests)} manifest types")
        deployment_package[env_name] = manifests
        
        # Generate Helm chart
        helm_generator = HelmChartGenerator(config)
        helm_chart = {
            "Chart.yaml": helm_generator.generate_chart_yaml(),
            "values.yaml": helm_generator.generate_values_yaml()
        }
        
        deployment_package[f"{env_name}_helm"] = helm_chart
        print(f"  ✓ Generated Helm chart")
    
    # Generate deployment scripts
    deployment_scripts = {
        "deploy.sh": generate_deploy_script(),
        "monitoring-setup.sh": generate_monitoring_script(),
        "cleanup.sh": generate_cleanup_script()
    }
    
    deployment_package["scripts"] = deployment_scripts
    
    print(f"\n📊 Package Summary:")
    print(f"  Environments: {len(configs)}")
    print(f"  Total manifests: {sum(len(v) for k, v in deployment_package.items() if not k.endswith('_helm') and k != 'scripts')}")
    print(f"  Helm charts: {len([k for k in deployment_package.keys() if k.endswith('_helm')])}")
    print(f"  Scripts: {len(deployment_scripts)}")
    
    print(f"\n{'='*60}")
    print("✅ Kubernetes deployment package generated!")
    
    return deployment_package


def generate_deploy_script() -> str:
    """Generate deployment script."""
    return '''#!/bin/bash
set -e

ENV=${1:-development}
NAMESPACE="probneural-operator"

echo "🚀 Deploying ProbNeural-Operator-Lab to $ENV environment"

# Create namespace
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply manifests
echo "📦 Applying Kubernetes manifests..."
kubectl apply -f ${ENV}/ -n $NAMESPACE

# Wait for deployment
echo "⏳ Waiting for deployment to be ready..."
kubectl rollout status deployment/probneural-operator -n $NAMESPACE --timeout=300s

# Verify deployment
echo "✅ Verifying deployment..."
kubectl get pods -n $NAMESPACE -l app=probneural-operator

echo "🎉 Deployment completed successfully!"
echo "Access URL: http://$(kubectl get ingress probneural-operator-ingress -n $NAMESPACE -o jsonpath='{.spec.rules[0].host}')"
'''


def generate_monitoring_script() -> str:
    """Generate monitoring setup script."""
    return '''#!/bin/bash
set -e

NAMESPACE="probneural-operator"

echo "📊 Setting up monitoring for ProbNeural-Operator-Lab"

# Install Prometheus Operator (if not already installed)
kubectl apply -f https://raw.githubusercontent.com/prometheus-operator/prometheus-operator/main/bundle.yaml

# Apply ServiceMonitor
kubectl apply -f monitoring/service-monitor.yaml -n $NAMESPACE

# Install Grafana dashboards
kubectl create configmap probneural-grafana-dashboard \\
    --from-file=monitoring/grafana-dashboard.json \\
    -n monitoring --dry-run=client -o yaml | kubectl apply -f -

echo "✅ Monitoring setup completed!"
echo "Grafana URL: http://grafana.monitoring.svc.cluster.local:3000"
'''


def generate_cleanup_script() -> str:
    """Generate cleanup script."""
    return '''#!/bin/bash
set -e

ENV=${1:-development}
NAMESPACE="probneural-operator"

echo "🧹 Cleaning up ProbNeural-Operator-Lab deployment"

# Delete manifests
kubectl delete -f ${ENV}/ -n $NAMESPACE --ignore-not-found=true

# Delete namespace (optional)
read -p "Delete namespace $NAMESPACE? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    kubectl delete namespace $NAMESPACE --ignore-not-found=true
fi

echo "✅ Cleanup completed!"
'''


def main():
    """Generate deployment package."""
    package = generate_deployment_package()
    
    # In a real implementation, you would write these files to disk
    print("\n💾 Files that would be generated:")
    for env, manifests in package.items():
        if isinstance(manifests, dict):
            for name in manifests.keys():
                print(f"  {env}/{name}.yaml")


if __name__ == "__main__":
    main()