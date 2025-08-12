# Terraform variables for ProbNeural Operator Lab

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "kubernetes_version" {
  description = "Kubernetes version for EKS cluster"
  type        = string
  default     = "1.28"
}

variable "gpu_instance_type" {
  description = "EC2 instance type for GPU nodes"
  type        = string
  default     = "p3.2xlarge"
}

variable "cpu_instance_type" {
  description = "EC2 instance type for CPU nodes"
  type        = string
  default     = "c5.2xlarge"
}

variable "gpu_node_desired_size" {
  description = "Desired number of GPU nodes"
  type        = number
  default     = 2
}

variable "gpu_node_max_size" {
  description = "Maximum number of GPU nodes"
  type        = number
  default     = 5
}

variable "gpu_node_min_size" {
  description = "Minimum number of GPU nodes"
  type        = number
  default     = 1
}

variable "cpu_node_desired_size" {
  description = "Desired number of CPU nodes"
  type        = number
  default     = 3
}

variable "cpu_node_max_size" {
  description = "Maximum number of CPU nodes"
  type        = number
  default     = 10
}

variable "cpu_node_min_size" {
  description = "Minimum number of CPU nodes"
  type        = number
  default     = 1
}

variable "enable_monitoring" {
  description = "Enable monitoring stack"
  type        = bool
  default     = true
}

variable "enable_logging" {
  description = "Enable centralized logging"
  type        = bool
  default     = true
}

variable "backup_retention_days" {
  description = "Number of days to retain backups"
  type        = number
  default     = 30
}