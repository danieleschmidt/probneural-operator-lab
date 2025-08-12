# Terraform configuration for ProbNeural Operator Lab on AWS
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
  
  backend "s3" {
    bucket = "probneural-terraform-state"
    key    = "terraform.tfstate"
    region = "us-west-2"
  }
}

provider "aws" {
  region = var.aws_region
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

# VPC
resource "aws_vpc" "probneural_vpc" {
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name        = "probneural-vpc"
    Environment = var.environment
  }
}

# Internet Gateway
resource "aws_internet_gateway" "probneural_igw" {
  vpc_id = aws_vpc.probneural_vpc.id
  
  tags = {
    Name        = "probneural-igw"
    Environment = var.environment
  }
}

# Public Subnets
resource "aws_subnet" "public_subnets" {
  count = 2
  
  vpc_id                  = aws_vpc.probneural_vpc.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  
  tags = {
    Name        = "probneural-public-subnet-${count.index + 1}"
    Environment = var.environment
  }
}

# Private Subnets
resource "aws_subnet" "private_subnets" {
  count = 2
  
  vpc_id            = aws_vpc.probneural_vpc.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 2)
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  tags = {
    Name        = "probneural-private-subnet-${count.index + 1}"
    Environment = var.environment
  }
}

# Route Tables
resource "aws_route_table" "public_rt" {
  vpc_id = aws_vpc.probneural_vpc.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.probneural_igw.id
  }
  
  tags = {
    Name        = "probneural-public-rt"
    Environment = var.environment
  }
}

resource "aws_route_table_association" "public_rta" {
  count = 2
  
  subnet_id      = aws_subnet.public_subnets[count.index].id
  route_table_id = aws_route_table.public_rt.id
}

# NAT Gateway
resource "aws_eip" "nat_eip" {
  domain = "vpc"
  
  tags = {
    Name        = "probneural-nat-eip"
    Environment = var.environment
  }
}

resource "aws_nat_gateway" "nat_gateway" {
  allocation_id = aws_eip.nat_eip.id
  subnet_id     = aws_subnet.public_subnets[0].id
  
  tags = {
    Name        = "probneural-nat-gateway"
    Environment = var.environment
  }
  
  depends_on = [aws_internet_gateway.probneural_igw]
}

# Private Route Table
resource "aws_route_table" "private_rt" {
  vpc_id = aws_vpc.probneural_vpc.id
  
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat_gateway.id
  }
  
  tags = {
    Name        = "probneural-private-rt"
    Environment = var.environment
  }
}

resource "aws_route_table_association" "private_rta" {
  count = 2
  
  subnet_id      = aws_subnet.private_subnets[count.index].id
  route_table_id = aws_route_table.private_rt.id
}

# Security Groups
resource "aws_security_group" "eks_cluster_sg" {
  name_prefix = "probneural-eks-cluster-"
  vpc_id      = aws_vpc.probneural_vpc.id
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name        = "probneural-eks-cluster-sg"
    Environment = var.environment
  }
}

resource "aws_security_group" "eks_nodes_sg" {
  name_prefix = "probneural-eks-nodes-"
  vpc_id      = aws_vpc.probneural_vpc.id
  
  ingress {
    from_port = 0
    to_port   = 65535
    protocol  = "tcp"
    self      = true
  }
  
  ingress {
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_cluster_sg.id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name        = "probneural-eks-nodes-sg"
    Environment = var.environment
  }
}

# IAM Roles
resource "aws_iam_role" "eks_cluster_role" {
  name = "probneural-eks-cluster-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "eks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
  role       = aws_iam_role.eks_cluster_role.name
}

resource "aws_iam_role" "eks_nodes_role" {
  name = "probneural-eks-nodes-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "eks_worker_node_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
  role       = aws_iam_role.eks_nodes_role.name
}

resource "aws_iam_role_policy_attachment" "eks_cni_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
  role       = aws_iam_role.eks_nodes_role.name
}

resource "aws_iam_role_policy_attachment" "eks_container_registry_policy" {
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
  role       = aws_iam_role.eks_nodes_role.name
}

# EKS Cluster
resource "aws_eks_cluster" "probneural_cluster" {
  name     = "probneural-cluster"
  role_arn = aws_iam_role.eks_cluster_role.arn
  version  = var.kubernetes_version
  
  vpc_config {
    subnet_ids              = concat(aws_subnet.public_subnets[*].id, aws_subnet.private_subnets[*].id)
    endpoint_private_access = true
    endpoint_public_access  = true
    security_group_ids      = [aws_security_group.eks_cluster_sg.id]
  }
  
  enabled_cluster_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_cluster_policy
  ]
  
  tags = {
    Name        = "probneural-cluster"
    Environment = var.environment
  }
}

# EKS Node Group - GPU Instances
resource "aws_eks_node_group" "gpu_nodes" {
  cluster_name    = aws_eks_cluster.probneural_cluster.name
  node_group_name = "probneural-gpu-nodes"
  node_role_arn   = aws_iam_role.eks_nodes_role.arn
  subnet_ids      = aws_subnet.private_subnets[*].id
  instance_types  = [var.gpu_instance_type]
  
  scaling_config {
    desired_size = var.gpu_node_desired_size
    max_size     = var.gpu_node_max_size
    min_size     = var.gpu_node_min_size
  }
  
  update_config {
    max_unavailable = 1
  }
  
  ami_type       = "AL2_x86_64_GPU"
  capacity_type  = "ON_DEMAND"
  disk_size      = 100
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy
  ]
  
  tags = {
    Name        = "probneural-gpu-nodes"
    Environment = var.environment
  }
}

# EKS Node Group - CPU Instances
resource "aws_eks_node_group" "cpu_nodes" {
  cluster_name    = aws_eks_cluster.probneural_cluster.name
  node_group_name = "probneural-cpu-nodes"
  node_role_arn   = aws_iam_role.eks_nodes_role.arn
  subnet_ids      = aws_subnet.private_subnets[*].id
  instance_types  = [var.cpu_instance_type]
  
  scaling_config {
    desired_size = var.cpu_node_desired_size
    max_size     = var.cpu_node_max_size
    min_size     = var.cpu_node_min_size
  }
  
  update_config {
    max_unavailable = 1
  }
  
  ami_type      = "AL2_x86_64"
  capacity_type = "SPOT"
  disk_size     = 50
  
  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node_policy,
    aws_iam_role_policy_attachment.eks_cni_policy,
    aws_iam_role_policy_attachment.eks_container_registry_policy
  ]
  
  tags = {
    Name        = "probneural-cpu-nodes"
    Environment = var.environment
  }
}

# S3 Bucket for model storage
resource "aws_s3_bucket" "model_storage" {
  bucket = "probneural-model-storage-${random_id.bucket_suffix.hex}"
  
  tags = {
    Name        = "probneural-model-storage"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_versioning" "model_storage_versioning" {
  bucket = aws_s3_bucket.model_storage.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "model_storage_encryption" {
  bucket = aws_s3_bucket.model_storage.id
  
  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        sse_algorithm = "AES256"
      }
    }
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# EFS for shared storage
resource "aws_efs_file_system" "shared_storage" {
  creation_token   = "probneural-efs"
  performance_mode = "generalPurpose"
  throughput_mode  = "provisioned"
  provisioned_throughput_in_mibps = 100
  
  tags = {
    Name        = "probneural-efs"
    Environment = var.environment
  }
}

resource "aws_efs_mount_target" "shared_storage_mount" {
  count = 2
  
  file_system_id  = aws_efs_file_system.shared_storage.id
  subnet_id       = aws_subnet.private_subnets[count.index].id
  security_groups = [aws_security_group.efs_sg.id]
}

resource "aws_security_group" "efs_sg" {
  name_prefix = "probneural-efs-"
  vpc_id      = aws_vpc.probneural_vpc.id
  
  ingress {
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [aws_security_group.eks_nodes_sg.id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name        = "probneural-efs-sg"
    Environment = var.environment
  }
}