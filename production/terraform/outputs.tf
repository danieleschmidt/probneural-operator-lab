# Terraform outputs for ProbNeural Operator Lab

output "cluster_endpoint" {
  description = "EKS cluster endpoint"
  value       = aws_eks_cluster.probneural_cluster.endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = aws_eks_cluster.probneural_cluster.vpc_config[0].cluster_security_group_id
}

output "cluster_iam_role_name" {
  description = "IAM role name associated with EKS cluster"
  value       = aws_iam_role.eks_cluster_role.name
}

output "cluster_iam_role_arn" {
  description = "IAM role ARN associated with EKS cluster"
  value       = aws_iam_role.eks_cluster_role.arn
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = aws_eks_cluster.probneural_cluster.certificate_authority[0].data
}

output "cluster_primary_security_group_id" {
  description = "The cluster primary security group ID created by EKS"
  value       = aws_eks_cluster.probneural_cluster.vpc_config[0].cluster_security_group_id
}

output "vpc_id" {
  description = "VPC ID where the cluster is deployed"
  value       = aws_vpc.probneural_vpc.id
}

output "private_subnets" {
  description = "List of IDs of private subnets"
  value       = aws_subnet.private_subnets[*].id
}

output "public_subnets" {
  description = "List of IDs of public subnets"
  value       = aws_subnet.public_subnets[*].id
}

output "s3_model_bucket_name" {
  description = "Name of the S3 bucket for model storage"
  value       = aws_s3_bucket.model_storage.bucket
}

output "s3_model_bucket_arn" {
  description = "ARN of the S3 bucket for model storage"
  value       = aws_s3_bucket.model_storage.arn
}

output "efs_file_system_id" {
  description = "EFS file system ID for shared storage"
  value       = aws_efs_file_system.shared_storage.id
}

output "efs_mount_target_dns_names" {
  description = "EFS mount target DNS names"
  value       = aws_efs_mount_target.shared_storage_mount[*].dns_name
}

output "kubeconfig_command" {
  description = "Command to configure kubectl"
  value       = "aws eks --region ${var.aws_region} update-kubeconfig --name ${aws_eks_cluster.probneural_cluster.name}"
}

output "cluster_name" {
  description = "Name of the EKS cluster"
  value       = aws_eks_cluster.probneural_cluster.name
}