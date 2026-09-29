terraform {
  required_version = ">= 1.5.0"
}

# Starter IaC boundary. Replace these placeholders with your cloud provider's
# Kubernetes/object-storage/database resources for the production submission.
variable "environment" { default = "dev" }

output "environment" {
  value = var.environment
}
