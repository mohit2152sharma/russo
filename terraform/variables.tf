variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "project_number" {
  description = "GCP project number"
  type        = string
}

variable "region" {
  description = "Default GCP region for Vertex AI"
  type        = string
  default     = "us-central1"
}

variable "github_repo" {
  description = "GitHub repository (owner/repo) allowed to authenticate via WIF"
  type        = string
}

