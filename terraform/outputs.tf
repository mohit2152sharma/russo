output "service_account_email" {
  description = "SA email — set as GCP_SERVICE_ACCOUNT GitHub secret"
  value       = google_service_account.russo_ci.email
}

output "wif_provider" {
  description = "WIF provider resource name — set as GCP_WIF_PROVIDER GitHub secret"
  value       = google_iam_workload_identity_pool_provider.russo_ci.name
}

output "github_secrets_summary" {
  description = "GitHub secrets to configure"
  value       = <<-EOT
    Add these GitHub Actions secrets (Settings > Secrets > Actions):

      GCP_WIF_PROVIDER     = ${google_iam_workload_identity_pool_provider.russo_ci.name}
      GCP_SERVICE_ACCOUNT  = ${google_service_account.russo_ci.email}
      GOOGLE_CLOUD_PROJECT = ${var.project_id}
  EOT
}
