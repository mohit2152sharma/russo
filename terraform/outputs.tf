output "service_account_email" {
  description = "SA email — set as GCP_SERVICE_ACCOUNT GitHub secret"
  value       = google_service_account.russo_ci.email
}

output "wif_provider" {
  description = "WIF provider resource name — set as GCP_WIF_PROVIDER GitHub secret"
  value       = google_iam_workload_identity_pool_provider.russo_ci.name
}