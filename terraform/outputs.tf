output "project_id" {
  description = "The ID of the Google Cloud project."
  value       = var.project_id
}

output "region" {
  description = "The Google Cloud region."
  value       = var.region
}

output "gcs_bucket_url" {
  description = "URL of the GCS bucket."
  value       = google_storage_bucket.data_bucket.self_link
}

output "service_account_email" {
  description = "Email of the service account."
  value       = google_service_account.trading_bot_sa.email
}

output "cloud_run_service_url" {
  description = "URL of the Cloud Run service."
  value       = google_cloud_run_v2_service.trading_bot_service.uri
}

output "cloud_function_url" {
  description = "URL of the Cloud Function."
  value       = google_cloudfunctions2_function.data_preprocessing_function.url
}

output "artifact_registry_repo_url" {
  description = "URL of the Artifact Registry repository."
  value       = google_artifact_registry_repository.longvpa_repo.repository_url
}
