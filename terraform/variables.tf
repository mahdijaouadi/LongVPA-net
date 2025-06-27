variable "project_id" {
  description = "The ID of the Google Cloud project."
  type        = string
}

variable "region" {
  description = "The Google Cloud region to deploy resources in."
  type        = string
  default     = "us-central1"
}

variable "bucket_name" {
  description = "Name of the GCS bucket for storing data and model artifacts."
  type        = string
}

variable "service_account_id" {
  description = "ID of the service account to be created for the trading bot."
  type        = string
  default     = "trading-bot-sa"
}

variable "cloud_run_service_name" {
  description = "Name of the Cloud Run service for the trading bot."
  type        = string
  default     = "longvpa-trading-bot"
}

variable "cloud_function_name" {
  description = "Name of the Cloud Function for data preprocessing/scheduling."
  type        = string
  default     = "longvpa-data-preprocess"
}

variable "scheduler_job_name" {
  description = "Name of the Cloud Scheduler job."
  type        = string
  default     = "longvpa-daily-trigger"
}

variable "scheduler_job_schedule" {
  description = "Schedule for the Cloud Scheduler job (e.g., '0 9 * * *' for 9 AM daily)."
  type        = string
  default     = "0 9 * * *"
}

variable "artifact_registry_repo_name" {
  description = "Name of the Artifact Registry repository."
  type        = string
  default     = "longvpa-repo"
}

variable "docker_image_name" {
  description = "Name of the Docker image for the trading bot."
  type        = string
  default     = "longvpa-bot-image"
}
