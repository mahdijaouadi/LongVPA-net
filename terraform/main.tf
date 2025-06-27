resource "google_project_service" "compute" {
  project = var.project_id
  service = "compute.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "storage" {
  project = var.project_id
  service = "storage.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "artifactregistry" {
  project = var.project_id
  service = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "run" {
  project = var.project_id
  service = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudbuild" {
  project = var.project_id
  service = "cloudbuild.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudresourcemanager" {
  project = var.project_id
  service = "cloudresourcemanager.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "iam" {
  project = var.project_id
  service = "iam.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudfunctions" {
  project = var.project_id
  service = "cloudfunctions.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "logging" {
  project = var.project_id
  service = "logging.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "monitoring" {
  project = var.project_id
  service = "monitoring.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudscheduler" {
  project = var.project_id
  service = "cloudscheduler.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "pubsub" {
  project = var.project_id
  service = "pubsub.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "sqladmin" {
  project = var.project_id
  service = "sqladmin.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "secretmanager" {
  project = var.project_id
  service = "secretmanager.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "containerregistry" {
  project = var.project_id
  service = "containerregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "dataflow" {
  project = var.project_id
  service = "dataflow.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "dataproc" {
  project = var.project_id
  service = "dataproc.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "bigquery" {
  project = var.project_id
  service = "bigquery.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudkms" {
  project = var.project_id
  service = "cloudkms.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudresourcemanager" {
  project = var.project_id
  service = "cloudresourcemanager.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "servicenetworking" {
  project = var.project_id
  service = "servicenetworking.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "vpcaccess" {
  project = var.project_id
  service = "vpcaccess.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "container" {
  project = var.project_id
  service = "container.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "datastore" {
  project = var.project_id
  service = "datastore.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "firestore" {
  project = var.project_id
  service = "firestore.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "appengine" {
  project = var.project_id
  service = "appengine.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudtasks" {
  project = var.project_id
  service = "cloudtasks.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "eventarc" {
  project = var.project_id
  service = "eventarc.googleapis.com"
  disable_on_destroy = false
}

# GCS Bucket for data and model artifacts
resource "google_storage_bucket" "data_bucket" {
  project       = var.project_id
  name          = var.bucket_name
  location      = var.region
  force_destroy = true # Set to false in production to prevent accidental deletion

  uniform_bucket_level_access = true
}

# Service Account for the trading bot
resource "google_service_account" "trading_bot_sa" {
  project      = var.project_id
  account_id   = var.service_account_id
  display_name = "Service Account for LongVPA Trading Bot"
}

# IAM binding for the service account to access the GCS bucket
resource "google_storage_bucket_iam_member" "data_bucket_iam" {
  bucket = google_storage_bucket.data_bucket.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.trading_bot_sa.email}"
}

# Artifact Registry for Docker images
resource "google_artifact_registry_repository" "longvpa_repo" {
  project       = var.project_id
  location      = var.region
  repository_id = var.artifact_registry_repo_name
  format        = "DOCKER"
  description   = "Docker repository for LongVPA trading bot images"
}

# Cloud Run service for the trading bot
resource "google_cloud_run_v2_service" "trading_bot_service" {
  project  = var.project_id
  name     = var.cloud_run_service_name
  location = var.region

  template {
    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_registry_repo_name}/${var.docker_image_name}:latest"
      resources {
        cpu_idle = true
        memory   = "512Mi"
        cpu      = "1"
      }
    }
    service_account = google_service_account.trading_bot_sa.email
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
}

# Cloud Function for data preprocessing/scheduling
resource "google_cloudfunctions2_function" "data_preprocessing_function" {
  project     = var.project_id
  name        = var.cloud_function_name
  location    = var.region

  build_config {
    runtime     = "python39"
    entry_point = "main"
    source {
      storage_source {
        bucket = google_storage_bucket.data_bucket.name
        object = "source.zip" # This will be the zipped source code of your Cloud Function
      }
    }
  }

  service_config {
    available_memory = "256Mi"
    timeout_seconds  = 300
    service_account_email = google_service_account.trading_bot_sa.email
  }
}

# Cloud Scheduler job to trigger the Cloud Function
resource "google_cloud_scheduler_job" "daily_trigger_job" {
  project    = var.project_id
  name       = var.scheduler_job_name
  region     = var.region
  schedule   = var.scheduler_job_schedule
  time_zone  = "America/New_York" # Or your desired timezone

  http_target {
    http_method = "POST"
    uri         = google_cloudfunctions2_function.data_preprocessing_function.url
    oauth_token {
      service_account_email = google_service_account.trading_bot_sa.email
    }
  }
}
  service = "datastore.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "firestore" {
  project = var.project_id
  service = "firestore.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "appengine" {
  project = var.project_id
  service = "appengine.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudtasks" {
  project = var.project_id
  service = "cloudtasks.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "eventarc" {
  project = var.project_id
  service = "eventarc.googleapis.com"
  disable_on_destroy = false
}

# GCS Bucket for data and model artifacts
resource "google_storage_bucket" "data_bucket" {
  project       = var.project_id
  name          = var.bucket_name
  location      = var.region
  force_destroy = true # Set to false in production to prevent accidental deletion

  uniform_bucket_level_access = true
}

# Service Account for the trading bot
resource "google_service_account" "trading_bot_sa" {
  project      = var.project_id
  account_id   = var.service_account_id
  display_name = "Service Account for LongVPA Trading Bot"
}

# IAM binding for the service account to access the GCS bucket
resource "google_storage_bucket_iam_member" "data_bucket_iam" {
  bucket = google_storage_bucket.data_bucket.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.trading_bot_sa.email}"
}

# Artifact Registry for Docker images
resource "google_artifact_registry_repository" "longvpa_repo" {
  project       = var.project_id
  location      = var.region
  repository_id = var.artifact_registry_repo_name
  format        = "DOCKER"
  description   = "Docker repository for LongVPA trading bot images"
}  service = "container.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "datastore" {
  project = var.project_id
  service = "datastore.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "firestore" {
  project = var.project_id
  service = "firestore.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "appengine" {
  project = var.project_id
  service = "appengine.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudtasks" {
  project = var.project_id
  service = "cloudtasks.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "eventarc" {
  project = var.project_id
  service = "eventarc.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "run_api" {
  project = var.project_id
  service = "run.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudfunctions_api" {
  project = var.project_id
  service = "cloudfunctions.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudbuild_api" {
  project = var.project_id
  service = "cloudbuild.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "artifactregistry_api" {
  project = var.project_id
  service = "artifactregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "compute_api" {
  project = var.project_id
  service = "compute.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "storage_api" {
  project = var.project_id
  service = "storage.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "iam_api" {
  project = var.project_id
  service = "iam.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "logging_api" {
  project = var.project_id
  service = "logging.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "monitoring_api" {
  project = var.project_id
  service = "monitoring.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudscheduler_api" {
  project = var.project_id
  service = "cloudscheduler.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "pubsub_api" {
  project = var.project_id
  service = "pubsub.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "sqladmin_api" {
  project = var.project_id
  service = "sqladmin.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "secretmanager_api" {
  project = var.project_id
  service = "secretmanager.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "containerregistry_api" {
  project = var.project_id
  service = "containerregistry.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "dataflow_api" {
  project = var.project_id
  service = "dataflow.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "dataproc_api" {
  project = var.project_id
  service = "dataproc.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "bigquery_api" {
  project = var.project_id
  service = "bigquery.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudkms_api" {
  project = var.project_id
  service = "cloudkms.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudresourcemanager_api" {
  project = var.project_id
  service = "cloudresourcemanager.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "servicenetworking_api" {
  project = var.project_id
  service = "servicenetworking.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "vpcaccess_api" {
  project = var.project_id
  service = "vpcaccess.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "container_api" {
  project = var.project_id
  service = "container.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "datastore_api" {
  project = var.project_id
  service = "datastore.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "firestore_api" {
  project = var.project_id
  service = "firestore.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "appengine_api" {
  project = var.project_id
  service = "appengine.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "cloudtasks_api" {
  project = var.project_id
  service = "cloudtasks.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "eventarc_api" {
  project = var.project_id
  service = "eventarc.googleapis.com"
  disable_on_destroy = false
}
