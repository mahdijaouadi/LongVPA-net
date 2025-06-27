terraform {
  backend "gcs" {
    bucket = "${var.project_id}-tf-state"
    prefix = "terraform/state"
  }
}