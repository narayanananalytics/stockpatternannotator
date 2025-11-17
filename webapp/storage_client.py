"""Cloud Storage client for models and results."""

import logging
import json
import os
from typing import Optional, Dict, Any
from datetime import timedelta
from google.cloud import storage

logger = logging.getLogger(__name__)


class StorageClient:
    """Client for Cloud Storage operations."""

    def __init__(self):
        """Initialize Storage client."""
        self.bucket_name = os.getenv("GCS_BUCKET", "stockpattern-results")

        try:
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(self.bucket_name)
            logger.info(f"Storage client initialized for bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Storage client: {e}")
            self.storage_client = None
            self.bucket = None

    def upload_model(self, job_id: str, local_path: str) -> Optional[str]:
        """Upload trained model to Cloud Storage."""
        if not self.bucket:
            logger.error("Storage client not available")
            return None

        try:
            blob_path = f"models/{job_id}/model.zip"
            blob = self.bucket.blob(blob_path)
            blob.upload_from_filename(local_path)

            logger.info(f"Uploaded model for job {job_id} to {blob_path}")
            return f"gs://{self.bucket_name}/{blob_path}"

        except Exception as e:
            logger.error(f"Failed to upload model: {e}")
            return None

    def upload_results(self, job_id: str, results_data: Dict[str, Any]) -> Optional[str]:
        """Upload results JSON to Cloud Storage."""
        if not self.bucket:
            return None

        try:
            blob_path = f"results/{job_id}/results.json"
            blob = self.bucket.blob(blob_path)
            blob.upload_from_string(
                json.dumps(results_data, indent=2),
                content_type='application/json'
            )

            logger.info(f"Uploaded results for job {job_id} to {blob_path}")
            return f"gs://{self.bucket_name}/{blob_path}"

        except Exception as e:
            logger.error(f"Failed to upload results: {e}")
            return None

    def upload_backtest_csv(self, job_id: str, local_path: str) -> Optional[str]:
        """Upload backtest CSV to Cloud Storage."""
        if not self.bucket:
            return None

        try:
            blob_path = f"results/{job_id}/backtest.csv"
            blob = self.bucket.blob(blob_path)
            blob.upload_from_filename(local_path)

            logger.info(f"Uploaded backtest CSV for job {job_id}")
            return f"gs://{self.bucket_name}/{blob_path}"

        except Exception as e:
            logger.error(f"Failed to upload backtest CSV: {e}")
            return None

    def get_results(self, gcs_path: str) -> Optional[Dict[str, Any]]:
        """Get results JSON from Cloud Storage."""
        if not self.bucket:
            return None

        try:
            # Extract blob path from gs:// URL
            blob_path = gcs_path.replace(f"gs://{self.bucket_name}/", "")
            blob = self.bucket.blob(blob_path)

            if not blob.exists():
                logger.warning(f"Results not found: {gcs_path}")
                return None

            content = blob.download_as_text()
            return json.loads(content)

        except Exception as e:
            logger.error(f"Failed to get results: {e}")
            return None

    def get_signed_url(self, gcs_path: str, expiration: int = 3600) -> Optional[str]:
        """Generate signed URL for downloading files."""
        if not self.bucket:
            return None

        try:
            blob_path = gcs_path.replace(f"gs://{self.bucket_name}/", "")
            blob = self.bucket.blob(blob_path)

            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(seconds=expiration),
                method="GET"
            )

            return url

        except Exception as e:
            logger.error(f"Failed to generate signed URL: {e}")
            return None

    def delete_results(self, gcs_path: str) -> bool:
        """Delete results from Cloud Storage."""
        if not self.bucket:
            return False

        try:
            blob_path = gcs_path.replace(f"gs://{self.bucket_name}/", "")

            # Delete all files in the job's results directory
            prefix = blob_path.rsplit('/', 1)[0] + '/'
            blobs = self.bucket.list_blobs(prefix=prefix)

            for blob in blobs:
                blob.delete()

            logger.info(f"Deleted results: {gcs_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete results: {e}")
            return False

    def delete_model(self, gcs_path: str) -> bool:
        """Delete model from Cloud Storage."""
        return self.delete_results(gcs_path)  # Same logic
