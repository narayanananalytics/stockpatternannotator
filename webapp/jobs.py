"""Cloud Run Jobs management."""

import os
import logging
from typing import Optional
from google.cloud import run_v2
from google.api_core import exceptions

from .models import TrainingJob, JobStatus
from .firestore_client import FirestoreClient
from .storage_client import StorageClient

logger = logging.getLogger(__name__)


class JobManager:
    """Manages Cloud Run Jobs for training."""

    def __init__(self, firestore_client: FirestoreClient, storage_client: StorageClient):
        self.firestore = firestore_client
        self.storage = storage_client

        self.project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        self.region = os.getenv("CLOUD_RUN_REGION", "us-central1")
        self.service_account = os.getenv("CLOUD_RUN_SERVICE_ACCOUNT")

        # Initialize Cloud Run Jobs client
        try:
            self.jobs_client = run_v2.JobsClient()
            self.executions_client = run_v2.ExecutionsClient()
        except Exception as e:
            logger.warning(f"Could not initialize Cloud Run clients: {e}")
            self.jobs_client = None
            self.executions_client = None

    async def submit_job(self, job: TrainingJob):
        """Submit job to Cloud Run Jobs."""
        if not self.jobs_client:
            logger.error("Cloud Run Jobs client not available")
            job.status = JobStatus.FAILED
            job.message = "Cloud Run Jobs not configured"
            self.firestore.update_job(job)
            return

        try:
            # Create or update Cloud Run Job
            job_name = f"training-job-{job.job_id}"
            parent = f"projects/{self.project_id}/locations/{self.region}"

            # Job configuration
            job_config = {
                "template": {
                    "template": {
                        "containers": [{
                            "image": f"gcr.io/{self.project_id}/stockpattern-worker:latest",
                            "env": [
                                {"name": "JOB_ID", "value": job.job_id},
                                {"name": "POLYGON_API_KEY", "value": job.polygon_api_key},
                                {"name": "SYMBOL", "value": job.symbol},
                                {"name": "TIMEFRAME", "value": job.timeframe},
                                {"name": "START_DATE", "value": job.start_date},
                                {"name": "END_DATE", "value": job.end_date},
                                {"name": "TOTAL_TIMESTEPS", "value": str(job.total_timesteps)},
                                {"name": "FORECAST_HORIZON", "value": str(job.forecast_horizon)},
                                {"name": "INITIAL_BALANCE", "value": str(job.initial_balance)},
                                {"name": "TRANSACTION_COST", "value": str(job.transaction_cost)},
                                {"name": "ALLOW_SHORT", "value": str(job.allow_short).lower()},
                                {"name": "COMPUTE_TYPE", "value": job.compute_type.value},
                                {"name": "GOOGLE_CLOUD_PROJECT", "value": self.project_id},
                            ],
                            "resources": {
                                "limits": {
                                    "cpu": "4" if job.compute_type == "cpu" else "8",
                                    "memory": "16Gi" if job.compute_type == "cpu" else "32Gi"
                                }
                            }
                        }],
                        "timeout": "3600s",  # 1 hour max
                        "service_account": self.service_account
                    }
                }
            }

            # Add GPU if requested (note: Cloud Run GPU support is limited)
            if job.compute_type == "gpu":
                logger.warning("GPU requested but Cloud Run GPU support is limited")
                # You would need to use Vertex AI or GCE for proper GPU support

            # Create job execution
            try:
                request = run_v2.CreateJobRequest(
                    parent=parent,
                    job_id=job_name,
                    job=job_config
                )
                operation = self.jobs_client.create_job(request=request)
                cloud_run_job = operation.result()
                logger.info(f"Created Cloud Run Job: {cloud_run_job.name}")
            except exceptions.AlreadyExists:
                # Job already exists, just run it
                logger.info(f"Cloud Run Job {job_name} already exists, running it")

            # Run the job
            execution_request = run_v2.RunJobRequest(
                name=f"{parent}/jobs/{job_name}"
            )
            operation = self.jobs_client.run_job(request=execution_request)
            execution = operation.result()

            # Update job status
            job.status = JobStatus.RUNNING
            job.cloud_run_job_name = job_name
            job.cloud_run_execution_name = execution.name
            job.message = "Training started"
            self.firestore.update_job(job)

            logger.info(f"Started execution for job {job.job_id}")

        except Exception as e:
            logger.error(f"Failed to submit job {job.job_id}: {e}", exc_info=True)
            job.status = JobStatus.FAILED
            job.message = f"Failed to start: {str(e)}"
            self.firestore.update_job(job)

    async def check_job_status(self, job: TrainingJob) -> JobStatus:
        """Check status of running job."""
        if not job.cloud_run_execution_name or not self.executions_client:
            return job.status

        try:
            execution = self.executions_client.get_execution(
                name=job.cloud_run_execution_name
            )

            # Map Cloud Run status to our status
            if execution.status.state == run_v2.Execution.State.RUNNING:
                return JobStatus.RUNNING
            elif execution.status.state == run_v2.Execution.State.SUCCEEDED:
                return JobStatus.COMPLETED
            elif execution.status.state == run_v2.Execution.State.FAILED:
                return JobStatus.FAILED
            elif execution.status.state == run_v2.Execution.State.CANCELLED:
                return JobStatus.CANCELLED
            else:
                return JobStatus.PENDING

        except Exception as e:
            logger.error(f"Failed to check job status: {e}")
            return job.status

    async def cancel_job(self, job: TrainingJob):
        """Cancel a running job."""
        if not job.cloud_run_execution_name or not self.executions_client:
            logger.warning(f"Cannot cancel job {job.job_id}: no execution name")
            return

        try:
            self.executions_client.cancel_execution(
                name=job.cloud_run_execution_name
            )

            job.status = JobStatus.CANCELLED
            job.message = "Cancelled by user"
            self.firestore.update_job(job)

            logger.info(f"Cancelled job {job.job_id}")

        except Exception as e:
            logger.error(f"Failed to cancel job {job.job_id}: {e}")
