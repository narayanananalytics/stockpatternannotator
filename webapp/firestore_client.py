"""Firestore client for job persistence."""

import logging
from typing import List, Optional
from datetime import datetime
from google.cloud import firestore

from .models import TrainingJob, JobStatus

logger = logging.getLogger(__name__)


class FirestoreClient:
    """Client for Firestore operations."""

    def __init__(self):
        """Initialize Firestore client."""
        try:
            self.db = firestore.Client()
            self.jobs_collection = self.db.collection('training_jobs')
            logger.info("Firestore client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Firestore: {e}")
            self.db = None
            self.jobs_collection = None

    def create_job(self, job: TrainingJob) -> bool:
        """Create a new job in Firestore."""
        if not self.jobs_collection:
            logger.error("Firestore not available")
            return False

        try:
            doc_ref = self.jobs_collection.document(job.job_id)
            doc_ref.set(job.to_dict())
            logger.info(f"Created job {job.job_id} in Firestore")
            return True
        except Exception as e:
            logger.error(f"Failed to create job in Firestore: {e}")
            return False

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        """Get job by ID."""
        if not self.jobs_collection:
            return None

        try:
            doc = self.jobs_collection.document(job_id).get()
            if doc.exists:
                return TrainingJob.from_dict(doc.to_dict())
            return None
        except Exception as e:
            logger.error(f"Failed to get job {job_id}: {e}")
            return None

    def update_job(self, job: TrainingJob) -> bool:
        """Update job in Firestore."""
        if not self.jobs_collection:
            return False

        try:
            job.updated_at = datetime.utcnow()
            doc_ref = self.jobs_collection.document(job.job_id)
            doc_ref.update(job.to_dict())
            logger.info(f"Updated job {job.job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to update job {job.job_id}: {e}")
            return False

    def delete_job(self, job_id: str) -> bool:
        """Delete job from Firestore."""
        if not self.jobs_collection:
            return False

        try:
            self.jobs_collection.document(job_id).delete()
            logger.info(f"Deleted job {job_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete job {job_id}: {e}")
            return False

    def get_user_jobs(self, user_email: str, limit: int = 20) -> List[TrainingJob]:
        """Get jobs for a user."""
        if not self.jobs_collection:
            return []

        try:
            query = (
                self.jobs_collection
                .where('user_email', '==', user_email)
                .order_by('created_at', direction=firestore.Query.DESCENDING)
                .limit(limit)
            )

            jobs = []
            for doc in query.stream():
                try:
                    job = TrainingJob.from_dict(doc.to_dict())
                    jobs.append(job)
                except Exception as e:
                    logger.error(f"Failed to parse job {doc.id}: {e}")

            return jobs

        except Exception as e:
            logger.error(f"Failed to get user jobs: {e}")
            return []

    def update_job_progress(
        self,
        job_id: str,
        progress: float,
        message: Optional[str] = None
    ) -> bool:
        """Update job progress."""
        if not self.jobs_collection:
            return False

        try:
            update_data = {
                'progress': progress,
                'updated_at': datetime.utcnow().isoformat()
            }
            if message:
                update_data['message'] = message

            self.jobs_collection.document(job_id).update(update_data)
            return True

        except Exception as e:
            logger.error(f"Failed to update progress for job {job_id}: {e}")
            return False

    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        message: Optional[str] = None
    ) -> bool:
        """Update job status."""
        if not self.jobs_collection:
            return False

        try:
            update_data = {
                'status': status.value,
                'updated_at': datetime.utcnow().isoformat()
            }

            if status == JobStatus.RUNNING and message is None:
                update_data['started_at'] = datetime.utcnow().isoformat()
            elif status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                update_data['completed_at'] = datetime.utcnow().isoformat()

            if message:
                update_data['message'] = message

            self.jobs_collection.document(job_id).update(update_data)
            return True

        except Exception as e:
            logger.error(f"Failed to update status for job {job_id}: {e}")
            return False

    def update_job_results(
        self,
        job_id: str,
        model_path: str,
        results_path: str,
        metrics: dict
    ) -> bool:
        """Update job with results."""
        if not self.jobs_collection:
            return False

        try:
            update_data = {
                'model_path': model_path,
                'results_path': results_path,
                'metrics': metrics,
                'status': JobStatus.COMPLETED.value,
                'progress': 100.0,
                'completed_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }

            self.jobs_collection.document(job_id).update(update_data)
            logger.info(f"Updated results for job {job_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update results for job {job_id}: {e}")
            return False
