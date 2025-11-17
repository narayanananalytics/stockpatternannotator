"""
Stock Pattern Annotator - Cloud Run Web Application

FastAPI web application for running RL trading simulations in Google Cloud.
Protected by Google Cloud IAP for authentication.
"""

from fastapi import FastAPI, Request, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from datetime import datetime, timedelta
import os
import json
import uuid
from typing import Optional, Dict, Any
import logging

from .jobs import JobManager
from .models import TrainingJob, JobStatus
from .firestore_client import FirestoreClient
from .storage_client import StorageClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Stock Pattern Annotator",
    description="RL Trading Strategy Development Platform",
    version="0.5.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="webapp/static"), name="static")

# Templates
templates = Jinja2Templates(directory="webapp/templates")

# Initialize clients
firestore_client = FirestoreClient()
storage_client = StorageClient()
job_manager = JobManager(firestore_client, storage_client)

# Get user from IAP headers
def get_user_email(request: Request) -> str:
    """Extract user email from IAP headers."""
    # In production with IAP, this header is set by Google
    user_email = request.headers.get("X-Goog-Authenticated-User-Email", "")
    if user_email.startswith("accounts.google.com:"):
        user_email = user_email.replace("accounts.google.com:", "")

    # For local development
    if not user_email:
        user_email = os.getenv("DEV_USER_EMAIL", "dev@example.com")

    return user_email


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with job submission form."""
    user_email = get_user_email(request)

    # Get user's recent jobs
    recent_jobs = firestore_client.get_user_jobs(user_email, limit=5)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "user_email": user_email,
            "recent_jobs": recent_jobs
        }
    )


@app.post("/jobs/create")
async def create_job(
    request: Request,
    background_tasks: BackgroundTasks,
    polygon_api_key: str = Form(...),
    symbol: str = Form(...),
    timeframe: str = Form(...),
    start_date: str = Form(...),
    end_date: str = Form(...),
    total_timesteps: int = Form(100000),
    compute_type: str = Form("cpu"),
    forecast_horizon: int = Form(5),
    initial_balance: float = Form(10000.0),
    transaction_cost: float = Form(0.001),
    allow_short: bool = Form(False)
):
    """Create a new training job."""
    user_email = get_user_email(request)

    # Validate inputs
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        if end <= start:
            raise ValueError("End date must be after start date")

        if total_timesteps < 1000 or total_timesteps > 5000000:
            raise ValueError("Timesteps must be between 1,000 and 5,000,000")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Create job
    job_id = str(uuid.uuid4())

    job = TrainingJob(
        job_id=job_id,
        user_email=user_email,
        polygon_api_key=polygon_api_key,
        symbol=symbol.upper(),
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        total_timesteps=total_timesteps,
        compute_type=compute_type,
        forecast_horizon=forecast_horizon,
        initial_balance=initial_balance,
        transaction_cost=transaction_cost,
        allow_short=allow_short,
        status=JobStatus.PENDING,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    # Save to Firestore
    firestore_client.create_job(job)

    logger.info(f"Created job {job_id} for user {user_email}")

    # Submit to Cloud Run Jobs (async)
    background_tasks.add_task(job_manager.submit_job, job)

    # Redirect to job status page
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
async def get_job(request: Request, job_id: str):
    """Get job status and results."""
    user_email = get_user_email(request)

    # Get job from Firestore
    job = firestore_client.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # Check authorization
    if job.user_email != user_email:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Get results if completed
    results = None
    if job.status == JobStatus.COMPLETED and job.results_path:
        results = storage_client.get_results(job.results_path)

    return templates.TemplateResponse(
        "job.html",
        {
            "request": request,
            "job": job,
            "results": results,
            "user_email": user_email
        }
    )


@app.get("/api/jobs/{job_id}/status")
async def get_job_status(request: Request, job_id: str):
    """Get job status (for polling)."""
    user_email = get_user_email(request)

    job = firestore_client.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.user_email != user_email:
        raise HTTPException(status_code=403, detail="Not authorized")

    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "progress": job.progress,
        "message": job.message,
        "updated_at": job.updated_at.isoformat() if job.updated_at else None
    }


@app.get("/jobs", response_class=HTMLResponse)
async def list_jobs(request: Request, limit: int = 20):
    """List user's jobs."""
    user_email = get_user_email(request)

    jobs = firestore_client.get_user_jobs(user_email, limit=limit)

    return templates.TemplateResponse(
        "jobs.html",
        {
            "request": request,
            "jobs": jobs,
            "user_email": user_email
        }
    )


@app.delete("/jobs/{job_id}")
async def delete_job(request: Request, job_id: str):
    """Delete a job."""
    user_email = get_user_email(request)

    job = firestore_client.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.user_email != user_email:
        raise HTTPException(status_code=403, detail="Not authorized")

    # Delete from Firestore
    firestore_client.delete_job(job_id)

    # Delete from Cloud Storage
    if job.results_path:
        storage_client.delete_results(job.results_path)
    if job.model_path:
        storage_client.delete_model(job.model_path)

    return {"message": "Job deleted"}


@app.get("/api/jobs/{job_id}/download/model")
async def download_model(request: Request, job_id: str):
    """Download trained model."""
    user_email = get_user_email(request)

    job = firestore_client.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.user_email != user_email:
        raise HTTPException(status_code=403, detail="Not authorized")

    if not job.model_path:
        raise HTTPException(status_code=404, detail="Model not available")

    # Generate signed URL
    signed_url = storage_client.get_signed_url(job.model_path, expiration=3600)

    return RedirectResponse(url=signed_url)


@app.get("/api/jobs/{job_id}/download/results")
async def download_results(request: Request, job_id: str):
    """Download backtest results CSV."""
    user_email = get_user_email(request)

    job = firestore_client.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.user_email != user_email:
        raise HTTPException(status_code=403, detail="Not authorized")

    if not job.results_path:
        raise HTTPException(status_code=404, detail="Results not available")

    # Generate signed URL
    signed_url = storage_client.get_signed_url(job.results_path, expiration=3600)

    return RedirectResponse(url=signed_url)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.5.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
