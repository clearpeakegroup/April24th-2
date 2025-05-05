from fastapi import APIRouter, HTTPException, status, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import ValidationError
from uuid import uuid4, UUID
from .job_models import JobRequest, JobResponse, JobPatchRequest, AccountResponse, HealthResponse
from .job_queue import enqueue_job, subscribe_progress, redis_cli
from .job_cache import JOB_CACHE
import time
import random

router = APIRouter()

# Endpoints will be implemented here in the next steps. 

# --- Job Submission Endpoints ---
@router.post("/jobs/train", response_model=JobResponse)
@router.post("/jobs/forward", response_model=JobResponse)
@router.post("/jobs/live", response_model=JobResponse)
async def submit_job(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
        job_req = JobRequest(**data)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=e.errors())
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    job_id = str(uuid4())
    job = {
        "id": job_id,
        "type": job_req.type,
        "params": job_req.params,
        "status": "pending",
        "percent_complete": 0.0,
        "result": None,
        "error": None
    }
    JOB_CACHE.set(job_id, job)
    enqueue_job(job)
    return JobResponse(id=job_id, type=job_req.type, status="pending", percent_complete=0.0, params=job_req.params)

# --- Job Control Endpoint ---
@router.patch("/jobs/{id}", response_model=JobResponse)
def patch_job(id: str, patch: JobPatchRequest):
    job = JOB_CACHE.get(id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if patch.action == "pause":
        job["status"] = "paused"
    elif patch.action == "resume":
        job["status"] = "running"
    elif patch.action == "cancel":
        job["status"] = "cancelled"
    else:
        raise HTTPException(status_code=400, detail="Invalid action")
    JOB_CACHE.set(id, job)
    return JobResponse(**job)

# --- Job Metadata Endpoint ---
@router.get("/jobs/{id}", response_model=JobResponse)
def get_job(id: str):
    job = JOB_CACHE.get(id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(**job)

# --- Job Progress Stream (SSE) ---
@router.get("/jobs/{id}/stream")
def stream_job_progress(id: str):
    def event_stream():
        for update in subscribe_progress(id):
            yield f"data: {update}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# --- Mock Account Endpoint ---
@router.get("/account", response_model=AccountResponse)
def get_account():
    # Return mock account data
    return AccountResponse(margin=100000.0, buying_power=200000.0, pnl=random.uniform(-1000, 1000))

# --- Health Endpoint ---
@router.get("/health", response_model=HealthResponse)
def health():
    try:
        redis_cli.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    return HealthResponse(status="ok", redis=redis_ok) 