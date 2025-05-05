from fastapi import FastAPI, APIRouter, File, UploadFile, WebSocket, Depends, HTTPException, status, Path, Form, BackgroundTasks, Body, Query
import logging
logger = logging.getLogger("finrl-api")
from fastapi.middleware.cors import CORSMiddleware
import logging
logger = logging.getLogger("finrl-api")
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import logging
logger = logging.getLogger("finrl-api")
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
import logging
logger = logging.getLogger("finrl-api")
import uvicorn
import os
import jwt
import io
import numpy as np
import asyncio
from backend.models import SessionLocal, Order, OrderStatus, User, AuditLog, StrategyConfig, Agent, AgentStatus, Fill, FillStatus, EquityCurve, ModelArtifact
import logging
logger = logging.getLogger("finrl-api")
from backend.models.tick_event import TickEvent
import logging
logger = logging.getLogger("finrl-api")
from backend.data_ingestion.historical_loader import load_zstd_dbn, batch_ingest_to_db
import logging
logger = logging.getLogger("finrl-api")
from backend.envs.multi_asset_env import MultiAssetTradingEnv
import logging
logger = logging.getLogger("finrl-api")
from backend.middleware.tasks import retrain_model, run_backtest, run_forwardtest, run_live_execution, batch_ingest_task
import logging
logger = logging.getLogger("finrl-api")
from celery.result import AsyncResult
import logging
logger = logging.getLogger("finrl-api")
from backend.middleware.celery_app import celery_app
import logging
logger = logging.getLogger("finrl-api")
import uuid
from prometheus_fastapi_instrumentator import Instrumentator
import logging
logger = logging.getLogger("finrl-api")
import logging
import tempfile
import pandas as pd
from sqlalchemy import func
import logging
logger = logging.getLogger("finrl-api")
from backend.execution.execution_service import ExecutionService
import logging
logger = logging.getLogger("finrl-api")
from backend.execution.risk_manager import RiskManager
import logging
logger = logging.getLogger("finrl-api")
import threading
import time
from sqlalchemy.orm import Session
import logging
logger = logging.getLogger("finrl-api")
import json
from pydantic import BaseModel, Field, field_validator
import logging
logger = logging.getLogger("finrl-api")
import datetime
from slowapi import Limiter
import logging
logger = logging.getLogger("finrl-api")
from slowapi.util import get_remote_address
import logging
logger = logging.getLogger("finrl-api")
from fastapi import Request
import logging
logger = logging.getLogger("finrl-api")
from slowapi.errors import RateLimitExceeded
import logging
logger = logging.getLogger("finrl-api")
from fastapi.responses import PlainTextResponse
import logging
logger = logging.getLogger("finrl-api")
from fastapi.exception_handlers import RequestValidationError
import logging
logger = logging.getLogger("finrl-api")
import hashlib
import redis
from backend.services.pipeline_orchestrator import PipelineOrchestrator
import logging
logger = logging.getLogger("finrl-api")
from backend.services.adaptive_trade_router import AdaptiveTradeRouter, RouterOutput
import logging
logger = logging.getLogger("finrl-api")
import queue
from backend.services.tick_history_service import get_ticks
import logging
logger = logging.getLogger("finrl-api")
from backend.orchestrator.orchestrator import SuperLiquidAvalancheOrchestrator
import logging
logger = logging.getLogger("finrl-api")
import gradio as gr
from frontend.gradio_app import demo as gradio_app # Import 'demo', rename to gradio_app
import logging
logger = logging.getLogger("finrl-api")

print("--- DEBUG: api.py STARTING IMPORT (Restored) --- ")

RL_JOBS = {}

# OAuth2/JWT stub
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# Harden secrets
SECRET_KEY = os.environ.get("JWT_SECRET_KEY")
REFRESH_SECRET_KEY = os.environ.get("JWT_REFRESH_SECRET_KEY")
if not SECRET_KEY or not REFRESH_SECRET_KEY:
    raise RuntimeError("JWT secrets must be set in environment variables!")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

print("--- DEBUG: Defining router --- ")
router = APIRouter()
print("--- DEBUG: router defined --- ")

# --- Define FastAPI App FIRST ---
print("--- DEBUG: Defining app (Restored) --- ")
app = FastAPI(title="FinRL API Gateway") # Restore original title
print("--- DEBUG: app defined (Restored) --- ")

# --- Mount Gradio directly on app --- 
print("--- DEBUG: Mounting Gradio App Directly on App (Restored) --- ")
app = gr.mount_gradio_app(app, gradio_app, path="/gradio") # Mount directly on app at / --- TEMPORARILY COMMENTED OUT FOR DEBUGGING
print("--- DEBUG: Gradio App Mount Temporarily Disabled --- ")

# --- Define Dependencies FIRST ---
print("--- DEBUG: Defining require_role --- ")
def require_role(required_role: str):
    def role_checker(user=Depends(get_current_user)):
        # Ensure user_utils is imported if needed for revoke checks later
        # from backend.services import user_utils
        # Example check (adjust based on your actual User model/payload):
        # --- TODO: This logic needs actual user role check --- 
        # if user.get("role") not in ["admin", "trader", "viewer"] or user["role"] != required_role:
        #     raise HTTPException(status_code=403, detail="Insufficient permissions")
        # Optional: Add token revocation check here if needed
        # if user_utils.is_token_revoked(...):
        #    raise HTTPException(status_code=401, detail="Token revoked")
        return user # Placeholder - allows any valid token for now
    return role_checker
print("--- DEBUG: require_role defined --- ")

# --- Add Middleware ---
print("--- DEBUG: Adding middleware --- ")
# CORS whitelist
ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",") # Default to * for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

# Brute-force protection: rate limit login and registration
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return PlainTextResponse("Rate limit exceeded", status_code=429)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# File-based logging
logging.basicConfig(filename='finrl-backend.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

@app.middleware("http")
async def log_requests(request, call_next):
    logging.info(f"Request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logging.info(f"Response: {response.status_code} {request.url}")
        return response
    except Exception as e:
        logging.error(f"Error handling request {request.method} {request.url}: {e}")
        raise

print("--- DEBUG: Middleware added --- ")

# --- Startup event ---
@app.on_event("startup")
async def start_orchestrator():
    # asyncio.create_task(orchestrator.run()) # Keep orchestrator commented out for now
    pass

# --- REST Endpoints (Attached to router) ---
# (All endpoints are defined in the router section below)
print("--- DEBUG: Defining endpoints on router --- ")

# Define /jobs back on router
@router.get("/jobs")
def list_jobs(user=Depends(get_current_user)): # Add auth back
    """Placeholder endpoint to list jobs. Currently returns empty list."""
    # TODO: Implement actual job listing logic from DB/Redis
    return []

# Define /account back on router
@router.get("/account")
def get_account_info(user=Depends(get_current_user)): # Add auth back
    """Placeholder endpoint for account info. Returns dummy data."""
    # TODO: Implement actual account info retrieval
    return {"user_id": user.get("id"), "username": user.get("username", "N/A"), "email": "user@example.com", "balance": 100000.00}

# --- Uncomment Training/Testing Endpoints ---
@router.post("/upload")
async def upload_data(file: UploadFile = File(...), user=Depends(get_current_user)):
    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zst") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    # Parse file
    try:
        df = load_zstd_dbn(tmp_path)
    except Exception as e:
        return {"error": f"Failed to parse file: {e}"}
    # Insert into DB
    session = SessionLocal()
    try:
        events = [TickEvent(**row) for row in df.to_dict(orient="records")]
        session.bulk_save_objects(events)
        session.commit()
        count = len(events)
        # log_audit(session, user["id"], "upload_config", file.filename) # TODO: Uncomment audit log
    except Exception as e:
        session.rollback()
        return {"error": f"DB insert failed: {e}"}
    finally:
        session.close()
    os.remove(tmp_path)
    return {"filename": file.filename, "rows": count, "status": "uploaded and ingested"}

@router.post("/train")
# async def train_agent(params: dict): # Temporarily disable auth
async def train_agent(params: dict):
    # Add job_id to params if not present (API should ideally enforce this)
    if "job_id" not in params:
        params["job_id"] = str(uuid.uuid4()) # Generate job_id if missing
    job_id = params["job_id"] 
    
    logger.info(f"Received /train request for job {job_id} with params: {params}")# Debug log
    
    r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    r.set(f"jobs:{job_id}:progress", 0)
    # Pass the whole params dict
    task = retrain_model.apply_async(args=[params])
    return {"status": "training started", "task_id": task.id, "job_id": job_id, "params": params}

@router.post("/backtest")
async def backtest(params: dict, user=Depends(get_current_user)):
    # Add job_id to params if not present
    if "job_id" not in params:
        params["job_id"] = str(uuid.uuid4())
    job_id = params["job_id"] 
    
    r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    r.set(f"jobs:{job_id}:progress", 0)
    # Pass the whole params dict
    task = run_backtest.apply_async(args=[params])
    return {"status": "backtest started", "task_id": task.id, "job_id": job_id, "params": params}

@router.post("/forwardtest")
async def forwardtest(params: dict, user=Depends(get_current_user)):
    # Add job_id to params if not present
    if "job_id" not in params:
        params["job_id"] = str(uuid.uuid4())
    job_id = params["job_id"] 
    
    r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    r.set(f"jobs:{job_id}:progress", 0)
    # Pass the whole params dict
    task = run_forwardtest.apply_async(args=[params])
    return {"status": "forward test started", "task_id": task.id, "job_id": job_id, "params": params}

@router.post("/live")
async def live_exec(params: dict, ):
    # Add job_id to params if not present
    if "job_id" not in params:
        params["job_id"] = str(uuid.uuid4())
    job_id = params["job_id"] 
    
    r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    r.set(f"jobs:{job_id}:progress", 0)
    # Pass the whole params dict
    task = run_live_execution.apply_async(args=[params])
    return {"status": "live execution started", "task_id": task.id, "job_id": job_id, "params": params}

# --- TODO: Restore other endpoints if needed ---
@router.get("/task_status/{task_id}")
async def get_task_status(task_id: str = Path(...)):
    result = AsyncResult(task_id, app=celery_app)
    response = {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.successful() else None
    }
    return response

# --- Include the router --- 
print("--- DEBUG: Including router --- ")
app.include_router(router)
print("--- DEBUG: Router included --- ")

# --- Remove basic /test endpoint used for debugging ---
# @app.get("/test")
# async def read_root():
#     return {"Hello": "FastAPI after Gradio"}

print("--- DEBUG: Restored api.py EOF ---") 