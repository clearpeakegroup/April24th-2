from fastapi import FastAPI, APIRouter, File, UploadFile, WebSocket, Depends, HTTPException, status, Path, Form, BackgroundTasks, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
import uvicorn
import os
import jwt
import io
import numpy as np
import asyncio
from backend.models import SessionLocal, Order, OrderStatus, User, AuditLog, StrategyConfig, Agent, AgentStatus, Fill, FillStatus, EquityCurve, ModelArtifact
from backend.models.tick_event import TickEvent
from backend.data_ingestion.historical_loader import load_zstd_dbn, batch_ingest_to_db
from backend.envs.multi_asset_env import MultiAssetTradingEnv
from backend.middleware.tasks import retrain_model, run_backtest, run_forwardtest, run_live_execution
from celery.result import AsyncResult
from backend.middleware.celery_app import celery_app
import uuid
from prometheus_fastapi_instrumentator import Instrumentator
import logging
import tempfile
import pandas as pd
from sqlalchemy import func
from backend.execution.execution_service import ExecutionService
from backend.execution.risk_manager import RiskManager
import threading
import time
from backend.services import user_utils
from sqlalchemy.orm import Session
import json
from pydantic import BaseModel, ValidationError, validator
import datetime
from slowapi import Limiter
from slowapi.util import get_remote_address
from fastapi import Request
from slowapi.errors import RateLimitExceeded
from fastapi.responses import PlainTextResponse
from fastapi.exception_handlers import RequestValidationError
import hashlib

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

router = APIRouter()

# CORS whitelist
ALLOWED_ORIGINS = os.environ.get("CORS_ORIGINS", "https://your-frontend.com").split(",")
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

@app.middleware("http")
async def add_rate_limit(request: Request, call_next):
    response = await limiter(request, call_next)
    return response

# Brute-force protection: rate limit login and registration
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return PlainTextResponse("Rate limit exceeded", status_code=429)

# --- REST Endpoints ---
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
        log_audit(session, user["id"], "upload_config", file.filename)
    except Exception as e:
        session.rollback()
        return {"error": f"DB insert failed: {e}"}
    finally:
        session.close()
    os.remove(tmp_path)
    return {"filename": file.filename, "rows": count, "status": "uploaded and ingested"}

@router.post("/train")
async def train_agent(params: dict, user=Depends(get_current_user)):
    # Accept agent_type, agent_config, env_config in params
    task = retrain_model.apply_async(args=[params])
    return {"status": "training started", "task_id": task.id, "params": params}

@router.post("/backtest")
async def backtest(params: dict, user=Depends(get_current_user)):
    task = run_backtest.apply_async(args=[params])
    return {"status": "backtest started", "task_id": task.id, "params": params}

@router.post("/forwardtest")
async def forwardtest(params: dict, user=Depends(get_current_user)):
    task = run_forwardtest.apply_async(args=[params])
    return {"status": "forward test started", "task_id": task.id, "params": params}

@router.post("/live")
async def live_exec(params: dict, user=Depends(get_current_user)):
    task = run_live_execution.apply_async(args=[params])
    return {"status": "live execution started", "task_id": task.id, "params": params}

@router.get("/task_status/{task_id}")
async def get_task_status(task_id: str = Path(...)):
    result = AsyncResult(task_id, app=celery_app)
    response = {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.successful() else None
    }
    return response

@router.post("/ingest/batch")
def ingest_batch(
    file: UploadFile = File(...),
    version: int = Form(1),
    user: str = Form(None),
    source: str = Form("api")
):
    """
    Batch ingest tick data from uploaded CSV file.
    """
    df = pd.read_csv(file.file)
    status, message = batch_ingest_to_db(df, version=version, user=user, source=source)
    return {"status": status, "message": message}

@router.post("/rl/job")
def submit_rl_job(
    background_tasks: BackgroundTasks,
    config: dict
):
    """
    Submit an RL training job with the given config.
    """
    job_id = str(uuid.uuid4())
    RL_JOBS[job_id] = {"status": "pending", "result": None}
    def run_job():
        # TODO: Replace with actual RL training logic
        RL_JOBS[job_id]["status"] = "running"
        # Simulate training
        import time; time.sleep(2)
        RL_JOBS[job_id]["status"] = "finished"
        RL_JOBS[job_id]["result"] = {"reward": 123.45}
    background_tasks.add_task(run_job)
    return {"job_id": job_id, "status": "submitted"}

@router.get("/rl/job/{job_id}")
def get_rl_job_status(job_id: str):
    """
    Get the status and result of an RL job.
    """
    job = RL_JOBS.get(job_id)
    if not job:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    return job

execution_service = ExecutionService()
risk_manager = RiskManager()

# For demo: set default limits for all users (should be set per user in real system)
def ensure_user_limits(user_id):
    if user_id not in risk_manager.position_limits:
        risk_manager.set_limits(user_id, {"AAPL": 10, "MSFT": 10, "GOOG": 10}, 100000.0)

@router.post("/orders")
async def place_order(
    symbol: str = Body(...),
    qty: float = Body(...),
    side: str = Body(...),
    order_type: str = Body("market"),
    agent_id: int = Body(None),
    user=Depends(get_current_user)
):
    ensure_user_limits(user["id"])
    try:
        risk_manager.check_order(user["id"], symbol, qty, side)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    session = SessionLocal()
    try:
        order = Order(
            user_id=user["id"],
            agent_id=agent_id,
            symbol=symbol,
            qty=qty,
            side=side,
            order_type=order_type,
            status=OrderStatus.new
        )
        session.add(order)
        session.commit()
        session.refresh(order)
        # Simulate immediate transition to pending
        order.status = OrderStatus.pending
        session.commit()
        # Execute order via broker abstraction
        execution_service.execute_order(order.id)
        session.refresh(order)
        return {"order_id": order.id, "status": order.status.value}
    finally:
        session.close()

@router.get("/orders/{order_id}")
async def get_order(order_id: int, user=Depends(get_current_user)):
    session = SessionLocal()
    try:
        order = session.query(Order).filter_by(id=order_id, user_id=user["id"]).first()
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        return {
            "order_id": order.id,
            "symbol": order.symbol,
            "qty": order.qty,
            "side": order.side,
            "order_type": order.order_type,
            "status": order.status.value,
            "submitted_at": order.submitted_at,
            "filled_at": order.filled_at,
            "cancelled_at": order.cancelled_at,
            "rejected_at": order.rejected_at,
            "broker_order_id": order.broker_order_id,
            "meta": order.meta
        }
    finally:
        session.close()

@router.get("/orders")
async def list_orders(user=Depends(get_current_user)):
    session = SessionLocal()
    try:
        orders = session.query(Order).filter_by(user_id=user["id"]).order_by(Order.submitted_at.desc()).all()
        return [
            {
                "order_id": o.id,
                "symbol": o.symbol,
                "qty": o.qty,
                "side": o.side,
                "order_type": o.order_type,
                "status": o.status.value,
                "submitted_at": o.submitted_at
            } for o in orders
        ]
    finally:
        session.close()

@router.post("/orders/{order_id}/cancel")
async def cancel_order(order_id: int, user=Depends(get_current_user)):
    session = SessionLocal()
    try:
        order = session.query(Order).filter_by(id=order_id, user_id=user["id"]).first()
        if not order:
            raise HTTPException(status_code=404, detail="Order not found")
        if order.status in [OrderStatus.filled, OrderStatus.cancelled, OrderStatus.rejected]:
            raise HTTPException(status_code=400, detail="Order cannot be cancelled")
        order.status = OrderStatus.cancelled
        order.cancelled_at = func.now()
        session.commit()
        return {"order_id": order.id, "status": order.status.value}
    finally:
        session.close()

# --- WebSocket for real-time updates ---
@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    for i in range(100):
        # Simulate real-time tick and PnL
        await ws.send_json({"tick": np.random.randn(), "pnl": np.random.randn()})
        await asyncio.sleep(1)
    await ws.close()

app = FastAPI(title="FinRL API Gateway")

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Set up file-based logging
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

app.include_router(router)

# --- HTTPS self-signed cert for dev (uvicorn CLI) ---
# To run: uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000 --ssl-keyfile=certs/key.pem --ssl-certfile=certs/cert.pem

if __name__ == "__main__":
    # To run with HTTPS locally:
    # uvicorn backend.api:app --host 0.0.0.0 --port 8000 --ssl-keyfile=certs/key.pem --ssl-certfile=certs/cert.pem
    uvicorn.run(
        "backend.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        ssl_keyfile="certs/key.pem",
        ssl_certfile="certs/cert.pem"
    )

live_trading_status = {}

# Simulated live trading loop

def live_trading_loop(user_id, agent_id):
    live_trading_status[(user_id, agent_id)] = "running"
    for i in range(30):  # Simulate 30 cycles
        if live_trading_status.get((user_id, agent_id)) != "running":
            break
        # Simulate market data, order, fill
        # In real system: fetch data, generate signal, place order, handle fill
        time.sleep(1)
    live_trading_status[(user_id, agent_id)] = "stopped"

@router.post("/live/start")
async def start_live_trading(agent_id: int = Body(...), background_tasks: BackgroundTasks = None, user=Depends(get_current_user)):
    key = (user["id"], agent_id)
    if live_trading_status.get(key) == "running":
        return {"status": "already running"}
    if background_tasks is not None:
        background_tasks.add_task(live_trading_loop, user["id"], agent_id)
    else:
        threading.Thread(target=live_trading_loop, args=(user["id"], agent_id), daemon=True).start()
    return {"status": "started"}

@router.post("/live/stop")
async def stop_live_trading(agent_id: int = Body(...), user=Depends(get_current_user)):
    key = (user["id"], agent_id)
    if live_trading_status.get(key) != "running":
        return {"status": "not running"}
    live_trading_status[key] = "stopping"
    return {"status": "stopping"}

def log_audit(session: Session, user_id: int, action: str, details: str = None):
    audit = AuditLog(user_id=user_id, action=action, details=details)
    session.add(audit)
    session.commit()

# User registration
@router.post("/register")
@limiter.limit("3/minute")
def register(username: str = Body(...), email: str = Body(...), password: str = Body(...)):
    session = SessionLocal()
    if session.query(User).filter((User.username == username) | (User.email == email)).first():
        session.close()
        raise HTTPException(status_code=400, detail="Username or email already exists")
    user = User(username=username, email=email, hashed_password=user_utils.hash_password(password), role="viewer")
    session.add(user)
    session.commit()
    log_audit(session, user.id, "register")
    session.close()
    return {"status": "registered"}

# User login
@router.post("/login")
@limiter.limit("5/minute")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    session = SessionLocal()
    user = session.query(User).filter(User.username == form_data.username).first()
    if not user or not user_utils.verify_password(form_data.password, user.hashed_password):
        session.close()
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = user_utils.create_access_token({"id": user.id, "role": user.role})
    refresh_token = user_utils.create_refresh_token({"id": user.id, "role": user.role})
    log_audit(session, user.id, "login")
    session.close()
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}

# Password reset (stub: just sets new password, no email)
@router.post("/password_reset")
def password_reset(username: str = Body(...), new_password: str = Body(...)):
    session = SessionLocal()
    user = session.query(User).filter(User.username == username).first()
    if not user:
        session.close()
        raise HTTPException(status_code=404, detail="User not found")
    user.hashed_password = user_utils.hash_password(new_password)
    session.commit()
    log_audit(session, user.id, "password_reset")
    session.close()
    return {"status": "password reset"}

# JWT refresh
@router.post("/token/refresh")
def refresh_token(refresh_token: str = Body(...)):
    payload = user_utils.decode_token(refresh_token, refresh=True)
    if not payload or user_utils.is_token_revoked(refresh_token):
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    access_token = user_utils.create_access_token({"id": payload["id"], "role": payload["role"]})
    return {"access_token": access_token, "token_type": "bearer"}

# JWT revoke
@router.post("/token/revoke")
def revoke_token(token: str = Body(...)):
    user_utils.revoke_token(token)
    return {"status": "revoked"}

# RBAC dependency
def require_role(required_role: str):
    def role_checker(user=Depends(get_current_user)):
        if user.get("role") not in ["admin", "trader", "viewer"] or user["role"] != required_role:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return user
    return role_checker

# Example: enforce RBAC on config upload
@router.post("/upload")
async def upload_data(file: UploadFile = File(...), user=Depends(require_role("trader"))):
    # ... existing code ...
    session = SessionLocal()
    # ... existing code ...
    log_audit(session, user["id"], "upload_config", file.filename)
    # ... existing code ...
    session.close()
    # ... existing code ...

ALLOWED_TICKERS = {"AAPL", "MSFT", "GOOG"}

class StrategyConfigSchema(BaseModel):
    name: str
    tickers: list
    params: dict
    @validator("tickers")
    def check_tickers(cls, v):
        if not all(t in ALLOWED_TICKERS for t in v):
            raise ValueError("Invalid ticker in config")
        return v

# Create strategy config
@router.post("/strategy_configs")
def create_strategy_config(config: dict = Body(...), user=Depends(require_role("trader"))):
    try:
        parsed = StrategyConfigSchema(**config)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    session = SessionLocal()
    sc = StrategyConfig(
        user_id=user["id"],
        name=parsed.name,
        version=1,
        config_json=json.dumps(config),
        is_active=True
    )
    session.add(sc)
    session.commit()
    log_audit(session, user["id"], "create_strategy_config", parsed.name)
    session.close()
    return {"status": "created", "id": sc.id}

# List strategy configs
@router.get("/strategy_configs")
def list_strategy_configs(user=Depends(get_current_user)):
    session = SessionLocal()
    configs = session.query(StrategyConfig).filter_by(user_id=user["id"]).order_by(StrategyConfig.updated_at.desc()).all()
    result = [
        {"id": c.id, "name": c.name, "version": c.version, "is_active": c.is_active, "created_at": c.created_at, "updated_at": c.updated_at}
        for c in configs
    ]
    session.close()
    return result

# Get strategy config
@router.get("/strategy_configs/{config_id}")
def get_strategy_config(config_id: int, user=Depends(get_current_user)):
    session = SessionLocal()
    c = session.query(StrategyConfig).filter_by(id=config_id, user_id=user["id"]).first()
    if not c:
        session.close()
        raise HTTPException(status_code=404, detail="Config not found")
    result = {"id": c.id, "name": c.name, "version": c.version, "is_active": c.is_active, "config": json.loads(c.config_json)}
    session.close()
    return result

# Update strategy config (creates new version)
@router.put("/strategy_configs/{config_id}")
def update_strategy_config(config_id: int, config: dict = Body(...), user=Depends(require_role("trader"))):
    try:
        parsed = StrategyConfigSchema(**config)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    session = SessionLocal()
    old = session.query(StrategyConfig).filter_by(id=config_id, user_id=user["id"], is_active=True).first()
    if not old:
        session.close()
        raise HTTPException(status_code=404, detail="Config not found")
    old.is_active = False
    new_version = old.version + 1
    new = StrategyConfig(
        user_id=user["id"],
        name=parsed.name,
        version=new_version,
        config_json=json.dumps(config),
        is_active=True,
        previous_version_id=old.id
    )
    session.add(new)
    session.commit()
    log_audit(session, user["id"], "update_strategy_config", parsed.name)
    session.close()
    return {"status": "updated", "id": new.id}

# Delete strategy config
@router.delete("/strategy_configs/{config_id}")
def delete_strategy_config(config_id: int, user=Depends(require_role("trader"))):
    session = SessionLocal()
    c = session.query(StrategyConfig).filter_by(id=config_id, user_id=user["id"]).first()
    if not c:
        session.close()
        raise HTTPException(status_code=404, detail="Config not found")
    session.delete(c)
    session.commit()
    log_audit(session, user["id"], "delete_strategy_config", c.name)
    session.close()
    return {"status": "deleted"}

# Rollback strategy config
@router.post("/strategy_configs/{config_id}/rollback")
def rollback_strategy_config(config_id: int, user=Depends(require_role("trader"))):
    session = SessionLocal()
    c = session.query(StrategyConfig).filter_by(id=config_id, user_id=user["id"]).first()
    if not c or not c.previous_version_id:
        session.close()
        raise HTTPException(status_code=404, detail="No previous version to rollback to")
    prev = session.query(StrategyConfig).filter_by(id=c.previous_version_id).first()
    if not prev:
        session.close()
        raise HTTPException(status_code=404, detail="Previous version not found")
    c.is_active = False
    prev.is_active = True
    session.commit()
    log_audit(session, user["id"], "rollback_strategy_config", prev.name)
    session.close()
    return {"status": "rolled back", "id": prev.id}

# Create agent
@router.post("/agents")
def create_agent(name: str = Body(...), type: str = Body(...), config_id: int = Body(...), user=Depends(require_role("trader"))):
    session = SessionLocal()
    agent = Agent(
        user_id=user["id"],
        name=name,
        type=type,
        status=AgentStatus.training,
        config_id=config_id
    )
    session.add(agent)
    session.commit()
    log_audit(session, user["id"], "create_agent", name)
    session.close()
    return {"status": "created", "id": agent.id}

# List agents
@router.get("/agents")
def list_agents(user=Depends(get_current_user)):
    session = SessionLocal()
    agents = session.query(Agent).filter_by(user_id=user["id"]).order_by(Agent.updated_at.desc()).all()
    result = [
        {"id": a.id, "name": a.name, "type": a.type, "status": a.status.value, "config_id": a.config_id, "artifact_path": a.artifact_path, "created_at": a.created_at, "updated_at": a.updated_at}
        for a in agents
    ]
    session.close()
    return result

# Get agent
@router.get("/agents/{agent_id}")
def get_agent(agent_id: int, user=Depends(get_current_user)):
    session = SessionLocal()
    a = session.query(Agent).filter_by(id=agent_id, user_id=user["id"]).first()
    if not a:
        session.close()
        raise HTTPException(status_code=404, detail="Agent not found")
    result = {"id": a.id, "name": a.name, "type": a.type, "status": a.status.value, "config_id": a.config_id, "artifact_path": a.artifact_path, "created_at": a.created_at, "updated_at": a.updated_at, "error_message": a.error_message}
    session.close()
    return result

# Update agent (status, name, type, config_id)
@router.put("/agents/{agent_id}")
def update_agent(agent_id: int, updates: dict = Body(...), user=Depends(require_role("trader"))):
    session = SessionLocal()
    a = session.query(Agent).filter_by(id=agent_id, user_id=user["id"]).first()
    if not a:
        session.close()
        raise HTTPException(status_code=404, detail="Agent not found")
    for k, v in updates.items():
        if hasattr(a, k):
            setattr(a, k, v)
    session.commit()
    log_audit(session, user["id"], "update_agent", a.name)
    session.close()
    return {"status": "updated"}

# Delete agent
@router.delete("/agents/{agent_id}")
def delete_agent(agent_id: int, user=Depends(require_role("trader"))):
    session = SessionLocal()
    a = session.query(Agent).filter_by(id=agent_id, user_id=user["id"]).first()
    if not a:
        session.close()
        raise HTTPException(status_code=404, detail="Agent not found")
    session.delete(a)
    session.commit()
    log_audit(session, user["id"], "delete_agent", a.name)
    session.close()
    return {"status": "deleted"}

def calculate_file_hash(path):
    """Calculate SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()

# Upload model artifact (with metadata)
@router.post("/agents/{agent_id}/artifact")
def upload_agent_artifact(agent_id: int, file: UploadFile = File(...), version: int = Body(1), notes: str = Body(None), user=Depends(require_role("trader"))):
    session = SessionLocal()
    a = session.query(Agent).filter_by(id=agent_id, user_id=user["id"]).first()
    if not a:
        session.close()
        raise HTTPException(status_code=404, detail="Agent not found")
    artifact_dir = "model_artifacts"
    os.makedirs(artifact_dir, exist_ok=True)
    path = os.path.join(artifact_dir, f"agent_{agent_id}_v{version}_{file.filename}")
    with open(path, "wb") as f:
        f.write(file.file.read())
    file_hash = calculate_file_hash(path)
    artifact = ModelArtifact(agent_id=agent_id, version=version, path=path, hash=file_hash, is_valid=True, notes=notes)
    session.add(artifact)
    a.artifact_path = path
    session.commit()
    log_audit(session, user["id"], "upload_agent_artifact", path)
    session.close()
    return {"status": "uploaded", "path": path, "hash": file_hash}

# Download model artifact (with corruption check)
@router.get("/agents/{agent_id}/artifact")
def download_agent_artifact(agent_id: int, version: int = None, user=Depends(get_current_user)):
    session = SessionLocal()
    q = session.query(ModelArtifact).filter_by(agent_id=agent_id, is_valid=True)
    if version:
        q = q.filter_by(version=version)
    artifact = q.order_by(ModelArtifact.version.desc()).first()
    if not artifact or not os.path.exists(artifact.path):
        session.close()
        raise HTTPException(status_code=404, detail="Artifact not found")
    # Check hash
    file_hash = calculate_file_hash(artifact.path)
    if file_hash != artifact.hash:
        artifact.is_valid = False
        session.commit()
        session.close()
        raise HTTPException(status_code=500, detail="Artifact corrupted")
    session.close()
    return FileResponse(artifact.path, filename=os.path.basename(artifact.path))

# List model artifacts for an agent
@router.get("/agents/{agent_id}/artifacts")
def list_agent_artifacts(agent_id: int, user=Depends(get_current_user)):
    session = SessionLocal()
    artifacts = session.query(ModelArtifact).filter_by(agent_id=agent_id).order_by(ModelArtifact.version.desc()).all()
    result = [
        {"id": a.id, "version": a.version, "path": a.path, "hash": a.hash, "created_at": a.created_at, "is_valid": a.is_valid, "notes": a.notes}
        for a in artifacts
    ]
    session.close()
    return result

# Create fill
@router.post("/fills")
def create_fill(order_id: int = Body(...), agent_id: int = Body(None), symbol: str = Body(...), qty: float = Body(...), price: float = Body(...), side: str = Body(...), status: str = Body("filled"), user=Depends(require_role("trader"))):
    session = SessionLocal()
    fill = Fill(
        order_id=order_id,
        user_id=user["id"],
        agent_id=agent_id,
        symbol=symbol,
        qty=qty,
        price=price,
        side=side,
        status=FillStatus(status)
    )
    session.add(fill)
    session.commit()
    session.close()
    return {"status": "created", "id": fill.id}

# List fills
@router.get("/fills")
def list_fills(user=Depends(get_current_user)):
    session = SessionLocal()
    fills = session.query(Fill).filter_by(user_id=user["id"]).order_by(Fill.timestamp.desc()).all()
    result = [
        {"id": f.id, "order_id": f.order_id, "agent_id": f.agent_id, "symbol": f.symbol, "qty": f.qty, "price": f.price, "side": f.side, "timestamp": f.timestamp, "status": f.status.value}
        for f in fills
    ]
    session.close()
    return result

# Get fill
@router.get("/fills/{fill_id}")
def get_fill(fill_id: int, user=Depends(get_current_user)):
    session = SessionLocal()
    f = session.query(Fill).filter_by(id=fill_id, user_id=user["id"]).first()
    if not f:
        session.close()
        raise HTTPException(status_code=404, detail="Fill not found")
    result = {"id": f.id, "order_id": f.order_id, "agent_id": f.agent_id, "symbol": f.symbol, "qty": f.qty, "price": f.price, "side": f.side, "timestamp": f.timestamp, "status": f.status.value}
    session.close()
    return result

# Create equity curve point
@router.post("/equity_curves")
def create_equity_curve(agent_id: int = Body(None), equity: float = Body(...), pnl: float = Body(...), user=Depends(require_role("trader"))):
    session = SessionLocal()
    ec = EquityCurve(
        user_id=user["id"],
        agent_id=agent_id,
        equity=equity,
        pnl=pnl
    )
    session.add(ec)
    session.commit()
    session.close()
    return {"status": "created", "id": ec.id}

# List equity curve points
@router.get("/equity_curves")
def list_equity_curves(agent_id: int = None, user=Depends(get_current_user)):
    session = SessionLocal()
    q = session.query(EquityCurve).filter_by(user_id=user["id"])
    if agent_id:
        q = q.filter_by(agent_id=agent_id)
    ecs = q.order_by(EquityCurve.timestamp.desc()).all()
    result = [
        {"id": e.id, "agent_id": e.agent_id, "timestamp": e.timestamp, "equity": e.equity, "pnl": e.pnl}
        for e in ecs
    ]
    session.close()
    return result

# Get equity curve point
@router.get("/equity_curves/{ec_id}")
def get_equity_curve(ec_id: int, user=Depends(get_current_user)):
    session = SessionLocal()
    e = session.query(EquityCurve).filter_by(id=ec_id, user_id=user["id"]).first()
    if not e:
        session.close()
        raise HTTPException(status_code=404, detail="Equity curve point not found")
    result = {"id": e.id, "agent_id": e.agent_id, "timestamp": e.timestamp, "equity": e.equity, "pnl": e.pnl}
    session.close()
    return result

# WebSocket for real-time equity curve streaming
@router.websocket("/ws/equity_curve/{agent_id}")
async def equity_curve_ws(ws: WebSocket, agent_id: int, user_id: int = None):
    await ws.accept()
    # For demo: stream random equity curve points
    for i in range(30):
        now = datetime.datetime.utcnow().isoformat()
        equity = 100000 + i * 100 + (os.urandom(1)[0] % 20 - 10)
        pnl = i * 10 + (os.urandom(1)[0] % 5 - 2)
        await ws.send_json({"timestamp": now, "equity": equity, "pnl": pnl})
        await asyncio.sleep(1)
    await ws.close()

# Job status
@router.get("/jobs/{task_id}/status")
def job_status(task_id: str, user=Depends(get_current_user)):
    result = AsyncResult(task_id, app=celery_app)
    return {"task_id": task_id, "status": result.status, "result": result.result if result.successful() else None}

# Job log (stub: return log file tail)
@router.get("/jobs/{task_id}/log")
def job_log(task_id: str, user=Depends(get_current_user)):
    log_path = "finrl-backend.log"
    if not os.path.exists(log_path):
        return {"log": "No log file found."}
    with open(log_path, "r") as f:
        lines = f.readlines()[-50:]
    return {"log": "".join(lines)}

# Job error traceback (stub: from AsyncResult)
@router.get("/jobs/{task_id}/error")
def job_error(task_id: str, user=Depends(get_current_user)):
    result = AsyncResult(task_id, app=celery_app)
    if result.failed():
        return {"task_id": task_id, "error": str(result.result)}
    return {"task_id": task_id, "error": None}

# Job cancellation (revoke)
@router.post("/jobs/{task_id}/cancel")
def job_cancel(task_id: str, user=Depends(require_role("trader"))):
    result = AsyncResult(task_id, app=celery_app)
    result.revoke(terminate=True)
    return {"task_id": task_id, "status": "cancelled"}

# Job re-queue (stub: re-apply task with same args if possible)
@router.post("/jobs/{task_id}/requeue")
def job_requeue(task_id: str, user=Depends(require_role("trader"))):
    result = AsyncResult(task_id, app=celery_app)
    if not result.info or not hasattr(result.info, "args"):
        return {"error": "Cannot requeue: no args found."}
    # This is a stub; real implementation would depend on how args are stored
    task_name = result.task_name
    args = result.info.args if hasattr(result.info, "args") else []
    new_result = celery_app.send_task(task_name, args=args)
    return {"old_task_id": task_id, "new_task_id": new_result.id, "status": "requeued"} 