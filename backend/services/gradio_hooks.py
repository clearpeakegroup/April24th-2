import os
import json
from typing import Any, Dict
import redis
from backend.middleware.tasks import retrain_model, run_live_execution
from backend.db import crud

# Placeholder for DB CRUD and schema imports
# from backend.db import crud, schemas

redis_cli = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

def save_config_to_db(file_obj: Any) -> str:
    """Save uploaded config to DB and return status message."""
    payload = json.load(file_obj)
    # cfg = crud.create_strategy_config(schemas.StrategyConfigCreate(**payload))
    # return f"✔️ config saved with id `{cfg.id}`"
    # TEMP: Return dummy ID
    return "✔️ config saved with id `dummy_id`"

def trigger_training_job(config_id: str) -> Dict[str, Any]:
    """Trigger a Celery training job and initialize progress in Redis."""
    # job = retrain_model.delay(config_id=config_id)
    # For now, pass dummy params
    job = retrain_model.delay({"config_id": config_id})
    redis_cli.set(f"gradio:train:{job.id}", 0)
    return { "value": 0, "job_id": job.id }

def get_job_progress(job_id: str) -> int:
    """Fetch training progress from Redis."""
    pct = redis_cli.get(f"gradio:train:{job_id}")
    return int(pct) if pct else 0

def start_live_trading(agent_id: str, broker_id: str = "paper") -> str:
    """Start a live trading Celery job."""
    job = run_live_execution.delay({"agent_id": agent_id, "broker_id": broker_id})
    return f"Live bot started (celery id {job.id})"

def stream_equity_curve(agent_id: str):
    """Generator for Gradio live plot of equity curve."""
    for point in crud.iter_equity(agent_id):
        yield point  # (timestamp, equity) 