import os
import uuid
import redis
import time
from backend.middleware.tasks import preprocess_task, retrain_model, run_backtest, run_forwardtest, run_live_execution

class PipelineOrchestrator:
    def __init__(self, redis_url=None):
        self.redis = redis.Redis.from_url(redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0"))

    def start_pipeline(self, file_path, user, config):
        pipeline_id = str(uuid.uuid4())
        self.redis.hset(f"pipeline:{pipeline_id}", mapping={
            "stage": "preprocessing",
            "progress": 0,
            "elapsed": 0,
            "eta": 0,
            "status": "running"
        })
        # 1. Preprocess
        preprocess_job = preprocess_task.apply_async(args=[file_path, user, pipeline_id])
        self.redis.hset(f"pipeline:{pipeline_id}", "preprocess_task_id", preprocess_job.id)
        # The rest of the pipeline is chained via callbacks in the Celery task (see below)
        return pipeline_id

    def update_stage(self, pipeline_id, stage, progress, elapsed, eta):
        self.redis.hset(f"pipeline:{pipeline_id}", mapping={
            "stage": stage,
            "progress": progress,
            "elapsed": elapsed,
            "eta": eta
        })

    def set_status(self, pipeline_id, status):
        self.redis.hset(f"pipeline:{pipeline_id}", "status", status)

    def get_progress(self, pipeline_id):
        d = self.redis.hgetall(f"pipeline:{pipeline_id}")
        return {k.decode(): (int(v) if v.isdigit() else v.decode()) for k, v in d.items()}

    def get_results(self, pipeline_id):
        # Optionally fetch results/logs from Redis or DB
        return self.redis.hgetall(f"pipeline:{pipeline_id}:results") 