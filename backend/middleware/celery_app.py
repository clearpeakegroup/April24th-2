from celery import Celery
from kombu import Queue
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6380/0")

celery_app = Celery(
    "finrl_tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

# Enable Redis cluster support and prefetch tuning
celery_app.conf.update(
    task_queues=[
        Queue("backtest"),
        Queue("forwardtest"),
        Queue("retrain"),
        Queue("liveexec"),
        Queue("features"),
    ],
    worker_prefetch_multiplier=8,  # Tune for throughput
    broker_transport_options={
        "visibility_timeout": 3600,
        "fanout_prefix": True,
        "fanout_patterns": True,
        "max_connections": 100,
    },
    result_backend_transport_options={
        "retry_policy": {
            "timeout": 10.0
        }
    },
    task_routes={
        'features.sentiment_pipeline': {'queue': 'features'},
    }
) 