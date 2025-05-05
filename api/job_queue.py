import os
import redis
import json
from typing import Dict, Any, Generator

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_cli = redis.Redis.from_url(REDIS_URL, decode_responses=True)

JOBS_QUEUE = "jobs"


def enqueue_job(job: Dict[str, Any]):
    redis_cli.rpush(JOBS_QUEUE, json.dumps(job))


def subscribe_progress(job_id: str) -> Generator[Dict[str, Any], None, None]:
    pubsub = redis_cli.pubsub()
    channel = f"progress:{job_id}"
    pubsub.subscribe(channel)
    for message in pubsub.listen():
        if message["type"] == "message":
            yield json.loads(message["data"]) 