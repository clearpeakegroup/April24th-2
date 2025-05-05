import redis
import os
import json

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
AGENT_OUTPUT_STREAM = "agent_signals"

def publish_agent_signal(signal: dict):
    r = redis.Redis.from_url(REDIS_URL)
    # Ensure all values are strings for Redis
    signal_str = {k: json.dumps(v) if not isinstance(v, str) else v for k, v in signal.items()}
    r.xadd(AGENT_OUTPUT_STREAM, signal_str) 