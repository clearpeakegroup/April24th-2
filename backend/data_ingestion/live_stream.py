import os
import json
import asyncio
import websockets
import redis
import pandas as pd
from loguru import logger
from typing import Any, Dict

CREDENTIALS_PATH = os.path.expanduser("~/.config/finrl/credentials.json")
SCHEMA = ["ts_event", "instrument", "side", "price", "size", "exchange", "type"]
REDIS_STREAM = "ticks"


def load_credentials():
    if not os.path.exists(CREDENTIALS_PATH):
        logger.warning(f"Credentials file not found: {CREDENTIALS_PATH}")
        return None
    with open(CREDENTIALS_PATH, "r") as f:
        return json.load(f)

def retry_async(times=3):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(1, times + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Attempt {attempt} failed: {e}")
                    if attempt == times:
                        raise
        return wrapper
    return decorator

@retry_async(times=3)
async def ingest_live_ticks(ws_url: str, redis_url: str = "redis://localhost:6379/0"):
    creds = load_credentials()
    tls_auth = creds.get("tls_auth") if creds else None
    logger.info(f"Connecting to {ws_url} with TLS auth stub: {tls_auth}")
    r = redis.Redis.from_url(redis_url)
    async with websockets.connect(ws_url, ssl=True) as ws:
        async for msg in ws:
            # Assume JSON tick message
            tick = json.loads(msg)
            # Apply schema
            row = {col: tick.get(col) for col in SCHEMA}
            df = pd.DataFrame([row])
            # Publish to Redis stream
            r.xadd(REDIS_STREAM, row)
            logger.info(f"Published tick to Redis: {row}")

# Example usage (to be run in an async context):
# asyncio.run(ingest_live_ticks("wss://databento.example/ws")) 