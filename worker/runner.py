import asyncio
import os
import json
from typing import TypedDict, Literal, Dict, Any
from rq import Worker, Queue, Connection
import redis
import aioredis
from worker.strategy_engine import run_backtest, run_forward_test, run_live

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

class JobPayload(TypedDict):
    id: str
    type: Literal["train", "forward", "live"]
    params: Dict[str, Any]

async def publish_progress(job_id: str, payload: dict, redis_url=REDIS_URL):
    redis_conn = await aioredis.from_url(redis_url, decode_responses=True)
    channel = f"progress:{job_id}"
    await redis_conn.publish(channel, json.dumps({"id": job_id, **payload}))
    await redis_conn.close()

async def check_control(job_id: str, redis_url=REDIS_URL):
    redis_conn = await aioredis.from_url(redis_url, decode_responses=True)
    key = f"jobctl:{job_id}"
    while True:
        val = await redis_conn.get(key)
        if val == "cancel":
            await redis_conn.close()
            raise asyncio.CancelledError()
        elif val == "pause":
            while await redis_conn.get(key) == "pause":
                await asyncio.sleep(0.5)
        await asyncio.sleep(0.5)
    await redis_conn.close()

def job_handler(payload: JobPayload):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(_job_handler(payload))

async def _job_handler(payload: JobPayload):
    job_id = payload["id"]
    job_type = payload["type"]
    params = payload["params"]
    try:
        if job_type == "train":
            # Expect params['data'] (pd.DataFrame) and toggles
            async for update in run_backtest(**params):
                await publish_progress(job_id, update)
                await check_control(job_id)
        elif job_type == "forward":
            async for update in run_forward_test(**params):
                await publish_progress(job_id, update)
                await check_control(job_id)
        elif job_type == "live":
            async for update in run_live(**params):
                await publish_progress(job_id, update)
                await check_control(job_id)
        else:
            raise ValueError(f"Unknown job type: {job_type}")
    except asyncio.CancelledError:
        await publish_progress(job_id, {"msg": "Job cancelled", "pct": 0})
    except Exception as e:
        await publish_progress(job_id, {"msg": f"Error: {e}", "pct": 0})

if __name__ == "__main__":
    redis_sync = redis.from_url(REDIS_URL, decode_responses=True)
    with Connection(redis_sync):
        q = Queue("jobs")
        w = Worker([q])
        w.work() 