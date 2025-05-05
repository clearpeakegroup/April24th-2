import asyncio
import redis
import os
import json

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_STREAM = "ticks"

class TickStreamConsumer:
    def __init__(self, group: str, consumer: str):
        self.redis = redis.Redis.from_url(REDIS_URL)
        self.group = group
        self.consumer = consumer
        # Create the consumer group if it doesn't exist
        try:
            self.redis.xgroup_create(REDIS_STREAM, group, id='0', mkstream=True)
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    async def consume(self, handler):
        while True:
            resp = self.redis.xreadgroup(self.group, self.consumer, {REDIS_STREAM: '>'}, count=100, block=1000)
            for stream, messages in resp:
                for msg_id, msg in messages:
                    tick = {k.decode(): v.decode() for k, v in msg.items()}
                    await handler(tick)
                    self.redis.xack(REDIS_STREAM, self.group, msg_id)
            await asyncio.sleep(0.01) 