import asyncio
import websockets
import os
import json
from backend.models import SessionLocal
from backend.models.tick_event import TickEvent
import redis
from datetime import datetime

# Secure credentials placeholder
DATABENTO_API_KEY = os.getenv("DATABENTO_API_KEY", "<FILL_ME>")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_STREAM = "ticks"
WS_URL = "wss://live.databento.com/v1/marketdata"

class DatabentoIngestor:
    def __init__(self, dataset, symbols, schema="mbo"):
        self.dataset = dataset
        self.symbols = symbols
        self.schema = schema
        self.redis = redis.Redis.from_url(REDIS_URL)

    async def ingest_forever(self):
        while True:
            try:
                await self._stream_data()
            except Exception as e:
                print(f"[DatabentoIngestor] Error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def _stream_data(self):
        async with websockets.connect(WS_URL) as ws:
            # Authenticate
            await ws.send(json.dumps({"action": "auth", "key": DATABENTO_API_KEY}))
            auth_resp = await ws.recv()
            print(f"[DatabentoIngestor] Auth response: {auth_resp}")
            # Subscribe
            await ws.send(json.dumps({
                "action": "subscribe",
                "dataset": self.dataset,
                "symbols": self.symbols,
                "schema": self.schema
            }))
            sub_resp = await ws.recv()
            print(f"[DatabentoIngestor] Sub response: {sub_resp}")
            # Stream data
            while True:
                msg = await ws.recv()
                try:
                    data = json.loads(msg)
                    if isinstance(data, dict) and data.get("type") == "trade":
                        tick = self._parse_record(data)
                        if tick:
                            self.redis.xadd(REDIS_STREAM, tick)
                            await self._save_to_db(tick)
                except Exception as e:
                    print(f"[DatabentoIngestor] Parse error: {e}")

    def _parse_record(self, record):
        # Map Databento MBO fields to TickEvent schema
        try:
            return {
                "ts_event": int(record.get("ts_event", 0)),
                "instrument": record.get("symbol"),
                "side": record.get("side"),
                "price": float(record.get("price", 0)),
                "size": float(record.get("size", 0)),
                "exchange": record.get("exchange"),
                "type": record.get("type"),
                "source": "databento"
            }
        except Exception:
            return None

    async def _save_to_db(self, tick):
        session = SessionLocal()
        try:
            exists = session.query(TickEvent).filter_by(
                ts_event=tick["ts_event"],
                instrument=tick["instrument"],
                exchange=tick["exchange"],
                type=tick["type"]
            ).first()
            if not exists:
                session.add(TickEvent(**tick))
                session.commit()
        except Exception:
            session.rollback()
        finally:
            session.close() 