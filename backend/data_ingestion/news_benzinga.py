import os
import json
import asyncio
import websockets
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
BENZINGA_WS_URL = os.getenv("BENZINGA_WS_URL")
BENZINGA_API_KEY = os.getenv("BENZINGA_API_KEY")
RAW_BASE = "data/lake/raw/news"

async def ingest_benzinga_news():
    assert BENZINGA_WS_URL and BENZINGA_API_KEY, "Benzinga config missing in .env"
    while True:
        try:
            async with websockets.connect(BENZINGA_WS_URL, extra_headers={"Authorization": f"Bearer {BENZINGA_API_KEY}"}) as ws:
                logger.info("Connected to Benzinga Pro news feed.")
                async for msg in ws:
                    now = datetime.utcnow()
                    partition = f"{RAW_BASE}/{now.year:04d}/{now.month:02d}/{now.day:02d}"
                    Path(partition).mkdir(parents=True, exist_ok=True)
                    fname = f"{partition}/benzinga_{now.strftime('%H%M%S_%f')}.json.zst"
                    # Write raw JSON, zstd compressed
                    import zstandard as zstd
                    cctx = zstd.ZstdCompressor()
                    with open(fname, "wb") as f:
                        f.write(cctx.compress(msg.encode('utf-8')))
                    logger.info(f"Saved Benzinga news: {fname}")
        except Exception as e:
            logger.error(f"Benzinga ingest error: {e}")
            await asyncio.sleep(5)  # Backoff and retry 