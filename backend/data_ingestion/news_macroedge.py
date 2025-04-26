import os
import json
import asyncio
import httpx
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

load_dotenv()
MACROEDGE_API_URL = os.getenv("MACROEDGE_API_URL")
MACROEDGE_API_KEY = os.getenv("MACROEDGE_API_KEY")
RAW_BASE = "data/lake/raw/news"
POLL_INTERVAL = int(os.getenv("MACROEDGE_POLL_SEC", 10))

async def ingest_macroedge():
    assert MACROEDGE_API_URL and MACROEDGE_API_KEY, "MacroEdge config missing in .env"
    headers = {"Authorization": f"Bearer {MACROEDGE_API_KEY}"}
    last_seen = set()
    while True:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(MACROEDGE_API_URL, headers=headers)
                resp.raise_for_status()
                news_items = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else []
                for item in news_items:
                    news_id = item.get("id") or item.get("guid") or json.dumps(item)
                    if news_id in last_seen:
                        continue
                    now = datetime.utcnow()
                    partition = f"{RAW_BASE}/{now.year:04d}/{now.month:02d}/{now.day:02d}"
                    Path(partition).mkdir(parents=True, exist_ok=True)
                    fname = f"{partition}/macroedge_{now.strftime('%H%M%S_%f')}.json.zst"
                    import zstandard as zstd
                    cctx = zstd.ZstdCompressor()
                    with open(fname, "wb") as f:
                        f.write(cctx.compress(json.dumps(item).encode('utf-8')))
                    logger.info(f"Saved MacroEdge news: {fname}")
                    last_seen.add(news_id)
        except Exception as e:
            logger.error(f"MacroEdge ingest error: {e}")
        await asyncio.sleep(POLL_INTERVAL) 