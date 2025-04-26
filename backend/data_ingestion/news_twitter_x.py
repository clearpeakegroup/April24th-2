import os
import json
import asyncio
import websockets
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import re

load_dotenv()
TWITTER_X_WS_URL = os.getenv("TWITTER_X_WS_URL")
TWITTER_X_BEARER = os.getenv("TWITTER_X_BEARER")
RAW_BASE = "data/lake/raw/news"
CASHTAG_PATTERN = re.compile(r"\\$[A-Za-z]{1,6}")

async def ingest_twitter_x():
    assert TWITTER_X_WS_URL and TWITTER_X_BEARER, "Twitter X config missing in .env"
    while True:
        try:
            async with websockets.connect(TWITTER_X_WS_URL, extra_headers={"Authorization": f"Bearer {TWITTER_X_BEARER}"}) as ws:
                logger.info("Connected to Twitter X FinanceFirehose.")
                async for msg in ws:
                    try:
                        tweet = json.loads(msg)
                        text = tweet.get("text", "")
                        if not CASHTAG_PATTERN.search(text):
                            continue  # Skip if no cashtag
                        now = datetime.utcnow()
                        partition = f"{RAW_BASE}/{now.year:04d}/{now.month:02d}/{now.day:02d}"
                        Path(partition).mkdir(parents=True, exist_ok=True)
                        fname = f"{partition}/twitterx_{now.strftime('%H%M%S_%f')}.json.zst"
                        import zstandard as zstd
                        cctx = zstd.ZstdCompressor()
                        with open(fname, "wb") as f:
                            f.write(cctx.compress(msg.encode('utf-8')))
                        logger.info(f"Saved Twitter X news: {fname}")
                    except Exception as inner:
                        logger.error(f"Twitter X message error: {inner}")
        except Exception as e:
            logger.error(f"Twitter X ingest error: {e}")
            await asyncio.sleep(5) 