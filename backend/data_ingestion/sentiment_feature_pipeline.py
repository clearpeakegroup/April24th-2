import os
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd

load_dotenv()
RAW_BASE = os.getenv("RAW_NEWS_BASE", "data/lake/raw/news")
FEATURE_BASE = os.getenv("SENTIMENT_FEATURE_BASE", "data/lake/parquet/features/sentiment")
BATCH_SEC = int(os.getenv("SENTIMENT_BATCH_SEC", 60))

# Dummy sentiment extractor (replace with production model)
def extract_sentiment(news_json):
    text = news_json.get("headline", "") + " " + news_json.get("body", "")
    # Placeholder: real model should be used
    return {"sentiment": 0.0, "length": len(text)}

async def run_sentiment_feature_pipeline():
    while True:
        now = datetime.utcnow()
        partition = f"{now.year:04d}/{now.month:02d}/{now.day:02d}"
        raw_dir = Path(RAW_BASE) / partition
        feat_dir = Path(FEATURE_BASE) / partition
        feat_dir.mkdir(parents=True, exist_ok=True)
        if not raw_dir.exists():
            await asyncio.sleep(10)
            continue
        files = sorted(raw_dir.glob("*.zst"))
        if not files:
            await asyncio.sleep(10)
            continue
        # Group by 60s window
        window_start = now.replace(second=0, microsecond=0)
        window_end = window_start + timedelta(seconds=BATCH_SEC)
        batch = []
        for f in files:
            ts = datetime.utcfromtimestamp(f.stat().st_mtime)
            if window_start <= ts < window_end:
                with open(f, "rb") as zf:
                    dctx = zstd.ZstdDecompressor()
                    data = dctx.decompress(zf.read())
                    try:
                        news_json = json.loads(data.decode())
                    except Exception:
                        continue
                    features = extract_sentiment(news_json)
                    features["source_file"] = str(f)
                    features["timestamp"] = ts.isoformat()
                    batch.append(features)
        if batch:
            table = pa.Table.from_pylist(batch)
            out_path = feat_dir / f"sentiment_{window_start.strftime('%H%M%S')}.parquet"
            pq.write_table(table, out_path, compression="zstd")
            logger.info(f"Wrote sentiment features: {out_path} ({len(batch)} rows)")
        await asyncio.sleep(BATCH_SEC) 