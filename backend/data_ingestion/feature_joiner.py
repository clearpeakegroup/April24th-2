import os
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import pyarrow.parquet as pq
import pandas as pd
from pydantic import BaseModel, StrictStr, StrictFloat, ValidationError, Field
from trader_core.config_loader import ConfigLoader

load_dotenv()
ORDERBOOK_BASE = os.getenv("ORDERBOOK_FEATURE_BASE", "data/lake/parquet/features/orderbook")
SENTIMENT_BASE = os.getenv("SENTIMENT_FEATURE_BASE", "data/lake/parquet/features/sentiment")
UNIFIED_BASE = os.getenv("UNIFIED_FEATURE_BASE", "data/lake/parquet/features/unified")
BATCH_SEC = int(os.getenv("UNIFIED_BATCH_SEC", 60))

class UnifiedFeature(BaseModel):
    asset: StrictStr
    timestamp: StrictStr
    best_bid: StrictFloat
    best_ask: StrictFloat
    spread: StrictFloat
    depth_bid: StrictFloat
    depth_ask: StrictFloat
    sentiment: StrictFloat
    news_length: int
    # Extend with more features as needed

def align_and_join_features(orderbook_df, sentiment_df):
    # Join on asset and timestamp (nearest within window)
    # Assume both have 'asset' and 'timestamp' columns
    orderbook_df['timestamp'] = pd.to_datetime(orderbook_df['timestamp'])
    sentiment_df['timestamp'] = pd.to_datetime(sentiment_df['timestamp'])
    # For now, do a left join on asset and nearest timestamp within 60s
    joined = pd.merge_asof(
        orderbook_df.sort_values('timestamp'),
        sentiment_df.sort_values('timestamp'),
        by='asset',
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta(seconds=BATCH_SEC)
    )
    joined = joined.dropna(subset=['sentiment', 'length'])
    joined = joined.rename(columns={'length': 'news_length'})
    return joined

async def run_feature_joiner():
    assets = ConfigLoader.get_assets("config/default.yml")
    while True:
        now = datetime.utcnow()
        partition = f"{now.year:04d}/{now.month:02d}/{now.day:02d}"
        orderbook_dir = Path(ORDERBOOK_BASE) / partition
        sentiment_dir = Path(SENTIMENT_BASE) / partition
        unified_dir = Path(UNIFIED_BASE) / partition
        unified_dir.mkdir(parents=True, exist_ok=True)
        if not orderbook_dir.exists() or not sentiment_dir.exists():
            await asyncio.sleep(10)
            continue
        # Load all orderbook and sentiment parquet files for the window
        window_start = now.replace(second=0, microsecond=0)
        window_end = window_start + timedelta(seconds=BATCH_SEC)
        ob_files = sorted(orderbook_dir.glob("*.parquet"))
        sent_files = sorted(sentiment_dir.glob("*.parquet"))
        if not ob_files or not sent_files:
            await asyncio.sleep(10)
            continue
        ob_df = pd.concat([pq.read_table(f).to_pandas() for f in ob_files], ignore_index=True)
        sent_df = pd.concat([pq.read_table(f).to_pandas() for f in sent_files], ignore_index=True)
        # Filter by window
        ob_df['timestamp'] = pd.to_datetime(ob_df['timestamp'])
        sent_df['timestamp'] = pd.to_datetime(sent_df['timestamp'])
        ob_df = ob_df[(ob_df['timestamp'] >= window_start) & (ob_df['timestamp'] < window_end)]
        sent_df = sent_df[(sent_df['timestamp'] >= window_start) & (sent_df['timestamp'] < window_end)]
        # Align and join
        joined = align_and_join_features(ob_df, sent_df)
        batch = []
        for _, row in joined.iterrows():
            record = row.to_dict()
            try:
                feat = UnifiedFeature(**record)
                batch.append(feat.model_dump())
            except ValidationError as ve:
                logger.error(f"Unified feature validation error: {ve}")
        if batch:
            out_path = unified_dir / f"unified_{window_start.strftime('%H%M%S')}.parquet"
            pq.write_table(pd.DataFrame(batch), out_path, compression="zstd")
            logger.info(f"Wrote unified features: {out_path} ({len(batch)} rows)")
        await asyncio.sleep(BATCH_SEC)

# Schema: UnifiedFeature (asset, timestamp, best_bid, best_ask, spread, depth_bid, depth_ask, sentiment, news_length)
# Extend this joiner for more features/sources as needed. 