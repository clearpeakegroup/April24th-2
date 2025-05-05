import os
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from pydantic import BaseModel, StrictStr, StrictFloat, StrictInt, ValidationError, Field
from trader_core.config_loader import ConfigLoader
from backend.data_ingestion.decompression import decompress_zstd

load_dotenv()
RAW_BASE = os.getenv("RAW_MARKET_BASE", "data/lake/raw/market")
FEATURE_BASE = os.getenv("ORDERBOOK_FEATURE_BASE", "data/lake/parquet/features/orderbook")
BATCH_SEC = int(os.getenv("ORDERBOOK_BATCH_SEC", 60))

class OrderBookFeature(BaseModel):
    asset: StrictStr
    timestamp: StrictStr
    best_bid: StrictFloat
    best_ask: StrictFloat
    spread: StrictFloat
    depth_bid: StrictFloat
    depth_ask: StrictFloat
    # Extend with more features as needed

def extract_orderbook_features(df: pd.DataFrame, asset: str, ts: datetime) -> dict:
    # Assumes df is filtered for a single asset and time window
    if df.empty:
        return None
    best_bid = df[df['side'] == 'buy']['price'].max()
    best_ask = df[df['side'] == 'sell']['price'].min()
    spread = best_ask - best_bid if pd.notnull(best_bid) and pd.notnull(best_ask) else None
    depth_bid = df[df['side'] == 'buy']['size'].sum()
    depth_ask = df[df['side'] == 'sell']['size'].sum()
    return {
        'asset': asset,
        'timestamp': ts.isoformat(),
        'best_bid': float(best_bid) if pd.notnull(best_bid) else 0.0,
        'best_ask': float(best_ask) if pd.notnull(best_ask) else 0.0,
        'spread': float(spread) if spread is not None else 0.0,
        'depth_bid': float(depth_bid) if pd.notnull(depth_bid) else 0.0,
        'depth_ask': float(depth_ask) if pd.notnull(depth_ask) else 0.0,
    }

def validate_config():
    missing = []
    if not RAW_BASE:
        missing.append("RAW_MARKET_BASE")
    if not FEATURE_BASE:
        missing.append("ORDERBOOK_FEATURE_BASE")
    if missing:
        logger.error(f"Missing required orderbook feature config: {', '.join(missing)}")
        raise RuntimeError(f"Missing required orderbook feature config: {', '.join(missing)}")

def health_check():
    try:
        validate_config()
        # Check if output directory exists or can be created
        Path(FEATURE_BASE).mkdir(parents=True, exist_ok=True)
        logger.info("Orderbook feature pipeline health check passed.")
        return True
    except Exception as e:
        logger.error(f"Orderbook feature pipeline health check failed: {e}")
        return False

async def run_orderbook_feature_pipeline():
    validate_config()
    assets = ConfigLoader.get_assets("config/default.yml")
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
        window_start = now.replace(second=0, microsecond=0)
        window_end = window_start + timedelta(seconds=BATCH_SEC)
        batch = []
        for asset in assets:
            asset_files = [f for f in files if asset in f.name]
            dfs = []
            for f in asset_files:
                with open(f, "rb") as fh:
                    data = decompress_zstd(fh.read())
                    try:
                        df = pd.read_csv(pd.compat.StringIO(data.decode("utf-8")))
                        dfs.append(df)
                    except Exception:
                        continue
            if dfs:
                df_all = pd.concat(dfs, ignore_index=True)
                # Filter by window
                df_all['ts_event'] = pd.to_datetime(df_all['ts_event'], unit='ns', errors='coerce')
                df_window = df_all[(df_all['ts_event'] >= window_start) & (df_all['ts_event'] < window_end)]
                features = extract_orderbook_features(df_window, asset, window_start)
                if features:
                    try:
                        feat = OrderBookFeature(**features)
                        batch.append(feat.model_dump())
                    except ValidationError as ve:
                        logger.error(f"OrderBook feature validation error: {ve}")
        if batch:
            table = pa.Table.from_pylist(batch)
            out_path = feat_dir / f"orderbook_{window_start.strftime('%H%M%S')}.parquet"
            pq.write_table(table, out_path, compression="zstd")
            logger.info(f"Wrote orderbook features: {out_path} ({len(batch)} rows)")
        await asyncio.sleep(BATCH_SEC)

# Schema: OrderBookFeature (asset: str, timestamp: str, best_bid: float, best_ask: float, spread: float, depth_bid: float, depth_ask: float)
# Extend this schema and pipeline for additional features and sources as needed. 