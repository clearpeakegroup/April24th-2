import os
import json
import pandas as pd
import zstandard as zstd
from loguru import logger
from typing import List
from pathlib import Path

CREDENTIALS_PATH = os.path.expanduser("~/.config/finrl/credentials.json")

SCHEMA = [
    "ts_event", "instrument", "side", "price", "size", "exchange", "type"
]


def load_credentials():
    if not os.path.exists(CREDENTIALS_PATH):
        logger.warning(f"Credentials file not found: {CREDENTIALS_PATH}")
        return None
    with open(CREDENTIALS_PATH, "r") as f:
        return json.load(f)


def retry(times=3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(1, times + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Attempt {attempt} failed: {e}")
                    if attempt == times:
                        raise
        return wrapper
    return decorator


@retry(times=3)
def load_zstd_dbn(file_path: str) -> pd.DataFrame:
    logger.info(f"Loading ZSTD-compressed DBN file: {file_path}")
    with open(file_path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader:
            data = reader.read()
            # Assume CSV inside for this example
            df = pd.read_csv(pd.compat.StringIO(data.decode("utf-8")))
    # Ensure schema
    for col in SCHEMA:
        if col not in df.columns:
            df[col] = None
    df = df[SCHEMA]
    logger.info(f"Loaded {len(df)} rows from {file_path}")
    return df 