import os
import json
import pandas as pd
import zstandard as zstd
from loguru import logger
from typing import List
from pathlib import Path
from backend.models.tick_event import TickEvent
from backend.models import SessionLocal, IngestionAudit
from sqlalchemy.exc import IntegrityError
from io import StringIO

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
            df = pd.read_csv(StringIO(data.decode("utf-8")))
    # Ensure schema
    for col in SCHEMA:
        if col not in df.columns:
            df[col] = None
    df = df[SCHEMA]
    logger.info(f"Loaded {len(df)} rows from {file_path}")
    return df 

def batch_ingest_to_db(df, version=1, user=None, source="unknown"):
    """
    Ingests a DataFrame of tick data into the database with deduplication, versioning, and audit logging.
    """
    session = SessionLocal()
    try:
        # Deduplicate: only insert if not already present (by ts_event, instrument, exchange, type, version)
        new_records = []
        for row in df.to_dict(orient="records"):
            exists = session.query(TickEvent).filter_by(
                ts_event=row["ts_event"],
                instrument=row["instrument"],
                exchange=row["exchange"],
                type=row["type"],
                version=version
            ).first()
            if not exists:
                new_records.append(TickEvent(
                    ts_event=row["ts_event"],
                    instrument=row["instrument"],
                    side=row["side"],
                    price=row["price"],
                    size=row["size"],
                    exchange=row["exchange"],
                    type=row["type"],
                    version=version
                ))
        session.add_all(new_records)
        session.commit()
        status = "success"
        message = f"Inserted {len(new_records)} new records."
    except Exception as e:
        session.rollback()
        status = "fail"
        message = str(e)
    finally:
        # Audit log
        audit = IngestionAudit(
            user=user,
            source=source,
            record_count=len(df),
            status=status,
            message=message
        )
        session.add(audit)
        session.commit()
        session.close()
    return status, message 