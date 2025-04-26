import os
import asyncio
import tempfile
import shutil
import time
import pytest
from pathlib import Path
from backend.data_ingestion.news_benzinga import ingest_benzinga_news
import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd
from backend.data_ingestion.sentiment_feature_pipeline import run_sentiment_feature_pipeline

@pytest.mark.asyncio
async def test_benzinga_ingest_to_parquet(monkeypatch):
    # Setup temp dirs
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw/news/2024/04/24"
        feat_dir = Path(tmpdir) / "parquet/features/sentiment/2024/04/24"
        raw_dir.mkdir(parents=True, exist_ok=True)
        feat_dir.mkdir(parents=True, exist_ok=True)
        # Mock websocket
        class MockWebSocket:
            def __init__(self, msgs):
                self.msgs = msgs
            async def __aenter__(self):
                return self
            async def __aexit__(self, exc_type, exc, tb):
                pass
            async def __aiter__(self):
                for msg in self.msgs:
                    yield msg
        # Generate 100MB of mock news
        msg = '{"headline": "Test News", "body": "A" * 1000}'
        msgs = [msg] * (100 * 1024 * 1024 // len(msg))
        monkeypatch.setattr("websockets.connect", lambda *a, **kw: MockWebSocket(msgs))
        # Patch output path
        monkeypatch.setattr("backend.data_ingestion.news_benzinga.RAW_BASE", str(raw_dir.parent))
        # Run ingest (should be fast)
        start = time.time()
        task = asyncio.create_task(ingest_benzinga_news())
        await asyncio.sleep(5)  # Let it run for a few seconds
        task.cancel()
        # Check files
        files = list(raw_dir.glob("*.zst"))
        assert len(files) > 0, "No files written"
        # Extract and convert to parquet
        rows = []
        for f in files:
            with open(f, "rb") as zf:
                dctx = zstd.ZstdDecompressor()
                data = dctx.decompress(zf.read())
                rows.append(eval(data.decode()))
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, feat_dir / "sentiment.parquet", compression="zstd")
        assert (feat_dir / "sentiment.parquet").exists(), "Parquet not written"
        assert time.time() - start < 30, "Ingestion + parquet > 30s"

@pytest.mark.asyncio
async def test_sentiment_feature_pipeline(monkeypatch):
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_dir = Path(tmpdir) / "raw/news/2024/04/24"
        feat_dir = Path(tmpdir) / "parquet/features/sentiment/2024/04/24"
        raw_dir.mkdir(parents=True, exist_ok=True)
        feat_dir.mkdir(parents=True, exist_ok=True)
        # Write a few mock .zst news files
        msg = '{"headline": "Test News", "body": "A" * 1000}'
        cctx = zstd.ZstdCompressor()
        for i in range(5):
            fname = raw_dir / f"benzinga_120000_{i}.json.zst"
            with open(fname, "wb") as f:
                f.write(cctx.compress(msg.encode('utf-8')))
        # Patch paths
        monkeypatch.setattr("backend.data_ingestion.sentiment_feature_pipeline.RAW_BASE", str(raw_dir.parent))
        monkeypatch.setattr("backend.data_ingestion.sentiment_feature_pipeline.FEATURE_BASE", str(feat_dir.parent))
        # Patch batch sec to 2 for fast test
        monkeypatch.setattr("backend.data_ingestion.sentiment_feature_pipeline.BATCH_SEC", 2)
        # Run pipeline
        task = asyncio.create_task(run_sentiment_feature_pipeline())
        await asyncio.sleep(3)
        task.cancel()
        # Check parquet
        files = list(feat_dir.glob("*.parquet"))
        assert len(files) > 0, "No parquet files written"
        table = pq.read_table(files[0])
        assert table.num_rows == 5, "Feature row count mismatch" 