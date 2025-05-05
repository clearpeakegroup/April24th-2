import json
import zstandard as zstd
import pytest
from backend.data_ingestion.decompression import decompress_zstd

def test_news_zstd_decompression_and_json():
    payload = {"headline": "Test News", "body": "Some text.", "source": "test"}
    raw = json.dumps(payload).encode()
    cctx = zstd.ZstdCompressor()
    compressed = cctx.compress(raw)
    decompressed = decompress_zstd(compressed)
    loaded = json.loads(decompressed.decode())
    assert loaded == payload 