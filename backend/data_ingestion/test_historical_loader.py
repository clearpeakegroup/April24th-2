import io
import pandas as pd
import zstandard as zstd
import pytest
from data_ingestion import historical_loader

def test_load_zstd_dbn(monkeypatch, tmp_path):
    # Create a fake CSV
    csv_content = "ts_event,instrument,side,price,size,exchange,type\n1,ABC,buy,100,10,NYSE,trade\n"
    compressed = zstd.ZstdCompressor().compress(csv_content.encode("utf-8"))
    file_path = tmp_path / "test.dbn.zst"
    with open(file_path, "wb") as f:
        f.write(compressed)
    df = historical_loader.load_zstd_dbn(str(file_path))
    assert isinstance(df, pd.DataFrame)
    assert set(historical_loader.SCHEMA).issubset(df.columns)
    assert df.iloc[0]["instrument"] == "ABC" 