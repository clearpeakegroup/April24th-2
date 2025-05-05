import pytest
from backend.data_ingestion.live_stream import validate_config

def test_live_stream_validate_config_missing(monkeypatch):
    monkeypatch.delenv("REDIS_STREAM", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        validate_config()
    assert "REDIS_STREAM" in str(excinfo.value)

def test_live_stream_validate_config_present(monkeypatch):
    monkeypatch.setenv("REDIS_STREAM", "ticks")
    # Should not raise
    validate_config() 