import pytest
from backend.data_ingestion.databento_ingest import validate_config

def test_databento_validate_config_missing(monkeypatch):
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.delenv("WS_URL", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        validate_config()
    assert "DATABENTO_API_KEY" in str(excinfo.value)
    assert "REDIS_URL" in str(excinfo.value)
    assert "WS_URL" in str(excinfo.value)

def test_databento_validate_config_present(monkeypatch):
    monkeypatch.setenv("DATABENTO_API_KEY", "testkey")
    monkeypatch.setenv("REDIS_URL", "redis://test")
    monkeypatch.setenv("WS_URL", "wss://test")
    # Should not raise
    validate_config() 