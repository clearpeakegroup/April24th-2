import pytest
import os
from backend.data_ingestion.news_benzinga import validate_config
from loguru import logger


def test_benzinga_validate_config_missing(monkeypatch):
    monkeypatch.delenv("BENZINGA_WS_URL", raising=False)
    monkeypatch.delenv("BENZINGA_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        validate_config()
    assert "BENZINGA_WS_URL" in str(excinfo.value)
    assert "BENZINGA_API_KEY" in str(excinfo.value)


def test_benzinga_validate_config_present(monkeypatch):
    monkeypatch.setenv("BENZINGA_WS_URL", "wss://test")
    monkeypatch.setenv("BENZINGA_API_KEY", "testkey")
    # Should not raise
    validate_config() 