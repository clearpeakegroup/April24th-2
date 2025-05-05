import pytest
from backend.data_ingestion.news_marketwatch import validate_config

def test_marketwatch_validate_config_missing(monkeypatch):
    monkeypatch.delenv("MARKETWATCH_API_URL", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        validate_config()
    assert "MARKETWATCH_API_URL" in str(excinfo.value)

def test_marketwatch_validate_config_present(monkeypatch):
    monkeypatch.setenv("MARKETWATCH_API_URL", "https://test")
    # Should not raise
    validate_config() 