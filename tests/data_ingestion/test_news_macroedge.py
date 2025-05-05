import pytest
from backend.data_ingestion.news_macroedge import validate_config

def test_macroedge_validate_config_missing(monkeypatch):
    monkeypatch.delenv("MACROEDGE_API_URL", raising=False)
    monkeypatch.delenv("MACROEDGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        validate_config()
    assert "MACROEDGE_API_URL" in str(excinfo.value)
    assert "MACROEDGE_API_KEY" in str(excinfo.value)

def test_macroedge_validate_config_present(monkeypatch):
    monkeypatch.setenv("MACROEDGE_API_URL", "https://test")
    monkeypatch.setenv("MACROEDGE_API_KEY", "testkey")
    # Should not raise
    validate_config() 