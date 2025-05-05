import pytest
from backend.data_ingestion.news_twitter_x import validate_config

def test_twitter_x_validate_config_missing(monkeypatch):
    monkeypatch.delenv("TWITTER_X_WS_URL", raising=False)
    monkeypatch.delenv("TWITTER_X_BEARER", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        validate_config()
    assert "TWITTER_X_WS_URL" in str(excinfo.value)
    assert "TWITTER_X_BEARER" in str(excinfo.value)

def test_twitter_x_validate_config_present(monkeypatch):
    monkeypatch.setenv("TWITTER_X_WS_URL", "wss://test")
    monkeypatch.setenv("TWITTER_X_BEARER", "testbearer")
    # Should not raise
    validate_config() 