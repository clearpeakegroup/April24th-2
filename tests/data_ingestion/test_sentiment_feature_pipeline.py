import pytest
from backend.data_ingestion.sentiment_feature_pipeline import validate_config

def test_sentiment_feature_validate_config_missing(monkeypatch):
    monkeypatch.delenv("RAW_NEWS_BASE", raising=False)
    monkeypatch.delenv("SENTIMENT_FEATURE_BASE", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        validate_config()
    assert "RAW_NEWS_BASE" in str(excinfo.value)
    assert "SENTIMENT_FEATURE_BASE" in str(excinfo.value)

def test_sentiment_feature_validate_config_present(monkeypatch):
    monkeypatch.setenv("RAW_NEWS_BASE", "data/lake/raw/news")
    monkeypatch.setenv("SENTIMENT_FEATURE_BASE", "data/lake/parquet/features/sentiment")
    # Should not raise
    validate_config() 