import pytest
import pandas as pd
from backend.data_ingestion.orderbook_feature_pipeline import OrderBookFeature, extract_orderbook_features, validate_config
from datetime import datetime

def test_orderbook_feature_schema_valid():
    df = pd.DataFrame({
        'side': ['buy', 'sell', 'buy', 'sell'],
        'price': [100.0, 101.0, 99.5, 102.0],
        'size': [10, 5, 7, 3],
        'ts_event': [1, 2, 3, 4]
    })
    ts = datetime.utcnow()
    features = extract_orderbook_features(df, 'MES', ts)
    feat = OrderBookFeature(**features)
    assert feat.asset == 'MES'
    assert isinstance(feat.best_bid, float)
    assert isinstance(feat.best_ask, float)
    assert isinstance(feat.spread, float)
    assert isinstance(feat.depth_bid, float)
    assert isinstance(feat.depth_ask, float)

def test_orderbook_feature_empty_df():
    df = pd.DataFrame(columns=['side', 'price', 'size', 'ts_event'])
    ts = datetime.utcnow()
    features = extract_orderbook_features(df, 'MNQ', ts)
    assert features is None

def test_orderbook_feature_missing_fields():
    df = pd.DataFrame({
        'side': ['buy'],
        'price': [100.0],
        # 'size' missing
        'ts_event': [1]
    })
    ts = datetime.utcnow()
    with pytest.raises(Exception):
        extract_orderbook_features(df, 'ZF', ts)

def test_orderbook_feature_validate_config_missing(monkeypatch):
    monkeypatch.delenv("RAW_MARKET_BASE", raising=False)
    monkeypatch.delenv("ORDERBOOK_FEATURE_BASE", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        validate_config()
    assert "RAW_MARKET_BASE" in str(excinfo.value)
    assert "ORDERBOOK_FEATURE_BASE" in str(excinfo.value)

def test_orderbook_feature_validate_config_present(monkeypatch):
    monkeypatch.setenv("RAW_MARKET_BASE", "data/lake/raw/market")
    monkeypatch.setenv("ORDERBOOK_FEATURE_BASE", "data/lake/parquet/features/orderbook")
    # Should not raise
    validate_config() 