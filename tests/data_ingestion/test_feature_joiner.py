import pytest
import pandas as pd
from backend.data_ingestion.feature_joiner import UnifiedFeature, align_and_join_features, BATCH_SEC
from datetime import datetime, timedelta

def make_orderbook_df(asset, ts):
    return pd.DataFrame({
        'asset': [asset],
        'timestamp': [ts],
        'best_bid': [100.0],
        'best_ask': [101.0],
        'spread': [1.0],
        'depth_bid': [10.0],
        'depth_ask': [5.0],
    })

def make_sentiment_df(asset, ts):
    return pd.DataFrame({
        'asset': [asset],
        'timestamp': [ts],
        'sentiment': [0.5],
        'length': [42],
    })

def test_unified_feature_schema_valid():
    ts = pd.Timestamp(datetime.utcnow())
    ob_df = make_orderbook_df('MES', ts)
    sent_df = make_sentiment_df('MES', ts)
    joined = align_and_join_features(ob_df, sent_df)
    assert not joined.empty
    record = joined.iloc[0].to_dict()
    feat = UnifiedFeature(**record)
    assert feat.asset == 'MES'
    assert isinstance(feat.sentiment, float)
    assert isinstance(feat.news_length, int)

def test_unified_feature_misaligned_timestamps():
    ts1 = pd.Timestamp(datetime.utcnow())
    ts2 = ts1 + timedelta(seconds=BATCH_SEC + 10)
    ob_df = make_orderbook_df('MNQ', ts1)
    sent_df = make_sentiment_df('MNQ', ts2)
    joined = align_and_join_features(ob_df, sent_df)
    assert joined.empty

def test_unified_feature_missing_sentiment():
    ts = pd.Timestamp(datetime.utcnow())
    ob_df = make_orderbook_df('ZF', ts)
    sent_df = pd.DataFrame(columns=['asset', 'timestamp', 'sentiment', 'length'])
    joined = align_and_join_features(ob_df, sent_df)
    assert joined.empty 