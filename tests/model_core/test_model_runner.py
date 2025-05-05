import pytest
from backend.model_core.model_runner import ModelRunner
from backend.models.model_io import ModelInputBatch, ModelInputFeature, ModelOutputBatch
from datetime import datetime

def make_input_batch():
    feats = [
        ModelInputFeature(
            asset=asset,
            timestamp=datetime.utcnow().isoformat(),
            best_bid=100.0,
            best_ask=101.0,
            spread=1.0,
            depth_bid=10.0,
            depth_ask=5.0,
            sentiment=0.5,
            news_length=42
        ) for asset in ['MES', 'MNQ', 'ZF', 'ZN', 'UB']
    ]
    return ModelInputBatch(features=feats)

def test_model_runner_output_schema():
    runner = ModelRunner()
    batch = make_input_batch()
    output = runner.run_inference(batch)
    assert isinstance(output, ModelOutputBatch)
    for action in output.actions:
        assert action.asset in {'MES', 'MNQ'}
        assert action.action in {'buy', 'sell', 'hold'}
        assert isinstance(action.confidence, float)

def test_model_runner_only_mes_mnq():
    runner = ModelRunner()
    batch = make_input_batch()
    output = runner.run_inference(batch)
    assets = {a.asset for a in output.actions}
    assert assets.issubset({'MES', 'MNQ'}) 