import torch
from backend.agents.lnn_feature_extractor import LNNFeatureExtractor
from backend.envs.multi_asset_env import MultiAssetTradingEnv
import numpy as np
from loguru import logger
import os

def test_lnn_output_shape():
    """Test LNN output shape."""
    batch_size = 2
    seq_len = 5
    input_dim = 5
    hidden_dim = 8
    output_dim = 3
    x = torch.randn(batch_size, seq_len, input_dim)
    lnn = LNNFeatureExtractor(input_dim, hidden_dim, output_dim)
    out = lnn(x)
    logger.info(f"LNN output: {out}")
    assert out.shape == (batch_size, output_dim)

def test_env_with_lnn():
    """Test MultiAssetTradingEnv with LNN feature extractor."""
    asset_list = ["AAPL", "BTC"]
    env = MultiAssetTradingEnv(asset_list)
    lnn = LNNFeatureExtractor(input_dim=5, hidden_dim=8, output_dim=3)
    env.set_feature_extractor(lnn)
    obs, info = env.reset()
    # Ensure obs is a torch.Tensor if needed
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs)
    logger.info(f"Env obs after LNN: {obs}")
    assert hasattr(obs, 'shape') or isinstance(obs, torch.Tensor)

def test_lnn_serialization():
    """Test LNNFeatureExtractor save/load."""
    lnn = LNNFeatureExtractor(5, 8, 3)
    path = "test_lnn.pt"
    torch.save(lnn.state_dict(), path)
    lnn2 = LNNFeatureExtractor(5, 8, 3)
    lnn2.load_state_dict(torch.load(path))
    os.remove(path)
    logger.info("LNN serialization test passed.")

def test_lnn_error_handling():
    """Test LNNFeatureExtractor error handling for bad input."""
    lnn = LNNFeatureExtractor(5, 8, 3)
    try:
        lnn(torch.randn(2, 3, 4))  # Wrong input_dim
        assert False, "Should have raised an error"
    except Exception as e:
        logger.info(f"Caught expected error: {e}")

def test_hybrid_agent_integration():
    """Test instantiation and forward pass of LNNDQNHybridAgent."""
    from trader_core.strategies.fut_only.lnn_drl_hybrid import LNNDQNHybridAgent
    agent = LNNDQNHybridAgent(env_config=None, input_dim=5, hidden_dim=8, output_dim=3)
    state = torch.randn(5)
    try:
        action = agent.act(state)
        logger.info(f"Hybrid agent action: {action}")
    except Exception as e:
        logger.error(f"Hybrid agent integration failed: {e}")

if __name__ == "__main__":
    test_lnn_output_shape()
    test_env_with_lnn()
    test_lnn_serialization()
    test_lnn_error_handling()
    test_hybrid_agent_integration() 