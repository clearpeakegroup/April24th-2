import pytest
import numpy as np
from backend.envs.multi_asset_quad_env import MultiAssetQuadEnv

def test_env_smoke():
    env = MultiAssetQuadEnv()
    obs, info = env.reset()
    assert obs is not None
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, float)
        if terminated or truncated:
            break
    env.render() 