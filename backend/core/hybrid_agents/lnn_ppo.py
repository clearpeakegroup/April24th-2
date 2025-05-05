from finrl.model.models import DRLAgent
from stable_baselines3 import PPO
from backend.core.lnn_models import LNNFeatureExtractor
import torch.nn as nn

class LNNPolicy(nn.Module):
    """Custom policy with LNN feature extractor for PPO."""
    def __init__(self, obs_space, action_space, net_arch=None):
        super().__init__()
        in_dim = obs_space.shape[-1]
        self.lnn = LNNFeatureExtractor(in_dim, 64, 32)
        self.pi = nn.Linear(32, action_space.n)
        self.vf = nn.Linear(32, 1)
    def forward(self, x):
        h = self.lnn(x)
        return self.pi(h), self.vf(h)

class LNNPPO(DRLAgent):
    """FinRL DRLAgent using LNNPolicy with PPO."""
    def get_model(self, env, **kwargs):
        return PPO(policy=LNNPolicy, env=env, **kwargs) 