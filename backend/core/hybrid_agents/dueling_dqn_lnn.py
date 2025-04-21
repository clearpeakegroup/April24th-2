from finrl.models import DRLAgent
from stable_baselines3 import DQN
from backend.core.lnn_models import LNNFeatureExtractor
import torch.nn as nn

class DuelingLNNQNetwork(nn.Module):
    """Dueling DQN with LNN feature extractor."""
    def __init__(self, obs_space, action_space, net_arch=None):
        super().__init__()
        in_dim = obs_space.shape[-1]
        self.lnn = LNNFeatureExtractor(in_dim, 64, 32)
        self.adv = nn.Linear(32, action_space.n)
        self.val = nn.Linear(32, 1)
    def forward(self, x):
        h = self.lnn(x)
        adv = self.adv(h)
        val = self.val(h)
        return val + adv - adv.mean(dim=1, keepdim=True)

class DuelingDQNLNN(DRLAgent):
    """FinRL DRLAgent using DuelingLNNQNetwork with DQN."""
    def get_model(self, env, **kwargs):
        return DQN(policy=DuelingLNNQNetwork, env=env, **kwargs) 