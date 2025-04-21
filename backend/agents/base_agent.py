import torch

class BaseAgent:
    def __init__(self, env_config=None, device=None, **kwargs):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.env_config = env_config
    def train(self, *args, **kwargs):
        raise NotImplementedError
    def act(self, state):
        raise NotImplementedError
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError
    def save(self, path):
        raise NotImplementedError
    def load(self, path):
        raise NotImplementedError 