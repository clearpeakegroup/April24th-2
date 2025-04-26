import torch
import os

def get_device():
    role = os.getenv("FINRL_DEVICE_ROLE", "live")
    if torch.cuda.is_available():
        if role == "train" and torch.cuda.device_count() > 1:
            return torch.device("cuda:1")
        return torch.device("cuda:0")
    return torch.device("cpu")

class BaseAgent:
    def __init__(self, env_config=None, device=None, **kwargs):
        self.device = device or get_device()
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