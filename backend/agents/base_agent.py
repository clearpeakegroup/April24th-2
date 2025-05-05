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
        raise RuntimeError("train() must be implemented by a subclass of BaseAgent.")
    def act(self, *args, **kwargs):
        raise RuntimeError("act() must be implemented by a subclass of BaseAgent.")
    def save(self, *args, **kwargs):
        raise RuntimeError("save() must be implemented by a subclass of BaseAgent.")
    def load(self, *args, **kwargs):
        raise RuntimeError("load() must be implemented by a subclass of BaseAgent.")
    def evaluate(self, *args, **kwargs):
        raise RuntimeError("evaluate() must be implemented by a subclass of BaseAgent.") 