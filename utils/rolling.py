import numpy as np
from collections import deque

class RollingStats:
    def __init__(self, window):
        self.window = window
        self.data = deque(maxlen=window)
    def update(self, value):
        self.data.append(value)
    @property
    def mean(self):
        return np.mean(self.data) if self.data else 0.0
    @property
    def std(self):
        return np.std(self.data) if self.data else 1.0
    def zscore(self, value):
        return (value - self.mean) / (self.std if self.std > 1e-8 else 1.0)
