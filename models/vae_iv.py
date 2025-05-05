import numpy as np

class VAEIV:
    def __init__(self, model_path=None, device='cuda'):
        self.model_path = model_path
        self.device = device
        # In real code, load model weights here
    def encode(self, iv_surface):
        # iv_surface: np.ndarray shape (37, n)
        # For demo, return a fake percentile based on mean IV
        mean_iv = np.mean(iv_surface)
        # Simulate percentile: 0 (low IV) to 100 (high IV)
        return int(np.clip((mean_iv - 0.1) / 0.4 * 100, 0, 100))
