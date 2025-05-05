import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict

@dataclass
class Allocation:
    mes: float
    mnq: float
    zn: float
    regime: str
    ts: int

class TabNetRegime:
    """TabNet loader and inference for macro regime classification."""
    def __init__(self, model_path='models/tabnet_regime.pt', device='cpu'):
        self.device = device
        self.model = self._load_model(model_path)
        self.last_regime = None
        self.last_alloc = None
    def _load_model(self, path):
        # Stub: in real code, load TabNet weights
        class DummyTabNet:
            def eval(self):
                pass
            def __call__(self, x):
                # Return softmaxed logits for 3 classes
                # For demo: random regime
                out = np.zeros((x.shape[0], 3))
                out[:, np.random.choice(3)] = 1.0
                return torch.tensor(out)
        return DummyTabNet()
    def predict(self, features: np.ndarray) -> str:
        # features: shape (batch, n_features)
        self.model.eval()
        logits = self.model(torch.tensor(features, dtype=torch.float32))
        regime_idx = int(torch.argmax(logits, dim=1)[0])
        return ['risk-on', 'neutral', 'risk-off'][regime_idx]

class Head5MacroRegimeStrategy:
    """
    Macro regime allocator: TabNet 3-state classifier, allocation map, rebalance logic.
    """
    ALLOC_MAP = {
        'risk-on':   {'mes': 0.6,  'mnq': 0.4,  'zn': 0.0},
        'neutral':   {'mes': 0.3,  'mnq': 0.2,  'zn': 0.5},
        'risk-off':  {'mes': -0.3, 'mnq': -0.4, 'zn': 0.9},
    }
    def __init__(self, model_path='models/tabnet_regime.pt', device='cpu', margin_limit=0.6):
        self.tabnet = TabNetRegime(model_path, device)
        self.margin_limit = margin_limit
        self.last_regime = None
        self.last_alloc = None
    def act(self, features: Dict[str, float], netliq: float, ts: int) -> Allocation:
        # features: dict of all required features (1-min bars, book pressure, econ_calendar)
        # netliq: current net liquidation value
        # ts: current timestamp
        # Compose feature vector (order must match model)
        feat_vec = np.array([features[k] for k in sorted(features.keys())]).reshape(1, -1)
        regime = self.tabnet.predict(feat_vec)
        alloc = self.ALLOC_MAP[regime].copy()
        # Leverage cap
        gross = sum(abs(v) for v in alloc.values())
        if gross > self.margin_limit:
            scale = self.margin_limit / gross
            for k in alloc:
                alloc[k] *= scale
        # Rebalance if regime changed or drift > 5%
        rebalance = (
            self.last_regime != regime or
            (self.last_alloc and any(abs(alloc[k] - self.last_alloc[k]) > 0.05 for k in alloc))
        )
        if rebalance:
            self.last_regime = regime
            self.last_alloc = alloc.copy()
            return Allocation(alloc['mes'], alloc['mnq'], alloc['zn'], regime, ts)
        return None
