import numpy as np
import time
from dataclasses import dataclass
from utils.kalman import KalmanBeta
from collections import deque

@dataclass
class Action:
    mnq_side: str
    mes_side: str
    mnq_qty: int
    mes_qty: int
    entry_ts: int
    zscore: float
    spread: float
    beta: float
    exit_reason: str = ''

class Head6IntradaySpreadStrategy:
    """
    MNQ-β·MES z-score scalp: Kalman/OLS beta, rolling spread, zscore, entry/exit/stop logic.
    """
    def __init__(self, config=None):
        cfg = config or {}
        self.ols_window = cfg.get('ols_window', 20*60)  # 20 min in seconds
        self.lookback = cfg.get('lookback', 2*60*60)   # 2h in seconds
        self.zscore_entry = cfg.get('zscore_entry', 2.0)
        self.zscore_exit = cfg.get('zscore_exit', 0.3)
        self.zscore_stop = cfg.get('zscore_stop', 3.5)
        self.max_horizon = cfg.get('max_horizon', 10*60)  # 10 min in seconds
        self.kalman = KalmanBeta()
        self.mnq_prices = deque(maxlen=self.lookback)
        self.mes_prices = deque(maxlen=self.lookback)
        self.spreads = deque(maxlen=self.lookback)
        self.last_entry_ts = 0
        self.position = None
        self.round_trips = 0
        self.pnl = 0.0
    def _zscore(self, spread):
        arr = np.array(self.spreads)
        mu = np.mean(arr) if len(arr) > 10 else 0.0
        sigma = np.std(arr) if len(arr) > 10 else 1.0
        return (spread - mu) / (sigma if sigma > 1e-8 else 1.0)
    def act(self, mnq_px, mes_px, ts=None):
        ts = ts or int(time.time())
        self.mnq_prices.append(mnq_px)
        self.mes_prices.append(mes_px)
        # Update beta
        if len(self.mnq_prices) > 20:
            beta = self.kalman.update(np.array(self.mes_prices)[-20:], np.array(self.mnq_prices)[-20:])
        else:
            beta = 1.0
        spread = mnq_px - beta * mes_px
        self.spreads.append(spread)
        z = self._zscore(spread)
        # Entry
        if not self.position:
            if z >= self.zscore_entry:
                # short MNQ, long MES (qty ratio beta)
                mnq_side, mes_side = 'SELL', 'BUY'
                mnq_qty = 1
                mes_qty = int(round(beta))
                self.position = Action(mnq_side, mes_side, mnq_qty, mes_qty, ts, z, spread, beta)
                self.last_entry_ts = ts
                return self.position
            elif z <= -self.zscore_entry:
                # long MNQ, short MES
                mnq_side, mes_side = 'BUY', 'SELL'
                mnq_qty = 1
                mes_qty = int(round(beta))
                self.position = Action(mnq_side, mes_side, mnq_qty, mes_qty, ts, z, spread, beta)
                self.last_entry_ts = ts
                return self.position
        # Exit
        if self.position:
            elapsed = ts - self.last_entry_ts
            if abs(z) < self.zscore_exit:
                self.position.exit_reason = 'zscore mean-revert'
                self.position = None
                self.round_trips += 1
                return None
            if abs(z) > self.zscore_stop:
                self.position.exit_reason = 'stop-loss'
                self.position = None
                self.round_trips += 1
                return None
            if elapsed > self.max_horizon:
                self.position.exit_reason = 'max horizon'
                self.position = None
                self.round_trips += 1
                return None
        return None
