import numpy as np
import time
from dataclasses import dataclass
from utils.rolling import RollingStats

@dataclass
class Action:
    mes_side: str
    mnq_side: str
    mes_qty: int
    mnq_qty: int
    entry_ts: int
    zscore: float
    rel: float
    sigma_mes: float
    sigma_mnq: float
    exit_reason: str = ''

class Head4CrossVolSpreadStrategy:
    """
    MES vs MNQ relative-vol scalper: trade straddle spread on z-score of IV ratio.
    """
    def __init__(self, config=None):
        cfg = config or {}
        self.vega_size = cfg.get('vega_size', 2000)
        self.zscore_entry = cfg.get('zscore_entry', 2.0)
        self.zscore_exit = cfg.get('zscore_exit', 0.5)
        self.rolling_window = cfg.get('rolling_window', 30)
        self.dte_target = cfg.get('dte_target', 7)
        self.hard_stop_sigma = cfg.get('hard_stop_sigma', 1.0)
        self.stats = RollingStats(self.rolling_window)
        self.last_entry_ts = 0
        self.cooldown_min = 10
        self.position = None
        self.pnl = 0.0

    def _atm_iv(self, greeks):
        # Find ATM IV for given DTE
        atm = min((g for g in greeks if abs(g['dte'] - self.dte_target) <= 1), key=lambda g: abs(g['delta']), default=None)
        return atm['iv'] if atm else 0.0

    def act(self, mes_greeks, mnq_greeks, mes_tick=None, mnq_tick=None):
        now_ns = int(time.time() * 1e9)
        sigma_mes = self._atm_iv(mes_greeks)
        sigma_mnq = self._atm_iv(mnq_greeks)
        if sigma_mes <= 0 or sigma_mnq <= 0:
            return None
        rel = sigma_mnq / sigma_mes
        self.stats.update(rel)
        zscore = self.stats.zscore(rel)
        # Entry logic
        if not self.position:
            if zscore >= self.zscore_entry:
                # long MES straddle, short MNQ straddle
                mes_side, mnq_side = 'BUY', 'SELL'
                mes_qty = mnq_qty = int(self.vega_size / sigma_mes)
                self.position = Action(mes_side, mnq_side, mes_qty, mnq_qty, now_ns, zscore, rel, sigma_mes, sigma_mnq)
                self.last_entry_ts = now_ns
                return self.position
            elif zscore <= -self.zscore_entry:
                # short MES straddle, long MNQ straddle
                mes_side, mnq_side = 'SELL', 'BUY'
                mes_qty = mnq_qty = int(self.vega_size / sigma_mes)
                self.position = Action(mes_side, mnq_side, mes_qty, mnq_qty, now_ns, zscore, rel, sigma_mes, sigma_mnq)
                self.last_entry_ts = now_ns
                return self.position
        # Exit logic
        if self.position:
            if abs(zscore) < self.zscore_exit:
                self.position.exit_reason = 'zscore mean-revert'
                self.position = None
                return None
            # Hard stop: loss > 1σ move (stub, would need PnL tracking)
            # if self.pnl < -self.hard_stop_sigma * self.stats.std:
            #     self.position.exit_reason = 'hard stop'
            #     self.position = None
            #     return None
        return None
