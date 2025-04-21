import numpy as np
import time
from dataclasses import dataclass
from models.vae_iv import VAEIV

@dataclass
class Action:
    qqq_strike: float
    qqq_qty: int
    mes_qty: int
    mnq_qty: int
    entry_ts: int
    order_type: str
    order_style: str
    vega: float
    iv_pct: float
    zn_prob_drop: float
    exit_conds: dict

class LNNZN:
    """Stub for LNN yield drop estimator."""
    def __init__(self):
        pass
    def __call__(self, zn_ticks):
        # For demo: return 0.8 if mean price drops, else 0.2
        if len(zn_ticks) < 2:
            return 0.2
        return 0.8 if np.mean([t['px'] for t in zn_ticks[-10:]]) < np.mean([t['px'] for t in zn_ticks[:10]]) else 0.2

class Head3VegaPivotStrategy:
    """
    Rate-shock cross-asset vol arb: ZN yield drop + low QQQ IV triggers vega-neutral straddle.
    """
    def __init__(self, config=None, vae_model=None):
        cfg = config or {}
        self.target_vega = cfg.get('target_vega', 3000)
        self.dte_min = cfg.get('dte_min', 3)
        self.dte_max = cfg.get('dte_max', 5)
        self.funding_ratio = cfg.get('funding_ratio', 0.5)
        self.order_type = cfg.get('order_type', 'LMT')
        self.order_style = cfg.get('order_style', 'mid')
        self.exit_iv_pct = cfg.get('exit_iv_pct', 40)
        self.exit_iv_spread = cfg.get('exit_iv_spread', 3.0)
        self.max_horizon_min = cfg.get('max_horizon_min', 240)
        self.last_entry_ts = 0
        self.cooldown_min = 30
        self.vae = vae_model or VAEIV()
        self.lnn_zn = LNNZN()
        self.metrics = {}

    def act(self, zn_ticks, qqq_greeks, mes_greeks, mnq_greeks, iv_hist=None, runtime_cfg=None):
        now_ns = zn_ticks[-1]['ts'] if zn_ticks else int(time.time() * 1e9)
        # Option flag check
        if runtime_cfg and (runtime_cfg.get('futures_only') or not runtime_cfg.get('options_on')):
            return None
        # Feature extraction
        zn_prob_drop = self.lnn_zn(zn_ticks[-100:])  # last 500ms
        # Build IV surface tensor (37 x n)
        iv_surface = np.array([[g['iv'] for g in qqq_greeks]]) if qqq_greeks else np.zeros((37,1))
        iv_pct = self.vae.encode(iv_surface)
        # Trigger
        if not (zn_prob_drop >= 0.7 and iv_pct <= 20):
            return None
        # Find ATM QQQ strike, 3-5 DTE
        atm_qqq = min((g for g in qqq_greeks if self.dte_min <= g['dte'] <= self.dte_max), key=lambda g: abs(g['delta']), default=None)
        if not atm_qqq or atm_qqq['vega'] <= 0:
            return None
        qqq_qty = int(np.ceil(self.target_vega / atm_qqq['vega']))
        mes_qty = int(-0.5 * qqq_qty)
        mnq_qty = int(-0.5 * qqq_qty)
        # Risk/exit
        exit_conds = {
            'iv_pct': self.exit_iv_pct,
            'iv_spread': self.exit_iv_spread,
            'max_horizon_min': self.max_horizon_min
        }
        # Cooldown
        if now_ns - self.last_entry_ts < self.cooldown_min * 60 * 1_000_000_000:
            return None
        self.last_entry_ts = now_ns
        # Metrics (stub)
        self.metrics['vega_net'] = qqq_qty * atm_qqq['vega'] + mes_qty + mnq_qty
        # Output Action
        return Action(
            qqq_strike=atm_qqq['strike'],
            qqq_qty=qqq_qty,
            mes_qty=mes_qty,
            mnq_qty=mnq_qty,
            entry_ts=now_ns,
            order_type=self.order_type,
            order_style=self.order_style,
            vega=atm_qqq['vega'],
            iv_pct=iv_pct,
            zn_prob_drop=zn_prob_drop,
            exit_conds=exit_conds
        )
