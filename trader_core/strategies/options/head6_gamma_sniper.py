import numpy as np
import time
from dataclasses import dataclass
from collections import deque

@dataclass
class Action:
    symbol: str
    side: str
    qty: int
    entry_ts: int
    strike: float
    option_type: str
    hedge_qty: int
    stop_premium: float
    tgt_premium: float
    conf: float
    fut_side: str
    fut_qty: int

class LNNStub:
    """Lightweight LNN filter stub for confidence scoring."""
    def __init__(self, input_dim=3, hidden=16):
        self.input_dim = input_dim
        self.hidden = hidden
    def __call__(self, tensor):
        # For demo: confidence is sigmoid of mean feature
        return float(1 / (1 + np.exp(-np.mean(tensor))))

class Head6GammaSniperStrategy:
    """
    0-DTE QQQ gamma scalp, driven by MNQ/MES order-flow, with LNN filter and delta hedge.
    """
    def __init__(self, config=None):
        cfg = config or {}
        # Configurable params
        self.window_ms = cfg.get('window_ms', 250)
        self.stride_ms = cfg.get('stride_ms', 125)
        self.lnn_hidden = cfg.get('lnn_hidden', 16)
        self.conf_thresh = cfg.get('conf_thresh', 0.65)
        self.bull_qi_mnq = cfg.get('bull_qi_mnq', 0.25)
        self.bull_qi_mes = cfg.get('bull_qi_mes', 0.15)
        self.bear_qi_mnq = cfg.get('bear_qi_mnq', -0.25)
        self.bear_qi_mes = cfg.get('bear_qi_mes', -0.15)
        self.time_stop = cfg.get('time_stop', 180)
        self.hard_stop_opt = cfg.get('hard_stop_opt', -0.4)
        self.tgt_take = cfg.get('tgt_take', 0.6)
        self.hedge_tol = cfg.get('hedge_tol', 0.05)
        self.last_entry_ts = 0
        self.cooldown_ms = 60_000  # 1 min per side
        self.last_side = None
        # Rolling windows
        self.mnq_ticks = deque()
        self.mes_ticks = deque()
        self.qqq_greeks = deque()
        self.lnn = LNNStub(input_dim=3, hidden=self.lnn_hidden)
        # Metrics (stub)
        self.pnl_total = 0.0
        self.hit_rate = 0
        self.latency_ms = []
        self.avg_gamma_per_usd = 0.0

    def _now(self):
        return int(time.time() * 1e9)  # nanoseconds

    def _window(self, events, now_ns, window_ms):
        cutoff = now_ns - window_ms * 1_000_000
        while events and events[0]['ts'] < cutoff:
            events.popleft()
        return list(events)

    def _queue_imbalance(self, ticks):
        # QI = (Σbid_qty-Σask_qty)/(Σbid_qty+Σask_qty)
        bid_qty = sum(t['bid_qty'] for t in ticks)
        ask_qty = sum(t['ask_qty'] for t in ticks)
        denom = bid_qty + ask_qty
        return (bid_qty - ask_qty) / denom if denom else 0.0

    def _vwap_shift(self, greeks):
        # VWAP_QQQ = (Σtrd_px*qty / Σqty) - last_mid_px
        trades = [g for g in greeks if 'trd_px' in g]
        if not trades:
            return 0.0
        pxs = [g['trd_px'] for g in trades]
        qtys = [g['qty'] for g in trades]
        vwap = np.average(pxs, weights=qtys)
        last_mid = trades[-1].get('mid_px', vwap)
        return vwap - last_mid

    def _feature_tensor(self, now_ns):
        # Stack last 8 slices (stride 125ms) of [ΔQI_MNQ, ΔQI_MES, VWAP_QQQ]
        slices = []
        for i in range(8):
            t0 = now_ns - (self.window_ms - i * self.stride_ms) * 1_000_000
            mnq = [t for t in self.mnq_ticks if t['ts'] >= t0]
            mes = [t for t in self.mes_ticks if t['ts'] >= t0]
            greeks = [g for g in self.qqq_greeks if g['ts'] >= t0]
            dq_mnq = self._queue_imbalance(mnq)
            dq_mes = self._queue_imbalance(mes)
            vwap_q = self._vwap_shift(greeks)
            slices.append([dq_mnq, dq_mes, vwap_q])
        return np.array(slices)

    def _pick_option(self, greeks, side):
        # Only 0-DTE, pick argmax(gamma/ask_px)
        candidates = [g for g in greeks if g.get('ttm_s', 0) <= 6*3600 and g['ask_px'] > 0]
        if not candidates:
            return None
        if side == 'CALL':
            filtered = [g for g in candidates if g['type'] == 'call']
        else:
            filtered = [g for g in candidates if g['type'] == 'put']
        if not filtered:
            return None
        return max(filtered, key=lambda g: g['gamma']/g['ask_px'])

    def act(self, mnq_tick, mes_tick, qqq_greek_snapshot, runtime_cfg=None):
        """
        Args:
            mnq_tick: dict with ts, bid_qty, ask_qty, ...
            mes_tick: dict with ts, bid_qty, ask_qty, ...
            qqq_greek_snapshot: list of dicts (greeks)
            runtime_cfg: dict with futures_only, options_on
        Returns:
            Action or None
        """
        now_ns = mnq_tick['ts']
        # Option flag check
        if runtime_cfg and (runtime_cfg.get('futures_only') or not runtime_cfg.get('options_on')):
            return None
        # Maintain rolling windows
        self.mnq_ticks.append(mnq_tick)
        self.mes_ticks.append(mes_tick)
        for g in qqq_greek_snapshot:
            g['ts'] = now_ns  # tag with current time
            self.qqq_greeks.append(g)
        self._window(self.mnq_ticks, now_ns, self.window_ms)
        self._window(self.mes_ticks, now_ns, self.window_ms)
        self._window(self.qqq_greeks, now_ns, self.window_ms)
        # Feature extraction
        dq_mnq = self._queue_imbalance(self.mnq_ticks)
        dq_mes = self._queue_imbalance(self.mes_ticks)
        tensor = self._feature_tensor(now_ns)
        # Signal engine
        bullish = dq_mnq >= self.bull_qi_mnq and dq_mes >= self.bull_qi_mes
        bearish = dq_mnq <= self.bear_qi_mnq and dq_mes <= self.bear_qi_mes
        conf = self.lnn(tensor)
        fire = ((bullish or bearish) and conf >= self.conf_thresh)
        if not fire:
            return None
        side = 'CALL' if bullish else 'PUT'
        opt = self._pick_option(self.qqq_greeks, side)
        if not opt:
            return None
        # Execution proto
        opt_qty = 1
        fut_qty = int(round(opt_qty * opt['delta'] * 40))
        fut_side = 'SELL' if bullish else 'BUY'
        # Risk/exit params
        stop_premium = opt['ask_px'] * (1 + self.hard_stop_opt)
        tgt_premium = opt['ask_px'] * (1 + self.tgt_take)
        # Cooldown
        if now_ns - self.last_entry_ts < self.cooldown_ms * 1_000_000:
            return None
        self.last_entry_ts = now_ns
        self.last_side = side
        # Metrics (stub)
        self.latency_ms.append(0)  # placeholder
        # Output Action
        return Action(
            symbol=f"QQQ-{opt['strike']}-0DTE",
            side='BUY',
            qty=opt_qty,
            entry_ts=now_ns,
            strike=opt['strike'],
            option_type=side,
            hedge_qty=fut_qty,
            stop_premium=stop_premium,
            tgt_premium=tgt_premium,
            conf=conf,
            fut_side=fut_side,
            fut_qty=fut_qty
        ) 