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
    stop_px: float
    tgt_px: float

class Head1LeadLagStrategy:
    def __init__(self, config=None):
        # Config defaults
        cfg = config or {}
        self.tb_threshold = cfg.get('tb_threshold', 12)
        self.dq_threshold = cfg.get('dq_threshold', 0.30)
        self.confirm_max_lag_ms = cfg.get('confirm_max_lag_ms', 20)
        self.base_size_mes = cfg.get('base_size_mes', 2)
        self.base_size_mnq = cfg.get('base_size_mnq', 4)
        self.max_outrun_coeff = cfg.get('max_outrun_coeff', 3)
        self.stop_ticks = cfg.get('stop_ticks', 4)
        self.tgt_ticks = cfg.get('tgt_ticks', 6)
        self.max_horizon_ms = cfg.get('max_horizon_ms', 2000)
        self.tick_size = cfg.get('tick_size', 0.25)  # Assume 0.25 for MES/MNQ
        self.last_entry_ts = 0
        self.cooldown_ms = 250
        # Rolling windows for features
        self.zn_events = deque()
        self.mes_events = deque()
        self.mnq_events = deque()
        self.last_action = None

    def _now(self):
        return int(time.time() * 1e9)  # nanoseconds

    def _window(self, events, now_ns, window_ms):
        cutoff = now_ns - window_ms * 1_000_000
        while events and events[0]['ts'] < cutoff:
            events.popleft()
        return list(events)

    def _queue_imbalance(self, book):
        bid = book['bid1']
        ask = book['ask1']
        if bid + ask == 0:
            return 0.0
        return (bid - ask) / (bid + ask)

    def _delta_qi(self, qi_series):
        if len(qi_series) < 2:
            return 0.0
        return qi_series[-1] - qi_series[0]

    def _sweep_depth(self, trades, best_px):
        return sum(abs(tr['price'] - best_px) for tr in trades)

    def _trade_burst(self, trades):
        return len(trades)

    def _entropy(self, trades):
        if not trades:
            return 0.0
        sides = [tr['side'] for tr in trades]
        p_buy = sides.count('buy') / len(sides)
        p_sell = 1 - p_buy
        H = 0
        for p in [p_buy, p_sell]:
            if p > 0:
                H -= p * np.log(p)
        return H

    def _features(self, zn_events, now_ns):
        # Only use last 100ms for features, 50ms for deltas
        window_100 = self._window(zn_events, now_ns, 100)
        window_50 = self._window(zn_events, now_ns, 50)
        if not window_100:
            return None
        # QI series for delta
        qi_series = [self._queue_imbalance(e['book']) for e in window_50]
        delta_qi = self._delta_qi(qi_series)
        sweep_depth = self._sweep_depth([e for e in window_50 if e['type']=='trade'], window_50[-1]['book']['bid1'])
        trade_burst = self._trade_burst([e for e in window_50 if e['type']=='trade'])
        entropy = self._entropy([e for e in window_50 if e['type']=='trade'])
        return {
            'QI': qi_series[-1] if qi_series else 0.0,
            'delta_QI': delta_qi,
            'SweepDepth': sweep_depth,
            'TradeBurst': trade_burst,
            'Entropy': entropy
        }

    def act(self, zn_tick, mes_tick, mnq_tick):
        now_ns = zn_tick['ts']
        # Maintain rolling windows
        self.zn_events.append(zn_tick)
        self.mes_events.append(mes_tick)
        self.mnq_events.append(mnq_tick)
        self._window(self.zn_events, now_ns, 100)
        self._window(self.mes_events, now_ns, 100)
        self._window(self.mnq_events, now_ns, 100)
        # Cooldown
        if now_ns - self.last_entry_ts < self.cooldown_ms * 1_000_000:
            return None
        # Feature extraction
        feats = self._features(self.zn_events, now_ns)
        if not feats:
            return None
        # Trigger logic
        buy_trigger = (
            feats['TradeBurst'] >= self.tb_threshold and
            feats['delta_QI'] >= self.dq_threshold and
            feats['SweepDepth'] >= 3
        )
        sell_trigger = (
            feats['TradeBurst'] >= self.tb_threshold and
            feats['delta_QI'] <= -self.dq_threshold and
            feats['SweepDepth'] >= 3
        )
        if not (buy_trigger or sell_trigger):
            return None
        # Confirmation within 20ms
        mes_qi = self._queue_imbalance(mes_tick['book'])
        mnq_qi = self._queue_imbalance(mnq_tick['book'])
        if buy_trigger:
            if not (mes_qi <= -0.10 and mnq_qi <= -0.10):
                return None
            side = 'buy'
        elif sell_trigger:
            if not (mes_qi >= 0.10 and mnq_qi >= 0.10):
                return None
            side = 'sell'
        # Entry sizing
        outrun_coeff = min(abs(feats['delta_QI']) / self.dq_threshold, self.max_outrun_coeff)
        size_mes = int(round(self.base_size_mes * outrun_coeff))
        size_mnq = int(round(self.base_size_mnq * outrun_coeff))
        # Entry px, stop, tgt
        entry_px_mes = mes_tick['book']['ask1'] if side == 'buy' else mes_tick['book']['bid1']
        entry_px_mnq = mnq_tick['book']['ask1'] if side == 'buy' else mnq_tick['book']['bid1']
        stop_px_mes = entry_px_mes - self.stop_ticks * self.tick_size if side == 'buy' else entry_px_mes + self.stop_ticks * self.tick_size
        tgt_px_mes = entry_px_mes + self.tgt_ticks * self.tick_size if side == 'buy' else entry_px_mes - self.tgt_ticks * self.tick_size
        stop_px_mnq = entry_px_mnq - self.stop_ticks * self.tick_size if side == 'buy' else entry_px_mnq + self.stop_ticks * self.tick_size
        tgt_px_mnq = entry_px_mnq + self.tgt_ticks * self.tick_size if side == 'buy' else entry_px_mnq - self.tgt_ticks * self.tick_size
        # Record entry
        self.last_entry_ts = now_ns
        self.last_action = [
            Action('MES', side, size_mes, now_ns, stop_px_mes, tgt_px_mes),
            Action('MNQ', side, size_mnq, now_ns, stop_px_mnq, tgt_px_mnq)
        ]
        return self.last_action 