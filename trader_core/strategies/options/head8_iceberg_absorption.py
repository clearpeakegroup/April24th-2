import time
from collections import deque, defaultdict
from typing import List, Dict, Optional

class IcebergEvent:
    """
    Represents an iceberg absorption event detected in the order book.
    """
    def __init__(self, side: str, price: float, size_est: float, timestamp: float):
        """
        Args:
            side: 'bid' or 'ask'.
            price: Price level of the event.
            size_est: Estimated iceberg size.
            timestamp: Event timestamp.
        """
        self.side = side  # 'bid' or 'ask'
        self.price = price
        self.size_est = size_est
        self.timestamp = timestamp

class Head8IcebergAbsorptionStrategy:
    """
    Detects iceberg absorption patterns in real-time order book data.
    - Monitors queue depth deltas within 5ms windows.
    - Flags iceberg if >5 successive executions at same price and visible depth doesn't drop.
    - Emits buy/sell signals based on detected absorption.
    """
    def __init__(self, window_ms: int = 5):
        """
        Args:
            window_ms: Window in milliseconds for grouping executions.
        """
        self.window_ms = window_ms
        self.last_depths: Dict[float, int] = {}  # price -> visible depth
        self.exec_history: deque = deque(maxlen=10)  # recent executions
        self.icebergs: List[IcebergEvent] = []
        self.active_signal: Optional[Dict] = None
        self.last_signal_time: Optional[float] = None

    def on_depth_update(self, price: float, depth: int, side: str, timestamp: float):
        """
        Update the visible depth for a given price and side.
        """
        self.last_depths[(side, price)] = depth

    def on_execution(self, price: float, qty: int, side: str, timestamp: float):
        """
        Record an execution and check for iceberg detection.
        """
        self.exec_history.append((price, qty, side, timestamp))
        self._detect_iceberg()

    def _detect_iceberg(self):
        """
        Detect iceberg absorption events based on recent executions and depth.
        """
        now = time.time()
        grouped = defaultdict(list)
        for price, qty, side, ts in self.exec_history:
            if now - ts < self.window_ms / 1000.0:
                grouped[(side, price)].append((qty, ts))
        for (side, price), events in grouped.items():
            if len(events) > 5:
                depth = self.last_depths.get((side, price), None)
                if depth is not None and all(self.last_depths.get((side, price), depth) == depth for _, _ in events):
                    size_est = sum(qty for qty, _ in events)
                    iceberg = IcebergEvent(side, price, size_est, now)
                    self.icebergs.append(iceberg)
                    self._emit_signal(iceberg)

    def _emit_signal(self, iceberg: IcebergEvent):
        """
        Emit a buy/sell signal based on the detected iceberg event.
        """
        if self.active_signal and time.time() - self.last_signal_time < 30:
            return
        if iceberg.side == 'bid':
            signal = {'action': 'buy', 'symbol': 'MES', 'qty': iceberg.size_est / 2, 'timestamp': iceberg.timestamp}
        else:
            signal = {'action': 'sell', 'symbol': 'MES', 'qty': iceberg.size_est, 'timestamp': iceberg.timestamp}
        self.active_signal = signal
        self.last_signal_time = time.time()

    def check_stop(self, new_iceberg: Optional[IcebergEvent] = None):
        """
        Check if the strategy should stop due to an opposite iceberg or timeout.
        """
        if not self.active_signal:
            return False
        if new_iceberg and ((self.active_signal['action'] == 'buy' and new_iceberg.side == 'ask') or (self.active_signal['action'] == 'sell' and new_iceberg.side == 'bid')):
            self.active_signal = None
            return True
        if time.time() - self.last_signal_time > 30:
            self.active_signal = None
            return True
        return False

class DepthQueueListener:
    """
    Utility to listen to real-time depth queue updates and executions.
    """
    def __init__(self, strategy: Head8IcebergAbsorptionStrategy):
        """
        Args:
            strategy: Instance of Head8IcebergAbsorptionStrategy.
        """
        self.strategy = strategy

    def on_depth(self, price: float, depth: int, side: str, timestamp: float):
        """
        Pass depth update to the strategy.
        """
        self.strategy.on_depth_update(price, depth, side, timestamp)

    def on_execution(self, price: float, qty: int, side: str, timestamp: float):
        """
        Pass execution event to the strategy.
        """
        self.strategy.on_execution(price, qty, side, timestamp) 