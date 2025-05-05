import time
from typing import List, Dict, Optional

class LiquidityVacuumEvent:
    def __init__(self, direction: str, swept_levels: int, timestamp: float, pre_depth: int):
        self.direction = direction  # 'buy' or 'sell'
        self.swept_levels = swept_levels
        self.timestamp = timestamp
        self.pre_depth = pre_depth
        self.active = True
        self.entry_price = None
        self.exit_price = None
        self.exit_time = None

class Head9LiquidityVacuumStrategy:
    """
    Detects liquidity vacuum events in MNQ order book and trades momentum sweeps.
    - Triggers on empty price level (0 qty bid+ask), then ≥3 levels swept in <2ms.
    - Enters trade in sweep direction, 2x avg trade size, trails stop 3 ticks.
    - Exits when depth restores ≥150% of pre-vacuum or after 60s.
    - Tracks vacuum_count, win_rate, avg_slip.
    """
    def __init__(self):
        self.last_depths: Dict[float, Dict[str, int]] = {}  # price -> {'bid': qty, 'ask': qty}
        self.vacuum_events: List[LiquidityVacuumEvent] = []
        self.active_trade: Optional[LiquidityVacuumEvent] = None
        self.avg_trade_size = 1  # Placeholder, should be updated with real stats
        self.vacuum_count = 0
        self.win_count = 0
        self.total_slip = 0.0
        self.trade_count = 0

    def on_depth_update(self, price: float, bid_qty: int, ask_qty: int, timestamp: float):
        self.last_depths[price] = {'bid': bid_qty, 'ask': ask_qty}
        # Detect empty level
        if bid_qty == 0 and ask_qty == 0:
            self._empty_level_detected(price, timestamp)

    def _empty_level_detected(self, price: float, timestamp: float):
        # Wait for next market order to sweep ≥3 levels in <2ms
        self.pending_vacuum = {'price': price, 'timestamp': timestamp}

    def on_market_order(self, direction: str, swept_levels: int, price: float, timestamp: float):
        # Check for pending vacuum
        if hasattr(self, 'pending_vacuum'):
            dt = (timestamp - self.pending_vacuum['timestamp'])*1000
            if swept_levels >= 3 and dt < 2:
                event = LiquidityVacuumEvent(direction, swept_levels, timestamp, self._pre_vacuum_depth())
                self.vacuum_events.append(event)
                self.vacuum_count += 1
                self._enter_trade(event, price)
            del self.pending_vacuum

    def _pre_vacuum_depth(self):
        # Estimate pre-vacuum depth (sum of all levels)
        return sum(qty['bid']+qty['ask'] for qty in self.last_depths.values())

    def _enter_trade(self, event: LiquidityVacuumEvent, price: float):
        event.entry_price = price
        self.active_trade = event
        # Trade size: 2x average
        event.trade_size = 2 * self.avg_trade_size
        event.stop_price = price - 3 if event.direction == 'buy' else price + 3
        event.entry_time = time.time()

    def on_tick(self, price: float, depth: int, timestamp: float):
        # Trail stop logic
        if self.active_trade and self.active_trade.active:
            if (self.active_trade.direction == 'buy' and price <= self.active_trade.stop_price) or \
               (self.active_trade.direction == 'sell' and price >= self.active_trade.stop_price):
                self._exit_trade(price, timestamp)
            # Exit if depth restores
            elif self._current_depth() >= 1.5 * self.active_trade.pre_depth:
                self._exit_trade(price, timestamp)
            # Exit after 60s
            elif time.time() - self.active_trade.entry_time > 60:
                self._exit_trade(price, timestamp)

    def _current_depth(self):
        return sum(qty['bid']+qty['ask'] for qty in self.last_depths.values())

    def _exit_trade(self, price: float, timestamp: float):
        event = self.active_trade
        event.exit_price = price
        event.exit_time = timestamp
        event.active = False
        slip = (event.exit_price - event.entry_price) if event.direction == 'buy' else (event.entry_price - event.exit_price)
        self.total_slip += slip
        self.trade_count += 1
        if slip > 0:
            self.win_count += 1
        self.active_trade = None

    @property
    def win_rate(self):
        return self.win_count / self.trade_count if self.trade_count else 0.0

    @property
    def avg_slip(self):
        return self.total_slip / self.trade_count if self.trade_count else 0.0 