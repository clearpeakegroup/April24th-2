import logging
from typing import Dict

class RiskManager:
    def __init__(self, position_limits: Dict[str, int], pnl_limit: float):
        self.position_limits = position_limits  # e.g., {'MES': 2, 'NQ': 1}
        self.pnl_limit = pnl_limit
        self.positions = {k: 0 for k in position_limits}
        self.realized_pnl = 0.0
        self.kill_switch = False

    def update_position(self, symbol: str, qty: int):
        self.positions[symbol] += qty
        logging.info(f"Updated position: {symbol} = {self.positions[symbol]}")
        if abs(self.positions[symbol]) > self.position_limits[symbol]:
            self.trigger_kill_switch(f"Position limit exceeded for {symbol}")

    def update_pnl(self, pnl: float):
        self.realized_pnl += pnl
        logging.info(f"Updated PnL: {self.realized_pnl}")
        if abs(self.realized_pnl) > self.pnl_limit:
            self.trigger_kill_switch("PnL limit exceeded")

    def trigger_kill_switch(self, reason: str):
        self.kill_switch = True
        logging.error(f"KILL SWITCH ACTIVATED: {reason}")

    def check(self):
        if self.kill_switch:
            raise Exception("Trading halted by risk manager.") 