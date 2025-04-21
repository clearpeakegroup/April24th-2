import logging
from typing import Dict

class RiskManager:
    def __init__(self):
        self.position_limits = {}  # {user_id: {symbol: limit}}
        self.pnl_limits = {}       # {user_id: limit}
        self.positions = {}        # {user_id: {symbol: pos}}
        self.realized_pnl = {}     # {user_id: pnl}
        self.kill_switch = set()   # user_ids with kill switch

    def set_limits(self, user_id, symbol_limits: Dict[str, int], pnl_limit: float):
        self.position_limits[user_id] = symbol_limits
        self.pnl_limits[user_id] = pnl_limit
        self.positions[user_id] = {k: 0 for k in symbol_limits}
        self.realized_pnl[user_id] = 0.0

    def update_position(self, user_id, symbol, qty):
        self.positions[user_id][symbol] += qty
        logging.info(f"User {user_id} updated position: {symbol} = {self.positions[user_id][symbol]}")
        if abs(self.positions[user_id][symbol]) > self.position_limits[user_id][symbol]:
            self.trigger_kill_switch(user_id, f"Position limit exceeded for {symbol}")

    def update_pnl(self, user_id, pnl):
        self.realized_pnl[user_id] += pnl
        logging.info(f"User {user_id} updated PnL: {self.realized_pnl[user_id]}")
        if abs(self.realized_pnl[user_id]) > self.pnl_limits[user_id]:
            self.trigger_kill_switch(user_id, "PnL limit exceeded")

    def trigger_kill_switch(self, user_id, reason):
        self.kill_switch.add(user_id)
        logging.error(f"KILL SWITCH ACTIVATED for user {user_id}: {reason}")

    def check(self, user_id):
        if user_id in self.kill_switch:
            raise Exception(f"Trading halted by risk manager for user {user_id}.")

    def check_order(self, user_id, symbol, qty, side):
        if user_id in self.kill_switch:
            raise Exception(f"Trading halted by risk manager for user {user_id}.")
        # Check position limit
        pos = self.positions[user_id][symbol] + (qty if side == "buy" else -qty)
        if abs(pos) > self.position_limits[user_id][symbol]:
            self.trigger_kill_switch(user_id, f"Order would exceed position limit for {symbol}")
            raise Exception(f"Order would exceed position limit for {symbol}")
        # Check PnL limit (stub, real logic would be more complex)
        if abs(self.realized_pnl[user_id]) > self.pnl_limits[user_id]:
            self.trigger_kill_switch(user_id, "PnL limit exceeded")
            raise Exception("PnL limit exceeded") 