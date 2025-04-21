import numpy as np
import pandas as pd
from typing import Dict, Any

class Order:
    def __init__(self, symbol: str, qty: float, price: float, side: str, ts=None):
        self.symbol = symbol
        self.qty = qty
        self.price = price
        self.side = side  # 'buy' or 'sell'
        self.ts = ts

    def to_dict(self):
        return dict(symbol=self.symbol, qty=self.qty, price=self.price, side=self.side, ts=self.ts)

def calc_pnl(trades: list, prices: pd.Series) -> float:
    # Simple P&L: sum((sell - buy) * qty)
    pnl = 0.0
    pos = 0.0
    entry = 0.0
    for t in trades:
        if t['side'] == 'buy':
            entry = t['price']
            pos += t['qty']
        elif t['side'] == 'sell' and pos > 0:
            pnl += (t['price'] - entry) * t['qty']
            pos -= t['qty']
    return pnl

def apply_slippage(price: float, slip_bps: float = 2) -> float:
    # Simulate slippage in basis points
    return price * (1 + np.random.uniform(-slip_bps, slip_bps) / 10000) 