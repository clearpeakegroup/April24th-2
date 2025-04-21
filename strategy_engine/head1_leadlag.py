import pandas as pd
import numpy as np
from typing import Iterable, Iterator, AsyncIterable, AsyncIterator
from .base import BaseStrategy, BacktestResult, Progress
from .utils import Order, calc_pnl, apply_slippage

class StrategyHead(BaseStrategy):
    def backtest(self, data: pd.DataFrame) -> BacktestResult:
        # Stub: Buy MES if MNQ leads up, sell if ZN leads down
        trades = []
        equity_curve = []
        cash = 100000
        pos = 0
        for i, row in data.iterrows():
            if row['MNQ'] > row['MNQ'].mean() and pos == 0:
                price = apply_slippage(row['MES'])
                trades.append({'side': 'buy', 'qty': 1, 'price': price})
                pos = 1
                cash -= price
            elif row['ZN'] < row['ZN'].mean() and pos == 1:
                price = apply_slippage(row['MES'])
                trades.append({'side': 'sell', 'qty': 1, 'price': price})
                pos = 0
                cash += price
            equity_curve.append(cash + (row['MES'] if pos else 0))
        pnl = calc_pnl(trades, data['MES'])
        return BacktestResult(pnl, trades, equity_curve)

    def forward(self, data_iter: Iterable) -> Iterator[Progress]:
        # Simulate progress
        for i, row in enumerate(data_iter):
            yield Progress(pct=min(100, (i+1)*10), msg=f"Step {i+1}")

    async def live(self, feed: AsyncIterable) -> AsyncIterator[Order]:
        async for row in feed:
            # Dummy: always yield a buy order
            yield Order(symbol='MES', qty=1, price=row['MES'], side='buy') 