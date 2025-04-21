import pandas as pd
from typing import Iterable, Iterator, AsyncIterable, AsyncIterator
from .base import BaseStrategy, BacktestResult, Progress
from .utils import Order, calc_pnl, apply_slippage

class StrategyHead(BaseStrategy):
    def backtest(self, data: pd.DataFrame) -> BacktestResult:
        # Stub: Try different params, pick best
        best_pnl = -float('inf')
        best_trades = []
        equity_curve = []
        for param in [1, 2, 3]:
            trades = []
            cash = 100000
            pos = 0
            for i, row in data.iterrows():
                if row['MES'] % param == 0 and pos == 0:
                    price = apply_slippage(row['MES'])
                    trades.append({'side': 'buy', 'qty': 1, 'price': price})
                    pos = 1
                    cash -= price
                elif row['MES'] % param != 0 and pos == 1:
                    price = apply_slippage(row['MES'])
                    trades.append({'side': 'sell', 'qty': 1, 'price': price})
                    pos = 0
                    cash += price
            pnl = calc_pnl(trades, data['MES'])
            if pnl > best_pnl:
                best_pnl = pnl
                best_trades = trades
        equity_curve = [100000 + best_pnl] * len(data)
        return BacktestResult(best_pnl, best_trades, equity_curve)

    def forward(self, data_iter: Iterable) -> Iterator[Progress]:
        for i, row in enumerate(data_iter):
            yield Progress(pct=min(100, (i+1)*10), msg=f"Step {i+1}")

    async def live(self, feed: AsyncIterable) -> AsyncIterator[Order]:
        async for row in feed:
            yield Order(symbol='MES', qty=1, price=row['MES'], side='buy') 