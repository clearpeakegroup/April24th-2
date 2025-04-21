import pandas as pd
from typing import Iterable, Iterator, AsyncIterable, AsyncIterator
from .base import BaseStrategy, BacktestResult, Progress
from .utils import Order, calc_pnl, apply_slippage

class StrategyHead(BaseStrategy):
    def backtest(self, data: pd.DataFrame) -> BacktestResult:
        # Stub: Buy if GDP growth, sell if recession
        trades = []
        equity_curve = []
        cash = 100000
        pos = 0
        for i, row in data.iterrows():
            if row['macro'] == 'growth' and pos == 0:
                price = apply_slippage(row['ES'])
                trades.append({'side': 'buy', 'qty': 1, 'price': price})
                pos = 1
                cash -= price
            elif row['macro'] == 'recession' and pos == 1:
                price = apply_slippage(row['ES'])
                trades.append({'side': 'sell', 'qty': 1, 'price': price})
                pos = 0
                cash += price
            equity_curve.append(cash + (row['ES'] if pos else 0))
        pnl = calc_pnl(trades, data['ES'])
        return BacktestResult(pnl, trades, equity_curve)

    def forward(self, data_iter: Iterable) -> Iterator[Progress]:
        for i, row in enumerate(data_iter):
            yield Progress(pct=min(100, (i+1)*10), msg=f"Step {i+1}")

    async def live(self, feed: AsyncIterable) -> AsyncIterator[Order]:
        async for row in feed:
            yield Order(symbol='ES', qty=1, price=row['ES'], side='buy') 