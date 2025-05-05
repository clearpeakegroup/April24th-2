import pandas as pd
from typing import Iterable, Iterator, AsyncIterable, AsyncIterator
from .base import BaseStrategy, BacktestResult, Progress
from .utils import Order, calc_pnl, apply_slippage

class StrategyHead(BaseStrategy):
    def backtest(self, data: pd.DataFrame) -> BacktestResult:
        # Stub: Buy if vol spread widens, sell if narrows
        trades = []
        equity_curve = []
        cash = 100000
        pos = 0
        for i, row in data.iterrows():
            spread = row['VIX'] - row['VVIX']
            if spread > 2 and pos == 0:
                price = apply_slippage(row['VIX'])
                trades.append({'side': 'buy', 'qty': 1, 'price': price})
                pos = 1
                cash -= price
            elif spread < -2 and pos == 1:
                price = apply_slippage(row['VIX'])
                trades.append({'side': 'sell', 'qty': 1, 'price': price})
                pos = 0
                cash += price
            equity_curve.append(cash + (row['VIX'] if pos else 0))
        pnl = calc_pnl(trades, data['VIX'])
        return BacktestResult(pnl, trades, equity_curve)

    def forward(self, data_iter: Iterable) -> Iterator[Progress]:
        for i, row in enumerate(data_iter):
            yield Progress(pct=min(100, (i+1)*10), msg=f"Step {i+1}")

    async def live(self, feed: AsyncIterable) -> AsyncIterator[Order]:
        async for row in feed:
            yield Order(symbol='VIX', qty=1, price=row['VIX'], side='buy') 