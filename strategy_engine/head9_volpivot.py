import pandas as pd
import numpy as np
from typing import Iterable, Iterator, AsyncIterable, AsyncIterator
from .base import BaseStrategy, BacktestResult, Progress
from .utils import Order, calc_pnl, apply_slippage

class StrategyHead(BaseStrategy):
    def backtest(self, data: pd.DataFrame) -> BacktestResult:
        # Stub: diff-QP risk-parity allocation
        trades = []
        equity_curve = []
        cash = 100000
        # Use a simple QP: min w'Vw s.t. sum(w)=1, w>=0
        cov = data.cov()
        inv_vol = 1 / np.sqrt(np.diag(cov))
        weights = inv_vol / inv_vol.sum()
        for i, row in data.iterrows():
            alloc = (cash * weights).to_dict()
            for sym, amt in alloc.items():
                price = apply_slippage(row[sym])
                trades.append({'side': 'buy', 'qty': amt/price, 'price': price, 'symbol': sym})
            equity_curve.append(cash)
        pnl = sum([t['qty'] * (data[t['symbol']].iloc[-1] - t['price']) for t in trades])
        return BacktestResult(pnl, trades, equity_curve)

    def forward(self, data_iter: Iterable) -> Iterator[Progress]:
        for i, row in enumerate(data_iter):
            yield Progress(pct=min(100, (i+1)*10), msg=f"Step {i+1}")

    async def live(self, feed: AsyncIterable) -> AsyncIterator[Order]:
        async for row in feed:
            for sym in row:
                yield Order(symbol=sym, qty=1, price=row[sym], side='buy') 