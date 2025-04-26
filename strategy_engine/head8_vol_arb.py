import pandas as pd
from typing import Iterable, Iterator, AsyncIterable, AsyncIterator
from .base import BaseStrategy, BacktestResult, Progress
from .utils import Order, calc_pnl, apply_slippage
import numpy as np
from backend.agents.base_agent import get_device

class StrategyHead(BaseStrategy):
    def _signal(self, mes_gamma, mnq_sigma):
        # Signal: go long MES if gamma > threshold and MNQ sigma < threshold
        if mes_gamma > 1.5 and mnq_sigma < 1.0:
            return 1  # long
        elif mes_gamma < -1.5 and mnq_sigma > 1.0:
            return -1  # short
        return 0
    def backtest(self, data: pd.DataFrame) -> BacktestResult:
        trades = []
        pos = 0
        cash = 100000
        for i, row in data.iterrows():
            signal = self._signal(row['MES_gamma'], row['MNQ_sigma'])
            if signal == 1 and pos == 0:
                price = apply_slippage(row['MES'])
                trades.append({'side': 'buy', 'qty': 1, 'price': price})
                pos = 1
                cash -= price
            elif signal == -1 and pos == 1:
                price = apply_slippage(row['MES'])
                trades.append({'side': 'sell', 'qty': 1, 'price': price})
                pos = 0
                cash += price
        pnl = calc_pnl(trades, data['MES'])
        equity_curve = [100000 + pnl] * len(data)
        return BacktestResult(pnl, trades, equity_curve)
    def forward(self, data_iter: Iterable) -> Iterator[Progress]:
        for i, row in enumerate(data_iter):
            yield Progress(pct=min(100, (i+1)*10), msg=f"Step {i+1}")
    async def live(self, feed: AsyncIterable) -> AsyncIterator[Order]:
        pos = 0
        async for row in feed:
            signal = self._signal(row['MES_gamma'], row['MNQ_sigma'])
            if signal == 1 and pos == 0:
                yield Order(symbol='MES', qty=1, price=row['MES'], side='buy')
                pos = 1
            elif signal == -1 and pos == 1:
                yield Order(symbol='MES', qty=1, price=row['MES'], side='sell')
                pos = 0