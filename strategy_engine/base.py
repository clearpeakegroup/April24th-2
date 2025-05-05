import pandas as pd
from typing import Any, Iterator, AsyncIterator, Iterable, AsyncIterable, NamedTuple

class BacktestResult(NamedTuple):
    pnl: float
    trades: list
    equity_curve: list

class Progress(NamedTuple):
    pct: float
    msg: str

class BaseStrategy:
    def backtest(self, data: pd.DataFrame) -> BacktestResult:
        raise NotImplementedError
    def forward(self, data_iter: Iterable) -> Iterator[Progress]:
        raise NotImplementedError
    async def live(self, feed: AsyncIterable) -> AsyncIterator[Any]:
        raise NotImplementedError 