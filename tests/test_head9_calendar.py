import pytest
import pandas as pd
import asyncio
from strategy_engine.head9_calendar import StrategyHead

@pytest.mark.asyncio
async def test_head9_calendar_live():
    head = StrategyHead()
    row = {"MES_FM": 4200, "MES_NM": 4203}
    async def feed():
        yield row
    trades = []
    async for order in head.live(feed()):
        trades.append(order)
        break
    assert trades, "No trade produced"
    assert trades[0].side in ("buy", "sell")
    assert trades[0].symbol == "MES_FM"

def test_head9_calendar_backtest():
    head = StrategyHead()
    df = pd.DataFrame([
        {"MES_FM": 4200, "MES_NM": 4203},
        {"MES_FM": 4205, "MES_NM": 4200},
    ])
    result = head.backtest(df)
    assert isinstance(result.pnl, float)
    assert isinstance(result.trades, list)
    assert isinstance(result.equity_curve, list) 