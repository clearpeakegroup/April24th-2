import pytest
import pandas as pd
import asyncio
from strategy_engine.head8_vol_arb import StrategyHead

@pytest.mark.asyncio
async def test_head8_vol_arb_live():
    head = StrategyHead()
    row = {"MES": 4200, "MES_gamma": 2.0, "MNQ_sigma": 0.8}
    async def feed():
        yield row
    trades = []
    async for order in head.live(feed()):
        trades.append(order)
        break
    assert trades, "No trade produced"
    assert trades[0].side in ("buy", "sell")
    assert trades[0].symbol == "MES"

def test_head8_vol_arb_backtest():
    head = StrategyHead()
    df = pd.DataFrame([
        {"MES": 4200, "MES_gamma": 2.0, "MNQ_sigma": 0.8},
        {"MES": 4190, "MES_gamma": -2.0, "MNQ_sigma": 1.2},
    ])
    result = head.backtest(df)
    assert isinstance(result.pnl, float)
    assert isinstance(result.trades, list)
    assert isinstance(result.equity_curve, list) 