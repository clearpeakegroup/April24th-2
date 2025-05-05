import pytest
import pandas as pd
import asyncio
from strategy_engine.head10_news_trigger import StrategyHead

@pytest.mark.asyncio
async def test_head10_news_trigger_live():
    head = StrategyHead()
    # Mock news row with strong positive signal
    row = {"headline": "Fed cuts rates, stocks surge", "body": "Market expects rally.", "MES": 5000}
    async def feed():
        yield row
    trades = []
    async for order in head.live(feed()):
        trades.append(order)
        break  # Only need first trade
    assert trades, "No trade produced"
    assert trades[0].side in ("buy", "sell")
    assert trades[0].symbol == "MES"

def test_head10_news_trigger_backtest():
    head = StrategyHead()
    df = pd.DataFrame([
        {"headline": "Fed cuts rates", "body": "Market expects rally.", "MES": 5000},
        {"headline": "Earnings miss", "body": "Tech stocks fall.", "MES": 4950},
    ])
    result = head.backtest(df)
    assert isinstance(result.pnl, float)
    assert isinstance(result.trades, list)
    assert isinstance(result.equity_curve, list) 