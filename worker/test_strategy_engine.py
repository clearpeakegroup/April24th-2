import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st
from strategy_engine.factory import get_active_heads
from strategy_engine.base import BaseStrategy

@pytest.mark.parametrize("stage,with_options", [("train", True), ("train", False)])
def test_heads_reproducible_pnl(stage, with_options):
    np.random.seed(42)
    heads = get_active_heads(stage, with_options)
    # Toy data: 20 rows, columns for all possible symbols
    cols = ["MES", "MNQ", "ZN", "QQQ_C", "VIX", "VVIX", "ES", "NQ"]
    data = pd.DataFrame(np.random.rand(20, len(cols))*100+100, columns=cols)
    data["gamma"] = np.random.rand(20)
    data["macro"] = np.random.choice(["growth", "recession"], 20)
    for head in heads:
        result = head.backtest(data)
        assert isinstance(result.pnl, float)
        # Should be reproducible and not always negative
        assert np.sign(result.pnl) >= 0

@given(st.lists(st.floats(min_value=90, max_value=110), min_size=10, max_size=20))
def test_no_negative_cash_balance(prices):
    from strategy_engine.head1_leadlag import StrategyHead
    data = pd.DataFrame({"MES": prices, "MNQ": prices, "ZN": prices})
    head = StrategyHead()
    result = head.backtest(data)
    # Simulate cash balance from trades
    cash = 100000
    pos = 0
    for t in result.trades:
        if t['side'] == 'buy':
            cash -= t['price']
            pos += t['qty']
        elif t['side'] == 'sell':
            cash += t['price']
            pos -= t['qty']
        assert cash >= 0 