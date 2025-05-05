import time
import pytest
from trader_core.strategies.options.head9_liquidity_vacuum import Head9LiquidityVacuumStrategy
from utils.orderbook import detect_gaps

def test_detect_gaps():
    orderbook = [
        (100.0, 10, 5),
        (101.0, 0, 0),  # gap
        (102.0, 8, 7),
        (103.0, 0, 0),  # gap
    ]
    gaps = detect_gaps(orderbook)
    assert gaps == [101.0, 103.0]

@pytest.fixture
def strategy():
    return Head9LiquidityVacuumStrategy()

def test_liquidity_vacuum_triggers_once(strategy):
    # Simulate a gap event and a market order sweep
    now = time.time()
    strategy.on_depth_update(101.0, 0, 0, now)
    # Simulate a market order that sweeps 3 levels in <2ms
    strategy.on_market_order('buy', 3, 101.0, now + 0.001)
    assert strategy.vacuum_count == 1
    # Simulate another sweep at the same gap (should not double count)
    strategy.on_market_order('buy', 3, 101.0, now + 0.0015)
    assert strategy.vacuum_count == 1

def test_risk_guards(strategy):
    now = time.time()
    strategy.on_depth_update(101.0, 0, 0, now)
    strategy.on_market_order('buy', 3, 101.0, now + 0.001)
    # Simulate price moving to stop
    strategy.on_tick(98.0, 10, now + 0.002)
    assert strategy.active_trade is None
    # Simulate depth restoration
    strategy.on_depth_update(101.0, 100, 100, now + 0.003)
    strategy.on_tick(101.0, 200, now + 0.004)
    # No active trade, so nothing to exit
    assert strategy.active_trade is None 