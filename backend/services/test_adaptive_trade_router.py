import pytest
from backend.services.adaptive_trade_router import AdaptiveTradeRouter
from backend.data_ingestion.live_stream import FlowSignal

def make_test_signal():
    return FlowSignal(
        ts_ns=1234567890,
        actor=0,  # HFT_MM
        confidence=0.95,
        direction=True,
        expected_horizon_sec=0.1
    )

def test_router_latency(benchmark):
    router = AdaptiveTradeRouter()
    signal = make_test_signal()
    policy_pi0 = 0.5
    def run():
        router.route(policy_pi0, signal)
    result = benchmark(run)
    # Assert mean latency < 150 microseconds
    assert benchmark.stats.stats.mean < 0.00015, f"Router latency too high: {benchmark.stats.stats.mean*1e6:.2f} µs"

def test_router_correctness():
    router = AdaptiveTradeRouter()
    # MKT-MM
    s = FlowSignal(0, 0, 1.0, True, 0.1)
    out = router.route(1.0, s)
    assert out.signal_weight == 1.0 and out.order_type == "MARKET" and out.size_multiplier == 2.0
    # TOXIC-FLO
    s = FlowSignal(0, 1, 1.0, True, 0.1)
    out = router.route(1.0, s)
    assert out.signal_weight == 0.0 and out.order_type == "HALT" and out.size_multiplier == 0.0
    # BLOCK-TREND
    s = FlowSignal(0, 2, 1.0, True, 0.1)
    out = router.route(1.0, s)
    assert out.signal_weight == 1.0 and out.order_type == "MARKET"
    # GAMMA-PIN
    s = FlowSignal(0, 3, 1.0, True, 0.1)
    out = router.route(1.0, s)
    assert out.signal_weight == -0.3 and out.order_type == "LIMIT"
    # FED-FLOW
    s = FlowSignal(0, 4, 1.0, False, 0.1)
    out = router.route(1.0, s)
    assert out.signal_weight == 1.0 and out.order_type == "MARKET" 