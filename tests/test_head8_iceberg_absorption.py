import time
import pytest
from trader_core.strategies.options.head8_iceberg_absorption import Head8IcebergAbsorptionStrategy, DepthQueueListener

# Simulated PCAP sample events (timestamp, price, depth, side, qty, is_exec)
SAMPLE_EVENTS = [
    # (timestamp, price, depth, side, qty, is_exec)
    (0.00, 4200, 100, 'bid', 0, False),
    (0.01, 4200, 100, 'bid', 10, True),
    (0.02, 4200, 100, 'bid', 10, True),
    (0.03, 4200, 100, 'bid', 10, True),
    (0.04, 4200, 100, 'bid', 10, True),
    (0.05, 4200, 100, 'bid', 10, True),
    (0.06, 4200, 100, 'bid', 10, True),  # Should trigger iceberg
    (0.10, 4200, 90, 'bid', 0, False),   # Depth drops
    (0.20, 4199, 80, 'ask', 0, False),
    (0.21, 4199, 80, 'ask', 10, True),
    (0.22, 4199, 80, 'ask', 10, True),
    (0.23, 4199, 80, 'ask', 10, True),
    (0.24, 4199, 80, 'ask', 10, True),
    (0.25, 4199, 80, 'ask', 10, True),
    (0.26, 4199, 80, 'ask', 10, True),   # Should trigger iceberg
]

@pytest.fixture
def strategy():
    return Head8IcebergAbsorptionStrategy(window_ms=100)

def test_iceberg_absorption_detection_precision(strategy):
    listener = DepthQueueListener(strategy)
    detected = 0
    expected = 2  # We have 2 iceberg events in the sample
    for event in SAMPLE_EVENTS:
        ts, price, depth, side, qty, is_exec = event
        now = time.time()
        if not is_exec:
            listener.on_depth(price, depth, side, now)
        else:
            listener.on_execution(price, qty, side, now)
        # Check if a new iceberg was detected
        if len(strategy.icebergs) > detected:
            detected += 1
    precision = detected / expected
    assert precision >= 0.8, f"Detection precision {precision:.2f} < 0.8" 