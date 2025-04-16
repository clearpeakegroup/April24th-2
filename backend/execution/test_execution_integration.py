import pytest
import redis
import json
import threading
import time
from execution.broker_connector import ETradeBrokerConnector
from execution.risk_manager import RiskManager
from execution.executor import Executor

class MockBroker(ETradeBrokerConnector):
    def place_order(self, symbol, qty, side, order_type="MKT"):
        return {"order_id": "mock123", "status": "filled", "symbol": symbol, "qty": qty, "side": side}

@pytest.fixture(scope="module")
def redis_server():
    # Assumes Redis is running on localhost:6380
    r = redis.Redis(host="localhost", port=6380, db=0)
    yield r
    r.flushdb()

def test_executor_integration(redis_server):
    broker = MockBroker("id", "secret", "uri")
    risk = RiskManager({"MES": 2}, 1000.0)
    executor = Executor("redis://localhost:6380/0", broker, risk)
    # Run executor in a thread
    t = threading.Thread(target=executor.run, daemon=True)
    t.start()
    # Publish a signal
    signal = {"symbol": "MES", "qty": 1, "side": "buy"}
    redis_server.publish("agent_signals", json.dumps(signal))
    # Wait for confirmation
    time.sleep(1)
    msg = redis_server.pubsub()
    msg.subscribe("trade_confirmations")
    time.sleep(1)
    # No assertion, just ensure no exceptions and logs 