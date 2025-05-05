"""
CRUD operations for strategy configs and agents.
"""
from typing import Any

def create_strategy_config(config: Any) -> Any:
    """Stub for creating a strategy config in the DB."""
    pass 

def iter_equity(agent_id: str):
    """Mock generator yielding (timestamp, equity) tuples for live plot."""
    import time
    import random
    equity = 100000.0
    for i in range(100):
        equity += random.uniform(-500, 500)
        yield (int(time.time()) + i, equity)
        time.sleep(0.1)  # Simulate real-time 