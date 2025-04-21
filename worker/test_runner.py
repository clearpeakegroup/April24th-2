import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from worker.runner import _job_handler, publish_progress, check_control

@pytest.mark.asyncio
async def test_job_handler_progress(monkeypatch):
    # Mock strategy_engine to yield progress
    async def fake_gen(**params):
        for i in range(3):
            yield {"pct": i*50, "msg": f"Step {i}"}
    monkeypatch.setattr("worker.strategy_engine.strategy_engine.run_backtest", fake_gen)
    monkeypatch.setattr("worker.strategy_engine.strategy_engine.run_forward_test", fake_gen)
    monkeypatch.setattr("worker.strategy_engine.strategy_engine.run_live", fake_gen)
    # Patch publish_progress and check_control to just count calls
    progress_calls = []
    async def fake_publish(job_id, pct, msg, redis_url=None):
        progress_calls.append((pct, msg))
    async def fake_check(job_id, redis_url=None):
        return
    with patch("worker.runner.publish_progress", fake_publish), patch("worker.runner.check_control", fake_check):
        await _job_handler({"id": "abc", "type": "train", "params": {}})
    assert progress_calls == [(0, "Step 0"), (50, "Step 1"), (100, "Step 2")]

@pytest.mark.asyncio
async def test_check_control_cancel(monkeypatch):
    # Simulate Redis returning 'cancel' immediately
    class FakeRedis:
        async def get(self, key):
            return "cancel"
        async def close(self):
            pass
    monkeypatch.setattr("aioredis.from_url", lambda *a, **kw: FakeRedis())
    with pytest.raises(asyncio.CancelledError):
        await check_control("abc")

@pytest.mark.asyncio
async def test_check_control_pause(monkeypatch):
    # Simulate Redis returning 'pause' then None
    class FakeRedis:
        def __init__(self):
            self.calls = 0
        async def get(self, key):
            self.calls += 1
            if self.calls < 3:
                return "pause"
            return None
        async def close(self):
            pass
    monkeypatch.setattr("aioredis.from_url", lambda *a, **kw: FakeRedis())
    # Should not raise
    await check_control("abc") 