import asyncio
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from data_ingestion import live_stream

@pytest.mark.asyncio
async def test_ingest_live_ticks(monkeypatch):
    # Mock credentials
    monkeypatch.setattr(live_stream, "load_credentials", lambda: {"tls_auth": "dummy"})
    # Mock Redis
    mock_redis = MagicMock()
    monkeypatch.setattr(live_stream.redis, "Redis", lambda *a, **kw: mock_redis)
    # Mock WebSocket
    class FakeWebSocket:
        def __init__(self, messages):
            self._messages = messages
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
        def __aiter__(self):
            return self
        async def __anext__(self):
            if not self._messages:
                raise StopAsyncIteration
            return self._messages.pop(0)
    fake_tick = json.dumps({"ts_event": 1, "instrument": "ABC", "side": "buy", "price": 100, "size": 10, "exchange": "NYSE", "type": "trade"})
    monkeypatch.setattr(live_stream.websockets, "connect", lambda *a, **kw: FakeWebSocket([fake_tick]))
    await live_stream.ingest_live_ticks("wss://fake", redis_url="redis://localhost:6379/0")
    assert mock_redis.xadd.called 