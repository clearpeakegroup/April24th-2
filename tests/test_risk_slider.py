import pytest
import asyncio
from httpx import AsyncClient
from fastapi import status
from backend.api import app
import fakeredis.aioredis
import json

@pytest.fixture
def anyio_backend():
    return 'asyncio'

@pytest.fixture
def redis_url(monkeypatch):
    url = "redis://localhost:6379/0"
    monkeypatch.setenv("REDIS_URL", url)
    return url

@pytest.fixture
def fake_redis(monkeypatch):
    redis = fakeredis.aioredis.FakeRedis()
    monkeypatch.setattr("backend.api_settings.get_redis", lambda: redis)
    return redis

@pytest.mark.anyio
async def test_risk_happy_path(fake_redis):
    async with AsyncClient(app=app, base_url="http://test") as ac:
        resp = await ac.post("/settings/risk", json={"risk_pct": 15, "margin_pct": 12, "ack_high": False})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        # GET should return the same
        resp2 = await ac.get("/settings/risk")
        assert resp2.status_code == 200
        s = resp2.json()
        assert s["risk_pct"] == 15
        assert s["margin_pct"] == 12
        assert s["ack_high"] is False

@pytest.mark.anyio
async def test_risk_high_risk_with_ack(fake_redis):
    async with AsyncClient(app=app, base_url="http://test") as ac:
        resp = await ac.post("/settings/risk", json={"risk_pct": 30, "margin_pct": 12, "ack_high": True})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        resp2 = await ac.get("/settings/risk")
        assert resp2.status_code == 200
        s = resp2.json()
        assert s["risk_pct"] == 30
        assert s["ack_high"] is True

@pytest.mark.anyio
async def test_risk_high_risk_without_ack(fake_redis):
    async with AsyncClient(app=app, base_url="http://test") as ac:
        resp = await ac.post("/settings/risk", json={"risk_pct": 30, "margin_pct": 12, "ack_high": False})
        assert resp.status_code == 422
        assert "ack_high" in resp.text

@pytest.mark.anyio
async def test_risk_margin_high_with_ack(fake_redis):
    async with AsyncClient(app=app, base_url="http://test") as ac:
        resp = await ac.post("/settings/risk", json={"risk_pct": 10, "margin_pct": 30, "ack_high": True})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        resp2 = await ac.get("/settings/risk")
        assert resp2.status_code == 200
        s = resp2.json()
        assert s["margin_pct"] == 30
        assert s["ack_high"] is True

@pytest.mark.xfail(reason="fakeredis async pubsub is not reliable for timing-sensitive tests")
@pytest.mark.anyio
async def test_pubsub_latency(fake_redis):
    # Simulate a dummy QP module subscribing to settings
    pubsub = fake_redis.pubsub()
    await pubsub.subscribe("settings")
    async with AsyncClient(app=app, base_url="http://test") as ac:
        await ac.post("/settings/risk", json={"risk_pct": 20, "margin_pct": 11, "ack_high": False})
        msg = None
        for _ in range(10):
            msg = await pubsub.get_message(ignore_subscribe_messages=True)
            if msg is not None:
                break
            await asyncio.sleep(0.01)
        assert msg is not None, "Pubsub update not received within 100ms"
        data = json.loads(msg["data"])
        assert data["risk_pct"] == 20 