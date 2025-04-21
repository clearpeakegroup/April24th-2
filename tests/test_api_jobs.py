import pytest
from fastapi.testclient import TestClient
from api.router import router
from fastapi import FastAPI
import uuid

app = FastAPI()
app.include_router(router)
client = TestClient(app)

def test_post_jobs_train():
    resp = client.post("/jobs/train", json={"type": "train", "params": {"foo": "bar"}})
    assert resp.status_code == 200
    data = resp.json()
    assert data["type"] == "train"
    assert data["status"] == "pending"
    assert data["percent_complete"] == 0.0
    assert "id" in data
    job_id = data["id"]
    # GET job
    get_resp = client.get(f"/jobs/{job_id}")
    assert get_resp.status_code == 200
    # PATCH job
    patch_resp = client.patch(f"/jobs/{job_id}", json={"action": "pause"})
    assert patch_resp.status_code == 200
    assert patch_resp.json()["status"] == "paused"
    # PATCH invalid action
    bad_patch = client.patch(f"/jobs/{job_id}", json={"action": "bad"})
    assert bad_patch.status_code == 400
    # PATCH non-existent job
    notfound_patch = client.patch(f"/jobs/{uuid.uuid4()}", json={"action": "pause"})
    assert notfound_patch.status_code == 404
    # GET non-existent job
    notfound_get = client.get(f"/jobs/{uuid.uuid4()}")
    assert notfound_get.status_code == 404

def test_post_jobs_forward():
    resp = client.post("/jobs/forward", json={"type": "forward", "params": {"foo": "bar"}})
    assert resp.status_code == 200
    assert resp.json()["type"] == "forward"

def test_post_jobs_live():
    resp = client.post("/jobs/live", json={"type": "live", "params": {"foo": "bar"}})
    assert resp.status_code == 200
    assert resp.json()["type"] == "live"

def test_account_and_health():
    acc = client.get("/account")
    assert acc.status_code == 200
    data = acc.json()
    assert "margin" in data and "buying_power" in data and "pnl" in data
    health = client.get("/health")
    assert health.status_code == 200
    assert "status" in health.json() and "redis" in health.json()

def test_stream_job_progress():
    # Create a job
    resp = client.post("/jobs/train", json={"type": "train", "params": {"foo": "bar"}})
    job_id = resp.json()["id"]
    # Just check that the stream endpoint returns 200 and is a stream
    stream_resp = client.get(f"/jobs/{job_id}/stream", stream=True)
    assert stream_resp.status_code == 200
    # Not reading the stream fully in test (would block) 