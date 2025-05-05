from fastapi.testclient import TestClient
from backend.api import app
import io
import pandas as pd
import time
import pytest
from loguru import logger

client = TestClient(app)

@pytest.mark.skip(reason="Postgres not running")
def test_batch_ingest():
    # Create a small CSV in memory
    df = pd.DataFrame({
        "ts_event": [1, 2],
        "instrument": ["AAPL", "BTC"],
        "side": ["buy", "sell"],
        "price": [100.0, 20000.0],
        "size": [10, 0.5],
        "exchange": ["NASDAQ", "BINANCE"],
        "type": ["trade", "trade"]
    })
    csv_bytes = df.to_csv(index=False).encode()
    response = client.post(
        "/ingest/batch",
        files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
        data={"version": 1, "user": "test", "source": "test"}
    )
    logger.info(f"Batch ingest response: {response.json()}")
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_rl_job_submission_and_status():
    # Submit RL job
    response = client.post("/rl/job", json={"config": {"param": 1}})
    logger.info(f"RL job submission response: {response.json()}")
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    # Poll for status
    for _ in range(5):
        status_resp = client.get(f"/rl/job/{job_id}")
        logger.info(f"RL job status response: {status_resp.json()}")
        if status_resp.json()["status"] == "finished":
            break
        time.sleep(1)
    assert status_resp.json()["status"] == "finished"
    assert "reward" in status_resp.json()["result"]

if __name__ == "__main__":
    test_batch_ingest()
    test_rl_job_submission_and_status() 