import pytest
import asyncio
from unittest.mock import MagicMock, patch
from backend.model_core.live_inference_loop import live_inference_loop

@pytest.mark.asyncio
async def test_live_inference_loop(monkeypatch):
    # Mock ModelDataLoader
    class DummyLoader:
        def __init__(self, *a, **kw): pass
        def iter_batches(self, partition):
            class DummyBatch:
                features = []
            yield DummyBatch()
    # Mock ModelRunner
    class DummyRunner:
        def run_inference(self, batch):
            class DummyOutput:
                actions = []
            return DummyOutput()
    monkeypatch.setattr("backend.model_core.live_inference_loop.ModelDataLoader", DummyLoader)
    monkeypatch.setattr("backend.model_core.live_inference_loop.ModelRunner", DummyRunner)
    # Patch logger to capture output
    with patch("backend.model_core.live_inference_loop.logger") as mock_logger:
        # Run the loop for one iteration
        task = asyncio.create_task(live_inference_loop())
        await asyncio.sleep(1.5)
        task.cancel()
        assert mock_logger.info.called 