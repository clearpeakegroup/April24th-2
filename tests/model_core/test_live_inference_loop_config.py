import pytest
from backend.model_core.live_inference_loop import validate_config

def test_live_inference_loop_validate_config_missing(monkeypatch):
    monkeypatch.delenv("UNIFIED_FEATURE_BASE", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        validate_config()
    assert "UNIFIED_FEATURE_BASE" in str(excinfo.value)

def test_live_inference_loop_validate_config_present(monkeypatch):
    monkeypatch.setenv("UNIFIED_FEATURE_BASE", "data/lake/parquet/features/unified")
    # Should not raise
    validate_config() 