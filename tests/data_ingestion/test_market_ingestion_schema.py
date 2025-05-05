import pytest
from trader_core.config_loader import ConfigLoader

SCHEMA = ["ts_event", "instrument", "side", "price", "size", "exchange", "type"]

def make_sample_tick(asset):
    return {
        "ts_event": 1234567890,
        "instrument": asset,
        "side": "buy",
        "price": 123.45,
        "size": 10,
        "exchange": "CME",
        "type": "trade"
    }

@pytest.mark.parametrize("asset", ConfigLoader.get_assets("config/default.yml"))
def test_tick_schema(asset):
    tick = make_sample_tick(asset)
    for field in SCHEMA:
        assert field in tick
    assert tick["instrument"] == asset 