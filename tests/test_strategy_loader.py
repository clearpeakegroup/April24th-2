import os
import sys
import yaml
import importlib
from pathlib import Path
import pytest

sys.path.insert(0, os.path.abspath('.'))
from backend.agents.registry import get_strategies
from trader_core.cli import save_user_config, load_config

CONFIG_PATH = Path('config/default.yml')
USER_CONFIG_PATH = Path.home() / '.april24th' / 'user.yml'

@pytest.fixture(autouse=True)
def cleanup_user_config():
    # Remove user config before and after each test
    if USER_CONFIG_PATH.exists():
        USER_CONFIG_PATH.unlink()
    yield
    if USER_CONFIG_PATH.exists():
        USER_CONFIG_PATH.unlink()

def test_train_stage_loads_nine():
    save_user_config('train', True)
    cfg = load_config()['stages']['train']
    strats = get_strategies(cfg)
    assert len(strats) == 9, f"Expected 9 strategies for train, got {len(strats)}"

def test_forward_stage_loads_five():
    save_user_config('forward', False)
    cfg = load_config()['stages']['forward']
    strats = get_strategies(cfg)
    assert len(strats) == 5, f"Expected 5 strategies for forward, got {len(strats)}"

def test_live_stage_toggle():
    # Initially off
    save_user_config('live', False)
    cfg = load_config()['stages']['live']
    strats = get_strategies(cfg)
    assert len(strats) == 5, f"Expected 5 strategies for live (off), got {len(strats)}"
    # Toggle on
    save_user_config('live', True)
    cfg = load_config()['stages']['live']
    strats = get_strategies(cfg)
    assert len(strats) == 9, f"Expected 9 strategies for live (on), got {len(strats)}"
    # Toggle persists
    cfg2 = load_config()['stages']['live']
    strats2 = get_strategies(cfg2)
    assert len(strats2) == 9, f"Expected 9 strategies for live (on, persisted), got {len(strats2)}"
