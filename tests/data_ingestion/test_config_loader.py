import tempfile
import yaml
import os
import pytest
from trader_core.config_loader import ConfigLoader

def test_get_assets_from_config():
    assets = ['AAA', 'BBB', 'CCC']
    with tempfile.NamedTemporaryFile('w+', delete=False) as tf:
        yaml.dump({'assets': assets}, tf)
        tf.flush()
        loaded = ConfigLoader.get_assets(tf.name)
        assert loaded == assets
    os.unlink(tf.name)

def test_get_assets_default():
    with tempfile.NamedTemporaryFile('w+', delete=False) as tf:
        yaml.dump({'not_assets': [1,2,3]}, tf)
        tf.flush()
        loaded = ConfigLoader.get_assets(tf.name)
        assert loaded == ['MES', 'MNQ', 'ZF', 'ZN', 'UB']
    os.unlink(tf.name) 