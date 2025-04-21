import os
import sys
import yaml
from pathlib import Path
from loguru import logger

CONFIG_PATH = Path('config/default.yml')
USER_CONFIG_DIR = Path.home() / '.april24th'
USER_CONFIG_PATH = USER_CONFIG_DIR / 'user.yml'


def load_config():
    with open(CONFIG_PATH, 'r') as f:
        base_cfg = yaml.safe_load(f)
    if USER_CONFIG_PATH.exists():
        with open(USER_CONFIG_PATH, 'r') as f:
            user_cfg = yaml.safe_load(f) or {}
        # Merge user config into base config
        for stage, opts in (user_cfg.get('stages', {}) or {}).items():
            if stage in base_cfg['stages']:
                base_cfg['stages'][stage].update(opts)
    return base_cfg

def save_user_config(stage, enabled):
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    user_cfg = {}
    if USER_CONFIG_PATH.exists():
        with open(USER_CONFIG_PATH, 'r') as f:
            user_cfg = yaml.safe_load(f) or {}
    if 'stages' not in user_cfg:
        user_cfg['stages'] = {}
    if stage not in user_cfg['stages']:
        user_cfg['stages'][stage] = {}
    user_cfg['stages'][stage]['options_enabled'] = enabled
    with open(USER_CONFIG_PATH, 'w') as f:
        yaml.safe_dump(user_cfg, f)

def main():
    if len(sys.argv) != 4 or sys.argv[1] != 'options':
        logger.info('Usage: python -m trader_core.cli options <stage> <on|off>')
        sys.exit(1)
    _, _, stage, value = sys.argv
    if value not in ('on', 'off'):
        logger.info('Value must be "on" or "off"')
        sys.exit(1)
    enabled = value == 'on'
    save_user_config(stage, enabled)
    logger.info(f"Set options_enabled for stage '{stage}' to {enabled} (persisted in {USER_CONFIG_PATH})")

if __name__ == '__main__':
    main()
