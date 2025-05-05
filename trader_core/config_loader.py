import os
import yaml
from typing import Any, Dict

class ConfigLoader:
    """
    Loads configuration from YAML files and overlays environment variables.
    Use load_config(path) to load a config dict from a YAML file, with environment variable overrides.
    """
    @staticmethod
    def load_config(path: str = None) -> Dict[str, Any]:
        """
        Load configuration from a YAML file and overlay environment variables.
        Args:
            path: Path to the YAML config file.
        Returns:
            dict: Configuration dictionary.
        """
        config = {}
        if path:
            with open(path, 'r') as f:
                config = yaml.safe_load(f)
        # Overlay environment variables (flat keys only)
        for key, value in os.environ.items():
            if key in config:
                config[key] = value
        return config

# Usage example:
# config = ConfigLoader.load_config('config/default.yml') 