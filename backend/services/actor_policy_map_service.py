import redis
import yaml
import os
from pydantic import BaseModel, StrictStr, StrictInt, StrictFloat, ValidationError, Field
from typing import Dict, Any
from backend.data_ingestion.live_stream import FlowSignal

class RegimeMapping(BaseModel):
    actor: int = Field(..., strict=True, ge=0, le=10)
    regime: str = Field(..., min_length=1, max_length=32)
    edge: str = Field(..., min_length=1, max_length=128)
    stance: str = Field(..., min_length=1, max_length=128)

class ActorPolicyMapService:
    def __init__(self, redis_url: str = None, yaml_path: str = None):
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.yaml_path = yaml_path or os.getenv("REGIME_YAML", "regime_map.yaml")
        self._redis = redis.Redis.from_url(self.redis_url)
        self._mapping: Dict[int, RegimeMapping] = {}
        self.reload()

    def reload(self):
        # Try Redis first
        try:
            data = self._redis.get("regime_map")
            if data:
                mapping = yaml.safe_load(data)
                self._mapping = {int(k): RegimeMapping(**v) for k, v in mapping.items()}
                return
        except Exception:
            pass
        # Fallback to YAML
        if not os.path.exists(self.yaml_path):
            # Write default mapping if missing
            default_yaml = '''
0:
  actor: 0
  regime: "MKT-MM"
  edge: "Tight spread, high replenish"
  stance: "Aggressive take for scalps"
1:
  actor: 1
  regime: "TOXIC-FLO"
  edge: "High cancel/add, adverse selection"
  stance: "Stand-aside; widen stop"
2:
  actor: 2
  regime: "BLOCK-TREND"
  edge: "Large size, low cancel"
  stance: "Ride with trend until volume dries (time stop 2 min)"
3:
  actor: 3
  regime: "GAMMA-PIN"
  edge: "Mean-reversion, dampened vol"
  stance: "Provide liquidity – place passive bids/asks"
4:
  actor: 4
  regime: "FED-FLOW"
  edge: "One-way pressure but scheduled"
  stance: "Fade after 30 % of daily quota prints"
'''
            with open(self.yaml_path, "w") as f:
                f.write(default_yaml)
        with open(self.yaml_path, "r") as f:
            mapping = yaml.safe_load(f)
            self._mapping = {int(k): RegimeMapping(**v) for k, v in mapping.items()}

    def lookup_regime(self, flow_signal: FlowSignal) -> str:
        actor = int(flow_signal.actor)
        if actor not in self._mapping:
            self.reload()
        if actor in self._mapping:
            return self._mapping[actor].regime
        return "UNKNOWN"

    def get_full_mapping(self) -> Dict[int, RegimeMapping]:
        return self._mapping 