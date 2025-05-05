from typing import Tuple, Any
import numpy as np
from pydantic import BaseModel, StrictFloat, StrictStr, Field
import os
try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    INFLUXDB_URL = os.getenv("INFLUXDB_URL")
    INFLUXDB_TOKEN = os.getenv("INFLUXDB_TOKEN")
    INFLUXDB_ORG = os.getenv("INFLUXDB_ORG", "finrl")
    INFLUXDB_BUCKET = os.getenv("INFLUXDB_BUCKET", "finrl_metrics")
    influx_client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG) if INFLUXDB_URL else None
    influx_writer = influx_client.write_api(write_options=None) if influx_client else None
except ImportError:
    influx_client = None
    influx_writer = None

def log_router_to_influx(regime, signal_weight, order_type, size_multiplier, ts_ns):
    if not influx_writer:
        return
    try:
        p = Point("router_decision") \
            .tag("regime", regime) \
            .field("signal_weight", float(signal_weight)) \
            .field("order_type", str(order_type)) \
            .field("size_multiplier", float(size_multiplier)) \
            .time(int(ts_ns), WritePrecision.NS)
        influx_writer.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=p)
    except Exception:
        pass

class RouterOutput(BaseModel):
    signal_weight: float = Field(..., strict=True)
    order_type: str = Field(..., min_length=1, max_length=16)
    size_multiplier: float = Field(..., strict=True)

class AdaptiveTradeRouter:
    def __init__(self, policy_map_service=None):
        from backend.services.actor_policy_map_service import ActorPolicyMapService
        self.policy_map_service = policy_map_service or ActorPolicyMapService()

    def route(self, policy_pi0: float, flow_signal) -> RouterOutput:
        from backend.data_ingestion.live_stream import FlowSignal
        regime = self.policy_map_service.lookup_regime(flow_signal)
        direction = 1.0 if flow_signal.direction else -1.0
        if regime == "MKT-MM":
            # keep original PPO but allow double size
            out = RouterOutput(signal_weight=policy_pi0 * 1.0, order_type="MARKET", size_multiplier=2.0)
        elif regime == "TOXIC-FLO":
            out = RouterOutput(signal_weight=0.0, order_type="HALT", size_multiplier=0.0)
        elif regime == "BLOCK-TREND":
            out = RouterOutput(signal_weight=policy_pi0 * 0.5 + direction * 0.5, order_type="MARKET", size_multiplier=1.0)
        elif regime == "GAMMA-PIN":
            out = RouterOutput(signal_weight=-abs(policy_pi0) * 0.3, order_type="LIMIT", size_multiplier=1.0)
        elif regime == "FED-FLOW":
            out = RouterOutput(signal_weight=-direction * 1.0, order_type="MARKET", size_multiplier=1.0)
        else:
            # Default: pass through
            out = RouterOutput(signal_weight=policy_pi0, order_type="MARKET", size_multiplier=1.0)
        log_router_to_influx(regime, out.signal_weight, out.order_type, out.size_multiplier, getattr(flow_signal, 'ts_ns', 0))
        return out 