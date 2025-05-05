import os
import json
import asyncio
import websockets
import redis
import pandas as pd
from loguru import logger
from typing import Any, Dict, Optional
import zmq
import zmq.asyncio
from google.protobuf import message as pb_message
import numpy as np
import time
from collections import deque, defaultdict
try:
    import cupy as cp
    import cudf
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
from backend.services.adaptive_trade_router import AdaptiveTradeRouter

CREDENTIALS_PATH = os.path.expanduser("~/.config/finrl/credentials.json")
SCHEMA = ["ts_event", "instrument", "side", "price", "size", "exchange", "type"]
REDIS_STREAM = "ticks"

# Protobuf definition (inline for now, move to .proto if needed)
class FlowSignal:
    class Actor:
        HFT_MM = 0
        HFT_Spoofer = 1
        Macro_Block = 2
        Dealer_Hedge = 3
        Fed_OMO = 4
    def __init__(self, ts_ns: int, actor: int, confidence: float, direction: bool, expected_horizon_sec: float):
        self.ts_ns = ts_ns
        self.actor = actor
        self.confidence = confidence
        self.direction = direction
        self.expected_horizon_sec = expected_horizon_sec
    def to_dict(self):
        return {
            'ts_ns': self.ts_ns,
            'actor': self.actor,
            'confidence': self.confidence,
            'direction': self.direction,
            'expected_horizon_sec': self.expected_horizon_sec,
        }

# ZeroMQ PUB socket setup (singleton)
_zmq_ctx = zmq.asyncio.Context()
_zmq_pub = _zmq_ctx.socket(zmq.PUB)
import os
if os.environ.get("ENABLE_ZMQ_PUB", "0") == "1":
    try:
        _zmq_pub.bind("tcp://*:5557")
    except Exception as e:
        print(f"Warning: Could not bind to tcp://*:5557: {e}")


async def publish_flow_signal(signal: FlowSignal):
    # In production, use protobuf serialization
    await _zmq_pub.send_json(signal.to_dict())

# --- Rolling caches for detectors ---
ORDERBOOK_DEPTH = 10  # L3 depth to scan for queue vacuum
QUEUE_VACUUM_WINDOW_MS = 50
CANCEL_BURST_WINDOW_MS = 100
TRADE_SIZE_WINDOW = 500  # Number of trades to keep for Z-score
GAMMA_HEDGE_WINDOW_S = 1.0
REPO_GC_UPDATE_S = 3600

orderbook_snapshots = deque(maxlen=2)  # Store last 2 snapshots for diff
cancel_add_events = deque(maxlen=1000)  # (ts_ns, type: 'add'/'cancel')
trade_sizes = deque(maxlen=TRADE_SIZE_WINDOW)
trade_timestamps = deque(maxlen=TRADE_SIZE_WINDOW)
equity_basket_trades = deque(maxlen=1000)  # (ts_ns, symbol, side, size)
last_repo_gc_update = 0
sofr_fra_spread = 0.0

# --- Detector implementations ---
async def detect_queue_vacuum(orderbook_snapshots: deque) -> Optional[FlowSignal]:
    if len(orderbook_snapshots) < 2:
        return None
    snap0, snap1 = orderbook_snapshots[0], orderbook_snapshots[1]
    ts0, book0 = snap0['ts_ns'], snap0['depth']
    ts1, book1 = snap1['ts_ns'], snap1['depth']
    dt_ms = (ts1 - ts0) / 1e6
    if dt_ms > QUEUE_VACUUM_WINDOW_MS:
        return None
    # book: {'bid': [levels], 'ask': [levels]}
    depth0 = np.array(book0['bid'][:ORDERBOOK_DEPTH] + book0['ask'][:ORDERBOOK_DEPTH])
    depth1 = np.array(book1['bid'][:ORDERBOOK_DEPTH] + book1['ask'][:ORDERBOOK_DEPTH])
    total0 = np.sum(depth0)
    total1 = np.sum(depth1)
    if total0 == 0:
        return None
    drop = (total0 - total1) / total0
    if drop >= 0.8:
        return FlowSignal(
            ts_ns=ts1,
            actor=FlowSignal.Actor.HFT_MM,
            confidence=float(drop),
            direction=True if total1 > total0 else False,
            expected_horizon_sec=0.1
        )
    return None

async def detect_cancel_burst(cancel_add_events: deque) -> Optional[FlowSignal]:
    now_ns = time.time_ns()
    window_ns = CANCEL_BURST_WINDOW_MS * 1_000_000
    cancels = adds = 0
    for ts, typ in list(cancel_add_events):
        if now_ns - ts > window_ns:
            continue
        if typ == 'cancel':
            cancels += 1
        elif typ == 'add':
            adds += 1
    if adds == 0:
        return None
    ratio = cancels / adds
    if ratio > 5:
        toxicity = min(1.0, (ratio - 5) / 10 + 0.5)
        return FlowSignal(
            ts_ns=now_ns,
            actor=FlowSignal.Actor.HFT_Spoofer,
            confidence=toxicity,
            direction=False,
            expected_horizon_sec=0.2
        )
    return None

async def detect_trade_size_zscore(trade_sizes: deque, trade_timestamps: deque) -> Optional[FlowSignal]:
    if len(trade_sizes) < 30:
        return None
    arr = np.array(trade_sizes)
    log_sizes = np.log1p(arr)
    median = np.median(log_sizes)
    mad = np.median(np.abs(log_sizes - median))
    if mad == 0:
        return None
    zscores = (log_sizes - median) / (mad * 1.4826)
    idx = np.argmax(zscores)
    if zscores[idx] > 4:
        ts_ns = trade_timestamps[idx]
        return FlowSignal(
            ts_ns=ts_ns,
            actor=FlowSignal.Actor.Macro_Block,
            confidence=min(1.0, (zscores[idx] - 4) / 4 + 0.5),
            direction=True,
            expected_horizon_sec=60.0
        )
    return None

async def detect_gamma_hedge(equity_basket_trades: deque) -> Optional[FlowSignal]:
    # Look for sequence: BUY future, then SELL equity basket within 1s
    now_ns = time.time_ns()
    for i in range(len(equity_basket_trades) - 1):
        ts0, sym0, side0, size0 = equity_basket_trades[i]
        ts1, sym1, side1, size1 = equity_basket_trades[i + 1]
        if sym0 in {'MES', 'MNQ', 'M2K', 'ZN'} and side0 == 'buy':
            if sym1 not in {'MES', 'MNQ', 'M2K', 'ZN'} and side1 == 'sell':
                if 0 < (ts1 - ts0) < 1_000_000_000:
                    return FlowSignal(
                        ts_ns=ts1,
                        actor=FlowSignal.Actor.Dealer_Hedge,
                        confidence=1.0,
                        direction=False,
                        expected_horizon_sec=10.0
                    )
    return None

async def detect_repo_gc_shock(sofr_fra_spread: float, last_spread: float, last_update: float) -> Optional[FlowSignal]:
    now = time.time()
    if now - last_update > REPO_GC_UPDATE_S:
        # In production, subscribe to CME SOFR 3M + ZQ, here just simulate
        # If spread spikes > threshold, trigger
        if abs(sofr_fra_spread - last_spread) > 0.1:  # Example threshold
            return FlowSignal(
                ts_ns=int(now * 1e9),
                actor=FlowSignal.Actor.Fed_OMO,
                confidence=min(1.0, abs(sofr_fra_spread - last_spread)),
                direction=True if sofr_fra_spread > last_spread else False,
                expected_horizon_sec=300.0
            )
    return None

def load_credentials():
    if not os.path.exists(CREDENTIALS_PATH):
        logger.warning(f"Credentials file not found: {CREDENTIALS_PATH}")
        return None
    with open(CREDENTIALS_PATH, "r") as f:
        return json.load(f)

def retry_async(times=3):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for attempt in range(1, times + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Attempt {attempt} failed: {e}")
                    if attempt == times:
                        raise
        return wrapper
    return decorator

@retry_async(times=3)
async def ingest_live_ticks(ws_url: str, redis_url: str = "redis://localhost:6379/0"):
    from backend.api import FlowState

    creds = load_credentials()
    tls_auth = creds.get("tls_auth") if creds else None
    logger.info(f"Connecting to {ws_url} with TLS auth stub: {tls_auth}")
    r = redis.Redis.from_url(redis_url)
    router = AdaptiveTradeRouter()
    async with websockets.connect(ws_url, ssl=True) as ws:
        async for msg in ws:
            tick = json.loads(msg)
            row = {col: tick.get(col) for col in SCHEMA}
            df = pd.DataFrame([row])
            r.xadd(REDIS_STREAM, row)
            logger.info(f"Published tick to Redis: {row}")

            # --- Update rolling caches ---
            ts_ns = int(tick.get('ts_event', time.time_ns()))
            if 'depth' in tick:
                orderbook_snapshots.append({'ts_ns': ts_ns, 'depth': tick['depth']})
            if tick.get('type') == 'cancel':
                cancel_add_events.append((ts_ns, 'cancel'))
            elif tick.get('type') == 'add':
                cancel_add_events.append((ts_ns, 'add'))
            if tick.get('type') == 'trade':
                trade_sizes.append(float(tick.get('size', 0)))
                trade_timestamps.append(ts_ns)
                equity_basket_trades.append((ts_ns, tick.get('instrument', ''), tick.get('side', ''), float(tick.get('size', 0))))

            # --- Run detectors ---
            detectors = [
                await detect_queue_vacuum(orderbook_snapshots),
                await detect_cancel_burst(cancel_add_events),
                await detect_trade_size_zscore(trade_sizes, trade_timestamps),
                await detect_gamma_hedge(equity_basket_trades),
                await detect_repo_gc_shock(sofr_fra_spread, sofr_fra_spread, last_repo_gc_update),
            ]
            for signal in detectors:
                if signal is not None:
                    await publish_flow_signal(signal)
                    logger.info(f"Emitted FlowSignal: {signal.to_dict()}")
                    # Route and update FlowState for API/dashboard
                    router_out = router.route(0.0, signal)  # Use 0.0 as default policy_pi0 for detector-only signals
                    FlowState.update(signal, router_out)

# Example usage (to be run in an async context):
# asyncio.run(ingest_live_ticks("wss://databento.example/ws")) 