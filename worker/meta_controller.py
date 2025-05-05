import asyncio
import os
import json
from redis.asyncio import Redis
from datetime import datetime, timedelta, time as dtime
import logging

# --- Global risk/margin caps (updated live) ---
RISK = 0.10  # Default 10%
MARGIN = 0.11  # Default 11%

# --- CME settle time (16:00 ET = 21:00 UTC) ---
CME_SETTLE_UTC = dtime(21, 0)
FLAT_T_MINUS = 15  # minutes before settle

# --- Redis config ---
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
SETTINGS_HASH = "settings"
SETTINGS_CHANNEL = "settings"

logger = logging.getLogger("meta_controller")

async def update_risk_margin_from_redis(redis: Redis):
    global RISK, MARGIN
    settings = await redis.hgetall(SETTINGS_HASH)
    try:
        RISK = float(settings.get("risk_pct", 10)) / 100.0
        MARGIN = float(settings.get("margin_pct", 11)) / 100.0
        logger.info(f"🔥 Risk caps live-updated: VaR {RISK*100:.1f} % Margin {MARGIN*100:.1f} %")
    except Exception as e:
        logger.error(f"Failed to update risk/margin: {e}")

async def settings_listener():
    redis = Redis.from_url(REDIS_URL, decode_responses=True)
    pubsub = redis.pubsub()
    await pubsub.subscribe(SETTINGS_CHANNEL)
    logger.info("Meta-controller listening for risk/margin updates...")
    async for msg in pubsub.listen():
        if msg["type"] == "message":
            try:
                data = json.loads(msg["data"])
                await update_risk_margin_from_redis(redis)
            except Exception as e:
                logger.error(f"Error processing settings message: {e}")

async def auto_reset_caps():
    redis = Redis.from_url(REDIS_URL, decode_responses=True)
    while True:
        now = datetime.utcnow()
        settle_dt = datetime.combine(now.date(), CME_SETTLE_UTC)
        reset_time = settle_dt - timedelta(minutes=FLAT_T_MINUS)
        if now > reset_time:
            # If past today's reset, schedule for next day
            reset_time += timedelta(days=1)
        sleep_sec = (reset_time - now).total_seconds()
        logger.info(f"Next risk cap auto-reset scheduled for {reset_time} UTC ({sleep_sec/60:.1f} min)")
        await asyncio.sleep(max(sleep_sec, 0))
        # Reset caps to 10/11%
        await redis.hset(SETTINGS_HASH, mapping={"risk_pct": 10, "margin_pct": 11, "ack_high": 0})
        msg = json.dumps({"risk_pct": 10, "margin_pct": 11, "ack_high": False})
        await redis.publish(SETTINGS_CHANNEL, msg)
        logger.info("⏳ Session reset: risk caps returned to 10 / 11 %")

# --- Gear promoter logic ---
GEAR_LADDER = [
    {"equity": 3500,  "lots": 3,  "margin": 390,  "net": 145,  "var": -380},
    {"equity": 7000,  "lots": 4,  "margin": 520,  "net": 192,  "var": -510},
    {"equity": 10000, "lots": 6,  "margin": 780,  "net": 288,  "var": -760},
    {"equity": 16000, "lots": 8,  "margin": 1040, "net": 384,  "var": -1010},
    {"equity": 23000, "lots": 10, "margin": 1300, "net": 480,  "var": -1260},
    {"equity": 32000, "lots": 12, "margin": 1560, "net": 576,  "var": -1510},
]

def gear_promoter(equity: float) -> dict:
    """Promote to next gear if margin/equity <= MARGIN."""
    for gear in reversed(GEAR_LADDER):
        if equity >= gear["equity"] and (gear["margin"] / equity) <= MARGIN:
            return gear
    return GEAR_LADDER[0]

# --- QP sizer logic ---
def qp_sizer(current_var: float, equity: float) -> float:
    """Constrain position so current_VaR <= RISK * equity."""
    cap = RISK * equity
    return min(current_var, cap)

# --- Entrypoint ---
async def main():
    await asyncio.gather(
        settings_listener(),
        auto_reset_caps(),
    )

if __name__ == "__main__":
    asyncio.run(main()) 