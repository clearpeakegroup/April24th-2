from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, Field, ValidationError, field_validator
from redis.asyncio import Redis
from typing import Annotated
import os
import json

# Dependency injection for Redis
async def get_redis() -> Redis:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    return Redis.from_url(redis_url, decode_responses=True)

# Pydantic v2 model for risk settings
class RiskSettings(BaseModel):
    risk_pct: float = Field(..., ge=5, le=40, description="VaR-99% cap as percent of equity")
    margin_pct: float = Field(..., ge=5, le=40, description="Margin cap as percent of equity")
    ack_high: bool = Field(default=False, description="User acknowledged high risk")

    @field_validator('risk_pct', 'margin_pct')
    @classmethod
    def check_bounds(cls, v):
        if not (5 <= v <= 40):
            raise ValueError('Value must be between 5 and 40')
        return v

    @field_validator('ack_high')
    @classmethod
    def require_ack_if_high(cls, v, info):
        data = info.data
        risk = data.get('risk_pct', 0) if data else 0
        margin = data.get('margin_pct', 0) if data else 0
        if (risk > 25 or margin > 25) and not v:
            raise ValueError('ack_high must be True for risk_pct or margin_pct > 25')
        return v

router = APIRouter(prefix="/settings", tags=["settings"])

@router.post("/risk", status_code=200)
async def set_risk_settings(
    settings: RiskSettings,
    redis: Annotated[Redis, Depends(get_redis)]
):
    # Persist to Redis hash
    await redis.hset("settings", mapping={
        "risk_pct": settings.risk_pct,
        "margin_pct": settings.margin_pct,
        "ack_high": int(settings.ack_high)
    })
    # Publish to Redis channel
    msg = json.dumps({
        "risk_pct": settings.risk_pct,
        "margin_pct": settings.margin_pct,
        "ack_high": settings.ack_high
    })
    await redis.publish("settings", msg)
    return {"status": "ok", "detail": "Risk settings updated."}

@router.get("/risk", response_model=RiskSettings)
async def get_risk_settings(redis: Annotated[Redis, Depends(get_redis)]):
    settings = await redis.hgetall("settings")
    if not settings or "risk_pct" not in settings or "margin_pct" not in settings:
        raise HTTPException(status_code=404, detail="Risk settings not found")
    return RiskSettings(
        risk_pct=float(settings["risk_pct"]),
        margin_pct=float(settings["margin_pct"]),
        ack_high=bool(int(settings.get("ack_high", 0)))
    ) 