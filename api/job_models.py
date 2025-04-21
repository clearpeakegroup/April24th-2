from pydantic import BaseModel, Field, StrictStr, StrictInt, StrictFloat, StrictBool
from typing import Any, Dict, Optional, Literal
from uuid import UUID

class JobType(str):
    TRAIN = "train"
    FORWARD = "forward"
    LIVE = "live"

class JobParams(BaseModel):
    # Accepts arbitrary params, but must be a dict
    __root__: Dict[str, Any]

class JobRequest(BaseModel):
    type: Literal["train", "forward", "live"]
    params: Dict[str, Any]

class JobResponse(BaseModel):
    id: UUID
    type: Literal["train", "forward", "live"]
    status: Literal["pending", "running", "paused", "cancelled", "completed", "failed"]
    percent_complete: float = Field(..., ge=0, le=100)
    params: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class JobPatchRequest(BaseModel):
    action: Literal["pause", "resume", "cancel"]

class AccountResponse(BaseModel):
    margin: StrictFloat
    buying_power: StrictFloat
    pnl: StrictFloat

class HealthResponse(BaseModel):
    status: StrictStr
    redis: StrictBool 