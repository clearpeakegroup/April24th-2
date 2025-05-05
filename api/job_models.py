from pydantic import BaseModel, Field, StrictStr, StrictInt, StrictFloat, StrictBool, RootModel
from typing import Any, Dict, Optional, Literal
from uuid import UUID

class JobType(str):
    TRAIN = "train"
    FORWARD = "forward"
    LIVE = "live"

class JobParams(RootModel[Dict[str, Any]]):
    pass

class JobRequest(BaseModel):
    type: Literal["train", "forward", "live"] = Field(...)
    params: Dict[str, Any] = Field(...)

class JobResponse(BaseModel):
    id: UUID = Field(...)
    type: Literal["train", "forward", "live"] = Field(...)
    status: Literal["pending", "running", "paused", "cancelled", "completed", "failed"] = Field(...)
    percent_complete: float = Field(..., ge=0, le=100, strict=True)
    params: Dict[str, Any] = Field(...)
    result: Optional[Dict[str, Any]] = Field(default=None)
    error: Optional[str] = Field(default=None)

class JobPatchRequest(BaseModel):
    action: Literal["pause", "resume", "cancel"] = Field(...)

class AccountResponse(BaseModel):
    margin: float = Field(..., strict=True)
    buying_power: float = Field(..., strict=True)
    pnl: float = Field(..., strict=True)

class HealthResponse(BaseModel):
    status: str = Field(..., min_length=1, max_length=32)
    redis: bool = Field(..., strict=True) 