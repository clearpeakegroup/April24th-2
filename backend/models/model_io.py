from pydantic import BaseModel, StrictStr, StrictFloat, StrictInt, Field
from typing import List, Dict

class ModelInputFeature(BaseModel):
    asset: StrictStr
    timestamp: StrictStr
    best_bid: StrictFloat
    best_ask: StrictFloat
    spread: StrictFloat
    depth_bid: StrictFloat
    depth_ask: StrictFloat
    sentiment: StrictFloat
    news_length: StrictInt
    # Extend with more features as needed

class ModelInputBatch(BaseModel):
    features: List[ModelInputFeature]
    # All five assets, aligned by timestamp

class ModelOutputAction(BaseModel):
    asset: StrictStr  # Only MES or MNQ
    timestamp: StrictStr
    action: StrictStr = Field(..., pattern='^(buy|sell|hold)$')
    confidence: StrictFloat
    # Extend with more outputs as needed

class ModelOutputBatch(BaseModel):
    actions: List[ModelOutputAction]
    # Only MES & MNQ actions allowed

# Usage:
# ModelInputBatch(features=[...]) for input to model
# ModelOutputBatch(actions=[...]) for output from model
# Use .model_validate() to validate before/after model inference 