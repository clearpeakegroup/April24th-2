from pydantic import BaseModel, field_validator
from typing import Literal

class Order(BaseModel):
    symbol: str
    side: Literal["BUY", "SELL"]
    qty: int
    price: float | None = None

    @field_validator('qty')
    @classmethod
    def qty_positive(cls, v):
        if v <= 0:
            raise ValueError('qty must be positive')
        return v

    @field_validator('symbol')
    @classmethod
    def symbol_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('symbol must not be empty')
        return v 