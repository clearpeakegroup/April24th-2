from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum, func
from backend.models.base import Base
import enum

class FillStatus(enum.Enum):
    filled = "filled"
    partial = "partial"
    cancelled = "cancelled"

class Fill(Base):
    __tablename__ = "fills"
    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=True)
    symbol = Column(String(32), nullable=False)
    qty = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    side = Column(String(8), nullable=False)
    timestamp = Column(DateTime, server_default=func.now(), nullable=False)
    status = Column(Enum(FillStatus), nullable=False, default=FillStatus.filled) 