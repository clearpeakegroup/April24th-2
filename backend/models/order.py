from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum, func
from sqlalchemy.orm import relationship
from backend.models.base import Base
import enum

class OrderStatus(enum.Enum):
    new = "new"
    pending = "pending"
    filled = "filled"
    rejected = "rejected"
    cancelled = "cancelled"

class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    agent_id = Column(Integer, nullable=True)
    symbol = Column(String(32), nullable=False)
    qty = Column(Float, nullable=False)
    side = Column(String(8), nullable=False)  # buy/sell
    order_type = Column(String(16), nullable=False, default="market")
    status = Column(Enum(OrderStatus), nullable=False, default=OrderStatus.new)
    broker_order_id = Column(String(64), nullable=True)
    submitted_at = Column(DateTime, server_default=func.now(), nullable=False)
    filled_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)
    rejected_at = Column(DateTime, nullable=True)
    meta = Column(String, nullable=True)  # JSON string for extra data

    user = relationship("User") 