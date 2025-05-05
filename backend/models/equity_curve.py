from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey, func
from backend.models.base import Base

class EquityCurve(Base):
    __tablename__ = "equity_curves"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=True)
    timestamp = Column(DateTime, server_default=func.now(), nullable=False)
    equity = Column(Float, nullable=False)
    pnl = Column(Float, nullable=False) 