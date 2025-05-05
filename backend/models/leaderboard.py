from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, BigInteger
from sqlalchemy.sql import func
from backend.models.base import Base

class LeaderboardEntry(Base):
    __tablename__ = 'leaderboard_entries'

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey('agents.id'), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    symbol = Column(String, nullable=True, index=True) # Nullable if representing overall portfolio
    stage = Column(String, nullable=False, index=True) # e.g., 'backtest', 'forwardtest', 'live'
    pnl = Column(Float, nullable=False, default=0.0)
    sharpe = Column(Float, nullable=True)
    drawdown = Column(Float, nullable=True)
    latency_ns = Column(BigInteger, nullable=True) # Nanoseconds for precision
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    def __repr__(self):
        return f"<LeaderboardEntry(id={self.id}, user={self.user_id}, agent={self.agent_id}, stage='{self.stage}', pnl={self.pnl})>" 