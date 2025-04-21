from sqlalchemy import Column, Integer, String, Float, BigInteger, DateTime, JSON
from .base import Base
import datetime

class BacktestRun(Base):
    __tablename__ = "backtest_runs"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    parameters = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    notes = Column(String, nullable=True)

class ForwardTestRun(Base):
    __tablename__ = "forwardtest_runs"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    started_at = Column(DateTime, default=datetime.datetime.utcnow)
    finished_at = Column(DateTime, nullable=True)
    parameters = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    notes = Column(String, nullable=True) 