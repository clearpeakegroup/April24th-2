from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from sqlalchemy import Column, Integer, String, DateTime, Text, func
from .base import Base
from .user import User
from .order import Order, OrderStatus
from .audit_log import AuditLog
from .strategy_config import StrategyConfig
from .agent import Agent, AgentStatus
from .fill import Fill, FillStatus
from .equity_curve import EquityCurve
from .model_artifact import ModelArtifact

# Use the same database URL as alembic.ini
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://finrl:finrlpass@localhost:5432/finrl_db")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class IngestionAudit(Base):
    __tablename__ = "ingestion_audit"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, server_default=func.now(), nullable=False)
    user = Column(String, nullable=True)
    source = Column(String, nullable=False)
    record_count = Column(Integer, nullable=False)
    status = Column(String, nullable=False)  # e.g., 'success', 'fail'
    message = Column(Text, nullable=True)

# Removed TestTable dummy model after Alembic migration test 