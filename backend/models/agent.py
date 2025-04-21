from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Enum, func
from backend.models.base import Base
import enum

class AgentStatus(enum.Enum):
    training = "training"
    ready = "ready"
    live = "live"
    error = "error"

class Agent(Base):
    __tablename__ = "agents"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(128), nullable=False)
    type = Column(String(64), nullable=False)
    status = Column(Enum(AgentStatus), nullable=False, default=AgentStatus.training)
    config_id = Column(Integer, ForeignKey("strategy_configs.id"), nullable=False)
    artifact_path = Column(String(256), nullable=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False) 