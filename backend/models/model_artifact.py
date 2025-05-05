from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Text, func
from backend.models.base import Base

class ModelArtifact(Base):
    __tablename__ = "model_artifacts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    version = Column(Integer, nullable=False)
    path = Column(String(256), nullable=False)
    hash = Column(String(64), nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    is_valid = Column(Boolean, default=True)
    notes = Column(Text, nullable=True) 