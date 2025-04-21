from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, Text, func
from backend.models.base import Base

class StrategyConfig(Base):
    __tablename__ = "strategy_configs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(128), nullable=False)
    version = Column(Integer, nullable=False, default=1)
    config_json = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    is_active = Column(Boolean, default=True)
    previous_version_id = Column(Integer, ForeignKey("strategy_configs.id"), nullable=True) 