from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, func
from backend.models.base import Base

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    action = Column(String(64), nullable=False)
    details = Column(Text, nullable=True)
    timestamp = Column(DateTime, server_default=func.now(), nullable=False) 