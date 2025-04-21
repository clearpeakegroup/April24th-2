from sqlalchemy import Column, Integer, String, Float, BigInteger
from .base import Base

class TickEvent(Base):
    __tablename__ = "tick_events"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    ts_event = Column(BigInteger, index=True, nullable=False)
    instrument = Column(String, index=True, nullable=False)
    side = Column(String, nullable=True)
    price = Column(Float, nullable=False)
    size = Column(Float, nullable=True)
    exchange = Column(String, nullable=True)
    type = Column(String, nullable=True)
    version = Column(Integer, nullable=False, default=1) 