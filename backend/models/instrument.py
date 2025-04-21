from sqlalchemy import Column, Integer, String, Float, BigInteger
from .base import Base

class Instrument(Base):
    __tablename__ = "instruments"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    symbol = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=True)
    exchange = Column(String, nullable=True)
    tick_size = Column(Float, nullable=True)
    lot_size = Column(Float, nullable=True)
    currency = Column(String, nullable=True) 