from backend.models import SessionLocal
from backend.models.tick_event import TickEvent
from sqlalchemy.future import select

async def get_ticks(symbol: str, limit: int = 100, source: str = None):
    session = SessionLocal()
    try:
        q = select(TickEvent).where(TickEvent.instrument == symbol)
        if source:
            q = q.where(TickEvent.source == source)
        q = q.order_by(TickEvent.ts_event.desc()).limit(limit)
        result = await session.execute(q)
        return [row for row in result.scalars().all()]
    finally:
        session.close() 