from backend.models import SessionLocal
from backend.models.leaderboard import LeaderboardEntry
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.sql import func

def upsert_leaderboard(agent_id, user_id, symbol, stage, pnl, sharpe, drawdown, latency_ns):
    session = SessionLocal()
    try:
        stmt = insert(LeaderboardEntry).values(
            agent_id=agent_id, user_id=user_id, symbol=symbol, stage=stage,
            pnl=pnl, sharpe=sharpe, drawdown=drawdown, latency_ns=latency_ns
        ).on_conflict_do_update(
            index_elements=['agent_id', 'symbol', 'stage'],
            set_={
                "pnl": pnl, "sharpe": sharpe, "drawdown": drawdown,
                "latency_ns": latency_ns, "timestamp": func.now()
            }
        )
        session.execute(stmt)
        session.commit()
    finally:
        session.close() 