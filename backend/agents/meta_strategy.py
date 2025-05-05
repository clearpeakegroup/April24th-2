from backend.services.tick_stream_consumer import TickStreamConsumer
from backend.models.leaderboard import LeaderboardEntry
from backend.models import SessionLocal
import numpy as np
import asyncio

class MetaStrategyController:
    def __init__(self, leaderboard, state, top_n=3):
        self.leaderboard = leaderboard
        self.state = state
        self.top_n = top_n

    def get_top_agents(self, symbol):
        session = SessionLocal()
        try:
            q = session.query(LeaderboardEntry).filter_by(symbol=symbol, stage="live")
            q = q.order_by(LeaderboardEntry.pnl.desc())
            return [e.agent_id for e in q.limit(self.top_n).all()]
        finally:
            session.close()

    def update_weights(self):
        # Example: update weights based on leaderboard
        # (Replace with your blending logic)
        pass

    async def blend_decision(self, tick):
        top_agents = self.get_top_agents(tick["instrument"])
        weights = np.random.dirichlet(np.ones(len(top_agents)))
        blended_signal = np.dot(weights, np.random.randn(len(top_agents)))
        print(f"[META] {tick['instrument']} blended_signal={blended_signal:.4f} weights={weights}") 