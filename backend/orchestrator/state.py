from datetime import datetime, timedelta

class OrchestratorState:
    def __init__(self, config):
        self.equity = config.get("initial_equity", 3500.0)
        self.tier = 0
        self.block_size = (3, 3)
        self.var = 0.0
        self.emargin = 1.0
        self.pnl = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.session_open = False
        self.session_close_time = None
        self.last_equity_check = datetime.utcnow()
        self.config = config
        self.history = []

    def update_equity(self, delta):
        self.equity += delta
        self.history.append((datetime.utcnow(), self.equity))

    def promote_tier(self, new_tier, new_block):
        self.tier = new_tier
        self.block_size = new_block

    def set_session(self, open_status, close_time):
        self.session_open = open_status
        self.session_close_time = close_time 