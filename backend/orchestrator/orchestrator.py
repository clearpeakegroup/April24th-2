import asyncio
from datetime import datetime, timedelta
from backend.orchestrator.state import OrchestratorState
from backend.services.leaderboard_service import upsert_leaderboard
from backend.agents.meta_strategy import MetaStrategyController
from backend.models import SessionLocal, Agent
from backend.models.leaderboard import LeaderboardEntry
from backend.middleware.tasks import batch_ingest_task, preprocess_task, retrain_model, run_backtest, run_forwardtest, run_live_execution, flatten_positions_task
import logging
from backend.execution.execution_service import ExecutionService

class SuperLiquidAvalancheOrchestrator:
    def __init__(self, config):
        self.state = OrchestratorState(config)
        self.leaderboard = upsert_leaderboard
        self.meta_controller = MetaStrategyController(self.leaderboard, self.state)
        self.session_open_time = config.get("session_open_time", "09:30")
        self.session_close_time = config.get("session_close_time", "16:00")
        self.equity_ladder = config.get("equity_ladder", [
            (3500, (3, 3)), (4800, (4, 4)), (7200, (6, 6)), (12000, (8, 8)), (16000, (10, 10)), (20000, (12, 12))
        ])
        self.var_cap_pct = config.get("var_cap_pct", 0.15)
        self.e_margin_min = config.get("e_margin_min", 0.30)
        self.min_hold_sec = config.get("min_hold_sec", 15)
        self.last_equity_check = datetime.utcnow()

    async def run(self):
        while True:
            now = datetime.utcnow()
            if self._is_session_open(now):
                self.state.set_session(True, self._get_session_close(now))
                await self._run_day_trading_cycle()
            else:
                self.state.set_session(False, self._get_session_close(now))
                await self._flatten_positions()
                await asyncio.sleep(self._seconds_until_next_open(now))

    async def _run_day_trading_cycle(self):
        # 1. Ingest, preprocess, train, backtest, forward, live
        await self._ingest_data()
        await self._preprocess()
        await self._train_agents()
        await self._backtest_agents()
        await self._forward_test_agents()
        await self._live_trade()
        self._compound_and_promote()
        self.meta_controller.update_weights()
        # Equity check and tier promotion
        if (datetime.utcnow() - self.last_equity_check).total_seconds() > 15:
            self._check_equity_ladder()
            self.last_equity_check = datetime.utcnow()

    async def _ingest_data(self):
        # Example: Ingest a new file (replace with actual file path/user as needed)
        file_path = self.state.config.get("ingest_file_path")
        user = self.state.config.get("user")
        if file_path:
            result = batch_ingest_task.apply_async(args=[file_path, 1, user, "orchestrator"])
            result.get(timeout=300)  # Wait for completion or handle async

    async def _preprocess(self):
        # Example: Preprocess the ingested file (replace with actual file path/user as needed)
        file_path = self.state.config.get("preprocess_file_path")
        user = self.state.config.get("user")
        if file_path:
            result = preprocess_task.apply_async(args=[file_path, user, "orchestrator"])
            result.get(timeout=300)  # Wait for completion or handle async

    async def _train_agents(self):
        train_args = self.state.config.get("train_args", {"epochs": 5})
        data_path = self.state.config.get("train_data_path")
        user = self.state.config.get("user")
        pipeline_id = self.state.config.get("pipeline_id", "orchestrator")
        if data_path:
            params = {"data_path": data_path, "user": user, "train_args": train_args}
            result = retrain_model.apply_async(args=[params, pipeline_id, "training"])
            result.get(timeout=1800)

    async def _backtest_agents(self):
        model_path = self.state.config.get("model_path")
        user = self.state.config.get("user")
        pipeline_id = self.state.config.get("pipeline_id", "orchestrator")
        if model_path:
            params = {"model_path": model_path, "user": user}
            result = run_backtest.apply_async(args=[params, pipeline_id, "backtesting"])
            result.get(timeout=1800)

    async def _forward_test_agents(self):
        model_path = self.state.config.get("model_path")
        user = self.state.config.get("user")
        pipeline_id = self.state.config.get("pipeline_id", "orchestrator")
        if model_path:
            params = {"model_path": model_path, "user": user}
            result = run_forwardtest.apply_async(args=[params, pipeline_id, "forwardtesting"])
            result.get(timeout=1800)

    async def _live_trade(self):
        model_path = self.state.config.get("model_path")
        user = self.state.config.get("user")
        pipeline_id = self.state.config.get("pipeline_id", "orchestrator")
        if model_path:
            params = {"model_path": model_path, "user": user}
            result = run_live_execution.apply_async(args=[params, pipeline_id, "livetrading"])
            result.get(timeout=1800)

    async def _flatten_positions(self):
        user = self.state.config.get("user")
        if user is not None:
            result = flatten_positions_task.apply_async(args=[user])
            result.get(timeout=300)

    def _compound_and_promote(self): pass

    def _is_session_open(self, now):
        open_time = now.replace(hour=int(self.session_open_time[:2]), minute=int(self.session_open_time[3:]), second=0, microsecond=0)
        close_time = now.replace(hour=int(self.session_close_time[:2]), minute=int(self.session_close_time[3:]), second=0, microsecond=0)
        return open_time <= now < close_time

    def _get_session_close(self, now):
        return now.replace(hour=int(self.session_close_time[:2]), minute=int(self.session_close_time[3:]), second=0, microsecond=0)

    def _seconds_until_next_open(self, now):
        open_time = now.replace(hour=int(self.session_open_time[:2]), minute=int(self.session_open_time[3:]), second=0, microsecond=0)
        if now < open_time:
            return (open_time - now).total_seconds()
        return ((open_time + timedelta(days=1)) - now).total_seconds()

    def _check_equity_ladder(self):
        for eq, block in reversed(self.equity_ladder):
            if self.state.equity >= eq:
                if self.state.tier != eq:
                    self.state.promote_tier(eq, block)
                break 