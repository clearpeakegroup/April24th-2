from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import Optional
import threading
import logging
from trader_core.runner import StrategyRunner
from trader_core.data_connectors import LiveFeedConnector
from trader_core.execution import SimulatedExecution
from trader_core.middleware import MiddlewareManager, LoggingMiddleware, RiskManagementMiddleware, PnLMiddleware

app = FastAPI(title="FinRL Platform API")

# Global runner and middleware instances for demonstration (not production safe)
runner_instance = None
runner_thread = None
risk_middleware = RiskManagementMiddleware(max_position=10.0)
pnl_middleware = PnLMiddleware()

class RunRequest(BaseModel):
    strategy: str
    module: str
    symbol: Optional[str] = 'ES'
    interval: Optional[float] = 0.1
    max_position: Optional[float] = 10.0

@app.get("/status")
def status():
    """Get the status of the strategy runner."""
    global runner_instance
    return {"running": runner_instance is not None}

@app.post("/start")
def start(req: RunRequest, background_tasks: BackgroundTasks):
    """Start the strategy runner with the given parameters."""
    global runner_instance, runner_thread, risk_middleware, pnl_middleware
    if runner_instance is not None:
        raise HTTPException(status_code=400, detail="Runner already running")
    # Dynamically import strategy
    import importlib
    module = importlib.import_module(req.module)
    strategy_cls = getattr(module, req.strategy)
    data_connector = LiveFeedConnector(symbol=req.symbol, interval=req.interval)
    execution = SimulatedExecution()
    risk_middleware = RiskManagementMiddleware(max_position=req.max_position)
    pnl_middleware = PnLMiddleware()
    middleware_manager = MiddlewareManager([
        LoggingMiddleware(),
        risk_middleware,
        pnl_middleware
    ])
    runner_instance = StrategyRunner(strategy_cls, data_connector, execution, middleware_manager)
    def run():
        runner_instance.run()
    runner_thread = threading.Thread(target=run, daemon=True)
    runner_thread.start()
    return {"status": "started"}

@app.post("/stop")
def stop():
    """Stop the strategy runner."""
    global runner_instance, runner_thread
    runner_instance = None
    runner_thread = None
    return {"status": "stopped"}

@app.get("/metrics")
def metrics():
    """Return basic metrics including PnL and position."""
    global pnl_middleware
    return {
        "pnl": pnl_middleware.pnl,
        "position": pnl_middleware.position
    }

@app.get("/risk")
def get_risk():
    """Get current risk management settings."""
    global risk_middleware
    return {"max_position": risk_middleware.max_position, "current_position": risk_middleware.current_position}

class RiskUpdateRequest(BaseModel):
    max_position: float

@app.post("/risk")
def set_risk(req: RiskUpdateRequest):
    """Update max position for risk management."""
    global risk_middleware
    risk_middleware.max_position = req.max_position
    return {"max_position": risk_middleware.max_position}

@app.get("/logs")
def logs():
    """Return recent logs (stub)."""
    return {"logs": "Not implemented"} 