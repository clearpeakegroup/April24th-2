import pandas as pd
import asyncio
from typing import Any, AsyncGenerator, Dict, Iterable, AsyncIterable
from strategy_engine.factory import get_active_heads
from strategy_engine.base import BacktestResult, Progress

# Helper to extract toggles from params
def extract_toggles(params: Dict[str, Any]):
    stage = params.get('stage', 'train')
    with_options = params.get('with_options', False)
    return stage, with_options

def run_backtest(**params) -> AsyncGenerator[dict, None]:
    # Expect: params['data'] (pd.DataFrame), toggles
    data = params['data']
    stage, with_options = extract_toggles(params)
    heads = get_active_heads(stage, with_options)
    for i, head in enumerate(heads):
        result = head.backtest(data)
        yield {"head": head.__class__.__name__, "pnl": result.pnl, "progress": 100, "msg": f"{head.__class__.__name__} done"}

async def run_forward_test(**params) -> AsyncGenerator[dict, None]:
    data_iter = params['data_iter']
    stage, with_options = extract_toggles(params)
    heads = get_active_heads(stage, with_options)
    for i, head in enumerate(heads):
        for prog in head.forward(data_iter):
            yield {"head": head.__class__.__name__, "pct": prog.pct, "msg": prog.msg}

async def run_live(**params) -> AsyncGenerator[dict, None]:
    feed = params['feed']
    stage, with_options = extract_toggles(params)
    heads = get_active_heads(stage, with_options)
    for i, head in enumerate(heads):
        async for order in head.live(feed):
            yield {"head": head.__class__.__name__, "order": order.to_dict()} 