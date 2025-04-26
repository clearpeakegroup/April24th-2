from trader_core.strategies.fut_only import strategy_set as FUT_STRATS
from trader_core.strategies.options import strategy_set as OPT_STRATS
from trader_core.strategies.fut_only.ppo_agent import PPOAgent
from trader_core.strategies.fut_only.a2c_agent import A2CAgent
from trader_core.strategies.fut_only.lnn_agent import LNNAgent
from trader_core.strategies.fut_only.lnn_drl_hybrid import LNNDQNHybridAgent
from strategy_engine.head10_news_trigger import StrategyHead as Head10
from strategy_engine.head8_vol_arb import StrategyHead as Head8
from strategy_engine.head9_calendar import StrategyHead as Head9
# Add more as implemented

def get_strategies(cfg):
    if cfg.get('options_enabled', False):
        return FUT_STRATS + OPT_STRATS
    return FUT_STRATS

AGENT_REGISTRY = {
    "ppo": PPOAgent,
    "a2c": A2CAgent,
    "lnn": LNNAgent,
    "lnn_dqn_hybrid": LNNDQNHybridAgent,
    "head10_news_trigger": Head10,
    "head8_vol_arb": Head8,
    "head9_calendar": Head9,
    # ...
}

def agent_factory(config):
    """Instantiate an agent from a config dict.
    Config must include 'agent_type' and any required params for the agent class.
    """
    agent_type = config.get("agent_type")
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent_type: {agent_type}")
    agent_cls = AGENT_REGISTRY[agent_type]
    # Pass all config keys except agent_type
    kwargs = {k: v for k, v in config.items() if k != "agent_type"}
    return agent_cls(**kwargs) 