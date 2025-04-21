from .celery_app import celery_app
from backend.agents.registry import AGENT_REGISTRY
import random
import time

def get_agent_from_params(params):
    agent_type = params.get("agent_type", "ppo")
    agent_config = params.get("agent_config", {})
    env_config = params.get("env_config", {})
    agent_cls = AGENT_REGISTRY[agent_type]
    return agent_cls(env_config, **agent_config)

@celery_app.task(name="backtest.run")
def run_backtest(params):
    agent = get_agent_from_params(params)
    # Simulate slippage, latency, market impact
    time.sleep(random.uniform(0.5, 2.0))  # Simulate latency
    slippage = random.uniform(-0.05, 0.05)  # Simulate slippage
    impact = random.uniform(-0.1, 0.1)      # Simulate market impact
    # result = agent.backtest(slippage=slippage, impact=impact)  # Plug in real logic
    return {"status": "completed", "params": params, "slippage": slippage, "impact": impact}

@celery_app.task(name="forwardtest.run")
def run_forwardtest(params):
    agent = get_agent_from_params(params)
    time.sleep(random.uniform(0.5, 2.0))  # Simulate latency
    slippage = random.uniform(-0.05, 0.05)
    impact = random.uniform(-0.1, 0.1)
    # result = agent.forwardtest(slippage=slippage, impact=impact)  # Plug in real logic
    return {"status": "completed", "params": params, "slippage": slippage, "impact": impact}

@celery_app.task(name="retrain.model")
def retrain_model(params):
    agent = get_agent_from_params(params)
    agent.train(**params.get("train_args", {}))
    agent.save(params.get("model_path", f"models/{params.get('agent_type', 'ppo')}_latest.pt"))
    return {"status": "completed", "params": params}

@celery_app.task(name="liveexec.run")
def run_live_execution(params):
    agent = get_agent_from_params(params)
    agent.load(params.get("model_path", f"models/{params.get('agent_type', 'ppo')}_latest.pt"))
    # Implement live execution logic using agent.act()
    # result = agent.live_execute()  # Implement as needed
    return {"status": "completed", "params": params} 