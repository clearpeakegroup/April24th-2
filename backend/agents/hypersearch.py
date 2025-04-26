import ray
from ray import tune
from backend.agents.registry import AGENT_REGISTRY
from backend.agents.base_agent import get_device

def trainable(config):
    agent_cls = AGENT_REGISTRY[config['agent_name']]
    agent = agent_cls(config['env_config'], device=get_device(), **config)
    agent.train()
    return {"score": agent.evaluate()}

def run_hypersearch(agent_name, search_space, env_config):
    analysis = tune.run(
        trainable,
        config={
            "agent_name": agent_name,
            "env_config": env_config,
            **search_space
        },
        resources_per_trial={"gpu": 1}
    )
    return analysis.get_best_config(metric="score", mode="max") 