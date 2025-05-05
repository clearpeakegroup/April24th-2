import ray
from backend.agents.registry import AGENT_REGISTRY
from backend.agents.base_agent import get_device
import torch

@ray.remote(num_gpus=1)
def train_agent(agent_name, config, env_config):
    agent_cls = AGENT_REGISTRY[agent_name]
    device = get_device()
    agent = agent_cls(env_config, device=device, **config)
    agent.train()
    eval_score = agent.evaluate()
    agent.save(f"models/{agent_name}_{ray.get_runtime_context().task_id}.pt")
    return eval_score

def launch_parallel_training(agent_names, configs, env_config):
    ray.init(ignore_reinit_error=True)
    results = []
    for agent_name, config in zip(agent_names, configs):
        results.append(train_agent.remote(agent_name, config, env_config))
    scores = ray.get(results)
    ray.shutdown()
    return scores 