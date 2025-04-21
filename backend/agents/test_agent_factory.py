from backend.agents.registry import agent_factory
import torch
from loguru import logger

def test_agent_factory():
    configs = [
        {"agent_type": "ppo", "env_config": None},
        {"agent_type": "a2c", "env_config": None},
        {"agent_type": "lnn", "env_config": None, "input_dim": 5, "hidden_dim": 8, "output_dim": 3},
        {"agent_type": "lnn_dqn_hybrid", "env_config": None, "input_dim": 5, "hidden_dim": 8, "output_dim": 3},
    ]
    for cfg in configs:
        try:
            agent = agent_factory(cfg)
            logger.info(f"Instantiated agent: {cfg['agent_type']}")
            # Try a forward pass or act
            if hasattr(agent, "act"):
                state = torch.randn(cfg.get("input_dim", 5))
                action = agent.act(state)
                logger.info(f"Agent {cfg['agent_type']} act output: {action}")
        except Exception as e:
            logger.error(f"Agent {cfg['agent_type']} failed: {e}")

if __name__ == "__main__":
    test_agent_factory() 