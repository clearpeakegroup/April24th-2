from backend.core.hybrid_agents.lnn_ppo import LNNPPO
from backend.core.hybrid_agents.dueling_dqn_lnn import DuelingDQNLNN

AGENTS = {
    "lnn_ppo": LNNPPO,
    "dueling_dqn_lnn": DuelingDQNLNN,
}

def get_agent(name: str):
    """Factory to get agent class by name."""
    return AGENTS[name]() 