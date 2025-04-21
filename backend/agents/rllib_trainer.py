import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
import wandb
from backend.envs.multi_asset_env import MultiAssetTradingEnv
from loguru import logger

# Register custom environment for Ray RLlib
def env_creator(env_config):
    asset_list = env_config.get("asset_list", ["AAPL", "BTC"])
    return MultiAssetTradingEnv(asset_list)

register_env("MultiAssetTradingEnv-v0", env_creator)

# PPO config for distributed RL
ppo_config = (
    PPOConfig()
    .environment(env="MultiAssetTradingEnv-v0", env_config={"asset_list": ["AAPL", "BTC"]})
    .rollouts(num_rollout_workers=2)
    .framework("torch")
    .training(train_batch_size=2000, model={"fcnet_hiddens": [128, 128]})
)

# Weights & Biases (wandb) integration stub
def wandb_init(project="finrl-rl", config=None):
    # Call this at the start of training
    wandb.init(project=project, config=config)

# Training function
def train_with_rllib():
    # Optionally initialize wandb
    wandb_init(config=ppo_config.to_dict())
    algo = ppo_config.build()
    for i in range(5):
        result = algo.train()
        logger.info(f"Iteration {i}: reward={result['episode_reward_mean']}")
        # Log to wandb
        wandb.log({"reward": result["episode_reward_mean"]})
    algo.stop()

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    train_with_rllib()
    ray.shutdown() 