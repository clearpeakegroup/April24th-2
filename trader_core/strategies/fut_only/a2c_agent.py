from stable_baselines3 import A2C
from backend.agents.base_agent import BaseAgent

class A2CAgent(BaseAgent):
    def __init__(self, env_config=None, device=None, policy='MlpPolicy', **kwargs):
        super().__init__(env_config, device)
        self.model = A2C(policy, env_config, device=self.device, **kwargs)
    def train(self, timesteps=10000, checkpoint_path=None, checkpoint_interval=1000):
        for t in range(0, timesteps, checkpoint_interval):
            self.model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False)
            if checkpoint_path:
                self.save(f"{checkpoint_path}_step{t+checkpoint_interval}.zip")
    def act(self, state):
        return self.model.predict(state, deterministic=True)[0]
    def evaluate(self, *args, **kwargs):
        # Implement evaluation logic
        return 0.0
    def save(self, path):
        self.model.save(path)
    def load(self, path):
        self.model = A2C.load(path) 