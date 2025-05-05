from stable_baselines3 import PPO
from backend.agents.base_agent import BaseAgent
from backend.services.adaptive_trade_router import AdaptiveTradeRouter

class PPOAgent(BaseAgent):
    def __init__(self, env_config=None, device=None, policy='MlpPolicy', **kwargs):
        super().__init__(env_config, device)
        self.model = PPO(policy, env_config, device=self.device, **kwargs)
        self.router = AdaptiveTradeRouter()
    def train(self, timesteps=10000, checkpoint_path=None, checkpoint_interval=1000):
        for t in range(0, timesteps, checkpoint_interval):
            self.model.learn(total_timesteps=checkpoint_interval, reset_num_timesteps=False)
            if checkpoint_path:
                self.save(f"{checkpoint_path}_step{t+checkpoint_interval}.zip")
    def act(self, state, flow_signal=None):
        from backend.data_ingestion.live_stream import FlowSignal
        policy_pi0 = self.model.predict(state, deterministic=True)[0]
        if flow_signal is not None:
            router_out = self.router.route(float(policy_pi0), flow_signal)
            return router_out
        return policy_pi0
    def evaluate(self, *args, **kwargs):
        # Implement evaluation logic
        return 0.0
    def save(self, path):
        self.model.save(path)
    def load(self, path):
        self.model = PPO.load(path) 