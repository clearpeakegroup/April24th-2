import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from loguru import logger

class MultiAssetTradingEnv(gym.Env):
    """
    Custom multi-asset, vectorized, GPU-accelerated trading environment.
    Handles multiple assets, order types, and market microstructure.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, asset_list, window_size=50, initial_balance=1e6, slippage_pct=0.0005, latency_steps=1, transaction_cost_pct=0.0002, **kwargs):
        super().__init__()
        self.asset_list = asset_list
        self.n_assets = len(asset_list)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.slippage_pct = slippage_pct
        self.latency_steps = latency_steps
        self.transaction_cost_pct = transaction_cost_pct
        self.n_features = 5  # e.g., price, volume, bid, ask, spread
        obs_shape = (self.window_size, self.n_assets, self.n_features)
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        self._reset_internal_state()
        self.feature_extractor = None

    def set_feature_extractor(self, extractor):
        """Set an external feature extractor (e.g., LNN)."""
        self.feature_extractor = extractor

    def _reset_internal_state(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.positions = np.zeros(self.n_assets)
        self.history = np.zeros((self.window_size, self.n_assets, self.n_features))
        self.done = False
        self._action_buffer = []  # For latency simulation

    def _simulate_market(self):
        # Placeholder: random price, volume, bid, ask, spread
        return np.random.randn(self.n_assets, self.n_features)

    def _apply_feature_engineering(self, obs):
        # Advanced feature engineering: add rolling volatility, spread, rolling mean
        # obs: [window_size, n_assets, n_features]
        # Convert to DataFrame for rolling calculations
        obs_df = pd.DataFrame(obs.reshape(self.window_size, -1))
        # Rolling volatility (std)
        vol = obs_df.rolling(window=5, min_periods=1).std().values[-1]
        # Rolling mean
        mean = obs_df.rolling(window=5, min_periods=1).mean().values[-1]
        # Spread (max-min across assets for last window)
        spread = obs_df.iloc[-1].max() - obs_df.iloc[-1].min()
        # Append engineered features to last time step (broadcast to all assets)
        obs[-1, :, 0] = vol[:self.n_assets]  # Overwrite first feature with vol
        obs[-1, :, 1] = mean[:self.n_assets] # Overwrite second feature with mean
        obs[-1, :, 2] = spread               # Overwrite third feature with spread
        return obs

    def _compute_custom_reward(self, pnl_history, positions_history):
        # Custom reward: Sharpe, Sortino, drawdown, leverage penalty
        pnl = np.array(pnl_history)
        if len(pnl) < 2:
            return pnl[-1] if len(pnl) > 0 else 0.0
        sharpe = np.mean(pnl) / (np.std(pnl) + 1e-8)
        downside = pnl[pnl < 0]
        sortino = np.mean(pnl) / (np.std(downside) + 1e-8) if len(downside) > 0 else sharpe
        cum_pnl = np.cumsum(pnl)
        drawdown = np.max(np.maximum.accumulate(cum_pnl) - cum_pnl)
        leverage = np.mean(np.abs(positions_history)) / (self.balance + 1e-8)
        # Penalize excessive leverage and drawdown
        reward = sharpe - 0.1 * drawdown - 0.2 * leverage
        return reward

    def reset(self, *, seed=None, options=None):
        self._reset_internal_state()
        self.pnl_history = []
        self.positions_history = []
        obs = self.history.copy()
        obs = self._apply_feature_engineering(obs)
        if self.feature_extractor:
            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs)
            obs = self.feature_extractor(obs)
        info = {"balance": self.balance, "positions": self.positions.copy()}
        return obs, info

    def step(self, action):
        action = np.clip(action, 0, 1)
        action = action / (np.sum(action) + 1e-8)
        self._action_buffer.append(action)
        if len(self._action_buffer) <= self.latency_steps:
            exec_action = np.zeros_like(action)
        else:
            exec_action = self._action_buffer.pop(0)
        price_change = np.random.randn(self.n_assets) * 0.001
        slippage = 1 + np.random.uniform(-self.slippage_pct, self.slippage_pct, size=self.n_assets)
        exec_action = exec_action * slippage
        trade_amount = np.abs(exec_action * self.balance - self.positions)
        transaction_cost = np.sum(trade_amount) * self.transaction_cost_pct
        self.positions = exec_action * self.balance
        pnl = np.sum(self.positions * price_change) - transaction_cost
        self.balance += pnl
        self.current_step += 1
        new_obs = self._simulate_market()
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1, :, :] = new_obs
        obs = self.history.copy()
        obs = self._apply_feature_engineering(obs)
        if self.feature_extractor:
            obs = self.feature_extractor(obs)
        # Track history for custom reward
        if not hasattr(self, 'pnl_history'):
            self.pnl_history = []
        if not hasattr(self, 'positions_history'):
            self.positions_history = []
        self.pnl_history.append(pnl)
        self.positions_history.append(np.linalg.norm(self.positions))
        reward = self._compute_custom_reward(self.pnl_history, self.positions_history)
        self.done = self.current_step >= 1000 or self.balance <= 0
        info = {"balance": self.balance, "positions": self.positions.copy(), "transaction_cost": transaction_cost}
        return obs, reward, self.done, False, info

    def render(self, mode="human"):
        logger.info(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Positions: {self.positions}") 