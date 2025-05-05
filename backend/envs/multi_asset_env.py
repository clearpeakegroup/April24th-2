import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from loguru import logger

class MultiAssetTradingEnv(gym.Env):
    """
    Custom multi-asset, vectorized, GPU-accelerated trading environment.
    Handles multiple assets, order types, and market microstructure.
    Accepts historical data for backtesting/training.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, asset_list, window_size=50, initial_balance=1e6, slippage_pct=0.0005, latency_steps=1, transaction_cost_pct=0.0002, historical_data: pd.DataFrame = None, **kwargs):
        super().__init__()
        self.asset_list = asset_list
        self.n_assets = len(asset_list)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.slippage_pct = slippage_pct
        self.latency_steps = latency_steps
        self.transaction_cost_pct = transaction_cost_pct
        
        # --- Data Handling ---
        self.historical_data = historical_data
        self.is_live_mode = historical_data is None
        if not self.is_live_mode:
            if not isinstance(historical_data, pd.DataFrame):
                raise TypeError("historical_data must be a pandas DataFrame")
            # TODO: Validate DataFrame columns based on expected features
            self.data_len = len(historical_data)
            if self.data_len <= self.window_size:
                 raise ValueError("Historical data length must be greater than window_size")
            # Assuming columns are multi-indexed (asset, feature) or need reshaping
            # For now, assume data has shape (n_steps, n_assets * n_features)
            # Example: Flattened features per asset
            expected_cols = self.n_assets * self.n_features 
            if historical_data.shape[1] < expected_cols:
                 raise ValueError(f"Historical data needs at least {expected_cols} columns for {self.n_assets} assets and {self.n_features} features.")
            # Pre-reshape or select data if needed
            # Example: Reshape a flat DataFrame (steps, features_flat) into (steps, assets, features)
            # self.data_tensor = torch.tensor(historical_data.values).reshape(self.data_len, self.n_assets, self.n_features)
            self.data_values = historical_data.values # Keep as numpy for now
        else:
            logger.warning("No historical data provided, environment running in simulation/live mode.")
            self.data_len = 0 # Or infinity for true live

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
        # Initialize history based on mode
        if not self.is_live_mode:
            # Start from window_size index to have history
            self.current_step = self.window_size
            history_data = self.data_values[0:self.window_size]
            # Reshape if necessary based on how data is stored
            # Example assumes data_values is (steps, assets*features)
            if history_data.shape[1] == self.n_assets * self.n_features:
                self.history = history_data.reshape(self.window_size, self.n_assets, self.n_features)
            else:
                # Handle other data layouts or raise error
                raise ValueError("Data shape mismatch for history initialization")
        else:
            self.history = np.zeros((self.window_size, self.n_assets, self.n_features))
        self.done = False
        self._action_buffer = []  # For latency simulation

    def _simulate_market(self):
        # Placeholder: random price, volume, bid, ask, spread
        return np.random.randn(self.n_assets, self.n_features)

    def _get_next_market_state(self):
        """Gets the next market state from historical data or simulation."""
        if self.is_live_mode:
            # In live mode, simulate or fetch from live source
            # For now, use simulation
            logger.debug("Using simulated market state in live mode.")
            return self._simulate_market()
        else:
            # In historical mode, get data for the current step
            if self.current_step >= self.data_len:
                 # Should ideally be caught by self.done, but as fallback:
                 logger.warning("Attempted to access data beyond end of historical dataset.")
                 # Return last known state or zeros?
                 return self.history[-1] # Return last row of current history
            
            next_state_row = self.data_values[self.current_step]
            # Reshape if necessary (assuming flat features per asset)
            if next_state_row.shape[0] == self.n_assets * self.n_features:
                return next_state_row.reshape(self.n_assets, self.n_features)
            else:
                raise ValueError("Data shape mismatch for step data retrieval")

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
        # Feature engineering can be applied here if desired
        # obs = self._apply_feature_engineering(obs)
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
        # Get the next market state
        new_obs_features = self._get_next_market_state()
        
        # Update history
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1, :, :] = new_obs_features
        
        # Increment step counter *after* getting data for the current step
        # but *before* checking done condition based on the *next* step
        self.current_step += 1 
        
        obs = self.history.copy()
        # Feature engineering can be applied here if desired
        # obs = self._apply_feature_engineering(obs)
        if self.feature_extractor:
            obs = self.feature_extractor(obs)
        # Track history for custom reward
        if not hasattr(self, 'pnl_history'):
            self.pnl_history = []
        if not hasattr(self, 'positions_history'):
            self.positions_history = []
        # Ensure histories are tracked even if reward isn't calculated every step
        self.pnl_history.append(pnl)
        self.positions_history.append(np.linalg.norm(self.positions))
        
        reward = self._compute_custom_reward(self.pnl_history, self.positions_history)
        
        # Determine if done
        terminated = self.balance <= 0
        # Check if end of historical data reached (add 1 because current_step was incremented)
        truncated = (not self.is_live_mode and self.current_step >= self.data_len) or self.current_step >= 10000 # Add overall step limit
        self.done = terminated or truncated
        
        info = {"balance": self.balance, "positions": self.positions.copy(), "transaction_cost": transaction_cost}
        # Return terminated and truncated separately as per Gymnasium API
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        logger.info(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Positions: {self.positions}") 