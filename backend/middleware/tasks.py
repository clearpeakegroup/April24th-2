from .celery_app import celery_app
from backend.agents.registry import AGENT_REGISTRY
import random
import time
from backend.data_ingestion.sentiment_feature_pipeline import run_sentiment_feature_pipeline
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pyarrow.parquet as pq
import glob
import os
import boto3
from github import Github
from datetime import datetime
from loguru import logger
from backend.data_ingestion.historical_loader import SCHEMA, load_zstd_dbn
from backend.models.tick_event import TickEvent
from backend.models import SessionLocal, IngestionAudit
import pandas as pd
import redis
from backend.execution.execution_service import ExecutionService
from celery.utils.log import get_task_logger
from pathlib import Path
import numpy as np
import traceback

# --- Stable-Baselines3 & Gymnasium Imports ---
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3 # Import common agents
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# --- Custom Environment Import ---
from backend.envs.multi_asset_env import MultiAssetTradingEnv

# Use Celery logger
logger = get_task_logger(__name__)

load_dotenv()
BERT_MODEL = os.getenv("BERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
FEATURE_BASE = os.getenv("SENTIMENT_FEATURE_BASE", "data/lake/parquet/features/sentiment")
BERT_OUT_PATH = os.getenv("BERT_OUT_PATH", "models/news_bert_minilm_quant.pt")
S3_BUCKET = os.getenv("S3_BUCKET")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = os.getenv("GITHUB_REPO")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MODELS_DIR = Path("models/trained")
MODELS_DIR.mkdir(parents=True, exist_ok=True) # Ensure models directory exists

# --- Helper to get SB3 Agent Class ---
SB3_AGENT_MAP = {
    "ppo": PPO,
    "a2c": A2C,
    "ddpg": DDPG,
    "sac": SAC,
    "td3": TD3
}

# --- Custom Callback for Progress Reporting ---
class RedisProgressCallback(BaseCallback):
    def __init__(self, job_id, total_timesteps, redis_client, verbose=0):
        super().__init__(verbose)
        self.job_id = job_id
        self.total_timesteps = total_timesteps
        self.redis_client = redis_client
        self.last_report_percent = -1

    def _on_step(self) -> bool:
        current_percent = int(100 * self.num_timesteps / self.total_timesteps)
        # Report progress roughly every 1% or on the last step
        if current_percent > self.last_report_percent or self.num_timesteps == self.total_timesteps:
            try:
                self.redis_client.set(f"jobs:{self.job_id}:progress", current_percent)
                self.last_report_percent = current_percent
                if self.verbose > 0:
                     logger.info(f"Job {self.job_id}: Training progress {current_percent}% ({self.num_timesteps}/{self.total_timesteps})")
            except Exception as e:
                logger.error(f"Job {self.job_id}: Failed to report progress to Redis: {e}")
        return True # Continue training

def get_agent_from_params(params):
    agent_type = params.get("agent_type", "ppo")
    agent_config = params.get("agent_config", {})
    env_config = params.get("env_config", {})
    agent_cls = AGENT_REGISTRY[agent_type]
    return agent_cls(env_config, **agent_config)

@celery_app.task(name="preprocess.task")
def preprocess_task(file_path, user, pipeline_id):
    from backend.services.pipeline_orchestrator import PipelineOrchestrator
    orchestrator = PipelineOrchestrator()
    start_time = time.time()
    try:
        # Simulate preprocessing
        total_steps = 5
        for i in range(total_steps):
            time.sleep(1)
            elapsed = int(time.time() - start_time)
            eta = int((total_steps - (i + 1)) * (elapsed / (i + 1))) if i > 0 else total_steps
            pct = int(100 * (i + 1) / total_steps)
            orchestrator.update_stage(pipeline_id, "preprocessing", pct, elapsed, eta)
        # Save processed file (simulate)
        processed_path = file_path + ".processed.csv"
        with open(processed_path, "w") as f:
            f.write("processed data\n")
        # Trigger training
        retrain_model.apply_async(args=[{"data_path": processed_path, "user": user, "train_args": {"epochs": 5}}, pipeline_id, "training"])
    except Exception as e:
        orchestrator.set_status(pipeline_id, "failed")
        raise

@celery_app.task(bind=True, name="retrain.model")
@celery_app.task(bind=True, name="retrain_model")
@celery_app.task(bind=True, name="retrain_model")
def retrain_model(self, params, pipeline_id=None, stage="training"):
    """
    Trains a Stable-Baselines3 DRL agent based on the provided parameters.
    Replaces the previous placeholder implementation.
    """
    job_id = params.get("job_id")
    if not job_id:
        logger.error("Missing 'job_id' in params")
        # Optionally update Celery state to FAILURE
        self.update_state(state='FAILURE', meta={'error': "Missing 'job_id'"})
        return {"status": "error", "message": "Missing 'job_id' in params"}

    logger.info(f"Starting training job {job_id} with params: {params}")
    redis_client = redis.Redis.from_url(REDIS_URL)
    redis_client.set(f"jobs:{job_id}:progress", 0) # Initialize progress

    try:
        # 1. Parameter Parsing
        data_path = params.get("data_path")
        env_config = params.get("env_config", {})
        agent_type = params.get("agent_type", "ppo").lower()
        agent_config = params.get("agent_config", {})
        total_timesteps = params.get("total_timesteps", 10000) # Default timesteps
        n_envs = params.get("n_envs", 4) # Number of parallel environments

        if not data_path:
            raise ValueError("Missing 'data_path' in params")
        if "asset_list" not in env_config:
            raise ValueError("Missing 'asset_list' in env_config")
        if agent_type not in SB3_AGENT_MAP:
            raise ValueError(f"Unsupported agent_type: {agent_type}")

        # 2. Data Loading (Using Databento DBN Loader)
        logger.info(f"Job {job_id}: Loading Databento DBN data from {data_path}")
        try:
            df = load_zstd_dbn(data_path)
            logger.info(f"Job {job_id}: Loaded {len(df)} rows from {data_path}")
        except Exception as e:
            logger.error(f"Job {job_id}: Failed to load data from {data_path}: {e}")
            raise ValueError(f"Failed to load data from {data_path}: {e}") from e

        # 3. Environment Initialization
        logger.info(f"Job {job_id}: Initializing environment with config: {env_config}")
        # Ensure required args are present
        required_env_args = ["asset_list"]
        for arg in required_env_args:
            if arg not in env_config:
                 raise ValueError(f"Missing required env_config key: {arg}")
        
        # Use make_vec_env for parallel environments
        # Note: MultiAssetTradingEnv needs adaptation for VecEnv (e.g., handling data splits)
        # For now, using DummyVecEnv for simplicity - requires MultiAssetTradingEnv to be fully compliant
        # TODO: Adapt MultiAssetTradingEnv or implement custom VecEnv wrapper if needed
        # env = make_vec_env(lambda: MultiAssetTradingEnv(**env_config), n_envs=n_envs, vec_env_cls=DummyVecEnv)
        # --- Simpler non-vectorized version for initial implementation --- 
        env = MultiAssetTradingEnv(**env_config, historical_data=df) # Pass loaded data
        logger.info(f"Job {job_id}: Environment initialized.")

        # 4. Agent Initialization
        agent_cls = SB3_AGENT_MAP[agent_type]
        # Common SB3 params - TODO: Allow overriding via agent_config
        policy = agent_config.get("policy", "MlpPolicy")
        learning_rate = agent_config.get("learning_rate", 3e-4)
        batch_size = agent_config.get("batch_size", 64)
        n_epochs = agent_config.get("n_epochs", 10)
        n_steps = agent_config.get("n_steps", 2048 // n_envs if n_envs > 0 else 2048) # Adjust n_steps based on n_envs

        # TODO: Add agent-specific parameters based on agent_type
        agent_params = {
            "policy": policy,
            "env": env,
            "learning_rate": learning_rate,
            "verbose": agent_config.get("verbose", 1),
            # Add parameters specific to PPO, A2C etc. conditionally
            # Example for PPO:
            "batch_size": batch_size,
            "n_steps": n_steps,
            "n_epochs": n_epochs,
            # "gamma": ..., 
            # "gae_lambda": ..., 
            # "clip_range": ..., 
            # "ent_coef": ...,
            # "vf_coef": ...,
            # "max_grad_norm": ... 
        }
        # Filter agent_params based on actual agent_cls signature if needed
        logger.info(f"Job {job_id}: Initializing agent {agent_type} with config: {agent_params}")
        agent = agent_cls(**agent_params)
        logger.info(f"Job {job_id}: Agent initialized.")

        # 5. Training with Progress Callback
        logger.info(f"Job {job_id}: Starting training for {total_timesteps} timesteps.")
        progress_callback = RedisProgressCallback(job_id, total_timesteps, redis_client, verbose=1)
        agent.learn(total_timesteps=total_timesteps, callback=progress_callback, log_interval=10)
        logger.info(f"Job {job_id}: Training finished.")

        # 6. Model Saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{agent_type}_{job_id}_{timestamp}.zip"
        model_save_path = MODELS_DIR / model_filename
        agent.save(model_save_path)
        logger.info(f"Job {job_id}: Model saved to {model_save_path}")
        redis_client.set(f"jobs:{job_id}:progress", 100) # Ensure progress hits 100

        # 7. Result
        result = {
            "status": "completed",
            "message": f"Training completed successfully for job {job_id}",
            "model_artifact_path": str(model_save_path)
            # TODO: Add final training metrics if available from callbacks/agent
        }
        # Optionally update Celery state to SUCCESS
        self.update_state(state='SUCCESS', meta=result)
        return result

    except Exception as e:
        logger.exception(f"Job {job_id}: Training failed due to error: {e}")
        redis_client.set(f"jobs:{job_id}:progress", -1) # Indicate error
        # Update Celery state to FAILURE
        self.update_state(state='FAILURE', meta={'error': str(e), 'traceback': traceback.format_exc()})
        return {"status": "error", "message": str(e)}

@celery_app.task(bind=True, name="backtest.run")
@celery_app.task(bind=True, name="run_backtest")
@celery_app.task(bind=True, name="run_backtest")
def run_backtest(self, params, pipeline_id=None, stage="backtesting"):
    """
    Runs a backtest using a trained agent and historical data.
    Replaces the placeholder implementation.
    """
    job_id = params.get("job_id")
    if not job_id:
        logger.error("Missing 'job_id' in params for backtest")
        self.update_state(state='FAILURE', meta={'error': "Missing 'job_id'"})
        return {"status": "error", "message": "Missing 'job_id' for backtest"}

    logger.info(f"Starting backtest job {job_id} with params: {params}")
    redis_client = redis.Redis.from_url(REDIS_URL)
    redis_client.set(f"jobs:{job_id}:progress", 0)

    try:
        # 1. Parameter Parsing
        model_path = params.get("model_artifact_path")
        data_path = params.get("data_path") # Path to backtest data
        env_config = params.get("env_config", {})
        agent_type = params.get("agent_type", "ppo").lower() # Needed to load the correct class

        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(f"Model artifact not found at: {model_path}")
        if not data_path: # TODO: Uncomment when data loading is implemented
            raise ValueError("Missing 'data_path' for backtest data")
        if "asset_list" not in env_config:
            raise ValueError("Missing 'asset_list' in env_config")
        if agent_type not in SB3_AGENT_MAP:
            raise ValueError(f"Unsupported agent_type: {agent_type}")

        # 2. Load Data (Using Databento DBN Loader)
        logger.info(f"Job {job_id}: Loading backtest Databento DBN data from {data_path}")
        try:
            df = load_zstd_dbn(data_path)
            logger.info(f"Job {job_id}: Loaded {len(df)} rows for backtest from {data_path}")
        except Exception as e:
            logger.error(f"Job {job_id}: Failed to load backtest data from {data_path}: {e}")
            raise ValueError(f"Failed to load backtest data from {data_path}: {e}") from e

        # 3. Initialize Environment
        logger.info(f"Job {job_id}: Initializing backtest environment.")
        env = MultiAssetTradingEnv(**env_config, historical_data=df) # Pass loaded data

        # 4. Load Agent
        logger.info(f"Job {job_id}: Loading agent {agent_type} from {model_path}")
        agent_cls = SB3_AGENT_MAP[agent_type]
        agent = agent_cls.load(model_path, env=env)

        # 5. Run Backtest Evaluation Loop
        # Using a manual loop to simulate episode(s) and collect metrics
        # Note: SB3's evaluate_policy might be simpler if env conforms perfectly
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        max_steps = env_config.get("max_backtest_steps", 1000) # Limit steps
        rewards = []
        balances = [env.balance]
        positions_history = []

        logger.info(f"Job {job_id}: Starting backtest evaluation loop (max_steps={max_steps}).")
        redis_client.set(f"jobs:{job_id}:progress", 1) # Mark as started
        while not done and step_count < max_steps:
            action, _states = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
            rewards.append(reward)
            balances.append(info.get("balance", env.balance))
            positions_history.append(info.get("positions", np.zeros(env.n_assets)))
            
            # Update progress
            progress = int(100 * step_count / max_steps)
            if progress % 10 == 0 : # Report every 10%
                 redis_client.set(f"jobs:{job_id}:progress", progress)

        logger.info(f"Job {job_id}: Backtest loop finished after {step_count} steps.")
        redis_client.set(f"jobs:{job_id}:progress", 100)

        # 6. Calculate Metrics (Basic Example)
        final_balance = balances[-1]
        returns = pd.Series(balances).pct_change().dropna()
        pnl = final_balance - env.initial_balance
        sharpe_ratio = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252) if not returns.empty else 0 # Annualized Sharpe
        # TODO: Add Drawdown calculation

        metrics = {
            "steps": step_count,
            "total_reward": total_reward,
            "final_balance": final_balance,
            "profit_loss": pnl,
            "sharpe_ratio_annualized": sharpe_ratio
        }
        logger.info(f"Job {job_id}: Backtest Metrics: {metrics}")

        # 7. Result
        result = {
            "status": "completed",
            "message": f"Backtest completed successfully for job {job_id}",
            "metrics": metrics,
            # "equity_curve": balances # Optional: return full equity curve
        }
        self.update_state(state='SUCCESS', meta=result)
        return result

    except Exception as e:
        logger.exception(f"Job {job_id}: Backtest failed due to error: {e}")
        redis_client.set(f"jobs:{job_id}:progress", -1) # Indicate error
        self.update_state(state='FAILURE', meta={'error': str(e), 'traceback': traceback.format_exc()})
        return {"status": "error", "message": str(e)}

@celery_app.task(bind=True, name="forwardtest.run")
@celery_app.task(bind=True, name="run_forwardtest")
def run_forwardtest(self, params, pipeline_id=None, stage="forwardtesting"):
    """
    Runs a forward test using a trained agent and out-of-sample historical data.
    Replaces the placeholder implementation.
    """
    job_id = params.get("job_id")
    if not job_id:
        logger.error("Missing 'job_id' in params for forward test")
        self.update_state(state='FAILURE', meta={'error': "Missing 'job_id'"})
        return {"status": "error", "message": "Missing 'job_id' for forward test"}

    logger.info(f"Starting forward test job {job_id} with params: {params}")
    redis_client = redis.Redis.from_url(REDIS_URL)
    redis_client.set(f"jobs:{job_id}:progress", 0)

    try:
        # 1. Parameter Parsing (Similar to backtest)
        model_path = params.get("model_artifact_path")
        data_path = params.get("data_path") # Path to OOS forward test data
        env_config = params.get("env_config", {})
        agent_type = params.get("agent_type", "ppo").lower()

        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(f"Model artifact not found at: {model_path}")
        if not data_path: # TODO: Uncomment when data loading is implemented
            raise ValueError("Missing 'data_path' for forward test data")
        if "asset_list" not in env_config:
            raise ValueError("Missing 'asset_list' in env_config")
        if agent_type not in SB3_AGENT_MAP:
            raise ValueError(f"Unsupported agent_type: {agent_type}")

        # 2. Load Data (Using Databento DBN Loader)
        logger.info(f"Job {job_id}: Loading forward test Databento DBN data from {data_path}")
        try:
            df = load_zstd_dbn(data_path)
            logger.info(f"Job {job_id}: Loaded {len(df)} rows for forward test from {data_path}")
        except Exception as e:
            logger.error(f"Job {job_id}: Failed to load forward test data from {data_path}: {e}")
            raise ValueError(f"Failed to load forward test data from {data_path}: {e}") from e

        # 3. Initialize Environment
        logger.info(f"Job {job_id}: Initializing forward test environment.")
        env = MultiAssetTradingEnv(**env_config, historical_data=df) # Pass loaded data

        # 4. Load Agent
        logger.info(f"Job {job_id}: Loading agent {agent_type} from {model_path}")
        agent_cls = SB3_AGENT_MAP[agent_type]
        agent = agent_cls.load(model_path, env=env)

        # 5. Run Forward Test Evaluation Loop (Similar to backtest loop)
        obs, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        max_steps = env_config.get("max_forwardtest_steps", 500) # Limit steps
        rewards = []
        balances = [env.balance]
        positions_history = []

        logger.info(f"Job {job_id}: Starting forward test evaluation loop (max_steps={max_steps}).")
        redis_client.set(f"jobs:{job_id}:progress", 1) # Mark as started
        while not done and step_count < max_steps:
            action, _states = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            step_count += 1
            rewards.append(reward)
            balances.append(info.get("balance", env.balance))
            positions_history.append(info.get("positions", np.zeros(env.n_assets)))
            
            # Update progress
            progress = int(100 * step_count / max_steps)
            if progress % 10 == 0:
                redis_client.set(f"jobs:{job_id}:progress", progress)

        logger.info(f"Job {job_id}: Forward test loop finished after {step_count} steps.")
        redis_client.set(f"jobs:{job_id}:progress", 100)

        # 6. Calculate Metrics (Similar to backtest)
        final_balance = balances[-1]
        returns = pd.Series(balances).pct_change().dropna()
        pnl = final_balance - env.initial_balance
        sharpe_ratio = (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252) if not returns.empty else 0

        metrics = {
            "steps": step_count,
            "total_reward": total_reward,
            "final_balance": final_balance,
            "profit_loss": pnl,
            "sharpe_ratio_annualized": sharpe_ratio
        }
        logger.info(f"Job {job_id}: Forward Test Metrics: {metrics}")

        # 7. Result
        result = {
            "status": "completed",
            "message": f"Forward test completed successfully for job {job_id}",
            "metrics": metrics,
        }
        self.update_state(state='SUCCESS', meta=result)
        return result

    except Exception as e:
        logger.exception(f"Job {job_id}: Forward test failed due to error: {e}")
        redis_client.set(f"jobs:{job_id}:progress", -1)
        self.update_state(state='FAILURE', meta={'error': str(e), 'traceback': traceback.format_exc()})
        return {"status": "error", "message": str(e)}

@celery_app.task(bind=True, name="liveexec.run")
@celery_app.task(bind=True, name="run_live_execution")
def run_live_execution(self, params, pipeline_id=None, stage="livetrading"):
    """
    Runs live trading execution using a trained agent.
    Replaces the placeholder implementation.
    Requires integration with live data feed and execution service.
    """
    job_id = params.get("job_id")
    if not job_id:
        logger.error("Missing 'job_id' in params for live execution")
        self.update_state(state='FAILURE', meta={'error': "Missing 'job_id'"})
        return {"status": "error", "message": "Missing 'job_id' for live execution"}

    logger.info(f"Starting live execution job {job_id} with params: {params}")
    redis_client = redis.Redis.from_url(REDIS_URL)
    redis_client.set(f"jobs:{job_id}:progress", 0) # Progress indicates running state
    redis_client.set(f"jobs:{job_id}:status", "starting")

    # Flag in Redis to allow stopping the task externally
    stop_flag_key = f"jobs:{job_id}:stop_request"
    redis_client.delete(stop_flag_key) # Ensure no stale stop request

    try:
        # 1. Parameter Parsing
        model_path = params.get("model_artifact_path")
        env_config = params.get("env_config", {})
        agent_type = params.get("agent_type", "ppo").lower()
        live_config = params.get("live_config", {})
        trade_interval_seconds = live_config.get("trade_interval_seconds", 60) # Time between decisions

        if not model_path or not Path(model_path).exists():
            raise FileNotFoundError(f"Model artifact not found at: {model_path}")
        if "asset_list" not in env_config:
            raise ValueError("Missing 'asset_list' in env_config")
        if agent_type not in SB3_AGENT_MAP:
            raise ValueError(f"Unsupported agent_type: {agent_type}")

        # 2. Initialize Live Data Feed / Environment (Placeholder)
        logger.warning(f"Job {job_id}: Live data feed integration needed.")
        # TODO: Initialize connection to live data source (e.g., websocket, polling)
        # TODO: Adapt MultiAssetTradingEnv for live mode OR use a separate live environment class
        # env = LiveTradingEnv(**env_config)
        env = MultiAssetTradingEnv(**env_config) # Using simulation for now
        obs, info = env.reset()

        # 3. Load Agent
        logger.info(f"Job {job_id}: Loading agent {agent_type} from {model_path}")
        agent_cls = SB3_AGENT_MAP[agent_type]
        # Note: For live trading, might not need to pass env to load if using separate obs source
        agent = agent_cls.load(model_path) 

        # 4. Initialize Execution Service
        exec_service = ExecutionService() # Assumes it's configured via env vars or defaults
        logger.info(f"Job {job_id}: Execution service initialized.")
        # TODO: Potentially initialize RiskManager as well
        # risk_manager = RiskManager()
        # risk_manager.set_limits(...) # Load limits based on user/params

        # 5. Live Trading Loop
        logger.info(f"Job {job_id}: Entering live trading loop (interval: {trade_interval_seconds}s).")
        redis_client.set(f"jobs:{job_id}:status", "running")
        redis_client.set(f"jobs:{job_id}:progress", 1) # Indicate running
        step_count = 0

        while True:
            start_time = time.time()

            # Check for external stop request
            if redis_client.exists(stop_flag_key):
                logger.info(f"Job {job_id}: Stop request received. Exiting live loop.")
                redis_client.delete(stop_flag_key)
                break

            # a. Get Latest Observation
            # TODO: Replace with actual live data retrieval
            # obs = get_live_observation()
            # Using simulated step for now:
            sim_action = env.action_space.sample() # Simulate some action to advance env state
            obs, reward, terminated, truncated, info = env.step(sim_action)
            if terminated or truncated:
                 logger.warning(f"Job {job_id}: Simulated environment terminated/truncated unexpectedly.")
                 obs, info = env.reset() # Reset simulation if needed
            
            logger.debug(f"Job {job_id}: Fetched observation for step {step_count}")

            # b. Get Action from Agent
            action, _states = agent.predict(obs, deterministic=True)
            logger.info(f"Job {job_id}: Step {step_count} - Predicted Action: {action}")

            # c. TODO: Translate Action to Orders
            # This logic depends heavily on the action space definition and execution needs
            # Example: If action represents target portfolio weights:
            target_weights = action
            current_positions = exec_service.get_positions(user_id=params.get("user_id")) # Fetch current positions
            current_balance = exec_service.get_account_balance(user_id=params.get("user_id"))
            orders_to_place = calculate_orders_from_weights(target_weights, current_positions, current_balance, env_config["asset_list"])
            logger.info(f"Job {job_id}: Step {step_count} - Orders to place: {orders_to_place}")

            # d. TODO: Perform Risk Checks (Optional)
            # try:
            #     for order in orders_to_place:
            #         risk_manager.check_order(user_id, order['symbol'], order['qty'], order['side'])
            # except Exception as risk_e:
            #     logger.error(f"Job {job_id}: Risk check failed for order: {risk_e}")
            #     orders_to_place = [] # Cancel orders if risk check fails

            # e. TODO: Execute Orders
            # for order_params in orders_to_place:
            #     try:
            #         order_result = exec_service.place_order(**order_params) # Pass user_id if needed
            #         logger.info(f"Job {job_id}: Placed order: {order_result}")
            #     except Exception as exec_e:
            #         logger.error(f"Job {job_id}: Failed to place order {order_params}: {exec_e}")

            step_count += 1
            # Update status periodically (e.g., with current PnL)
            redis_client.set(f"jobs:{job_id}:status", f"running_step_{step_count}")
            redis_client.set(f"jobs:{job_id}:progress", step_count) # Use step count as progress

            # Wait for next interval
            elapsed = time.time() - start_time
            sleep_time = max(0, trade_interval_seconds - elapsed)
            time.sleep(sleep_time)

        # 6. Loop Exited (Stopped or Error)
        logger.info(f"Job {job_id}: Live execution loop finished.")
        # Final status update
        redis_client.set(f"jobs:{job_id}:status", "stopped")
        redis_client.set(f"jobs:{job_id}:progress", -2) # Use -2 for stopped normally

        result = {
            "status": "stopped",
            "message": f"Live execution stopped successfully for job {job_id}",
            "steps_executed": step_count
        }
        self.update_state(state='SUCCESS', meta=result) # Use SUCCESS for normal stop
        return result

    except Exception as e:
        logger.exception(f"Job {job_id}: Live execution failed due to error: {e}")
        redis_client.set(f"jobs:{job_id}:progress", -1) # Indicate error
        redis_client.set(f"jobs:{job_id}:status", "failed")
        self.update_state(state='FAILURE', meta={'error': str(e), 'traceback': traceback.format_exc()})
        return {"status": "error", "message": str(e)}

# --- Helper Function (Needs Implementation) ---
def calculate_orders_from_weights(target_weights, current_positions, current_balance, asset_list):
    """Translates target portfolio weights into concrete buy/sell orders."""
    logger.warning("calculate_orders_from_weights is a placeholder and needs implementation.")
    # TODO: Implement logic to compare target_weights with current_positions 
    #       and current_balance to generate a list of orders needed to rebalance.
    #       Consider asset prices, minimum order sizes, etc.
    # Example structure for return:
    # orders = [
    #    {"symbol": "AAPL", "qty": 10, "side": "buy", "order_type": "market"}, 
    #    {"symbol": "MSFT", "qty": 5, "side": "sell", "order_type": "market"}
    # ]
    return []

@celery_app.task(name="features.sentiment_pipeline")
def sentiment_feature_pipeline_task():
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run_sentiment_feature_pipeline())

@celery_app.task(name="train_news_bert")
def train_news_bert_task(params=None):
    # Load latest parquet features
    parquet_files = sorted(glob.glob(f"{FEATURE_BASE}/**/*.parquet", recursive=True), reverse=True)
    if not parquet_files:
        return {"status": "error", "msg": "No sentiment parquet found"}
    table = pq.read_table(parquet_files[0])
    df = table.to_pandas()
    texts = df.get("headline", "") + " " + df.get("body", "")
    labels = df.get("sentiment", 0)
    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    encodings = tokenizer(list(texts), truncation=True, padding=True)
    class NewsDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels.iloc[idx])
            return item
        def __len__(self):
            return len(self.labels)
    dataset = NewsDataset(encodings, labels)
    # Model
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=1)
    # Training
    args = TrainingArguments(output_dir="/tmp/bert_out", per_device_train_batch_size=32, num_train_epochs=1, logging_steps=10, save_steps=1000, fp16=True)
    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()
    # Quantize
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    torch.save(model.state_dict(), BERT_OUT_PATH)
    return {"status": "completed", "model_path": BERT_OUT_PATH, "num_samples": len(df)}

@celery_app.task(name="tag_model_nightly")
def tag_model_nightly():
    tag = datetime.utcnow().strftime("%Y%m%d_nightly")
    s3_key = f"models/{tag}.pt"
    # Upload to S3
    try:
        s3 = boto3.client("s3")
        s3.upload_file(BERT_OUT_PATH, S3_BUCKET, s3_key)
        logger.info(f"Uploaded model to S3: {S3_BUCKET}/{s3_key}")
    except Exception as e:
        logger.error(f"S3 upload failed: {e}")
        return {"status": "error", "msg": str(e)}
    # GitHub release
    try:
        g = Github(GITHUB_TOKEN)
        repo = g.get_repo(GITHUB_REPO)
        release = repo.create_git_release(tag, tag, f"Nightly model {tag}")
        with open(BERT_OUT_PATH, "rb") as f:
            release.upload_asset(BERT_OUT_PATH, name=f"{tag}.pt")
        logger.info(f"Created GitHub release: {tag}")
    except Exception as e:
        logger.error(f"GitHub release failed: {e}")
        return {"status": "error", "msg": str(e)}
    return {"status": "completed", "tag": tag}

@celery_app.task(name="ingest.batch")
def batch_ingest_task(tmp_path, version, user, source, job_id):
    r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    try:
        df = pd.read_csv(tmp_path)
        total = len(df)
        batch_size = 1000
        session = SessionLocal()
        new_records = []
        for i, row in enumerate(df.to_dict(orient="records")):
            exists = session.query(TickEvent).filter_by(
                ts_event=row["ts_event"],
                instrument=row["instrument"],
                exchange=row["exchange"],
                type=row["type"],
                version=version
            ).first()
            if not exists:
                new_records.append(TickEvent(
                    ts_event=row["ts_event"],
                    instrument=row["instrument"],
                    side=row["side"],
                    price=row["price"],
                    size=row["size"],
                    exchange=row["exchange"],
                    type=row["type"],
                    version=version
                ))
            # Commit in batches
            if len(new_records) >= batch_size or i == total - 1:
                session.add_all(new_records)
                session.commit()
                new_records = []
            # Progress update
            if (i + 1) % batch_size == 0 or i == total - 1:
                pct = int(100 * (i + 1) / total)
                r.set(f"ingest:{job_id}:progress", pct)
        status = "success"
        message = f"Inserted {total} records."
    except Exception as e:
        session.rollback()
        status = "fail"
        message = str(e)
        r.set(f"ingest:{job_id}:progress", -1)
    finally:
        # Audit log
        audit = IngestionAudit(
            user=user,
            source=source,
            record_count=total if 'total' in locals() else 0,
            status=status,
            message=message
        )
        session.add(audit)
        session.commit()
        session.close()
        os.remove(tmp_path)
    r.set(f"ingest:{job_id}:progress", 100 if status == "success" else -1)
    return {"status": status, "message": message}

@celery_app.task(name="positions.flatten")
def flatten_positions_task(user_id):
    exec_service = ExecutionService()
    exec_service.flatten_positions(user_id) 
import smtplib
from email.message import EmailMessage

def send_error_alert(subject, body, to="admin@example.com"):
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg["Subject"] = subject
        msg["From"] = "noreply@finrl.local"
        msg["To"] = to
        with smtplib.SMTP("localhost") as s:
            s.send_message(msg)
    except Exception as e:
        print(f"Failed to send alert: {e}")

# Example usage in a task:
# try:
#     ... task logic ...
# except Exception as e:
#     logger.error(f"Task failed: {e}")
#     send_error_alert("FinRL Task Failure", str(e))
#     raise
