import argparse
import os
import torch
import tensorflow as tf
import numpy as np
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from envs.multi_asset_quad_env import MultiAssetQuadEnv
from data_ingestion.historical_loader import load_zstd_dbn

# Enable GPU memory growth for TensorFlow (if used)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"TF GPU memory growth error: {e}")

def make_env(instruments):
    def _init():
        return MultiAssetQuadEnv(instrument_ids=instruments)
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', choices=['ppo', 'a2c'], default='ppo')
    parser.add_argument('--timesteps', type=int, default=10000)
    parser.add_argument('--checkpoint_freq', type=int, default=1000)
    parser.add_argument('--logdir', type=str, default='logs')
    parser.add_argument('--modeldir', type=str, default='models')
    parser.add_argument('--instruments', nargs='+', default=["MES", "NQ", "ES", "RTY", "QQQ"])
    parser.add_argument('--data', type=str, required=False)
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    os.makedirs(args.modeldir, exist_ok=True)

    # Load historical data (stub: not used in env yet)
    if args.data:
        df = load_zstd_dbn(args.data)
        print(f"Loaded data: {df.shape}")

    # Instantiate environment
    env = DummyVecEnv([make_env(args.instruments)])

    # Select algorithm
    algo_cls = PPO if args.algo == 'ppo' else A2C
    model = algo_cls(
        "MlpPolicy",  # Replace with custom LNN policy if implemented
        env,
        verbose=1,
        tensorboard_log=args.logdir,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Optional: periodic evaluation
    eval_env = DummyVecEnv([make_env(args.instruments)])
    eval_callback = EvalCallback(eval_env, best_model_save_path=args.modeldir,
                                 log_path=args.logdir, eval_freq=args.checkpoint_freq,
                                 deterministic=True, render=False)

    # Train
    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    model.save(os.path.join(args.modeldir, f"{args.algo}_final"))
    print(f"Model saved to {args.modeldir}")

if __name__ == "__main__":
    main() 