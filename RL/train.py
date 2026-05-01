import os

import gymnasium as gym
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from torch import nn

import RL.env  # noqa: F401
from RL.constant import stock_ids, train_end, train_start


def train_agent(env, total_timesteps: int = 2_000_000, seed: int = 0):
    network_architecture = dict(pi=[128, 128], vf=[128, 128])
    policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=network_architecture)

    print(f"Initializing PPO agent (seed={seed})...")
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=get_linear_fn(3e-4, 1e-5, 1.0),
        n_steps=1024,
        n_epochs=10,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,
        clip_range=get_linear_fn(0.2, 0.1, 1.0),
        verbose=1,
        target_kl=0.1,
        seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log="./tb_logs/",
    )
    env.reset()
    print(f"Starting training (seed={seed})...")
    model.learn(total_timesteps=total_timesteps, tb_log_name=f"PPO_v7_seed{seed}")

    model.save(f"ppo_trading_agent_v7_seed{seed}")
    env.save(f"vec_normalize_v7_seed{seed}.pkl")
    print(f"Training completed (seed={seed}). Saved ppo_trading_agent_v7_seed{seed}.zip + vec_normalize_v7_seed{seed}.pkl")


def load_data(data_dir: str = f"{os.getcwd()}/RL/data/") -> dict:
    historical_dfs = {}

    for stock_id in stock_ids:
        file_path = os.path.join(data_dir, f"{stock_id}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=["date"], index_col="date")
            df = df[~df.index.duplicated(keep="last")]
            df = df.sort_index()

            historical_dfs[stock_id] = df.loc[train_start:train_end].dropna()
            print(
                f"Successfully load {stock_id}.csv, data length = {len(historical_dfs[stock_id])}"
            )
        else:
            print(f"File path {file_path} not found")
    return historical_dfs


def make_env(stock_data, eval_mode=False):
    return lambda: gym.make(
        "TradingEnv-v0",
        stock_ids=stock_ids,
        stock_data=stock_data,
        eval_mode=eval_mode,
    )


if __name__ == "__main__":
    stock_data = load_data()
    n_envs = 8
    seeds = [0]
    for seed in seeds:
        print(f"\n{'=' * 60}\n=== Training seed {seed} ===\n{'=' * 60}")
        env = SubprocVecEnv([make_env(stock_data) for _ in range(n_envs)])
        env = VecMonitor(env)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)
        train_agent(env=env, seed=seed)
        env.close()
