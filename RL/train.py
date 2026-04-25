import os

import gymnasium as gym
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

import RL.env  # noqa: F401
from RL.constant import stock_ids, train_end, train_start


def train_agent(env: DummyVecEnv, total_timesteps: int = 300000, seed: int = 0):
    # Define MLP policy for PPO
    # policy: Actor to output action probabilities
    # value_function: Critic to estimate state values
    network_architecture = dict(pi=[256, 256], vf=[256, 128])
    policy_kwargs = dict(
        activation_fn=nn.ReLU,
        net_arch=network_architecture,
    )

    # Initialize PPO agent
    print(f"Initializing PPO agent (seed={seed})...")
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        n_steps=2048,
        n_epochs=4,
        batch_size=64,
        gamma=0.99,
        ent_coef=0.01,
        verbose=1,
        target_kl=0.05,
        seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log="./tb_logs/",
    )
    # Train the agent
    env.reset()
    print(f"Starting training (seed={seed})...")
    model.learn(total_timesteps=total_timesteps, tb_log_name=f"PPO_seed{seed}")

    model.save(f"ppo_trading_agent_v3_seed{seed}")
    print(f"Training completed (seed={seed}). Saved ppo_trading_agent_v3_seed{seed}.zip")


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


def make_env(stock_data):
    return lambda: gym.make("TradingEnv-v0", stock_ids=stock_ids, stock_data=stock_data)


if __name__ == "__main__":
    stock_data = load_data()
    seeds = [0, 1, 2]
    for seed in seeds:
        print(f"\n{'=' * 60}\n=== Training seed {seed} ===\n{'=' * 60}")
        env = DummyVecEnv([make_env(stock_data)])
        train_agent(env=env, seed=seed)
        env.close()
