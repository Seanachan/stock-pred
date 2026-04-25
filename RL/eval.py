import os

import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from RL.constant import stock_ids, val_end, val_start, test_end, test_start
from RL.env import TradingEnv


def val_agent(env: DummyVecEnv, total_timesteps: int = 100000):
    # Initialize PPO agent
    print("Initializing PPO agent...")
    model = PPO.load(
        "ppo_trading_agent_v1",
        env=env,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    # Evaluate the agent
    print("Starting evaluation...")
    obs = env.reset()
    actions_log = []
    done = False
    final = {}
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        actions_log.append(action)
        obs, rewards, done, info = env.step(action)
        final = info[0]

    print(f"Final return rate: {final['return_rate'] * 100:.2f}%")
    print(f"Total Trades: {final['total_trades']}")
    print(f"Total Assets: {final['total_asset']}")
    print(f"Final Inventory: {final['inventory']}")
    print(
        f"Asset History diff: {max(final['asset_history']) - min(final['asset_history']):.2f}"
    )

    print("Evaluation completed.")


def load_data(data_dir: str = f"{os.getcwd()}/RL/data/") -> dict:
    historical_dfs = {}

    for stock_id in stock_ids:
        file_path = os.path.join(data_dir, f"{stock_id}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=["date"], index_col="date")
            df = df[~df.index.duplicated(keep="last")]
            df = df.sort_index()

            historical_dfs[stock_id] = df.loc[test_start:test_end].dropna()
            print(
                f"Successfully load {stock_id}.csv, data length = {len(historical_dfs[stock_id])}"
            )
        else:
            print(f"File path {file_path} not found")
    return historical_dfs


if __name__ == "__main__":
    stock_data = load_data()
    env = DummyVecEnv([lambda: TradingEnv(stock_ids=stock_ids, stock_data=stock_data)])
    val_agent(env=env)
