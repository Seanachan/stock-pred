import os
from typing import Any

import gymnasium as gym
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import RL.env  # noqa: F401
from RL.constant import (  # noqa: F401
    stock_ids,
    test_end,
    test_start,
    val_end,
    val_start,
)


def val_agent(env: DummyVecEnv, model_path: str) -> dict:
    print(f"Loading {model_path}...")
    model = PPO.load(
        model_path,
        env=env,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    obs: Any = env.reset()
    actions_log = []
    done = False
    final: dict = {}
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        actions_log.append(action)
        obs, rewards, done, info = env.step(action)
        final = info[0]

    asset_diff = max(final["asset_history"]) - min(final["asset_history"])
    print(
        f"  return: {final['return_rate'] * 100:+.2f}%  "
        f"trades: {final['total_trades']}  "
        f"final: {final['total_asset']:,.0f}  "
        f"diff: {asset_diff:,.0f}"
    )
    print(f"  inventory: {final['inventory']}")
    return {
        "return": final["return_rate"],
        "trades": final["total_trades"],
        "final_asset": final["total_asset"],
        "asset_diff": asset_diff,
        "inventory": final["inventory"],
    }


def load_data(
    data_dir: str = f"{os.getcwd()}/RL/data/",
    start: str = val_start,
    end: str = val_end,
) -> dict:
    historical_dfs = {}

    for stock_id in stock_ids:
        file_path = os.path.join(data_dir, f"{stock_id}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=["date"], index_col="date")
            df = df[~df.index.duplicated(keep="last")]
            df = df.sort_index()

            historical_dfs[stock_id] = df.loc[start:end].dropna()
            print(
                f"Successfully load {stock_id}.csv, data length = {len(historical_dfs[stock_id])}"
            )
        else:
            print(f"File path {file_path} not found")
    return historical_dfs


def make_env(stock_data):
    return lambda: gym.make("TradingEnv-v0", stock_ids=stock_ids, stock_data=stock_data)


if __name__ == "__main__":
    import sys

    split = sys.argv[1] if len(sys.argv) > 1 else "val"
    if split == "test":
        start, end = test_start, test_end
    else:
        start, end = val_start, val_end
    print(f"Eval split: {split}  ({start} .. {end})")

    stock_data = load_data(start=start, end=end)

    # Compute EW basket return for benchmark comparison
    ew_returns = []
    for sid, df in stock_data.items():
        if len(df) > 1:
            ew_returns.append(df["close"].iloc[-1] / df["close"].iloc[0] - 1)
    ew_basket_return = sum(ew_returns) / max(1, len(ew_returns))
    print(f"EW-{len(ew_returns)} basket return on {split}: {ew_basket_return * 100:+.2f}%")

    seeds = [0, 1, 2]
    results = []
    for seed in seeds:
        print(f"\n{'=' * 60}\n=== Eval seed {seed} ({split}) ===\n{'=' * 60}")
        env = DummyVecEnv([make_env(stock_data)])
        model_path = f"ppo_trading_agent_v3_seed{seed}"
        result = val_agent(env=env, model_path=model_path)
        result["seed"] = seed
        results.append(result)
        env.close()

    print(f"\n{'=' * 60}\n=== Summary ===\n{'=' * 60}")
    print(
        f"{'seed':<6}{'return':>10}{'EW':>10}{'alpha':>10}"
        f"{'trades':>10}{'final_asset':>16}{'diff':>16}"
    )
    for r in results:
        alpha = r["return"] - ew_basket_return
        print(
            f"{r['seed']:<6}{r['return'] * 100:>9.2f}%"
            f"{ew_basket_return * 100:>9.2f}%{alpha * 100:>9.2f}%"
            f"{r['trades']:>10}"
            f"{r['final_asset']:>16,.0f}{r['asset_diff']:>16,.0f}"
        )
    returns = [r["return"] for r in results]
    mean_ret = sum(returns) / len(returns)
    mean_alpha = mean_ret - ew_basket_return
    print(
        f"\nmean return: {mean_ret * 100:+.2f}%  "
        f"min: {min(returns) * 100:+.2f}%  "
        f"max: {max(returns) * 100:+.2f}%"
    )
    print(f"mean alpha vs EW-{len(ew_returns)}: {mean_alpha * 100:+.2f}%")
