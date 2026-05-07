"""Deploy-oriented training: latest data, recent val window, single model.

Modes:
  default: train 2015-01-05..2025-09-30, val 2025-10-01..2026-04-22 (regime check)
  --full : train 2015-01-05..2026-04-22, no val (final deploy model)

Goal: maximize next ~1mo return with bear-tail protection. Not generalization.
"""

import argparse
import json
import os

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)
from torch import nn

import RL.env  # noqa: F401
from RL.constant import stock_ids
from RL.policy import PerAssetEncoder

TRAIN_START = "20150105"
SPLIT_TRAIN_END = "20250930"
SPLIT_VAL_START = "20251001"
SPLIT_VAL_END = "20260422"
FULL_TRAIN_END = "20260422"
TOTAL_TIMESTEPS = 3_000_000


def load_data(start, end):
    dfs = {}
    for sid in stock_ids:
        fp = f"RL/data/{sid}.csv"
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp, parse_dates=["date"], index_col="date")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df.loc[start:end].dropna()
        if len(df) > 30:
            dfs[sid] = df
    return dfs


def make_env(stock_data, eval_mode=False):
    return lambda: gym.make(
        "TradingEnv-v0",
        stock_ids=stock_ids,
        stock_data=stock_data,
        eval_mode=eval_mode,
    )


def train(stock_data, tag, seed):
    env = SubprocVecEnv([make_env(stock_data) for _ in range(8)])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            activation_fn=nn.ReLU,
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
            features_extractor_class=PerAssetEncoder,
            features_extractor_kwargs=dict(num_stocks=len(stock_ids)),
        ),
        learning_rate=get_linear_fn(3e-4, 1e-5, 1.0),
        n_steps=1024,
        n_epochs=10,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.02,
        clip_range=get_linear_fn(0.2, 0.1, 1.0),
        verbose=0,
        target_kl=0.1,
        seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log="./tb_logs/",
    )
    model.learn(total_timesteps=TOTAL_TIMESTEPS, tb_log_name=f"PPO_{tag}")

    model_path = f"ppo_{tag}"
    norm_path = f"vec_normalize_{tag}.pkl"
    model.save(model_path)
    env.save(norm_path)
    env.close()
    return model_path, norm_path


def evaluate(model_path, norm_path, val_data):
    env = DummyVecEnv([make_env(val_data, eval_mode=True)])
    env = VecNormalize.load(norm_path, env)
    env.training = False
    env.norm_reward = False

    model = PPO.load(
        model_path,
        env=env,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    obs = env.reset()
    done = False
    final = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, done, info = env.step(action)
        final = info[0]
    env.close()
    return final


def ew_basket_return(stock_data):
    rets = []
    for _sid, df in stock_data.items():
        if len(df) > 1:
            rets.append(df["close"].iloc[-1] / df["close"].iloc[0] - 1)
    return sum(rets) / max(1, len(rets))


def max_drawdown(asset_history):
    arr = np.asarray(asset_history, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd = (peak - arr) / np.maximum(peak, 1.0)
    return float(dd.max())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="train on all data (no val)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()

    if args.full:
        train_end = FULL_TRAIN_END
        tag = args.tag or "deploy_final"
    else:
        train_end = SPLIT_TRAIN_END
        tag = args.tag or (f"deploy_seed{args.seed}" if args.seed else "deploy")

    print(f"{'=' * 70}\nDEPLOY TRAINING [{tag}]\n{'=' * 70}")
    print(f"train: {TRAIN_START}..{train_end}")
    if not args.full:
        print(f"val:   {SPLIT_VAL_START}..{SPLIT_VAL_END}")
    print(f"seed: {args.seed}  timesteps: {TOTAL_TIMESTEPS:,}\n")

    train_data = load_data(TRAIN_START, train_end)
    print(f"train rows: {[(sid, len(df)) for sid, df in list(train_data.items())[:5]]}...")
    model_path, norm_path = train(train_data, tag, args.seed)

    if args.full:
        print(f"\nFull-data model saved: {model_path}.zip + {norm_path}")
    else:
        val_data = load_data(SPLIT_VAL_START, SPLIT_VAL_END)
        print(f"\nval rows: {[(sid, len(df)) for sid, df in list(val_data.items())[:5]]}...")
        ew = ew_basket_return(val_data)
        result = evaluate(model_path, norm_path, val_data)
        ret = result["return_rate"]
        alpha = ret - ew
        trades = result["total_trades"]
        mdd = max_drawdown(result["asset_history"])

        print(f"\n{'=' * 70}\nVAL RESULT [{tag}]\n{'=' * 70}")
        print(f"agent return : {ret * 100:+7.2f}%")
        print(f"EW baseline  : {ew * 100:+7.2f}%")
        print(f"alpha        : {alpha * 100:+7.2f}%")
        print(f"max drawdown : {mdd * 100:7.2f}%")
        print(f"trades       : {trades}")

        with open(f"deploy_val_{tag}.json", "w") as f:
            json.dump(
                {
                    "tag": tag,
                    "seed": args.seed,
                    "train": (TRAIN_START, train_end),
                    "val": (SPLIT_VAL_START, SPLIT_VAL_END),
                    "return": ret,
                    "ew": ew,
                    "alpha": alpha,
                    "max_drawdown": mdd,
                    "trades": trades,
                },
                f,
                indent=2,
                default=str,
            )
        print(f"\nSaved deploy_val_{tag}.json")
