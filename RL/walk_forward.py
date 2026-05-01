"""Walk-forward eval: train rolling cutoffs, val on next year. Distribution of alphas."""

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

FOLDS = [
    ("20150105", "20181231", "20190101", "20191231"),
    ("20150105", "20191231", "20200101", "20201231"),
    ("20150105", "20201231", "20210101", "20211231"),
    ("20150105", "20211231", "20220101", "20221231"),
    ("20150105", "20221231", "20230101", "20231231"),
]


def load_data(start, end):
    dfs = {}
    for sid in stock_ids:
        fp = f"RL/data/{sid}.csv"
        df = pd.read_csv(fp, parse_dates=["date"], index_col="date")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        dfs[sid] = df.loc[start:end].dropna()
    return dfs


def make_env(stock_data, eval_mode=False):
    return lambda: gym.make(
        "TradingEnv-v0",
        stock_ids=stock_ids,
        stock_data=stock_data,
        eval_mode=eval_mode,
    )


def train_fold(stock_data, fold_idx, total_timesteps=1_000_000):
    env = SubprocVecEnv([make_env(stock_data) for _ in range(8)])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            activation_fn=nn.ReLU,
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
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
        seed=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        tensorboard_log="./tb_logs/",
    )
    model.learn(total_timesteps=total_timesteps, tb_log_name=f"PPO_wf_fold{fold_idx}")

    model_path = f"ppo_wf_fold{fold_idx}"
    norm_path = f"vec_normalize_wf_fold{fold_idx}.pkl"
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


if __name__ == "__main__":
    results = []
    for i, (tr_s, tr_e, val_s, val_e) in enumerate(FOLDS):
        print(f"\n{'=' * 70}")
        print(f"FOLD {i}: train {tr_s}..{tr_e}  →  val {val_s}..{val_e}")
        print(f"{'=' * 70}")

        train_data = load_data(tr_s, tr_e)
        print(f"Train data: {[(sid, len(df)) for sid, df in train_data.items()]}")
        model_path, norm_path = train_fold(train_data, i)

        val_data = load_data(val_s, val_e)
        ew = ew_basket_return(val_data)
        result = evaluate(model_path, norm_path, val_data)
        ret = result["return_rate"]
        alpha = ret - ew
        trades = result["total_trades"]
        print(
            f"Fold {i} VAL: return={ret * 100:+.2f}%  "
            f"EW={ew * 100:+.2f}%  alpha={alpha * 100:+.2f}%  trades={trades}"
        )
        results.append(
            {
                "fold": i,
                "train": (tr_s, tr_e),
                "val": (val_s, val_e),
                "return": ret,
                "ew": ew,
                "alpha": alpha,
                "trades": trades,
            }
        )

    print(f"\n{'=' * 70}")
    print("WALK-FORWARD SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"{'fold':<6}{'val period':<24}{'return':>10}"
        f"{'EW':>10}{'alpha':>10}{'trades':>10}"
    )
    for r in results:
        val_label = f"{r['val'][0]}..{r['val'][1]}"
        print(
            f"{r['fold']:<6}{val_label:<24}"
            f"{r['return'] * 100:>9.2f}%"
            f"{r['ew'] * 100:>9.2f}%"
            f"{r['alpha'] * 100:>9.2f}%"
            f"{r['trades']:>10}"
        )

    alphas = [r["alpha"] for r in results]
    print(f"\nmean alpha: {np.mean(alphas) * 100:+.2f}%")
    print(f"min alpha:  {min(alphas) * 100:+.2f}%")
    print(f"max alpha:  {max(alphas) * 100:+.2f}%")
    print(f"std alpha:  {np.std(alphas) * 100:.2f}%")
    pos = sum(1 for a in alphas if a > 0)
    print(f"folds with positive alpha: {pos}/{len(alphas)}")

    with open("walk_forward_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nSaved walk_forward_results.json")
