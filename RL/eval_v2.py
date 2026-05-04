"""Eval v2 PPO models on 2024-2026 test split.

Restores v2 env semantics manually (no gym registration):
- 10 stocks (v2 universe)
- 8 features per stock + cash + inventory = 91-dim obs (raw, no VecNormalize)
- MultiDiscrete[5]^10: 0=sell_2 1=sell_1 2=hold 3=buy_1 4=buy_2 (fixed lots)
- Trades at prev-step price (anti-lookahead)

Goal: verify xlsx claim of PPO_v2_seed1 = +93.80% on 2024-01..2026-04.
"""

import json
import os

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from RL.feature import FeatureExtractor

V2_STOCKS = [
    "2330", "2317", "2454", "2382", "2308",
    "2891", "2881", "2412", "2303", "2882",
]
TEST_START = "20240101"
TEST_END = "20260401"
INITIAL_CASH = 100_000_000
FEE = 0.001425
TAX = 0.003
MIN_FEE = 20
N_FEAT = 8


def load_data():
    out = {}
    for sid in V2_STOCKS:
        fp = f"RL/data/{sid}.csv"
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)
        df = pd.read_csv(fp, parse_dates=["date"], index_col="date")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        out[sid] = df.loc[TEST_START:TEST_END].dropna()
    return out


def build_memory(stock_data):
    fx = FeatureExtractor(V2_STOCKS)
    market_dfs = fx.extract_features(stock_data)
    common_dates = sorted(
        set.intersection(*[set(df.index) for df in stock_data.values()])
    )
    n_steps = len(common_dates)
    n = len(V2_STOCKS)
    price_mem = np.zeros((n_steps, n))
    feat_mem = np.zeros((n_steps, n * N_FEAT))
    for i, date in enumerate(common_dates):
        feats = []
        for j, sid in enumerate(V2_STOCKS):
            df = market_dfs.get(sid)
            if df is not None and date in df.index:
                row = df.loc[date]
                price_mem[i, j] = row["close"]
                feats.extend([
                    row["return"], row["bias_5"], row["bias_20"], row["macd_h"],
                    row["rsi_14"], row["bb_pos"], row["atr"], row["capacity_change"],
                ])
            else:
                price_mem[i, j] = 0.0
                feats.extend([0.0] * N_FEAT)
        feat_mem[i] = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
    return common_dates, price_mem, feat_mem


def get_obs(feat_mem, step, cash, inventory):
    cash_norm = np.array([cash / INITIAL_CASH], dtype=np.float32)
    inv_norm = inventory.astype(np.float32) / 1000.0
    return np.concatenate([feat_mem[step], cash_norm, inv_norm]).astype(np.float32)


def step_v2(action, prices_used, cash, inventory):
    n_trades = 0
    for i, act in enumerate(action):
        p = prices_used[i]
        if p == 0 or np.isnan(p):
            continue
        a = int(act)
        if a >= 3:  # buy
            lots = 2 if a == 4 else 1
            cost = lots * p * 1000
            fee = max(cost * FEE, MIN_FEE)
            if cash >= cost + fee:
                cash -= cost + fee
                inventory[i] += 1000 * lots
                n_trades += 1
        elif a <= 1:  # sell
            lots_to_sell = 2 if a == 0 else 1
            avail = inventory[i] // 1000
            lots_to_sell = min(lots_to_sell, avail)
            if lots_to_sell > 0:
                shares = lots_to_sell * 1000
                revenue = shares * p * (1 - TAX - FEE)
                cash += revenue
                inventory[i] -= shares
                n_trades += 1
    return cash, inventory, n_trades


def eval_one(model_path, dates, price_mem, feat_mem):
    model = PPO.load(model_path, device="cpu")
    cash = float(INITIAL_CASH)
    inventory = np.zeros(len(V2_STOCKS), dtype=int)
    asset_history = [cash]
    total_trades = 0

    n_steps = len(dates)
    for t in range(n_steps - 1):
        obs = get_obs(feat_mem, t, cash, inventory)
        action, _ = model.predict(obs, deterministic=True)
        prices_used = price_mem[t - 1] if t > 0 else price_mem[0]
        cash, inventory, trades = step_v2(action, prices_used, cash, inventory)
        total_trades += trades
        next_prices = price_mem[t + 1]
        asset = cash + float(np.sum(inventory * next_prices))
        asset_history.append(asset)

    return {
        "return": asset_history[-1] / INITIAL_CASH - 1,
        "trades": total_trades,
        "asset_history": asset_history,
        "final_cash": cash,
        "final_inventory": dict(zip(V2_STOCKS, inventory.tolist())),
    }


def ew_basket(stock_data):
    rets = []
    for df in stock_data.values():
        if len(df) > 1:
            rets.append(float(df["close"].iloc[-1] / df["close"].iloc[0] - 1))
    return float(np.mean(rets))


def max_drawdown(values):
    arr = np.asarray(values, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd = (peak - arr) / np.maximum(peak, 1.0)
    return float(dd.max())


if __name__ == "__main__":
    stock_data = load_data()
    dates, price_mem, feat_mem = build_memory(stock_data)
    ew = ew_basket(stock_data)
    print(f"period: {TEST_START}..{TEST_END}  days: {len(dates)}  stocks: {len(V2_STOCKS)}")
    print(f"EW basket return: {ew * 100:+.2f}%\n")

    results = []
    for seed in [0, 1, 2]:
        path = f"ppo_trading_agent_v2_seed{seed}"
        if not os.path.exists(f"{path}.zip"):
            print(f"  [seed{seed}] missing model, skip")
            continue
        r = eval_one(path, dates, price_mem, feat_mem)
        alpha = r["return"] - ew
        mdd = max_drawdown(r["asset_history"])
        print(
            f"v2_seed{seed}: ret={r['return'] * 100:+7.2f}%  "
            f"alpha={alpha * 100:+7.2f}%  mdd={mdd * 100:6.2f}%  trades={r['trades']}"
        )
        results.append({
            "seed": seed,
            "return": r["return"],
            "alpha": alpha,
            "max_drawdown": mdd,
            "trades": r["trades"],
            "final_inventory": r["final_inventory"],
        })

    if results:
        alphas = [r["alpha"] for r in results]
        rets = [r["return"] for r in results]
        print(f"\n{'=' * 60}")
        print(f"mean return: {np.mean(rets) * 100:+.2f}%  std: {np.std(rets) * 100:.2f}%")
        print(f"mean alpha:  {np.mean(alphas) * 100:+.2f}%  std: {np.std(alphas) * 100:.2f}%")
        print(f"+ alpha:     {sum(1 for a in alphas if a > 0)}/{len(alphas)}")
        print(f"\nxlsx claim: PPO_v2_seed1 = +93.80%  (this run seed1: {results[1]['return'] * 100 if len(results) > 1 else float('nan'):+.2f}%)")

    with open("eval_v2_results.json", "w") as f:
        json.dump(
            {
                "period": (TEST_START, TEST_END),
                "stocks": V2_STOCKS,
                "ew_return": ew,
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )
    print("saved eval_v2_results.json")
