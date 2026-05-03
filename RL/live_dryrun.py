"""Phase 0 dry-run: predict actions on live data, print only.

No orders submitted. Validates feature pipeline + model loading + data fetch
parity before moving to paper-trading or live execution.
"""

import datetime
import os
import sys

import gymnasium as gym
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import RL.env  # noqa: F401
from RL.constant import stock_ids
from RL.feature import FeatureExtractor
from stock_api import Get_User_Stocks, get_taiwan_stock_data

ACTION_LABELS = {
    0: "SELL_100%",
    1: "SELL_50%",
    2: "SELL_25%",
    3: "HOLD",
    4: "BUY_5%",
    5: "BUY_15%",
    6: "BUY_30%",
}


def fetch_history(start_date: str, end_date: str) -> dict:
    """Fetch OHLCV per stock, return dict[sid] -> DataFrame indexed by date."""
    out = {}
    for sid in stock_ids:
        try:
            df = get_taiwan_stock_data(sid, start_date, end_date)
        except Exception as e:
            print(f"[{sid}] fetch failed: {e}")
            continue
        if df is None or df.empty:
            print(f"[{sid}] empty result")
            continue
        df = df.sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        out[sid] = df
        print(f"[{sid}] {len(df)} rows  {df.index[0].date()} .. {df.index[-1].date()}")
    return out


def build_obs(
    stock_data: dict,
    inventory: dict,
    cash_balance: float,
    initial_cash: float = 100_000_000,
) -> tuple[np.ndarray, dict]:
    """Mimic TradingEnv._get_obs on live data; return obs + last-day price map."""
    fx = FeatureExtractor(stock_ids)
    market_dfs = fx.extract_features(stock_data)

    common_dates = sorted(
        set.intersection(*[set(df.index) for df in stock_data.values() if not df.empty])
    )
    if not common_dates:
        raise RuntimeError("No common trading date across all stocks")
    last_date = common_dates[-1]
    print(f"\nUsing observation date: {last_date.date()}")

    feats = []
    last_prices = {}
    for sid in stock_ids:
        df = market_dfs.get(sid)
        if df is None or last_date not in df.index or len(df) < 30:
            feats.extend([0.0] * 12)
            last_prices[sid] = 0.0
            print(f"  [{sid}] insufficient data, zero features")
            continue
        row = df.loc[last_date]
        feats.extend(
            [
                row["return"],
                row["bias_5"],
                row["bias_20"],
                row["macd_h"],
                row["rsi_14"],
                row["bb_pos"],
                row["atr"],
                row["capacity_change"],
                row.get("return_rank", 0.5),
                row.get("rsi_14_rank", 0.5),
                row.get("bias_20_rank", 0.5),
                row.get("capacity_change_rank", 0.5),
            ]
        )
        last_prices[sid] = float(row["close"])

    feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
    cash_norm = np.array([cash_balance / initial_cash], dtype=np.float32)
    inv_arr = np.array(
        [inventory.get(sid, 0) / 1000.0 for sid in stock_ids], dtype=np.float32
    )
    obs = np.concatenate([feats, cash_norm, inv_arr]).astype(np.float32)
    return obs, last_prices


def parse_inventory(user_stocks_resp) -> dict:
    """Convert Get_User_Stocks response to {stock_id: total_shares}."""
    inv = {sid: 0 for sid in stock_ids}
    if not user_stocks_resp:
        return inv
    if isinstance(user_stocks_resp, dict):
        items = user_stocks_resp.values() if "stocks" not in user_stocks_resp else user_stocks_resp.get("stocks", [])
    elif isinstance(user_stocks_resp, list):
        items = user_stocks_resp
    else:
        items = []
    for item in items:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("stock_code") or item.get("stock_id") or "")
        shares = int(item.get("shares") or item.get("stock_shares") or 0)
        if sid in inv:
            inv[sid] += shares
    return inv


if __name__ == "__main__":
    load_dotenv()
    account = os.getenv("ACCOUNT")
    password = os.getenv("PASSWORD")
    if not account or not password:
        print("Missing ACCOUNT/PASSWORD in env. Aborting.")
        sys.exit(1)

    model_path = sys.argv[1] if len(sys.argv) > 1 else "ppo_deploy_final"
    norm_path = sys.argv[2] if len(sys.argv) > 2 else "vec_normalize_deploy_final.pkl"
    today = datetime.date.today()
    start = (today - datetime.timedelta(days=120)).strftime("%Y%m%d")
    end = today.strftime("%Y%m%d")

    print(f"=== Live dry run ===")
    print(f"Model:  {model_path}")
    print(f"Window: {start} .. {end}")
    print(f"Stocks: {stock_ids}\n")

    print("Fetching history...")
    stock_data = fetch_history(start, end)
    if not stock_data:
        print("No data fetched. Aborting.")
        sys.exit(1)

    print("\nFetching inventory...")
    raw_inv = Get_User_Stocks(account, password)
    print(f"Raw response: {raw_inv}")
    inventory = parse_inventory(raw_inv)
    print(f"Parsed inventory: {inventory}")
    held_value_proxy = sum(inventory.values())
    cash_balance = max(100_000_000 - held_value_proxy * 50, 0)
    print(f"Cash proxy (rough): {cash_balance:,.0f}")

    obs, last_prices = build_obs(stock_data, inventory, cash_balance)
    if not np.isfinite(obs).all():
        print("ERROR: obs contains non-finite values")
        sys.exit(1)
    print(f"\nObs shape: {obs.shape}, finite: {np.isfinite(obs).all()}")

    if os.path.exists(norm_path):
        print(f"\nLoading normalize stats from {norm_path}...")
        dummy = DummyVecEnv(
            [lambda: gym.make("TradingEnv-v0", stock_ids=stock_ids, stock_data=stock_data, eval_mode=True)]
        )
        venv = VecNormalize.load(norm_path, dummy)
        venv.training = False
        venv.norm_reward = False
        obs = venv.normalize_obs(obs.reshape(1, -1)).flatten().astype(np.float32)
        print(f"Applied normalize stats. Obs range: [{obs.min():.2f}, {obs.max():.2f}]")
    else:
        print(f"\nWARN: {norm_path} not found, predicting on raw obs (likely wrong if model trained with VecNormalize)")

    print(f"\nLoading model {model_path}...")
    model = PPO.load(model_path, device="cpu")
    action, _ = model.predict(obs, deterministic=True)
    print(f"Action raw: {action}")

    print("\n=== Decisions (DRY RUN, no orders sent) ===")
    print(f"{'sid':<6}{'price':>10}{'inv':>10}{'action':>10}")
    for i, act in enumerate(action):
        sid = stock_ids[i]
        price = last_prices.get(sid, 0.0)
        held = inventory.get(sid, 0)
        label = ACTION_LABELS.get(int(act), f"?{act}")
        print(f"{sid:<6}{price:>10.2f}{held:>10}{label:>10}")

    print("\nDRY RUN COMPLETE — no orders submitted.")
