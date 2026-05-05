"""Live RL deploy: ppo_deploy_final agent + drawdown safety overlay.

Each run (daily during TW market hours):
  1. Fetch latest 120d data per stock (TWSE/TPEX live API)
  2. Build 12-feat × 46-stock + cash + inv obs (matches v9/v10 env)
  3. Apply VecNormalize stats from training
  4. PPO predict (deterministic) -> MultiDiscrete[7]^46 actions
  5. Drawdown guard: if portfolio < (peak × (1 - DD_STOP)) -> liquidate + halt
  6. Else translate actions to Buy_Stock / Sell_Stock submissions
  7. Update local ledger (cash, inventory, peak) + persist state

State persists in deploy_state.json. GitHub Actions commits it back per run.

Modes:
  --paper  : print actions, do not submit
  --live   : submit via NCKU sim_stock API (needs ACCOUNT/PASSWORD env)
  --status : show ledger + drawdown only
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Literal

import gymnasium as gym
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import RL.env  # noqa: F401  registers TradingEnv-v0
from RL.constant import stock_ids
from RL.feature import FeatureExtractor
from stock_api import Buy_Stock, Get_User_Stocks, Sell_Stock, get_taiwan_stock_data

STATE_FILE = Path("deploy_state.json")
INITIAL_CASH = 100_000_000
DRAWDOWN_STOP = 0.10
HISTORY_DAYS = 180  # generous so we always cover 60d+ valid features

ACTION_LABELS = {
    0: "SELL_100", 1: "SELL_50", 2: "SELL_25", 3: "HOLD",
    4: "BUY_5", 5: "BUY_15", 6: "BUY_30",
}
BUY_PCT = {4: 0.05, 5: 0.15, 6: 0.30}
SELL_PCT = {0: 1.0, 1: 0.5, 2: 0.25}
FEE = 0.001425
TAX = 0.003
MIN_FEE = 20


def round_to_tick(price: float) -> float:
    """TW limit-order tick rules."""
    if price < 10:
        tick = 0.01
    elif price < 50:
        tick = 0.05
    elif price < 100:
        tick = 0.1
    elif price < 500:
        tick = 0.5
    elif price < 1000:
        tick = 1.0
    else:
        tick = 5.0
    return round(round(price / tick) * tick, 2)


def submit_price(sid: str, stock_data: dict, side: Literal["BUY", "SELL"]) -> float:
    """Pick limit price inside today's [low, high] band so order can fill.

    Returns 0 if today's bar unavailable.
    """
    df = stock_data.get(sid)
    if df is None or df.empty:
        return 0.0
    row = df.iloc[-1]
    open_ = float(row.get("open", 0) or 0)
    close = float(row.get("close", 0) or 0)
    if open_ <= 0 or close <= 0:
        return 0.0
    price = max(open_, close) if side == "BUY" else min(open_, close)
    return round_to_tick(price)


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "day_count": 0,
        "initial_cash": INITIAL_CASH,
        "cash_balance": INITIAL_CASH,
        "inventory": {sid: 0 for sid in stock_ids},
        "peak_value": INITIAL_CASH,
        "halted": False,
        "history": [],
    }


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2, default=str))


DATA_DIR = Path("RL/data")
CSV_STALE_DAYS = 7  # if CSV last_date older than this, refresh from API


def load_csv(sid: str) -> pd.DataFrame | None:
    """Return DataFrame indexed by date, or None if missing."""
    fp = DATA_DIR / f"{sid}.csv"
    if not fp.exists():
        return None
    df = pd.read_csv(fp, parse_dates=["date"], index_col="date")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def save_csv(sid: str, df: pd.DataFrame):
    """Persist refreshed data back to RL/data/<sid>.csv."""
    fp = DATA_DIR / f"{sid}.csv"
    out = df.reset_index().rename(columns={"index": "date"})
    out.to_csv(fp, index=False)


def fetch_history(days: int = HISTORY_DAYS, force_refresh: bool = False) -> dict:
    """Load price history. Prefer cached CSVs, refresh from TWSE if stale, write back."""
    today = pd.Timestamp(datetime.date.today())
    start_date = today - pd.Timedelta(days=days)
    out = {}
    n_from_csv, n_refreshed, n_written = 0, 0, 0
    for sid in stock_ids:
        df = load_csv(sid)
        need_refresh = force_refresh or df is None or df.index.max() < today - pd.Timedelta(days=CSV_STALE_DAYS)
        if need_refresh:
            try:
                # Only fetch the gap (last_date+1 → today), not full history
                fetch_start = (df.index.max() + pd.Timedelta(days=1)) if df is not None else start_date
                fresh = get_taiwan_stock_data(
                    sid, fetch_start.strftime("%Y%m%d"), today.strftime("%Y%m%d")
                )
                if fresh is not None and not fresh.empty:
                    fresh = fresh.sort_values("date").reset_index(drop=True)
                    fresh["date"] = pd.to_datetime(fresh["date"])
                    fresh = fresh.set_index("date")
                    if df is not None:
                        df = pd.concat([df, fresh])
                        df = df[~df.index.duplicated(keep="last")].sort_index()
                    else:
                        df = fresh
                    save_csv(sid, df)
                    n_written += 1
                n_refreshed += 1
            except Exception as e:
                print(f"  [{sid}] fetch err: {e}")
        else:
            n_from_csv += 1
        if df is not None:
            out[sid] = df.loc[df.index >= start_date]
    print(f"  data sources: csv={n_from_csv}, refreshed={n_refreshed}, csv_written={n_written}")
    return out


def build_obs(stock_data: dict, state: dict):
    fx = FeatureExtractor(stock_ids)
    market_dfs = fx.extract_features(stock_data)
    common_dates = sorted(
        set.intersection(*[set(df.index) for df in stock_data.values() if not df.empty])
    )
    if not common_dates:
        raise RuntimeError("no common date across all stocks")
    last_date = common_dates[-1]

    feats, last_prices = [], {}
    for sid in stock_ids:
        df = market_dfs.get(sid)
        if df is None or last_date not in df.index or len(df) < 30:
            feats.extend([0.0] * 12)
            last_prices[sid] = 0.0
            continue
        row = df.loc[last_date]
        feats.extend([
            row["return"], row["bias_5"], row["bias_20"], row["macd_h"],
            row["rsi_14"], row["bb_pos"], row["atr"], row["capacity_change"],
            row.get("return_rank", 0.5), row.get("rsi_14_rank", 0.5),
            row.get("bias_20_rank", 0.5), row.get("capacity_change_rank", 0.5),
        ])
        last_prices[sid] = float(row["close"])

    feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
    cash_norm = np.array([state["cash_balance"] / INITIAL_CASH], dtype=np.float32)
    inv_arr = np.array(
        [state["inventory"].get(sid, 0) / 1000.0 for sid in stock_ids], dtype=np.float32
    )
    obs = np.concatenate([feats, cash_norm, inv_arr]).astype(np.float32)
    return obs, last_prices, last_date


def normalize_obs(obs_raw: np.ndarray, norm_path: str, stock_data: dict) -> np.ndarray:
    dummy = DummyVecEnv([
        lambda: gym.make("TradingEnv-v0", stock_ids=stock_ids, stock_data=stock_data, eval_mode=True)
    ])
    venv = VecNormalize.load(norm_path, dummy)
    venv.training = False
    venv.norm_reward = False
    return venv.normalize_obs(obs_raw.reshape(1, -1)).flatten().astype(np.float32)


def execute_actions(actions, stock_data, state, account, password, live):
    log = []
    # Pass 1: sells (free up cash first)
    for i, sid in enumerate(stock_ids):
        a = int(actions[i])
        if a not in SELL_PCT:
            continue
        p = submit_price(sid, stock_data, "SELL")
        if p <= 0:
            continue
        held_lots = state["inventory"].get(sid, 0) // 1000
        sell_lots = int(held_lots * SELL_PCT[a])
        if sell_lots <= 0:
            continue
        shares = sell_lots * 1000
        gross = shares * p
        cost = gross * (TAX + FEE)
        if live and account:
            try:
                resp = Sell_Stock(account, password, sid, sell_lots, p)
            except Exception as e:
                resp = f"ERR:{e}"
        else:
            resp = "PAPER"
        log.append({"sid": sid, "action": "SELL", "label": ACTION_LABELS[a], "lots": sell_lots, "shares": shares, "price": p, "resp": str(resp)})
        state["cash_balance"] += gross - cost
        state["inventory"][sid] -= shares

    # Pass 2: buys
    for i, sid in enumerate(stock_ids):
        a = int(actions[i])
        if a not in BUY_PCT:
            continue
        p = submit_price(sid, stock_data, "BUY")
        if p <= 0:
            continue
        budget = state["cash_balance"] * BUY_PCT[a]
        lots = int(budget // (p * 1000))
        if lots <= 0:
            continue
        shares = lots * 1000
        gross = shares * p
        fee = max(gross * FEE, MIN_FEE)
        if state["cash_balance"] < gross + fee:
            continue
        if live and account:
            try:
                resp = Buy_Stock(account, password, sid, lots, p)
            except Exception as e:
                resp = f"ERR:{e}"
        else:
            resp = "PAPER"
        log.append({"sid": sid, "action": "BUY", "label": ACTION_LABELS[a], "lots": lots, "shares": shares, "price": p, "resp": str(resp)})
        state["cash_balance"] -= gross + fee
        state["inventory"][sid] = state["inventory"].get(sid, 0) + shares
    return log


def liquidate_all(stock_data, state, account, password, live):
    log = []
    for sid in stock_ids:
        held = state["inventory"].get(sid, 0)
        if held <= 0:
            continue
        p = submit_price(sid, stock_data, "SELL")
        if p <= 0:
            continue
        gross = held * p
        cost = gross * (TAX + FEE)
        held_lots = held // 1000
        if live and account:
            try:
                resp = Sell_Stock(account, password, sid, held_lots, p)
            except Exception as e:
                resp = f"ERR:{e}"
        else:
            resp = "PAPER"
        log.append({"sid": sid, "action": "LIQUIDATE", "lots": held_lots, "shares": held, "price": p, "resp": str(resp)})
        state["cash_balance"] += gross - cost
        state["inventory"][sid] = 0
    return log


def portfolio_value(prices, state) -> float:
    inv_val = sum(state["inventory"].get(sid, 0) * prices.get(sid, 0) for sid in stock_ids)
    return state["cash_balance"] + inv_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/ppo_deploy_final")
    parser.add_argument("--norm", default="models/vec_normalize_deploy_final.pkl")
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--refresh-data", action="store_true",
                        help="force-refresh all CSVs from TWSE before predicting")
    args = parser.parse_args()

    state = load_state()

    if args.status:
        print(f"day_count   : {state['day_count']}")
        print(f"cash        : {state['cash_balance']:,.0f}")
        print(f"peak        : {state['peak_value']:,.0f}")
        print(f"halted      : {state['halted']}")
        n_held = sum(1 for v in state["inventory"].values() if v > 0)
        print(f"holdings    : {n_held}/{len(stock_ids)} stocks")
        sys.exit(0)

    print(f"=== Deploy RL day {state['day_count'] + 1} ({datetime.date.today()}) ===")
    print(f"model: {args.model}  mode: {'LIVE' if args.live else 'PAPER'}")

    if state["halted"]:
        print("STATE: halted (drawdown stop hit prior). no actions.")
        save_state(state)
        sys.exit(0)

    print("\nloading data (csv preferred)...")
    stock_data = fetch_history(force_refresh=args.refresh_data)
    if len(stock_data) < 30:
        print(f"ERROR: only {len(stock_data)} stocks fetched, abort.")
        sys.exit(1)
    print(f"  fetched {len(stock_data)}/{len(stock_ids)} stocks")

    obs_raw, prices, obs_date = build_obs(stock_data, state)
    print(f"obs date: {obs_date.date()}, {sum(1 for p in prices.values() if p > 0)} prices")

    port_val = portfolio_value(prices, state)
    state["peak_value"] = max(state["peak_value"], port_val)
    drawdown = (state["peak_value"] - port_val) / max(state["peak_value"], 1)
    pnl = (port_val / state["initial_cash"] - 1) * 100
    print(f"portfolio   : {port_val:,.0f}  PnL: {pnl:+.2f}%  drawdown: {drawdown * 100:.2f}%")

    load_dotenv()
    account = os.getenv("ACCOUNT") if args.live else None
    password = os.getenv("PASSWORD") if args.live else None
    if args.live and (not account or not password):
        print("ERROR: ACCOUNT/PASSWORD missing for --live")
        sys.exit(1)

    if drawdown > DRAWDOWN_STOP:
        print(f"\n!!! DRAWDOWN TRIGGER ({drawdown * 100:.2f}% > {DRAWDOWN_STOP * 100:.0f}%) — LIQUIDATING !!!")
        log = liquidate_all(stock_data, state, account, password, args.live)
        for item in log:
            print(f"  LIQUIDATE {item['sid']:<6} x{item['shares']:>7}  resp={item['resp']}")
        state["halted"] = True
        state["day_count"] += 1
        state["history"].append({"date": str(obs_date.date()), "action": "halt", "port_val": port_val, "log": log})
        save_state(state)
        print(f"\nstate saved: HALTED at day {state['day_count']}")
        sys.exit(0)

    print("\nnormalizing obs + predicting...")
    obs = normalize_obs(obs_raw, args.norm, stock_data)
    model = PPO.load(args.model, device="cpu")
    actions, _ = model.predict(obs, deterministic=True)

    log = execute_actions(actions, stock_data, state, account, password, args.live)
    print(f"\n{len(log)} actions ({'LIVE' if args.live else 'PAPER'}):")
    print(f"  {'sid':<6}{'action':<6}{'label':<10}{'shares':>8}{'price':>10}{'resp':>20}")
    for item in log:
        print(f"  {item['sid']:<6}{item['action']:<6}{item['label']:<10}{item['shares']:>8}{item['price']:>10.2f}{str(item['resp'])[:20]:>20}")

    state["day_count"] += 1
    new_port = portfolio_value(prices, state)
    state["history"].append({
        "date": str(obs_date.date()),
        "actions": log,
        "port_val_before": port_val,
        "port_val_after": new_port,
        "drawdown": drawdown,
    })
    save_state(state)
    print(f"\nstate saved: day {state['day_count']}, cash {state['cash_balance']:,.0f}, port {new_port:,.0f}")
