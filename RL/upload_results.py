"""Upload PPO eval results to lab backtest API.

Runs an eval rollout, reconstructs orders + realized P&L from the action log,
writes the 3-sheet xlsx that BacktestSystem expects, then triggers uploads.
"""

import datetime
import os
import sys
from collections import deque
from typing import Any

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import RL.env  # noqa: F401
from backtest.backtest import BacktestSystem
from RL.constant import stock_ids, test_end, test_start, val_end, val_start
from RL.eval import load_data, make_env


def run_eval(model_path: str, stock_data: dict) -> dict:
    env = DummyVecEnv([make_env(stock_data)])
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
        action, _ = model.predict(obs, deterministic=True)
        actions_log.append(action[0])
        obs, _, done, info = env.step(action)
        final = info[0]
    env.close()
    return {"final": final, "actions": actions_log}


def build_price_memory(stock_data: dict) -> tuple[list, np.ndarray]:
    common_dates = sorted(
        set.intersection(*[set(df.index) for df in stock_data.values()])
    )
    price_memory = np.zeros((len(common_dates), len(stock_ids)))
    for t, d in enumerate(common_dates):
        for j, sid in enumerate(stock_ids):
            df = stock_data.get(sid)
            if df is not None and d in df.index:
                price_memory[t, j] = df.loc[d, "close"]
    return common_dates, price_memory


def build_orders(actions_log, dates, price_memory) -> list:
    """Reconstruct buy/sell orders from action log (5-level MultiDiscrete)."""
    orders = []
    for t, action in enumerate(actions_log):
        if t >= len(dates):
            break
        date_str = dates[t].strftime("%Y-%m-%d")
        for i, act in enumerate(action):
            sid = stock_ids[i]
            price = price_memory[t, i]
            if price == 0 or np.isnan(price):
                continue
            if act == 4 or act == 3:
                lots = 2 if act == 4 else 1
                orders.append(
                    {
                        "股票代碼": sid,
                        "交易日期": date_str,
                        "交易動作": "買入",
                        "成交價格": float(price),
                        "交易股數": 1000 * lots,
                    }
                )
            elif act == 1 or act == 0:
                lots = 2 if act == 0 else 1
                orders.append(
                    {
                        "股票代碼": sid,
                        "交易日期": date_str,
                        "交易動作": "賣出",
                        "成交價格": float(price),
                        "交易股數": 1000 * lots,
                    }
                )
    return orders


def build_realized(orders: list) -> list:
    """FIFO match buy/sell pairs to compute realized P&L."""
    lots_by_stock: dict = {}
    realized = []
    for o in orders:
        sid = o["股票代碼"]
        if sid not in lots_by_stock:
            lots_by_stock[sid] = deque()
        if o["交易動作"] == "買入":
            lots_by_stock[sid].append((o["成交價格"], o["交易股數"]))
        else:
            sell_shares = o["交易股數"]
            sell_price = o["成交價格"]
            while sell_shares > 0 and lots_by_stock[sid]:
                buy_price, buy_shares = lots_by_stock[sid][0]
                matched = min(buy_shares, sell_shares)
                profit = (sell_price - buy_price) * matched
                profit_rate = (
                    (profit / (buy_price * matched)) * 100 if buy_price > 0 else 0
                )
                realized.append(
                    {
                        "股票代碼": sid,
                        "交易日期": o["交易日期"],
                        "買入價格": buy_price,
                        "賣出價格": sell_price,
                        "交易股數": matched,
                        "獲利": profit,
                        "獲利率(%)": f"{profit_rate:.2f}",
                    }
                )
                if matched == buy_shares:
                    lots_by_stock[sid].popleft()
                else:
                    lots_by_stock[sid][0] = (buy_price, buy_shares - matched)
                sell_shares -= matched
    return realized


def write_xlsx(
    strategy_name: str,
    start: str,
    end: str,
    final: dict,
    orders: list,
    realized: list,
    last_prices: dict,
    fname: str = "backtest_performance.xlsx",
):
    init_cash = 100_000_000
    total_asset = final["total_asset"]
    realized_profit = sum(r["獲利"] for r in realized)
    unrealized_profit = (total_asset - init_cash) - realized_profit
    total_return = (total_asset - init_cash) / init_cash * 100

    inventory = final["inventory"]
    stock_value = sum(
        shares * last_prices.get(sid, 0.0) for sid, shares in inventory.items()
    )
    cash_balance = max(0.0, total_asset - stock_value)

    summary = {
        "策略名稱": [strategy_name],
        "起始期間": [start],
        "結束期間": [end],
        "更新時間": [datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "總收益率 (%)": [f"{total_return:.2f}%"],
        "已實現收益": [f"{realized_profit:.4f}"],
        "未實現收益": [f"{unrealized_profit:.4f}"],
        "最終現金餘額": [cash_balance],
        "最終總資產價值": [total_asset],
    }

    with pd.ExcelWriter(fname, engine="xlsxwriter") as w:
        pd.DataFrame(summary).to_excel(w, sheet_name="回測績效", index=False)
        pd.DataFrame(realized).to_excel(w, sheet_name="歷史已實現損益", index=False)
        pd.DataFrame(orders).to_excel(w, sheet_name="歷史交易委託", index=False)
    print(f"Wrote {fname}  (orders={len(orders)}, realized={len(realized)})")


if __name__ == "__main__":
    load_dotenv()
    account = os.getenv("ACCOUNT")
    password = os.getenv("PASSWORD")
    if not account or not password:
        print("Missing ACCOUNT or PASSWORD in env. Aborting.")
        sys.exit(1)

    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    version = sys.argv[3] if len(sys.argv) > 3 else "v2"

    if split == "test":
        start, end = test_start, test_end
    else:
        start, end = val_start, val_end

    strategy_name = f"PPO_{version}_seed{seed}_{split}"
    model_path = f"ppo_trading_agent_{version}_seed{seed}"

    print(f"Strategy: {strategy_name}")
    print(f"Model:    {model_path}")
    print(f"Window:   {start} .. {end}")

    stock_data = load_data(start=start, end=end)
    result = run_eval(model_path, stock_data)
    final = result["final"]
    print(
        f"Eval done. return={final['return_rate'] * 100:+.2f}%  "
        f"trades={final['total_trades']}  final={final['total_asset']:,.0f}"
    )

    dates, price_memory = build_price_memory(stock_data)
    last_prices = {sid: float(price_memory[-1, j]) for j, sid in enumerate(stock_ids)}

    orders = build_orders(result["actions"], dates, price_memory)
    realized = build_realized(orders)
    write_xlsx(strategy_name, start, end, final, orders, realized, last_prices)

    bt = BacktestSystem(account, password)
    print("Uploading summary...")
    bt.upload_performance_to_web()
    print("Uploading detail (orders)...")
    bt.upload_performance_detail_to_web()
    print("Uploading record (realized)...")
    bt.upload_performance_record_to_web()
    print("All uploaded.")
