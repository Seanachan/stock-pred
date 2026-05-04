"""Deploy 1-month plan: BH-EW (46 stocks) + daily pad trades to satisfy >100 rule.

Day 1     : buy 46 stocks equal weight (46 trades)
Day 2-21  : 3 pad pairs each day (sell 1 lot + buy 1 lot back, 6 trades/day)
Total     : 46 + 20 days × 6 = 166 trades over 21 trading days

Modes:
  --paper : print plan only (default)
  --live  : submit via NCKU sim_stock API (Buy_Stock / Sell_Stock)
  --day N : show actions for day N (1..21). Without --day prints full plan.

Cost estimate:
  Initial buys : ~0.14% (fee 0.1425% on full notional)
  Pad pairs    : 60 pairs × 1 lot × ~100K × 0.585% (sell tax+fee + buy fee)
                 = ~35K cost  (~0.035% on 100M base)
  Total drag   : ~0.18%

Expected month return: ~+2.5% (BH-EW 7mo historical was +17.75%, ~2.5%/mo)
"""

import argparse
import datetime
import json
import os
import sys
from typing import Dict, List

import pandas as pd

from RL.constant import stock_ids

INITIAL_CASH = 100_000_000
PAD_PAIRS_PER_DAY = 3  # 6 trades/day -> 20 days × 6 = 120 + 46 initial = 166
N_TRADING_DAYS = 21
DATA_DIR = "RL/data"


def load_latest_prices() -> Dict[str, float]:
    """Return latest close price per stock from CSVs."""
    out = {}
    for sid in stock_ids:
        fp = f"{DATA_DIR}/{sid}.csv"
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp, parse_dates=["date"], index_col="date")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        if len(df) > 0:
            out[sid] = float(df["close"].iloc[-1])
    return out


def build_initial_plan(prices: Dict[str, float], port_val: float = INITIAL_CASH) -> List[dict]:
    """Day 1: equal-weight buy across valid stocks. Lot rounded down."""
    valid = {sid: p for sid, p in prices.items() if p > 0}
    n = len(valid)
    per_stock = port_val / n
    plan = []
    total_cost = 0.0
    for sid, p in valid.items():
        lots = int(per_stock // (p * 1000))
        if lots > 0:
            shares = lots * 1000
            cost = shares * p
            plan.append({
                "action": "BUY", "stock_id": sid, "shares": shares,
                "price_ref": p, "notional": cost,
            })
            total_cost += cost
    return plan, total_cost


def build_pad_plan(holdings: Dict[str, int], prices: Dict[str, float], n_pairs: int) -> List[dict]:
    """Pick cheapest stocks with >=1 lot, generate sell-buyback pairs."""
    candidates = sorted(
        [(sid, prices[sid]) for sid in holdings if holdings[sid] >= 1000 and sid in prices and prices[sid] > 0],
        key=lambda x: x[1],
    )
    plan = []
    for i in range(n_pairs):
        if not candidates:
            break
        sid, p = candidates[i % len(candidates)]
        plan.append({"action": "SELL", "stock_id": sid, "shares": 1000, "price_ref": p, "notional": 1000 * p})
        plan.append({"action": "BUY", "stock_id": sid, "shares": 1000, "price_ref": p, "notional": 1000 * p})
    return plan


def submit_live(plan: List[dict], account: str, password: str, dry_run: bool = False):
    """Execute plan via NCKU API."""
    from stock_api import Buy_Stock, Sell_Stock
    for i, item in enumerate(plan):
        sid = item["stock_id"]
        shares = item["shares"]
        if item["action"] == "BUY":
            print(f"  [{i + 1}/{len(plan)}] BUY  {sid} x {shares}", end=" ")
            resp = "DRY" if dry_run else Buy_Stock(account, password, sid, shares)
        else:
            print(f"  [{i + 1}/{len(plan)}] SELL {sid} x {shares}", end=" ")
            resp = "DRY" if dry_run else Sell_Stock(account, password, sid, shares)
        print(f"-> {resp}")


def print_plan(plan: List[dict], label: str):
    print(f"\n{label}: {len(plan)} actions")
    print(f"  {'#':<4}{'action':<6}{'stock':<8}{'shares':>8}{'price':>10}{'notional':>14}")
    for i, item in enumerate(plan):
        print(
            f"  {i + 1:<4}{item['action']:<6}{item['stock_id']:<8}"
            f"{item['shares']:>8}{item['price_ref']:>10.2f}{item['notional']:>14,.0f}"
        )
    total_notional = sum(item["notional"] for item in plan)
    print(f"  TOTAL notional: {total_notional:,.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper", action="store_true", default=True, help="print only (default)")
    parser.add_argument("--live", action="store_true", help="submit via NCKU API")
    parser.add_argument("--day", type=int, default=None, help="day N (1-21); omit for full plan")
    args = parser.parse_args()

    prices = load_latest_prices()
    print(f"latest prices loaded: {len(prices)}/{len(stock_ids)} stocks")
    print(f"price range: {min(prices.values()):.1f} .. {max(prices.values()):.1f}")

    initial_plan, initial_notional = build_initial_plan(prices)
    print(f"\nDay 1 plan: {len(initial_plan)} buys, total notional {initial_notional:,.0f}")
    print(f"  cash leftover (lot rounding): {INITIAL_CASH - initial_notional:,.0f}")

    initial_holdings = {item["stock_id"]: item["shares"] for item in initial_plan}
    pad_plan = build_pad_plan(initial_holdings, prices, PAD_PAIRS_PER_DAY)

    total_trades_per_month = len(initial_plan) + (N_TRADING_DAYS - 1) * len(pad_plan)
    print(f"\nMonth schedule:")
    print(f"  Day 1     : {len(initial_plan)} initial buys")
    print(f"  Day 2-{N_TRADING_DAYS}  : {len(pad_plan)} pad actions/day × {N_TRADING_DAYS - 1} days = {(N_TRADING_DAYS - 1) * len(pad_plan)}")
    print(f"  TOTAL     : {total_trades_per_month} trades  (rule: >=100 → {'OK' if total_trades_per_month >= 100 else 'FAIL'})")

    pad_pair_cost = 0.00585 * 1000 * sum(item["price_ref"] for item in pad_plan if item["action"] == "SELL") / max(PAD_PAIRS_PER_DAY, 1)
    print(f"\nCost estimate:")
    print(f"  Initial fee : {initial_notional * 0.001425:,.0f}  ({initial_notional * 0.001425 / INITIAL_CASH * 100:.3f}% of cap)")
    pad_total_cost = (N_TRADING_DAYS - 1) * pad_pair_cost * PAD_PAIRS_PER_DAY
    print(f"  Pad trades  : {pad_total_cost:,.0f}  ({pad_total_cost / INITIAL_CASH * 100:.3f}% of cap)")
    print(f"  Total drag  : ~{(initial_notional * 0.001425 + pad_total_cost) / INITIAL_CASH * 100:.3f}%")

    if args.day is not None:
        if args.day == 1:
            plan = initial_plan
            label = f"DAY 1 ACTIONS"
        elif 2 <= args.day <= N_TRADING_DAYS:
            plan = pad_plan
            label = f"DAY {args.day} PAD ACTIONS"
        else:
            print(f"--day must be 1..{N_TRADING_DAYS}")
            sys.exit(1)
        print_plan(plan, label)

        if args.live:
            from dotenv import load_dotenv
            load_dotenv()
            account = os.getenv("ACCOUNT")
            password = os.getenv("PASSWORD")
            if not account or not password:
                print("\nMissing ACCOUNT/PASSWORD in env. Aborting --live.")
                sys.exit(1)
            print(f"\n--live: submitting via NCKU API as {account}")
            confirm = input("Confirm submit? [yes/NO]: ").strip().lower()
            if confirm == "yes":
                submit_live(plan, account, password)
            else:
                print("Cancelled.")
    else:
        print_plan(initial_plan, "DAY 1 INITIAL BUYS")
        print_plan(pad_plan, f"PAD ACTIONS (repeat each day 2..{N_TRADING_DAYS})")
        print(f"\nUse --day N --live to submit day N actions.")

    plan_out = {
        "date_generated": datetime.date.today().isoformat(),
        "n_trading_days": N_TRADING_DAYS,
        "pad_pairs_per_day": PAD_PAIRS_PER_DAY,
        "total_trades": total_trades_per_month,
        "initial_plan": initial_plan,
        "pad_plan_template": pad_plan,
    }
    with open("deploy_plan.json", "w") as f:
        json.dump(plan_out, f, indent=2, default=str)
    print(f"\nsaved deploy_plan.json")
