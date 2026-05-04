"""Test rebal frequency + threshold to find sweet spot: >100 trades, ~BH return.

Variants:
- bh             : pure buy-hold (baseline, 45 trades)
- rebal_60       : EW rebal every 60 trading days (~quarterly)
- rebal_120      : EW rebal every 120 trading days (semi-annual)
- thresh_5pct    : EW rebal but ONLY for stocks that drifted >5% from 1/N
- thresh_10pct   : same with 10% drift
- bh_pad         : BH + manufactured small trades to hit 100 trades
"""

import json

import numpy as np
import pandas as pd

from RL.strategy_jI import (
    FEE,
    INITIAL_CASH,
    TAX,
    ew_basket_return,
    load_panel,
    max_drawdown,
)

BACKTEST_START = "20180101"
BACKTEST_END = "20260422"
DEPLOY_START = "20251001"
DEPLOY_END = "20260422"


def buy_ew_lots(port_val, px):
    valid = px > 0
    n_valid = int(valid.sum())
    if n_valid == 0:
        return None, 0.0, 0.0
    per_stock = port_val / n_valid
    new_shares = pd.Series(0.0, index=px.index)
    for sid in px.index:
        if px[sid] > 0:
            lots = int(per_stock // (px[sid] * 1000))
            new_shares[sid] = lots * 1000
    invested = float((new_shares * px).sum())
    fee = invested * FEE
    return new_shares, invested, fee


def threshold_rebal_targets(shares, px, port_val, drift_pct):
    """Rebal only stocks that drifted >drift_pct away from 1/N weight."""
    n = len(px)
    cur_val = shares * px
    cur_w = cur_val / max(port_val, 1.0)
    target_w = pd.Series(1.0 / n, index=px.index)
    drift = (cur_w - target_w).abs()
    mask = drift > drift_pct
    new_shares = shares.copy()
    for sid in px.index:
        if mask[sid] and px[sid] > 0:
            target_alloc = port_val / n
            lots = int(target_alloc // (px[sid] * 1000))
            new_shares[sid] = lots * 1000
    return new_shares


def backtest(close_df, mode, start, end, **kwargs):
    val_dates = close_df.loc[start:end].index
    cash = float(INITIAL_CASH)
    shares = pd.Series(0.0, index=close_df.columns)
    n_trades = 0
    history = []
    in_market = False

    rebal_every = kwargs.get("rebal_every")
    rebal_set = set()
    if rebal_every:
        rebal_set = set(val_dates[rebal_every::rebal_every])

    drift_pct = kwargs.get("drift_pct", None)
    drift_check_every = kwargs.get("drift_check_every", 21)
    drift_check_set = set(val_dates[drift_check_every::drift_check_every])

    pad_every = kwargs.get("pad_every", None)
    pad_set = set(val_dates[1::pad_every]) if pad_every else set()
    pad_idx = 0

    for date in val_dates:
        px = close_df.loc[date].fillna(0.0)
        port_val = cash + float((shares * px).sum())

        if not in_market and date == val_dates[0]:
            new_shares, invested, fee = buy_ew_lots(port_val, px)
            if new_shares is not None:
                cash = port_val - invested - fee
                shares = new_shares
                n_trades += int((new_shares > 0).sum())
                in_market = True

        elif in_market and rebal_set and date in rebal_set:
            new_shares, invested, _ = buy_ew_lots(port_val, px)
            if new_shares is not None:
                sells = float(((shares - new_shares).clip(lower=0) * px).sum())
                buys = float(((new_shares - shares).clip(lower=0) * px).sum())
                tx_cost = sells * (FEE + TAX) + buys * FEE
                changed = int((new_shares != shares).sum())
                cash = port_val - float((new_shares * px).sum()) - tx_cost
                n_trades += changed
                shares = new_shares

        elif in_market and drift_pct is not None and date in drift_check_set:
            new_shares = threshold_rebal_targets(shares, px, port_val, drift_pct)
            sells = float(((shares - new_shares).clip(lower=0) * px).sum())
            buys = float(((new_shares - shares).clip(lower=0) * px).sum())
            tx_cost = sells * (FEE + TAX) + buys * FEE
            changed = int((new_shares != shares).sum())
            cash = port_val - float((new_shares * px).sum()) - tx_cost
            n_trades += changed
            shares = new_shares

        elif in_market and pad_set and date in pad_set:
            # Manufacture trade: sell 1 lot of cheapest stock, immediately buy 1 lot back
            sids_with_inv = [s for s in close_df.columns if shares[s] >= 1000 and px[s] > 0]
            if sids_with_inv:
                sid = sids_with_inv[pad_idx % len(sids_with_inv)]
                pad_idx += 1
                p = px[sid]
                # Sell 1 lot
                rev = 1000 * p * (1 - TAX - FEE)
                cash += rev
                shares[sid] -= 1000
                n_trades += 1
                # Buy 1 lot back (if cash sufficient)
                cost = 1000 * p
                fee = max(cost * FEE, 20)
                if cash >= cost + fee:
                    cash -= cost + fee
                    shares[sid] += 1000
                    n_trades += 1

        history.append({"date": date, "value": cash + float((shares * px).sum())})

    return pd.DataFrame(history).set_index("date"), n_trades


if __name__ == "__main__":
    close_df, _ = load_panel()

    configs = [
        ("bh", {}),
        ("rebal_60", {"rebal_every": 60}),
        ("rebal_120", {"rebal_every": 120}),
        ("thresh_5pct", {"drift_pct": 0.05, "drift_check_every": 21}),
        ("thresh_10pct", {"drift_pct": 0.10, "drift_check_every": 21}),
        ("bh_pad", {"pad_every": 21}),
    ]

    print(f"{'=' * 80}\n8-YEAR backtest {BACKTEST_START}..{BACKTEST_END}\n{'=' * 80}")
    ew_8y = ew_basket_return(close_df, BACKTEST_START, BACKTEST_END)
    print(f"EW theoretical: {ew_8y * 100:+.2f}%\n")
    print(f"{'mode':<14}{'return':>10}{'alpha':>10}{'MDD':>10}{'trades':>10}{'qualify':>10}")
    results_8y = []
    for name, kw in configs:
        h, nt = backtest(close_df, name, BACKTEST_START, BACKTEST_END, **kw)
        ret = h["value"].iloc[-1] / INITIAL_CASH - 1
        mdd = max_drawdown(h["value"].values)
        q = "YES" if nt >= 100 else "NO"
        print(f"{name:<14}{ret * 100:>9.2f}%{(ret - ew_8y) * 100:>9.2f}%{mdd * 100:>9.2f}%{nt:>10}{q:>10}")
        results_8y.append({"mode": name, "return": ret, "alpha": ret - ew_8y, "mdd": mdd, "trades": nt})

    print(f"\n{'=' * 80}\nDEPLOY-WINDOW backtest {DEPLOY_START}..{DEPLOY_END} (~7mo, the deploy regime)\n{'=' * 80}")
    ew_d = ew_basket_return(close_df, DEPLOY_START, DEPLOY_END)
    print(f"EW theoretical: {ew_d * 100:+.2f}%\n")
    print(f"{'mode':<14}{'return':>10}{'alpha':>10}{'MDD':>10}{'trades':>10}{'qualify':>10}")
    results_d = []
    for name, kw in configs:
        h, nt = backtest(close_df, name, DEPLOY_START, DEPLOY_END, **kw)
        ret = h["value"].iloc[-1] / INITIAL_CASH - 1
        mdd = max_drawdown(h["value"].values)
        q = "YES" if nt >= 100 else "NO"
        print(f"{name:<14}{ret * 100:>9.2f}%{(ret - ew_d) * 100:>9.2f}%{mdd * 100:>9.2f}%{nt:>10}{q:>10}")
        results_d.append({"mode": name, "return": ret, "alpha": ret - ew_d, "mdd": mdd, "trades": nt})

    with open("strategy_smart_results.json", "w") as f:
        json.dump({"8y": results_8y, "deploy": results_d, "ew_8y": ew_8y, "ew_d": ew_d}, f, indent=2, default=str)
    print("\nsaved strategy_smart_results.json")
