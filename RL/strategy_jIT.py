"""J+I+T: J+I (regime gate × momentum weight) + Turbulence index hard-stop.

Paper: Yang et al, AI4Finance ensemble (turbulence = Mahalanobis dist of returns).
When daily turbulence > threshold (99%-ile of pre-val history): liquidate, hold cash.

Backtest period 2018-2026 to include 2020 covid + 2022 bear.
"""

import json
import os

import numpy as np
import pandas as pd

from RL.constant import stock_ids
from RL.strategy_jI import (
    FEE,
    INITIAL_CASH,
    LOOKBACK_MOM,
    MA_WINDOW,
    REBAL_EVERY,
    TAX,
    ew_basket_return,
    load_panel,
    market_proxy,
    max_drawdown,
    momentum_weights,
    regime_series,
)

BACKTEST_START = "20180101"
BACKTEST_END = "20260422"
TURB_LOOKBACK = 252
TURB_PERCENTILE = 0.99


def turbulence_series(close_df, lookback=TURB_LOOKBACK):
    """Daily Mahalanobis turbulence vs rolling 252d mean/cov."""
    rets = close_df.pct_change().fillna(0.0).values
    dates = close_df.index
    n = rets.shape[0]
    turb = np.full(n, np.nan)
    for t in range(lookback, n):
        window = rets[t - lookback : t]
        cur = rets[t]
        valid_cols = (~np.isnan(window).any(axis=0)) & (~np.isnan(cur)) & (window.std(axis=0) > 0)
        if valid_cols.sum() < 5:
            continue
        w = window[:, valid_cols]
        r = cur[valid_cols]
        mu = w.mean(axis=0)
        cov = np.cov(w.T) + 1e-6 * np.eye(w.shape[1])
        try:
            cov_inv = np.linalg.pinv(cov)
            d = r - mu
            turb[t] = float(d @ cov_inv @ d)
        except np.linalg.LinAlgError:
            continue
    return pd.Series(turb, index=dates)


def backtest(close_df, regime, mom_w, turb, threshold, start, end, use_turbulence=True):
    val_dates = close_df.loc[start:end].index
    rebal_set = set(val_dates[::REBAL_EVERY])
    rebal_set.add(val_dates[0])

    cash = float(INITIAL_CASH)
    shares = pd.Series(0.0, index=close_df.columns)
    history, triggered = [], []

    for date in val_dates:
        px = close_df.loc[date].fillna(0.0)
        port_val = cash + float((shares * px).sum())

        in_turb = bool(use_turbulence and not np.isnan(turb.get(date, np.nan)) and turb.get(date) > threshold)

        if in_turb and (shares != 0).any():
            sell_val = float((shares * px).sum())
            tx_cost = sell_val * (FEE + TAX)
            cash = port_val - tx_cost
            shares = pd.Series(0.0, index=close_df.columns)
            triggered.append({"date": str(date.date()), "turb": float(turb.get(date)), "action": "liquidate"})
        elif date in rebal_set and not in_turb:
            target_invested = regime.get(date, 0) * port_val
            tgt_w = mom_w.loc[date] if date in mom_w.index else pd.Series(0.0, index=close_df.columns)
            tgt_w = tgt_w.fillna(0.0)
            tgt_alloc = tgt_w * target_invested

            new_shares = pd.Series(0.0, index=close_df.columns)
            for sid in close_df.columns:
                p = px[sid]
                if p > 0 and tgt_alloc[sid] > 0:
                    lots = int(tgt_alloc[sid] // (p * 1000))
                    new_shares[sid] = lots * 1000

            sells = float(((shares - new_shares).clip(lower=0) * px).sum())
            buys = float(((new_shares - shares).clip(lower=0) * px).sum())
            tx_cost = sells * (FEE + TAX) + buys * FEE
            cash = port_val - float((new_shares * px).sum()) - tx_cost
            shares = new_shares

        history.append({"date": date, "value": cash + float((shares * px).sum())})

    return pd.DataFrame(history).set_index("date"), triggered


def run_strategy(close_df, regime, mom_w, turb, threshold, start, end, label, use_turb):
    hist, trig = backtest(close_df, regime, mom_w, turb, threshold, start, end, use_turbulence=use_turb)
    final = hist["value"].iloc[-1]
    ret = final / INITIAL_CASH - 1
    ew = ew_basket_return(close_df, start, end)
    alpha = ret - ew
    mdd = max_drawdown(hist["value"].values)
    return {
        "label": label,
        "return": ret,
        "ew": ew,
        "alpha": alpha,
        "mdd": mdd,
        "trig_count": len(trig),
        "history": hist,
        "triggered": trig,
    }


if __name__ == "__main__":
    close_df, _ = load_panel()
    market = market_proxy(close_df)
    regime = regime_series(market)
    mom_w = momentum_weights(close_df)

    print("computing turbulence (Mahalanobis, 252d window)...")
    turb = turbulence_series(close_df)

    pre_start = pd.Timestamp(BACKTEST_START)
    pre_val_turb = turb.loc[turb.index < pre_start].dropna()
    if len(pre_val_turb) < 100:
        all_turb_v = turb.dropna()
        threshold = float(all_turb_v.quantile(TURB_PERCENTILE))
        print(f"  using full-history threshold (insufficient pre-val): {threshold:.1f}")
    else:
        threshold = float(pre_val_turb.quantile(TURB_PERCENTILE))
        print(f"  threshold (99%-ile, pre-{BACKTEST_START}): {threshold:.1f}")

    val_dates = close_df.loc[BACKTEST_START:BACKTEST_END].index
    days_above = int((turb.loc[val_dates] > threshold).sum())
    print(f"  days above threshold in {BACKTEST_START}..{BACKTEST_END}: {days_above}/{len(val_dates)}")

    print("\nrunning J+I (no T) ...")
    r_jI = run_strategy(close_df, regime, mom_w, turb, threshold, BACKTEST_START, BACKTEST_END, "J+I", use_turb=False)
    print("running J+I+T ...")
    r_jIT = run_strategy(close_df, regime, mom_w, turb, threshold, BACKTEST_START, BACKTEST_END, "J+I+T", use_turb=True)

    print(f"\n{'=' * 70}\nBACKTEST {BACKTEST_START}..{BACKTEST_END}  EW = {r_jI['ew'] * 100:+.2f}%\n{'=' * 70}")
    print(f"{'strategy':<10}{'return':>10}{'alpha':>10}{'MDD':>10}{'turb trig':>12}")
    for r in [r_jI, r_jIT]:
        print(f"{r['label']:<10}{r['return'] * 100:>9.2f}%{r['alpha'] * 100:>9.2f}%{r['mdd'] * 100:>9.2f}%{r['trig_count']:>12}")

    if r_jIT["triggered"]:
        print(f"\nturbulence trigger dates (first 15):")
        for t in r_jIT["triggered"][:15]:
            print(f"  {t['date']}  turb={t['turb']:.1f}")

    out = {
        "period": (BACKTEST_START, BACKTEST_END),
        "threshold": threshold,
        "days_above": days_above,
        "ew_return": r_jI["ew"],
        "results": [
            {k: v for k, v in r.items() if k not in ("history", "triggered")}
            for r in [r_jI, r_jIT]
        ],
        "turb_triggers": r_jIT["triggered"],
    }
    with open("strategy_jIT_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("saved strategy_jIT_results.json")
