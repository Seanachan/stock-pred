"""EW monthly rebalance: buy 46 EW, every 21d restore equal weights.

Generates trades while tracking BH-EW closely. Variants:
- ew_rebal     : pure monthly EW rebalance, no protection
- ew_rebal_T   : + turbulence hard-stop
- ew_rebal_dd  : + simple trailing-drawdown stop (-15% from peak)

Compare to BH-EW + theoretical EW.
"""

import json

import numpy as np
import pandas as pd

from RL.strategy_jI import (
    FEE,
    INITIAL_CASH,
    REBAL_EVERY,
    TAX,
    ew_basket_return,
    load_panel,
    max_drawdown,
)
from RL.strategy_jIT import TURB_PERCENTILE, turbulence_series

BACKTEST_START = "20180101"
BACKTEST_END = "20260422"
DD_STOP = 0.15  # liquidate if portfolio down 15% from peak
DD_REENTRY = 0.05  # re-enter when recovered to within 5% of peak


def buy_ew(port_val, px):
    valid = px > 0
    n_valid = int(valid.sum())
    if n_valid == 0:
        return None, 0.0, 0.0
    per_stock = port_val / n_valid
    new_shares = pd.Series(0.0, index=px.index)
    for sid in px.index:
        p = px[sid]
        if p > 0:
            lots = int(per_stock // (p * 1000))
            new_shares[sid] = lots * 1000
    invested = float((new_shares * px).sum())
    fee = invested * FEE
    return new_shares, invested, fee


def backtest(close_df, turb=None, threshold=None, start=BACKTEST_START, end=BACKTEST_END,
             mode="rebal", dd_stop=DD_STOP, dd_reentry=DD_REENTRY):
    """mode: 'bh' | 'rebal' | 'rebal_T' | 'rebal_dd'."""
    val_dates = close_df.loc[start:end].index
    rebal_set = set(val_dates[::REBAL_EVERY])
    rebal_set.add(val_dates[0])

    cash = float(INITIAL_CASH)
    shares = pd.Series(0.0, index=close_df.columns)
    history = []
    n_trades = 0
    triggers = []
    in_market = False
    peak_val = float(INITIAL_CASH)
    days_below_thresh = 0

    for date in val_dates:
        px = close_df.loc[date].fillna(0.0)
        port_val = cash + float((shares * px).sum())
        peak_val = max(peak_val, port_val)
        t_val = turb.get(date, np.nan) if turb is not None else np.nan
        in_turb = bool(mode == "rebal_T" and not np.isnan(t_val) and t_val > threshold)

        cur_dd = (peak_val - port_val) / max(peak_val, 1.0)
        in_dd = bool(mode == "rebal_dd" and cur_dd > dd_stop)

        # Day 1 entry
        if not in_market and date == val_dates[0]:
            new_shares, invested, fee = buy_ew(port_val, px)
            if new_shares is not None:
                cash = port_val - invested - fee
                shares = new_shares
                n_trades += int((new_shares > 0).sum())
                in_market = True

        # Turbulence / DD exit
        elif in_market and (in_turb or in_dd):
            sell_val = float((shares * px).sum())
            tx_cost = sell_val * (FEE + TAX)
            cash = port_val - tx_cost
            n_trades += int((shares > 0).sum())
            shares = pd.Series(0.0, index=close_df.columns)
            in_market = False
            days_below_thresh = 0
            triggers.append({
                "date": str(date.date()), "action": "exit",
                "turb": float(t_val) if not np.isnan(t_val) else None,
                "dd": float(cur_dd), "trades": int((shares == 0).sum()),
            })

        # Re-entry after exit
        elif not in_market and (mode == "rebal_T" or mode == "rebal_dd"):
            if mode == "rebal_T":
                if not in_turb:
                    days_below_thresh += 1
                else:
                    days_below_thresh = 0
                ready = days_below_thresh >= 5
            else:  # rebal_dd
                ready = cur_dd <= dd_reentry
            if ready:
                new_shares, invested, fee = buy_ew(port_val, px)
                if new_shares is not None:
                    cash = port_val - invested - fee
                    shares = new_shares
                    n_trades += int((new_shares > 0).sum())
                    in_market = True
                    triggers.append({"date": str(date.date()), "action": "reenter"})

        # Monthly EW rebalance (skip in pure BH)
        elif mode != "bh" and in_market and date in rebal_set and date != val_dates[0]:
            n_valid = int((px > 0).sum())
            if n_valid > 0:
                per_stock = port_val / n_valid
                new_shares = pd.Series(0.0, index=close_df.columns)
                for sid in close_df.columns:
                    p = px[sid]
                    if p > 0:
                        lots = int(per_stock // (p * 1000))
                        new_shares[sid] = lots * 1000
                sells = float(((shares - new_shares).clip(lower=0) * px).sum())
                buys = float(((new_shares - shares).clip(lower=0) * px).sum())
                tx_cost = sells * (FEE + TAX) + buys * FEE
                changed = (new_shares != shares).sum()
                cash = port_val - float((new_shares * px).sum()) - tx_cost
                n_trades += int(changed)
                shares = new_shares

        history.append({"date": date, "value": cash + float((shares * px).sum())})

    return pd.DataFrame(history).set_index("date"), n_trades, triggers


if __name__ == "__main__":
    close_df, _ = load_panel()
    print("computing turbulence...")
    turb = turbulence_series(close_df)
    pre = turb.loc[turb.index < pd.Timestamp(BACKTEST_START)].dropna()
    threshold = float(pre.quantile(TURB_PERCENTILE))
    print(f"turb threshold (99%-ile pre-{BACKTEST_START}): {threshold:.1f}\n")

    ew_theo = ew_basket_return(close_df, BACKTEST_START, BACKTEST_END)

    results = []
    for mode in ["bh", "rebal", "rebal_T", "rebal_dd"]:
        h, nt, trig = backtest(close_df, turb, threshold, mode=mode)
        ret = h["value"].iloc[-1] / INITIAL_CASH - 1
        mdd = max_drawdown(h["value"].values)
        results.append({"mode": mode, "return": ret, "alpha": ret - ew_theo, "mdd": mdd, "trades": nt, "triggers": len(trig)})

    print(f"{'=' * 75}\n8-yr backtest {BACKTEST_START}..{BACKTEST_END}  EW theory = {ew_theo * 100:+.2f}%\n{'=' * 75}")
    print(f"{'mode':<12}{'return':>10}{'alpha':>10}{'MDD':>10}{'trades':>10}{'triggers':>10}{'qualify':>10}")
    for r in results:
        q = "YES" if r["trades"] >= 100 else "NO"
        print(f"{r['mode']:<12}{r['return'] * 100:>9.2f}%{r['alpha'] * 100:>9.2f}%{r['mdd'] * 100:>9.2f}%"
              f"{r['trades']:>10}{r['triggers']:>10}{q:>10}")

    with open("strategy_ew_rebal_results.json", "w") as f:
        json.dump({"period": (BACKTEST_START, BACKTEST_END), "ew_theo": ew_theo,
                   "threshold": threshold, "results": results}, f, indent=2, default=str)
    print("\nsaved strategy_ew_rebal_results.json")
