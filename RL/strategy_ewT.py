"""EW + T: Equal-weight buy-hold 46 stocks + turbulence index hard-stop.

Simpler than J+I+T: drop momentum tilt + 200MA gate (both whipsaw / drag).
Just buy EW once, hold, liquidate only on turbulence spike. Re-enter when calm.

Compare:
- BH-EW (pure buy-hold, no T)
- EW+T (buy-hold + turbulence exits)
- J+I+T (regime gate + momentum + T)
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
    market_proxy,
    max_drawdown,
    momentum_weights,
    regime_series,
)
from RL.strategy_jIT import (
    BACKTEST_END,
    BACKTEST_START,
    TURB_PERCENTILE,
    backtest as backtest_jIT,
    turbulence_series,
)

REENTRY_COOLOFF = 5  # days below threshold before re-buying


def backtest_ew_t(close_df, turb, threshold, start, end, use_turbulence=True, cooloff=REENTRY_COOLOFF):
    val_dates = close_df.loc[start:end].index
    cash = float(INITIAL_CASH)
    shares = pd.Series(0.0, index=close_df.columns)
    history, triggered = [], []
    days_below = 0
    in_market = False

    for date in val_dates:
        px = close_df.loc[date].fillna(0.0)
        port_val = cash + float((shares * px).sum())
        t_val = turb.get(date, np.nan)
        in_turb = bool(use_turbulence and not np.isnan(t_val) and t_val > threshold)

        if use_turbulence and in_turb and in_market:
            sell_val = float((shares * px).sum())
            tx_cost = sell_val * (FEE + TAX)
            cash = port_val - tx_cost
            shares = pd.Series(0.0, index=close_df.columns)
            in_market = False
            days_below = 0
            triggered.append({"date": str(date.date()), "turb": float(t_val), "action": "liquidate"})
        elif use_turbulence and not in_turb and not in_market:
            days_below += 1
            if days_below >= cooloff:
                # Re-enter equal-weight
                valid = (px > 0).values
                n_valid = int(valid.sum())
                if n_valid > 0:
                    per_stock = port_val / n_valid
                    new_shares = pd.Series(0.0, index=close_df.columns)
                    for sid in close_df.columns:
                        p = px[sid]
                        if p > 0:
                            lots = int(per_stock // (p * 1000))
                            new_shares[sid] = lots * 1000
                    invested = float((new_shares * px).sum())
                    tx_cost = invested * FEE
                    cash = port_val - invested - tx_cost
                    shares = new_shares
                    in_market = True
                    triggered.append({"date": str(date.date()), "turb": float(t_val) if not np.isnan(t_val) else None, "action": "reenter"})
        elif not in_market and not use_turbulence:
            # Initial buy (BH-EW or first day of EW+T)
            valid = (px > 0).values
            n_valid = int(valid.sum())
            if n_valid > 0:
                per_stock = port_val / n_valid
                new_shares = pd.Series(0.0, index=close_df.columns)
                for sid in close_df.columns:
                    p = px[sid]
                    if p > 0:
                        lots = int(per_stock // (p * 1000))
                        new_shares[sid] = lots * 1000
                invested = float((new_shares * px).sum())
                tx_cost = invested * FEE
                cash = port_val - invested - tx_cost
                shares = new_shares
                in_market = True

        if not in_market and date == val_dates[0]:
            # Day 1 EW+T: buy initially, will liquidate later if turbulence
            valid = (px > 0).values
            n_valid = int(valid.sum())
            if n_valid > 0:
                per_stock = port_val / n_valid
                new_shares = pd.Series(0.0, index=close_df.columns)
                for sid in close_df.columns:
                    p = px[sid]
                    if p > 0:
                        lots = int(per_stock // (p * 1000))
                        new_shares[sid] = lots * 1000
                invested = float((new_shares * px).sum())
                tx_cost = invested * FEE
                cash = port_val - invested - tx_cost
                shares = new_shares
                in_market = True

        history.append({"date": date, "value": cash + float((shares * px).sum())})

    return pd.DataFrame(history).set_index("date"), triggered


if __name__ == "__main__":
    close_df, _ = load_panel()
    market = market_proxy(close_df)
    regime = regime_series(market)
    mom_w = momentum_weights(close_df)

    print("computing turbulence...")
    turb = turbulence_series(close_df)
    pre = turb.loc[turb.index < pd.Timestamp(BACKTEST_START)].dropna()
    threshold = float(pre.quantile(TURB_PERCENTILE))
    print(f"threshold (99%-ile pre-{BACKTEST_START}): {threshold:.1f}\n")

    ew = ew_basket_return(close_df, BACKTEST_START, BACKTEST_END)

    # BH-EW: pure buy-hold
    h_bh, _ = backtest_ew_t(close_df, turb, threshold, BACKTEST_START, BACKTEST_END, use_turbulence=False)
    bh_ret = h_bh["value"].iloc[-1] / INITIAL_CASH - 1
    bh_mdd = max_drawdown(h_bh["value"].values)

    # EW+T: BH with turbulence exits
    h_ewt, trig_ewt = backtest_ew_t(close_df, turb, threshold, BACKTEST_START, BACKTEST_END, use_turbulence=True)
    ewt_ret = h_ewt["value"].iloc[-1] / INITIAL_CASH - 1
    ewt_mdd = max_drawdown(h_ewt["value"].values)

    # J+I+T (re-run for comparison)
    h_jIT, trig_jIT = backtest_jIT(close_df, regime, mom_w, turb, threshold, BACKTEST_START, BACKTEST_END, use_turbulence=True)
    jIT_ret = h_jIT["value"].iloc[-1] / INITIAL_CASH - 1
    jIT_mdd = max_drawdown(h_jIT["value"].values)

    print(f"{'=' * 70}\nBACKTEST {BACKTEST_START}..{BACKTEST_END}  EW theoretical = {ew * 100:+.2f}%\n{'=' * 70}")
    print(f"{'strategy':<12}{'return':>10}{'alpha':>10}{'MDD':>10}{'triggers':>10}")
    print(f"{'BH-EW':<12}{bh_ret * 100:>9.2f}%{(bh_ret - ew) * 100:>9.2f}%{bh_mdd * 100:>9.2f}%{0:>10}")
    print(f"{'EW+T':<12}{ewt_ret * 100:>9.2f}%{(ewt_ret - ew) * 100:>9.2f}%{ewt_mdd * 100:>9.2f}%{len(trig_ewt):>10}")
    print(f"{'J+I+T':<12}{jIT_ret * 100:>9.2f}%{(jIT_ret - ew) * 100:>9.2f}%{jIT_mdd * 100:>9.2f}%{len(trig_jIT):>10}")

    if trig_ewt:
        print(f"\nEW+T triggers (first 10):")
        for t in trig_ewt[:10]:
            print(f"  {t['date']}  turb={t['turb']}  {t['action']}")

    out = {
        "period": (BACKTEST_START, BACKTEST_END),
        "threshold": threshold,
        "ew_theoretical": ew,
        "BH_EW": {"return": bh_ret, "mdd": bh_mdd},
        "EW_T": {"return": ewt_ret, "mdd": ewt_mdd, "triggers": len(trig_ewt)},
        "J_I_T": {"return": jIT_ret, "mdd": jIT_mdd, "triggers": len(trig_jIT)},
        "ewt_triggers": trig_ewt,
    }
    with open("strategy_ewT_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\nsaved strategy_ewT_results.json")
