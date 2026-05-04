"""J+I strategy: regime gate (200MA) × momentum-weighted (60d rank).

Rules:
- market_proxy = EW basket of 46 stocks (close)
- regime[t] = 1 if market_proxy[t] > 200MA[t] else 0
- on rebalance day:
    if regime: weight stock i = momentum_rank_60d / sum_ranks
    else:      all cash
- rebalance monthly (every 21 trading days)
- transaction cost: 0.1425% fee + 0.3% tax (sells only) -> 0.4425% on rotation
"""

import json
import os

import numpy as np
import pandas as pd

from RL.constant import stock_ids

VAL_START = "20251001"
VAL_END = "20260422"
INITIAL_CASH = 100_000_000
LOOKBACK_MOM = 60
MA_WINDOW = 200
REBAL_EVERY = 21  # trading days
FEE = 0.001425
TAX = 0.003


def load_panel():
    closes, opens = {}, {}
    for sid in stock_ids:
        fp = f"RL/data/{sid}.csv"
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp, parse_dates=["date"], index_col="date")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        closes[sid] = df["close"]
        opens[sid] = df["open"]
    close_df = pd.DataFrame(closes).sort_index()
    open_df = pd.DataFrame(opens).sort_index()
    return close_df, open_df


def market_proxy(close_df):
    """Equal-weighted basket level (rebased to 1.0 at start)."""
    norm = close_df.div(close_df.iloc[0])
    valid_mask = close_df.notna()
    valid_count = valid_mask.sum(axis=1).replace(0, 1)
    proxy = (norm * valid_mask).sum(axis=1) / valid_count
    return proxy


def regime_series(market, ma_window=MA_WINDOW):
    ma = market.rolling(ma_window).mean()
    return (market > ma).astype(int)


def momentum_weights(close_df, lookback=LOOKBACK_MOM):
    ret = close_df.pct_change(lookback)
    ranks = ret.rank(axis=1, pct=True)
    sums = ranks.sum(axis=1).replace(0, np.nan)
    weights = ranks.div(sums, axis=0).fillna(0.0)
    return weights


def backtest(close_df, regime, mom_w, val_start, val_end):
    val_dates = close_df.loc[val_start:val_end].index
    rebal_dates = set(val_dates[::REBAL_EVERY])
    rebal_dates.add(val_dates[0])

    cash = float(INITIAL_CASH)
    shares = pd.Series(0.0, index=close_df.columns)
    history, trade_log = [], []

    for date in val_dates:
        px = close_df.loc[date].fillna(0.0)
        port_val = cash + float((shares * px).sum())

        if date in rebal_dates:
            target_invested = regime.get(date, 0) * port_val
            tgt_w = mom_w.loc[date] if date in mom_w.index else pd.Series(
                0.0, index=close_df.columns
            )
            tgt_w = tgt_w.fillna(0.0)
            tgt_alloc = tgt_w * target_invested

            new_shares = pd.Series(0.0, index=close_df.columns)
            for sid in close_df.columns:
                p = px[sid]
                if p > 0 and tgt_alloc[sid] > 0:
                    lots = int(tgt_alloc[sid] // (p * 1000))
                    new_shares[sid] = lots * 1000

            sells_value = float(((shares - new_shares).clip(lower=0) * px).sum())
            buys_value = float(((new_shares - shares).clip(lower=0) * px).sum())
            tx_cost = sells_value * (FEE + TAX) + buys_value * FEE
            cash = port_val - float((new_shares * px).sum()) - tx_cost
            n_trades = int(((new_shares != shares)).sum())
            shares = new_shares
            trade_log.append({"date": str(date.date()), "trades": n_trades, "cost": tx_cost})

        history.append({"date": date, "value": cash + float((shares * px).sum())})

    return pd.DataFrame(history).set_index("date"), trade_log


def ew_basket_return(close_df, val_start, val_end):
    sub = close_df.loc[val_start:val_end]
    rets = []
    for sid in sub.columns:
        s = sub[sid].dropna()
        if len(s) > 1:
            rets.append(s.iloc[-1] / s.iloc[0] - 1)
    return float(np.mean(rets)) if rets else 0.0


def max_drawdown(values):
    arr = np.asarray(values, dtype=float)
    peak = np.maximum.accumulate(arr)
    dd = (peak - arr) / np.maximum(peak, 1.0)
    return float(dd.max())


if __name__ == "__main__":
    close_df, _ = load_panel()
    market = market_proxy(close_df)
    regime = regime_series(market)
    mom_w = momentum_weights(close_df)

    history, trade_log = backtest(close_df, regime, mom_w, VAL_START, VAL_END)
    final_val = history["value"].iloc[-1]
    ret = final_val / INITIAL_CASH - 1
    ew = ew_basket_return(close_df, VAL_START, VAL_END)
    alpha = ret - ew
    mdd = max_drawdown(history["value"].values)
    n_trades = sum(t["trades"] for t in trade_log)
    total_cost = sum(t["cost"] for t in trade_log)

    val_dates = close_df.loc[VAL_START:VAL_END].index
    bull_days = int(regime.loc[val_dates].sum())
    total_days = len(val_dates)

    print(f"{'=' * 70}\nJ+I STRATEGY VAL\n{'=' * 70}")
    print(f"period: {VAL_START}..{VAL_END} ({total_days} days)")
    print(f"regime: {bull_days} bull / {total_days - bull_days} bear")
    print()
    print(f"strategy return : {ret * 100:+7.2f}%")
    print(f"EW baseline     : {ew * 100:+7.2f}%")
    print(f"alpha           : {alpha * 100:+7.2f}%")
    print(f"max drawdown    : {mdd * 100:7.2f}%")
    print(f"total trades    : {n_trades}")
    print(f"total tx cost   : {total_cost:,.0f}")
    print()
    print("vs RL deploy: seed0=+1.62%, seed1=-8.11%, seed2=-27.10%, ensemble=-22.29%")

    with open("strategy_jI_val.json", "w") as f:
        json.dump(
            {
                "method": "regime_gate_x_momentum_weight",
                "val": (VAL_START, VAL_END),
                "return": ret,
                "ew": ew,
                "alpha": alpha,
                "max_drawdown": mdd,
                "trades": n_trades,
                "tx_cost": total_cost,
                "bull_days": bull_days,
                "total_days": total_days,
            },
            f,
            indent=2,
            default=str,
        )
    print("saved strategy_jI_val.json")
