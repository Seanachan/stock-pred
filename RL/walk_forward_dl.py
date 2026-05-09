"""Walk-forward eval for the supervised DL portfolio (v3 / v3.1)."""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from RL.constant import stock_ids
from RL.dl_portfolio import train_one_fold
from RL.feature import FeatureExtractor


def gen_rolling_folds(
    train_start="20150105",
    first_val="20190101",
    last_val_end="20231231",
    val_months=6,
):
    folds = []
    cur = pd.Timestamp(first_val)
    end_limit = pd.Timestamp(last_val_end)
    while cur + pd.DateOffset(months=val_months) - pd.Timedelta(days=1) <= end_limit:
        tr_e = (cur - pd.Timedelta(days=1)).strftime("%Y%m%d")
        v_s = cur.strftime("%Y%m%d")
        v_e = (
            cur + pd.DateOffset(months=val_months) - pd.Timedelta(days=1)
        ).strftime("%Y%m%d")
        folds.append((train_start, tr_e, v_s, v_e))
        cur += pd.DateOffset(months=val_months)
    return folds


FOLDS = gen_rolling_folds()


def load_data(start, end):
    dfs = {}
    for sid in stock_ids:
        fp = f"RL/data/{sid}.csv"
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp, parse_dates=["date"], index_col="date")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df = df.loc[start:end].dropna()
        if len(df) > 30:
            dfs[sid] = df
    return dfs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="v3")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--max-weight", type=float, default=0.10)
    parser.add_argument("--sparsemax", action="store_true")
    parser.add_argument("--entropy-lambda", type=float, default=0.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  tag: {args.tag}  cap: {args.max_weight}  "
          f"sparsemax: {args.sparsemax}  entropy_λ: {args.entropy_lambda}")
    fe = FeatureExtractor(stock_ids)

    results = []
    for i, (tr_s, tr_e, val_s, val_e) in enumerate(FOLDS):
        print(f"\n{'=' * 70}")
        print(f"FOLD {i}: train {tr_s}..{tr_e}  →  val {val_s}..{val_e}")
        print(f"{'=' * 70}")

        train_data = load_data(tr_s, tr_e)
        val_data = load_data(val_s, val_e)
        if not train_data or not val_data:
            print("  no data, skip")
            continue

        out = train_one_fold(
            train_data,
            val_data,
            feature_extractor=fe,
            stock_ids=stock_ids,
            epochs=args.epochs,
            lr=1e-3,
            hidden=64,
            emb=32,
            max_weight=args.max_weight,
            tx_cost=0.0042,
            use_sparsemax=args.sparsemax,
            entropy_lambda=args.entropy_lambda,
            device=device,
            log_every=50,
            seed=i,
        )
        out["fold"] = i
        out["train"] = (tr_s, tr_e)
        out["val"] = (val_s, val_e)

        print(
            f"Fold {i} VAL: return={out['return'] * 100:+.2f}%  "
            f"EW={out['ew'] * 100:+.2f}%  alpha={out['alpha'] * 100:+.2f}%  "
            f"sharpe={out['val_sharpe']:+.3f}  active={out['active_stocks_avg']:.1f}  "
            f"cash={out['cash_weight_avg'] * 100:.1f}%"
        )
        results.append(out)

    print(f"\n{'=' * 70}")
    print("WALK-FORWARD SUMMARY (v3 supervised DL)")
    print(f"{'=' * 70}")
    print(
        f"{'fold':<6}{'val period':<24}{'return':>10}"
        f"{'EW':>10}{'alpha':>10}{'sharpe':>10}{'active':>9}{'cash':>8}"
    )
    for r in results:
        val_label = f"{r['val'][0]}..{r['val'][1]}"
        print(
            f"{r['fold']:<6}{val_label:<24}"
            f"{r['return'] * 100:>9.2f}%{r['ew'] * 100:>9.2f}%"
            f"{r['alpha'] * 100:>9.2f}%{r['val_sharpe']:>+9.3f}"
            f"{r['active_stocks_avg']:>9.1f}{r['cash_weight_avg'] * 100:>7.1f}%"
        )

    alphas = [r["alpha"] for r in results]
    sharpes = [r["val_sharpe"] for r in results]
    print(f"\nmean alpha:  {np.mean(alphas) * 100:+.2f}%")
    print(f"mean sharpe: {np.mean(sharpes):+.3f}")
    print(f"min alpha:   {min(alphas) * 100:+.2f}%")
    print(f"max alpha:   {max(alphas) * 100:+.2f}%")
    print(f"std alpha:   {np.std(alphas) * 100:.2f}%")
    print(f"folds with positive alpha: {sum(1 for a in alphas if a > 0)}/{len(alphas)}")

    out_fp = f"walk_forward_{args.tag}_results.json"
    with open(out_fp, "w") as f:
        json.dump(
            [
                {k: v for k, v in r.items() if k != "stock_weights_avg"}
                for r in results
            ],
            f,
            indent=2,
            default=str,
        )
    print(f"\nSaved {out_fp}")
