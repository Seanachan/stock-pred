"""Walk-forward eval for the 5-seed v4 LSTM ensemble (matches deploy).

For each fold:
  1. Train N seeds with train_one_fold_lstm.
  2. Collect each seed's per-day val weight matrix (T_val, num_stocks+1).
  3. Aggregate across seeds via median + renormalize + per-stock cap
     (mirrors RL.deploy_rl.predict_dl_ensemble_weights).
  4. Recompute realized_returns on the aggregated weights.

This is the OOS analogue of the deployed ensemble decision rule.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from RL.constant import stock_ids
from RL.dl_portfolio import realized_returns, train_one_fold_lstm
from RL.feature import FeatureExtractor
from RL.walk_forward_dl import FOLDS, load_data


def aggregate_weights(
    per_seed_weights: list[np.ndarray], cap: float = 0.10
) -> np.ndarray:
    """Median-aggregate per-seed daily weight matrices, renormalize, re-cap.

    Each entry of per_seed_weights has shape (T_val, N+1) — daily portfolio
    weights including the cash slot at index -1, post-softmax + post-cap from
    its own seed. Return shape (T_val, N+1) with each row a valid simplex
    point obeying the per-stock cap (cash slot uncapped).

    Mirror RL.deploy_rl.predict_dl_ensemble_weights so backtest matches the
    decision rule used in live trading. Steps the deploy code performs daily:
      - per-dim median across seeds
      - clip negatives, renormalize sum=1 (fallback to all-cash if sum<=0)
      - clip each stock weight to `cap`, overflow → cash slot
    """
    stacked = np.stack(
        [w.astype(np.float64) for w in per_seed_weights], axis=0
    )  # (seeds, T, N+1)
    med = np.median(stacked, axis=0)  # (T, N+1)

    med = np.clip(med, 0.0, None)
    row_sum = med.sum(axis=1, keepdims=True)  # (T, 1)
    degenerate = (row_sum <= 0).flatten()
    safe_sum = np.where(row_sum <= 0, 1.0, row_sum)
    med = med / safe_sum
    if degenerate.any():
        med[degenerate] = 0.0
        med[degenerate, -1] = 1.0

    N = med.shape[1] - 1
    stocks = med[:, :N]
    excess = np.maximum(stocks - cap, 0.0).sum(axis=1)  # (T,)
    med[:, :N] = np.minimum(stocks, cap)
    med[:, -1] = med[:, -1] + excess
    return med


def run_fold(
    fold_idx: int,
    tr_s: str,
    tr_e: str,
    val_s: str,
    val_e: str,
    *,
    seeds: int,
    epochs: int,
    window: int,
    hidden: int,
    train_recent: int,
    max_weight: float,
    device: str,
    fe: FeatureExtractor,
) -> dict:
    train_data = load_data(tr_s, tr_e)
    prefix = (
        pd.Timestamp(val_s) - pd.Timedelta(days=window * 2)
    ).strftime("%Y%m%d")
    val_data = load_data(prefix, val_e)
    if not train_data or not val_data:
        return None

    per_seed_w = []
    per_seed_returns = []
    per_seed_ew = []
    rets_v_y = None
    tx_cost = None
    for s in range(seeds):
        print(f"  [fold {fold_idx} seed {s}] training...")
        out = train_one_fold_lstm(
            train_data,
            val_data,
            val_actual_start=val_s,
            feature_extractor=fe,
            stock_ids=stock_ids,
            epochs=epochs,
            lr=1e-3,
            window_len=window,
            hidden=hidden,
            max_weight=max_weight,
            tx_cost=0.0042,
            use_sparsemax=False,
            entropy_lambda=0.0,
            train_recent_days=train_recent,
            device=device,
            log_every=max(epochs, 1),  # quiet per-epoch logs
            seed=s,
        )
        per_seed_w.append(out["val_weights"])
        per_seed_returns.append(out["return"])
        per_seed_ew.append(out["ew"])
        if rets_v_y is None:
            rets_v_y = out["val_rets"]
            tx_cost = out["tx_cost"]
        print(
            f"  [fold {fold_idx} seed {s}] ret={out['return'] * 100:+.2f}% "
            f"alpha={out['alpha'] * 100:+.2f}%"
        )

    # Sanity: all seeds should have produced the same T_val grid.
    shapes = {w.shape for w in per_seed_w}
    if len(shapes) != 1:
        raise RuntimeError(f"seed weight shapes diverged: {shapes}")

    agg = aggregate_weights(per_seed_w, cap=max_weight)
    agg_t = torch.from_numpy(agg.astype(np.float32))
    rets_t = torch.from_numpy(rets_v_y.astype(np.float32))
    pnl = realized_returns(agg_t, rets_t, tx_cost=tx_cost)
    cum = (1 + pnl).cumprod(0)
    ens_ret = float(cum[-1].item() - 1)
    ens_sharpe = float(pnl.mean() / (pnl.std() + 1e-9))

    ew_ret = float(np.mean(per_seed_ew))  # identical across seeds (same val data)
    return {
        "fold": fold_idx,
        "train": (tr_s, tr_e),
        "val": (val_s, val_e),
        "seeds": seeds,
        "per_seed_returns": per_seed_returns,
        "ensemble_return": ens_ret,
        "ew": ew_ret,
        "alpha": ens_ret - ew_ret,
        "val_sharpe": ens_sharpe,
        "agg_cash_avg": float(agg[:, -1].mean()),
        "agg_active_avg": float((agg[:, :-1] > 0.005).sum(axis=1).mean()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="v4_5seed")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--train-recent", type=int, default=500)
    parser.add_argument("--max-weight", type=float, default=0.10)
    parser.add_argument("--fold-start", type=int, default=0)
    parser.add_argument("--fold-end", type=int, default=len(FOLDS))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"device: {device}  tag: {args.tag}  seeds: {args.seeds}  "
        f"epochs: {args.epochs}  window: {args.window}  hidden: {args.hidden}  "
        f"train_recent: {args.train_recent}  cap: {args.max_weight}"
    )
    fe = FeatureExtractor(stock_ids)

    results = []
    for i, fold in enumerate(FOLDS):
        if i < args.fold_start or i >= args.fold_end:
            continue
        tr_s, tr_e, val_s, val_e = fold
        print(f"\n{'=' * 70}")
        print(f"FOLD {i}: train {tr_s}..{tr_e}  →  val {val_s}..{val_e}")
        print(f"{'=' * 70}")
        out = run_fold(
            i, tr_s, tr_e, val_s, val_e,
            seeds=args.seeds,
            epochs=args.epochs,
            window=args.window,
            hidden=args.hidden,
            train_recent=args.train_recent,
            max_weight=args.max_weight,
            device=device,
            fe=fe,
        )
        if out is None:
            print("  no data, skip")
            continue
        print(
            f"Fold {i} ENSEMBLE: ret={out['ensemble_return'] * 100:+.2f}%  "
            f"EW={out['ew'] * 100:+.2f}%  alpha={out['alpha'] * 100:+.2f}%  "
            f"sharpe={out['val_sharpe']:+.3f}  cash={out['agg_cash_avg'] * 100:.1f}%  "
            f"active={out['agg_active_avg']:.1f}"
        )
        results.append(out)

    print(f"\n{'=' * 70}")
    print(f"WALK-FORWARD SUMMARY ({args.seeds}-seed v4 LSTM ensemble)")
    print(f"{'=' * 70}")
    print(
        f"{'fold':<6}{'val period':<24}{'ens_ret':>10}{'EW':>10}"
        f"{'alpha':>10}{'sharpe':>10}{'cash':>8}{'active':>9}"
    )
    for r in results:
        val_label = f"{r['val'][0]}..{r['val'][1]}"
        print(
            f"{r['fold']:<6}{val_label:<24}"
            f"{r['ensemble_return'] * 100:>9.2f}%{r['ew'] * 100:>9.2f}%"
            f"{r['alpha'] * 100:>9.2f}%{r['val_sharpe']:>+9.3f}"
            f"{r['agg_cash_avg'] * 100:>7.1f}%{r['agg_active_avg']:>9.1f}"
        )

    if results:
        alphas = [r["alpha"] for r in results]
        sharpes = [r["val_sharpe"] for r in results]
        rets = [r["ensemble_return"] for r in results]
        print(f"\nmean ens_ret: {np.mean(rets) * 100:+.2f}%")
        print(f"mean alpha:   {np.mean(alphas) * 100:+.2f}%")
        print(f"mean sharpe:  {np.mean(sharpes):+.3f}")
        print(f"min alpha:    {min(alphas) * 100:+.2f}%")
        print(f"max alpha:    {max(alphas) * 100:+.2f}%")
        print(f"std alpha:    {np.std(alphas) * 100:.2f}%")
        print(f"folds positive alpha: {sum(1 for a in alphas if a > 0)}/{len(alphas)}")

    out_fp = f"walk_forward_{args.tag}_results.json"
    with open(out_fp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {out_fp}")
