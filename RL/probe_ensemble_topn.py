"""Offline sweep of ensemble hard top-N truncation over cached walk-forward
seed weights. Post-aggregation concentration: keep the N highest-weighted
stocks in the aggregated book, zero the rest, renorm, re-cap (water-fill).

Because top-N is applied AFTER aggregation (training unchanged), replaying it
on walk_forward_v5_waterfill_cache.pkl reproduces the exact OOS result of a
walk-forward run with top-N at the aggregate step — no retrain needed.

Run: CUDA_VISIBLE_DEVICES="" uv run python -m RL.probe_ensemble_topn
"""
import pickle
import sys

import numpy as np
import torch

from RL.dl_portfolio import cap_water_fill_np, realized_returns


def aggregate_topn(per_seed_weights, per_seed_scores, cap, top_n):
    """Sharpe-weighted mean -> hard top-N stock truncation -> renorm -> cap."""
    stacked = np.stack([w.astype(np.float64) for w in per_seed_weights], axis=0)
    s = np.asarray(per_seed_scores, dtype=np.float64)
    ws = np.exp(s - s.max())
    ws /= ws.sum()
    agg = (stacked * ws[:, None, None]).sum(axis=0)  # (T, N+1)
    agg = np.clip(agg, 0.0, None)
    agg /= agg.sum(axis=-1, keepdims=True)
    n_stocks = agg.shape[-1] - 1
    if top_n is not None and top_n < n_stocks:
        stock = agg[:, :n_stocks]
        kth = np.sort(stock, axis=-1)[:, -top_n][:, None]  # top_n-th largest/row
        keep = stock >= kth
        agg[:, :n_stocks] = stock * keep
        agg /= agg.sum(axis=-1, keepdims=True)  # survivors + cash rescale
    return cap_water_fill_np(agg, cap)


def fold_metrics(agg, val_rets, tx_cost, ew):
    pnl = realized_returns(torch.from_numpy(agg.astype(np.float32)),
                           torch.from_numpy(val_rets.astype(np.float32)),
                           tx_cost=tx_cost)
    ens_ret = float((1 + pnl).cumprod(0)[-1].item() - 1)
    return {
        "alpha": ens_ret - ew,
        "cash": float(agg[:, -1].mean()),
        "active": float((agg[:, :-1] > 0.005).sum(axis=1).mean()),
    }


def main(cache_fp, cap=0.10):
    with open(cache_fp, "rb") as f:
        cache = pickle.load(f)
    print(f"loaded {cache_fp}: {len(cache)} folds\n")
    print(f"{'top_n':>6}{'mean_alpha':>12}{'median':>9}{'pos':>6}"
          f"{'std':>8}{'mean_cash':>11}{'mean_active':>12}{'beat+0.98%':>11}")
    for top_n in (None, 25, 20, 15, 12, 10, 8):
        rows = [fold_metrics(
            aggregate_topn(f["per_seed_weights"], f["per_seed_val_sharpe"], cap, top_n),
            f["val_rets"], f["tx_cost"], f["ew"]) for f in cache]
        a = np.array([r["alpha"] for r in rows])
        c = np.array([r["cash"] for r in rows])
        act = np.array([r["active"] for r in rows])
        label = "none" if top_n is None else str(top_n)
        print(f"{label:>6}{a.mean() * 100:>11.2f}%{np.median(a) * 100:>8.2f}%"
              f"{int((a > 0).sum()):>4}/{len(a)}{a.std() * 100:>7.1f}%"
              f"{c.mean() * 100:>10.1f}%{act.mean():>12.1f}"
              f"{str(a.mean() > 0.0098):>11}")


if __name__ == "__main__":
    fp = sys.argv[1] if len(sys.argv) > 1 else "walk_forward_v5_waterfill_cache.pkl"
    main(fp)
