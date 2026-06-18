"""Offline aggregation sweep over cached walk-forward seed weights.

Loads walk_forward_<tag>_cache.pkl (per-fold per-seed daily weight matrices)
produced by RL.walk_forward_dl_ensemble, then replays three ensemble
aggregation schemes WITHOUT retraining:

  - sharpe_weighted : softmax(val_sharpe)-weighted mean  (T1, deployed rule)
  - uniform         : equal-weight mean across seeds     (handoff fallback)
  - median          : per-dim median across seeds        (legacy v4 rule)

For each scheme it recomputes per-fold alpha / cash% / active and runs the
v5 acceptance gate. One walk-forward run -> all schemes compared apples-to-
apples (same seed weights, same val returns).
"""
import pickle
import sys

import numpy as np
import torch

from RL.dl_portfolio import realized_returns
from RL.walk_forward_dl_ensemble import aggregate_weights


def _renorm_and_cap(agg: np.ndarray, cap: float) -> np.ndarray:
    """Shared tail of aggregate_weights: clip>=0, renorm rows, per-stock cap,
    overflow -> cash slot. Mirrors RL.walk_forward_dl_ensemble exactly."""
    agg = np.clip(agg, 0.0, None)
    row_sum = agg.sum(axis=1, keepdims=True)
    degenerate = (row_sum <= 0).flatten()
    safe_sum = np.where(row_sum <= 0, 1.0, row_sum)
    agg = agg / safe_sum
    if degenerate.any():
        agg[degenerate] = 0.0
        agg[degenerate, -1] = 1.0
    n = agg.shape[1] - 1
    stocks = agg[:, :n]
    excess = np.maximum(stocks - cap, 0.0).sum(axis=1)
    agg[:, :n] = np.minimum(stocks, cap)
    agg[:, -1] = agg[:, -1] + excess
    return agg


def aggregate_median(per_seed_weights, cap: float) -> np.ndarray:
    stacked = np.stack([w.astype(np.float64) for w in per_seed_weights], axis=0)
    agg = np.median(stacked, axis=0)  # (T, N+1)
    return _renorm_and_cap(agg, cap)


def fold_metrics(agg: np.ndarray, val_rets: np.ndarray, tx_cost: float, ew: float):
    agg_t = torch.from_numpy(agg.astype(np.float32))
    rets_t = torch.from_numpy(val_rets.astype(np.float32))
    pnl = realized_returns(agg_t, rets_t, tx_cost=tx_cost)
    cum = (1 + pnl).cumprod(0)
    ens_ret = float(cum[-1].item() - 1)
    return {
        "alpha": ens_ret - ew,
        "ens_ret": ens_ret,
        "cash": float(agg[:, -1].mean()),
        "active": float((agg[:, :-1] > 0.005).sum(axis=1).mean()),
    }


def run_scheme(cache, scheme: str, cap: float):
    rows = []
    for fold in cache:
        w = fold["per_seed_weights"]
        if scheme == "sharpe_weighted":
            agg = aggregate_weights(w, per_seed_scores=fold["per_seed_val_sharpe"], cap=cap)
        elif scheme == "uniform":
            agg = aggregate_weights(w, per_seed_scores=None, cap=cap)
        elif scheme == "median":
            agg = aggregate_median(w, cap)
        else:
            raise ValueError(scheme)
        rows.append(fold_metrics(agg, fold["val_rets"], fold["tx_cost"], fold["ew"]))
    return rows


def gate(rows):
    alphas = np.array([r["alpha"] for r in rows])
    cash = np.array([r["cash"] for r in rows])
    active = np.array([r["active"] for r in rows])
    n = len(rows)
    m_alpha, s_alpha = float(alphas.mean()), float(alphas.std())
    pos = int((alphas > 0).sum())
    m_cash, m_active = float(cash.mean()), float(active.mean())
    checks = {
        "mean_alpha>=4%": (m_alpha >= 0.04, f"{m_alpha * 100:+.2f}%"),
        f"pos_folds>=7/{n}": (pos >= 7, f"{pos}/{n}"),
        "cash in[5,40]%": (0.05 <= m_cash <= 0.40, f"{m_cash * 100:.1f}%"),
        "active in[6,20]": (6 <= m_active <= 20, f"{m_active:.1f}"),
        "std_alpha<=8%": (s_alpha <= 0.08, f"{s_alpha * 100:.2f}%"),
    }
    return checks, alphas


def main(cache_fp: str, cap: float = 0.10):
    with open(cache_fp, "rb") as f:
        cache = pickle.load(f)
    print(f"loaded {cache_fp}: {len(cache)} folds, "
          f"{len(cache[0]['per_seed_weights'])} seeds each\n")
    for scheme in ("sharpe_weighted", "uniform", "median"):
        rows = run_scheme(cache, scheme, cap)
        checks, alphas = gate(rows)
        all_pass = all(ok for ok, _ in checks.values())
        print(f"=== {scheme} === {'PASS' if all_pass else 'FAIL'}")
        for name, (ok, val) in checks.items():
            print(f"   [{'P' if ok else 'F'}] {name:<18} {val}")
        print(f"   per-fold alpha: {[f'{a * 100:+.1f}' for a in alphas]}\n")


if __name__ == "__main__":
    fp = sys.argv[1] if len(sys.argv) > 1 else "walk_forward_v5_full_cache.pkl"
    main(fp)
