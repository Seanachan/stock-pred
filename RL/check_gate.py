"""v5 walk-forward acceptance gate checker.

Reads a walk_forward_*_results.json and prints PASS/FAIL per criterion.
Gate (all must pass):
  - mean alpha            >= +4.0%
  - positive-alpha folds  >= 7/10
  - mean cash%            in [5%, 40%]
  - mean active_stocks    in [6, 30]
  - std(alpha)            <= 8%
Exit code 0 if all pass, 1 otherwise.
"""
import json
import sys

import numpy as np


def main(fp: str) -> int:
    with open(fp) as f:
        results = json.load(f)
    n = len(results)
    alphas = np.array([r["alpha"] for r in results])
    cash = np.array([r["agg_cash_avg"] for r in results])
    active = np.array([r["agg_active_avg"] for r in results])

    mean_alpha = float(alphas.mean())
    std_alpha = float(alphas.std())
    pos_folds = int((alphas > 0).sum())
    mean_cash = float(cash.mean())
    mean_active = float(active.mean())

    checks = [
        ("mean alpha >= +4.0%", mean_alpha >= 0.04, f"{mean_alpha * 100:+.2f}%"),
        ("positive-alpha folds >= 7/10", pos_folds >= 7, f"{pos_folds}/{n}"),
        ("mean cash% in [5%,40%]", 0.05 <= mean_cash <= 0.40, f"{mean_cash * 100:.1f}%"),
        ("mean active in [6,30]", 6 <= mean_active <= 30, f"{mean_active:.1f}"),
        ("std(alpha) <= 8%", std_alpha <= 0.08, f"{std_alpha * 100:.2f}%"),
    ]

    print(f"=== GATE CHECK: {fp}  ({n} folds) ===")
    all_pass = True
    for name, ok, val in checks:
        all_pass &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<32} = {val}")
    print(f"\nper-fold alpha: {[f'{a * 100:+.1f}%' for a in alphas]}")
    print(f"\n=== {'GATE PASSED' if all_pass else 'GATE FAILED'} ===")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "walk_forward_v5_full_results.json"))
