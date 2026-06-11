# v5 Paper-Trade Cutover Assessment

Generated: 2026-06-11  
Branch: v5-iteration  
Paper commits reviewed: 5 (`paper: v5 state` commits 2026-06-05 through 2026-06-11)

---

## Paper-day table

| date       | day | port_val        | PnL%    | drawdown | holdings | orders |
|------------|-----|-----------------|---------|----------|----------|--------|
| 2026-06-02 |   1 |  99,862,991.665 | -0.14%  |  0.00%   |    15    |   15   |
| 2026-06-02 |   2 |  99,862,333.225 | -0.14%  |  0.14%   |    15    |    2   |
| 2026-06-02 |   3 |  99,862,333.225 | -0.14%  |  0.14%   |    15    |    0   |
| 2026-06-02 |   4 |  99,862,333.225 | -0.14%  |  0.14%   |    15    |    0   |
| 2026-06-02 |   5 |  99,862,333.225 | -0.14%  |  0.14%   |    15    |    0   |

**Final state (day 5):** day_count=5, cash_balance=3,864,933.22 TWD (3.86%), active holdings=15, halted=false  
Source prices: all 5 entries record `"date": "2026-06-02"` (see Concern below)

---

## Cutover assessment

### What is healthy

- **Holdings = 15 exactly** across all paper days. The `ensemble_top_n=15` truncation (v5's core change) is working: the portfolio enters with exactly 15 names and the two day-2 lot-rounding SELLs reduce share counts but do not drop any stock to zero.
- **Cash allocation = 3.86%** at end — near-fully invested as designed. The water-fill cap overflow is redistributing weight correctly, leaving only rounding residual in cash.
- **Never halted.** `halted` is false throughout; drawdown peaked at 0.14 %, far below the 8 % DRAWDOWN_STOP threshold and the 0.08 check criterion.
- **All paper orders returned `ok: true`** with `resp: "PAPER"`. No rejected submissions.

### Concern: prices are frozen — multi-day P&L is unobservable

All 5 history entries carry `"date": "2026-06-02"` and portfolio value has been flat at 99,862,333.225 for entries 3–5. Five CI runs (June 5 × 2, June 9, June 10, June 11) produced no price movement. The v5-iteration branch does not have the daily CI data-refresh commits that `main` receives (`deploy: state + data YYYY-MM-DD`), so the per-stock CSVs in `RL/data/` remain frozen at 2026-06-02 prices. Each paper run evaluates the portfolio at the same stale quotes, and the model produces identical target weights → no rebalancing. True mark-to-market PnL across the six Taiwan trading days (June 3–11) is not captured here.

This is not evidence that the model is broken; it is evidence that the paper harness on this branch lacks fresh price data. The live deploy on `main` does refresh CSVs daily (via `CSV_STALE_DAYS=7` in `deploy_rl.py`), so this issue would not carry over once the workflow flips to `main --live`.

### Verdict: **CONCERN**

The v5 system mechanics are structurally sound (correct top-N holdings, minimal cash residual, no halt, no significant drawdown), but frozen data means we have zero signal on whether live market prices over the past week would have caused the portfolio to rebalance, drift, or trigger a drawdown. We observed 1 real trading day of price action (2026-06-02 initial deploy + same-day lot adjustments) rather than the ~6 trading days the commit history implies. Proceeding to live without at least one confirmed mark-to-market price update on a trading day carries model-performance uncertainty.

**Recommended pre-cutover check:** manually inspect whether `RL/data/2330.csv` on `main` or in the latest retrain commit has data through at least 2026-06-09 (7-day stale window). If it does, the live CI will refresh prices on first `--live` run and the system will price correctly from day one. If the paper concern is about mechanics only (and walk-forward OOS results in `walk_forward_v4_5seed` already validated alpha), the operator may accept this risk.

---

User must authorize the live cutover — the v5 retrain is already done; only the CI flip to v5 (deploy_rl.yml + the retrain workflow) with --live, plus a push to main, remain.
