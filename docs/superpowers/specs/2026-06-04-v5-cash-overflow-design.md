# v5 LSTM — cap-overflow water-fill (conviction-preserving)

Date: 2026-06-04
Branch: `v5-iteration`
Status: design approved, pre-implementation

## Problem

The v5 LSTM (and v4) parks ~48% of the book in cash on bull folds, producing
large negative alpha versus the always-invested equal-weight (EW) benchmark.
This cash is **not** a deliberate hedge — it is mechanical **cap-overflow**.

Mechanism: per-stock scores + a learned `cash_logit` go through a softmax over
`[46 stocks + cash]`, then a cap step clamps each stock to `max_weight = 0.10`
and routes **all excess above the cap into the cash slot**. When the model is
confident it concentrates; softmax wants >0.10 on its top picks; the cap shaves
the excess straight into cash. Concentrate on ~8 names → ~25%+ leaks to cash.
Against a fully-invested EW benchmark this is a structural drag precisely when
the model has conviction.

Ruled out as the lever (proven this session via `RL/sweep_aggregation.py`):
ensemble aggregation (sharpe-weighted vs uniform vs median). Uniform ≈
sharpe-weighted because the val_sharpe spread (0.247–0.362) is too tight to
matter; median is worse. See `walk_forward_<tag>_cache.pkl` replay.

## Goal / success criterion

Target is **relative**: a 5-seed walk-forward ensemble with the water-fill fix
must beat v4's documented OOS alpha.

- Primary gate: **mean alpha > +0.98%** (v4 5-seed median ensemble baseline),
  stretch toward v4 single-seed +4.09%.
- Sanity: mean cash% drops from ~48% into roughly [5%, 40%].
- Robustness: re-run `sweep_aggregation` on the instrumented cache; the result
  should not depend on the aggregation scheme.

Out of scope this round: the strict original gate (≥7/10 positive folds,
mean α≥+4%) — likely unachievable by this architecture family (v4 never hit it).

## Approach

Replace "excess → cash" with **conviction-preserving water-fill**: redistribute
cap-overflow among the still-below-cap stocks **proportional to their current
weight**, so overflow flows to the model's next-favorite names (ranks 11, 12,
13 …) rather than to cash or to a flat EW spread. The model's **deliberate
softmax cash is preserved** — only the mechanical overflow is redirected.

### Core algorithm — `cap_water_fill(w, cap, mask=None)`

`w` has the cash slot at index `-1`; leading axes are batch/time (handles both
1-D `(N+1,)` for live deploy and 2-D `(T, N+1)` for training/OOS).

```
stock = w[..., :N]          # per-stock weights (post-softmax, post per-seed cap)
cash  = w[..., N]           # deliberate cash — preserved
for _ in range(N_ITERS):    # fixed count (8) — deterministic, autograd-safe
    excess = sum_over_stocks(max(stock - cap, 0))     # per batch/time
    stock  = min(stock, cap)                           # clamp capped names
    below  = (stock < cap) & unmasked                  # names with headroom
    pool   = sum_over_stocks(stock * below)            # current weight in 'below'
    stock  = stock + below * excess * stock / pool     # ∝ current weight
                                                        # (guard pool>0)
residual = original_stock_mass - sum_over_stocks(stock) # couldn't fit (rare)
cash = cash + max(residual, 0)
return concat(stock, cash)
```

- Iteration re-enforces the cap: redistributed mass can re-exceed `cap`; the
  next pass clamps it. 8 iterations converges for cap=0.10 over 46 names.
- `pool == 0` (degenerate) → skip redistribution; overflow falls to residual →
  cash. Keeps every row a valid simplex point.
- Masked stocks (no data) stay 0 and never receive overflow.
- Sum is conserved: `sum(stock) + cash == 1` at all times.

### Two implementations, one rule (parity-locked)

Both must produce identical numbers (`max|Δ| < 1e-6` on random input):

- `cap_water_fill_torch` — differentiable, used inside the training/inference
  forward pass. Fixed-count loop; `clamp`/`min`/division only → backprops
  cleanly (subgradient at the cap edge, like ReLU).
- `cap_water_fill_np` — used by the numpy aggregation sites.

Both live in `RL/dl_portfolio.py` (already imported by the other two modules),
so there is a single source of truth.

### Three call sites (must all switch together)

| Site | File / function | dtype | shape |
|---|---|---|---|
| 1 | `dl_portfolio.PortfolioNetLSTM._cap_renorm` | torch | `(T, N+1)` |
| 2 | `walk_forward_dl_ensemble.aggregate_weights` (cap tail) | numpy | `(T_val, N+1)` |
| 3 | `deploy_rl.predict_dl_ensemble_weights` (cap tail) | numpy | `(N+1,)` |

Site 1 applies per-seed inside each net's forward; sites 2 & 3 apply again on
the aggregated weights. The previous outage came from these three drifting out
of sync — a shared function plus a parity test makes divergence
impossible-by-construction.

### Backward compatibility

Add `cap_overflow` to the checkpoint `config` dict:
- `"waterfill"` → new behavior.
- absent / `"cash"` → legacy excess→cash (so any older checkpoint still loads
  and behaves as trained). Append-only config; never reorder.

How each site obtains the flag:
- **Site 1** (`_cap_renorm`): a `cap_overflow` constructor arg on
  `PortfolioNetLSTM`, stored into `config` at train time. Training builds the
  net with `"waterfill"`; `deploy_rl.predict_dl_weights` builds it with
  `cfg.get("cap_overflow", "cash")` so older ckpts replay their trained rule.
- **Site 2** (`walk_forward_dl_ensemble.aggregate_weights`): never loads a
  checkpoint — it is the live experiment, so it takes a `cap_overflow` parameter
  passed `"waterfill"` by `run_fold`.
- **Site 3** (`deploy_rl.predict_dl_ensemble_weights`): reads
  `ckpt["config"].get("cap_overflow", "cash")`, as it already reads
  `max_weight`.

`dl_train_deploy.py` writes `cap_overflow="waterfill"` into new v5 checkpoints.

## Test plan

1. **Unit** (`RL/`): for random `(T, N+1)` softmax rows —
   - every output row sums to 1;
   - every stock weight ≤ cap (+eps);
   - output cash ≥ input deliberate cash;
   - torch ↔ numpy parity `max|Δ| < 1e-6`.
2. **Fabricated deploy↔walk-forward parity** = 0 on a synthetic 5-seed input
   (mirrors the e59620d T1 parity smoke).
3. **Smoke**: 1-fold instrumented walk-forward — confirm cash% drops vs current
   v5 and the run completes + writes the cache pkl.
4. **Acceptance**: 5-seed × 10-fold instrumented walk-forward
   (`--tag v5_waterfill`, CPU, ~2.4h). Pass if mean alpha > +0.98%. Then
   `sweep_aggregation` for aggregation robustness.

## Scope guards

- All work on `v5-iteration`. **No changes to `main`, `deploy_rl.yml`, or
  `retrain_v4.yml`** until acceptance passes.
- `max_weight` stays 0.10 — isolate the water-fill effect; do not also raise the
  cap this round (would confound attribution).
- Keep T1 (sharpe-weighted aggregation) as-is; it is not this round's variable.

## Risks

- **Forced over-diversification**: if overflow always finds headroom, the book
  could drift toward EW. Mitigated by proportional-to-weight redistribution
  (concentrates on next-best names) — monitor mean active_stocks; if it balloons
  toward 46, reconsider.
- **Training instability** from the iterative step. Mitigated by fixed iteration
  count and smooth ops; watch train Sharpe curves on the smoke fold.
- **No alpha lift**: cash may not be the dominant drag. Acceptance gate catches
  this; fallback is the raise-cap probe or revisiting the feature set.

## Result (2026-06-04)

5-seed × 10-fold walk-forward, `--tag v5_waterfill`, deployed sharpe-weighted aggregation:

- **mean alpha = +4.27%** (beats v4 5-seed +0.98% and v4 single-seed +4.09% — the chosen target). median alpha = +2.63%. 6/10 positive. std(alpha) = 18.0%.
- mean cash = 2.5%, mean active = 25.9.
- Cash drag eliminated (pre-fix v5 ~48% → 2.5%); failing-v5 mean alpha ~−1.9% → +4.27%.
- Aggregation-robust: uniform +4.09%, median +5.41% (all beat +0.98%).

**Caveats (do not overclaim):**
- The +4.27% mean is dominated by fold 4 (+53.7%); excluding it, mean = −1.21%. The robust central number is the median, +2.63% — beats v4 5-seed but not single-seed.
- Near-zero cash (2.5%, below the [5,40] design band) means the book is ~always fully invested — no defensive cash. `active` drifted 16→26 (more EW-like). Water-fill over-corrected slightly.
- The strict original gate still fails (6/10 < 7, std 18% > 8%, active > 20, cash < 5%) — but that gate was explicitly out of scope this round.

**Verdict:** passes the chosen "beat v4 OOS alpha" bar on mean and median; genuine improvement, but one-fold-dependent and high-variance. Ship decision deferred to human review. Artifacts: `walk_forward_v5_waterfill_results.json`, `walk_forward_v5_waterfill_cache.pkl`, `logs/v5_waterfill_wf.log`.

## Refinement: top-K concentration (2026-06-05)

The water-fill result over-diversified (active 16→26, cash 2.5%). Refinement:
redistribute cap-overflow only to the **top-K highest-weighted below-cap
stocks**, leaving the tail at ~0, to pull `active` back into [6,20] while
keeping the cash-drag fix.

- `cap_water_fill_{np,torch}` gain `top_k: int | None = None` (None = current
  all-below-cap behavior, backward-compat). The recipient set is the top-K
  below-cap stocks by **initial** weight, fixed across iterations; overflow that
  the fixed K cannot absorb (all at cap) falls to cash (residual). This bounds
  the active set at ~(initially-capped + K).
- New config key `cap_top_k` (default None), plumbed exactly like `cap_overflow`
  through the 3 cap sites + train + deploy. Experiment uses **K=10**, exposed as
  `--cap-top-k` on `dl_train_deploy` and `walk_forward_dl_ensemble`.
- Validation: smoke (active drops toward band) → 5-seed×10-fold acceptance
  (`--tag v5_topk10`). Keep if mean alpha still beats v4 (+0.98%) with active in
  [6,20]; if mean alpha drops below v4, revert to plain water-fill.
