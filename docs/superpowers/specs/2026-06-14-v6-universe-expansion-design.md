# v6 — universe expansion (~90) + daily retrain

Date: 2026-06-14
Branch: `v6-universe` (off `v5-iteration` = v5 water-fill + top-N code)
Status: design approved, pre-implementation

## Context

v5 (water-fill cap + ensemble top-N=15 over **46** stocks) is the best model so
far — walk-forward mean alpha +6.96% / median +2.71% / 7-of-10 positive vs v4
+0.98%. But CLAUDE.md flags the empirical ceiling: *"Don't expect more from this
architecture without changing the feature set or universe."* Meanwhile v4 in
production hit its drawdown stop and halted (down ~25%). The user wants to break
the ceiling by **enlarging the stock universe** (more + better choices) and
**retraining more often**. This spec covers v6: ~90-stock universe, daily
weekday retrain, top-N re-tuned for the bigger pool. No architecture change —
the LSTM is per-stock-shared; only `stock_emb` grows.

## Goals / success criterion

- **Primary:** v6 5-seed walk-forward **median alpha (vs the 90-stock EW)** beats
  v5's +2.71%, ideally approaches the +6.96% mean. (The EW benchmark changes with
  the universe, so this is not a raw apples-to-apples number vs v5's 46-stock
  figures — it's "does the bigger pool let the model extract more edge.")
- **Sanity:** active-stock count and cash% stay sane for N=90; no halts in a
  paper/cutover check.
- **Cadence:** retrain runs daily on weekdays without hitting the per-job timeout.

## Design

### 1. Universe (~90, append-only)

- Source names from **authoritative TWSE 0050 + 0100 mid-cap** index holdings
  (WebFetch the ETF constituent lists at implementation time). Keep **TWSE + OTC
  (TPEX) only — exclude ESB** (proxy close, missing open price → silent NaN in
  features; `stock_api` flags `close_is_proxy`).
- Verify every code exists in `stock_api/stock_symbol_map.json` (2,565 symbols)
  or add it; `get_stock_market` raises on unknown codes.
- Dedup against the current 46; take the most-liquid new names to reach ~90.
- **Append** to `RL/constant.py:stock_ids` in new sector-commented blocks. NEVER
  reorder the existing 46 — `deploy_rl.py:658` (`if saved_ids != stock_ids`)
  compares the full ordered list and invalidates every checkpoint on any change.
  (A fresh full retrain is required and expected.)

### 2. Data acquisition

- Add the ~44 new `(code, name)` tuples to `RL/fetch_data.py:STOCKS`, set
  `END_DATE` = today, run `python -m RL.fetch_data`.
- Fetches full 2015→now history into `RL/data/<sid>.csv` (10-col schema). Cost
  ~6 min/stock (2.0s sleep × ~138 months) → ~4–5h for ~44 stocks; batchable.
  The 35-day coverage skip-logic leaves the existing 46 CSVs untouched.
- Stocks listed after 2015 get fewer early-fold samples and are silently dropped
  from folds where `len(df) ≤ 30` (in `walk_forward_dl.load_data`). Production
  retrain only needs `--start 20200101`. Acceptable.

### 3. Code changes (small — config + data, no architecture)

- `RL/constant.py`: append ~44 SIDs.
- `RL/dl_portfolio.py`: bump water-fill default `n_iters` 8 → 16 in
  `cap_water_fill_np`/`cap_water_fill_torch` (redistribution accuracy at N=90 with
  cap 0.10; the post-loop structural clamp already guarantees the cap, so this
  only sharpens redistribution, and is a no-op at N=46).
- `RL/check_gate.py`: widen the `active in [6,20]` band (e.g. `[6,30]`) for the
  bigger universe.
- `.github/workflows/`: retrain cron → `0 1 * * 1-5` (daily weekdays),
  `timeout-minutes: 120`, train command carries `--ensemble-top-n <tuned>`;
  deploy yaml points at the v6 ckpts. (Repo is PUBLIC → unlimited Actions
  minutes; only the per-job timeout binds, and N=90 retrain ~60–80 min fits.)
- Note (not blocking): `deploy_rl.py:232` / `live_dryrun.py:77` hardcode
  `[0.0]*12` — these are on the **legacy PPO** obs path, not the DL deploy path
  (which reads `feat_cols` from the checkpoint), so they don't affect v6.

### 4. top-N ("more choices") — empirically tuned, no retrain

After the v6 walk-forward writes its cache, run `RL/probe_ensemble_topn.py` on it
sweeping `top_n ∈ {15, 20, 26, 33, 40}` (biased higher per the "more choices"
intent — at N=90, top_n=15 is only 17% of the pool). Pick the best
mean/median-alpha point with active-count in band. Set that as `--ensemble-top-n`
in the retrain workflow and re-emit the deploy ckpts' config (the probe is a pure
post-aggregation replay — see the v5 precedent — so it needs no retrain).

### 5. Retrain + re-validate

- `dl_train_deploy --seeds 0,1,2,3,4 --epochs 500 --train-recent 500 --window 50
  --hidden 32 --max-weight 0.10 --emb-dim 4 --ensemble-top-n <tuned>` → 5-seed v6
  ckpts (`models/dl_v6_seed{0..4}.pt`), config carries the new `stock_ids`,
  `num_stocks`, `cap_overflow=waterfill`, tuned `ensemble_top_n`.
- `walk_forward_dl_ensemble --tag v6 ...` (~5h at N=90, CPU) → OOS alpha vs the
  90-stock EW + the cache for the top-N probe.

### 6. Cutover (gated)

Same consistent-set discipline as v5: promote v6 code + ckpts + workflows + a
fresh real-state ledger to main only on explicit user authorization, after the
walk-forward clears the success bar. v4/v5 ckpts kept for rollback.

## Phasing

1. **Universe + data** (source names → fetch CSVs, ~4–5h) — the long pole; can
   run overnight/batched.
2. **Code** (constant.py, n_iters, check_gate, workflows) — quick, parallel with (1).
3. **Retrain v6 + walk-forward** (~5h) → success bar + top-N probe.
4. **Cutover** — gated.

## Risks

- **New mid-caps carry no exploitable alpha** → v6 ≈ v5 or worse. The walk-forward
  is the gate; if it doesn't clear, keep v5's 46-stock universe.
- **Data-fetch flakiness** (TWSE HTML blocks / rate-limit) — `fetch_data.py` has
  backoff; batch + re-run, the skip-logic is idempotent.
- **Daily retrain churn** — daily ckpt commits to main; acceptable on a public
  repo, but watch that a bad fetch doesn't poison a retrain (the deploy reconciles
  + the drawdown stop remains the backstop).
- **Retrain timeout at N=90** — ~60–80 min vs 120-min budget; if it creeps,
  drop epochs (500→400) or seeds, or move to a self-hosted runner.

## Result (2026-06-15)

Universe expanded 46 → 88 (43 index-liquid names; 6446 dropped — 2024 IPO collapsed the train window). Two `build_tensors` date-intersection bugs fixed (universe drop + load_data cover-from-start guard). The intersection design does not scale cleanly past ~46 stocks (compounding per-stock data gaps); folds 7–9 (recent) could not be built — validated on **folds 0–6 (7 folds)** only. Proper fix (calendar-reindex + ffill / union+mask in build_tensors) deferred.

5-seed walk-forward, folds 0–6, top-N sweep (offline probe on the cache):

| top_n | mean α | median α | pos | active |
|---|---|---|---|---|
| 15 | +3.84% | −0.01% | 3/7 | 15 |
| 12 | +5.22% | +2.97% | 4/7 | 12 |
| **10** | **+6.11%** | **+5.65%** | 4/7 | 10 |
| none | −0.06% | −1.42% | 3/7 | 34 |

**Finding:** the bigger pool helps only with *stronger* concentration — **top_n=10** is best (median +5.65% beats v5's +2.71% / +2.6% on matched folds). Higher N (15–34) drags in marginal names and underperforms. So "more choices" = bigger candidate pool, **fewer** held positions.

**Caveats:** 7-fold (not 10) — recent 2022–2023 folds unvalidated; fold 4 (+52%) still lifts the mean (median is the robust figure); std ~24%. Not airtight vs v5's 10-fold numbers.

**Status:** v6 @ top_n=10 is a genuine improvement on the validatable folds, but ships only after a date-handling fix lets the full 10 folds validate (or via a paper period). Cutover gated on user. v6 deploy ckpts currently carry ensemble_top_n=15 — must be set to 10 before any cutover.
