# v6 Universe Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Grow the stock universe 46 → ~90 (index-liquid TWSE/OTC names), retrain a 5-seed v6 ensemble, re-tune top-N, move retrain to daily weekdays, and validate v6 beats v5 OOS before a gated cutover.

**Architecture:** No model change — `PortfolioNetLSTM` is per-stock-shared; only `stock_emb` grows by ~44 rows. The work is: source names → fetch data → append `stock_ids` (load-bearing, append-only) → retrain → walk-forward validate → tune top-N → cutover.

**Tech Stack:** Python 3.13 (`.venv`, uv), numpy, PyTorch 2.11 CPU, `stock_api` (TWSE/TPEX fetch). No pytest — tests are standalone `uv run python -m RL.<mod>` scripts. All Python runs prefix `CUDA_VISIBLE_DEVICES=""` (force CPU).

**Branch:** `v6-universe` only. No changes to `main` until the gated cutover (Task 8).

---

### Task 1: Source the ~44 new universe names

**Files:**
- Create: `RL/v6_new_stocks.py` (a reviewable list artifact, imported by Task 2)

- [ ] **Step 1: Fetch authoritative index constituents**

Use WebFetch on the Yuanta 0050 holdings and the FTSE TWSE Mid-Cap (0051 / "0100") holdings to collect candidate codes:
- `https://www.yuantaetfs.com/product/detail/0050/ratio` (Yuanta Taiwan Top-50 holdings)
- `https://www.yuantaetfs.com/product/detail/0051/ratio` (Yuanta Mid-Cap-100 holdings)

Extract every 4-digit stock code shown. If a page can't be parsed, fall back to a TWSE market-cap ranking page (`https://goodinfo.tw/tw/StockList.asp?MARKET_CAT=...`) — the goal is a candidate pool of ~120 large/mid-cap codes.

- [ ] **Step 2: Filter + verify against the symbol map**

Run this to filter candidates to valid, non-duplicate, non-ESB names and pull authoritative names:

```python
# scratch: paste CANDIDATES = ["2379", "3037", ...] from Step 1
from RL.constant import stock_ids
from stock_api.symbols import get_stock_market, get_stock_info
existing = set(stock_ids)
keep = []
for code in CANDIDATES:
    if code in existing:
        continue
    try:
        mkt = get_stock_market(code)            # raises if unknown
    except Exception:
        continue
    if mkt not in ("TWSE", "TPEX"):             # exclude ESB / unknown
        continue
    name = get_stock_info(code).get("name", "")
    keep.append((code, name))
print(len(keep), "candidates")
for c, n in keep:
    print(f'    ("{c}", "{n}"),')
```

Keep the first ~44 (0050-not-in-46 first, then mid-caps by index weight) so the final universe is ~90. Sanity: every kept code is TWSE or TPEX, in the symbol map, not already in `stock_ids`.

- [ ] **Step 3: Write the list artifact**

Create `RL/v6_new_stocks.py`:

```python
"""The ~44 new universe names added in v6 (sourced from TWSE 0050 + mid-cap
0100 holdings, verified TWSE/TPEX-only against stock_symbol_map.json, deduped
against the existing 46). (code, name) pairs; codes append to stock_ids in
order. See docs/superpowers/specs/2026-06-14-v6-universe-expansion-design.md."""

NEW_STOCKS: list[tuple[str, str]] = [
    # paste the verified (code, name) tuples from Step 2 here
]
```

- [ ] **Step 4: Verify the artifact**

Run: `CUDA_VISIBLE_DEVICES="" uv run python -c "from RL.v6_new_stocks import NEW_STOCKS; from RL.constant import stock_ids; s=set(stock_ids); assert all(c not in s for c,_ in NEW_STOCKS); from stock_api.symbols import get_stock_market; assert all(get_stock_market(c) in ('TWSE','TPEX') for c,_ in NEW_STOCKS); print(len(NEW_STOCKS),'new; total',len(stock_ids)+len(NEW_STOCKS))"`
Expected: prints `~44 new; total ~90`, no assertion error.

- [ ] **Step 5: Commit**

```bash
git add RL/v6_new_stocks.py
git commit -m "feat(rl): v6 new universe names (~44 index-liquid TWSE/OTC)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Append new names to `stock_ids` + `fetch_data.py`

**Files:**
- Modify: `RL/constant.py:13-28` (append to `stock_ids`)
- Modify: `RL/fetch_data.py:24` (`STOCKS`) and `:83` (`END_DATE`)

- [ ] **Step 1: Append to `stock_ids`**

In `RL/constant.py`, the list currently ends with:
```python
    # Transport + retail + tech (10)
    "2603", "2609", "2615", "2618", "2912", "5880", "6415", "9921",
    "3231",
]
```
Change the closing to append the new codes (paste the codes from `NEW_STOCKS`, preserving order; group with a comment). Example shape (replace with the real codes):
```python
    # Transport + retail + tech (10)
    "2603", "2609", "2615", "2618", "2912", "5880", "6415", "9921",
    "3231",
    # v6 universe expansion (~44 — 0050 + mid-cap 0100, TWSE/OTC)
    "2379", "3037", "3017", "2357", "4938",  # ... all NEW_STOCKS codes
]
```
**Do NOT reorder or remove any of the existing 46** — `deploy_rl.py:658` compares the full ordered list.

- [ ] **Step 2: Append to `fetch_data.py:STOCKS` + bump `END_DATE`**

In `RL/fetch_data.py`, append the same `(code, name)` tuples from `NEW_STOCKS` to the `STOCKS` list (before the closing `]`), following the existing `("2330", "台積電"),` format. Then change line 83:
```python
END_DATE = "20260422"
```
to today's date:
```python
END_DATE = "20260614"
```

- [ ] **Step 3: Verify imports + lengths line up**

Run: `CUDA_VISIBLE_DEVICES="" uv run python -c "from RL.constant import stock_ids; from RL.fetch_data import STOCKS; sc={c for c,_ in STOCKS}; missing=[s for s in stock_ids if s not in sc]; print('stock_ids',len(stock_ids),'| STOCKS',len(STOCKS),'| stock_ids missing from STOCKS:',missing)"`
Expected: `stock_ids ~90 | STOCKS ~94 | stock_ids missing from STOCKS: []` (every universe code is fetchable; STOCKS may be a small superset).

- [ ] **Step 4: Commit**

```bash
git add RL/constant.py RL/fetch_data.py
git commit -m "feat(rl): expand universe to ~90 (append v6 names; fetch_data END_DATE)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Fetch full price history for the new stocks

**Files:** none (data acquisition — long-running, ~4–5h, batchable)

- [ ] **Step 1: Run the bulk fetcher**

Run (backgroundable; the 35-day coverage skip-logic leaves existing 46 CSVs untouched, fetches only new codes):
```bash
CUDA_VISIBLE_DEVICES="" uv run python -m RL.fetch_data 2>&1 | tee logs/v6_fetch.log
```
Expected: per-stock `=== [i/N] <code> <name> ===` lines; new stocks fetch 2015→today (~6 min each); existing skipped. If TWSE returns HTML blocks, `safe_get_json` backs off and retries; transient failures can be re-run (idempotent).

- [ ] **Step 2: Verify coverage for the new stocks**

Run:
```bash
CUDA_VISIBLE_DEVICES="" uv run python - <<'PY'
import os, pandas as pd
from RL.v6_new_stocks import NEW_STOCKS
miss, short = [], []
for c, n in NEW_STOCKS:
    fp = f"RL/data/{c}.csv"
    if not os.path.exists(fp):
        miss.append(c); continue
    df = pd.read_csv(fp, parse_dates=["date"])
    if len(df) < 250:  # < ~1y of bars
        short.append((c, len(df)))
print("missing CSVs:", miss)
print("short (<250 rows, recent IPOs):", short)
print("ok:", len(NEW_STOCKS) - len(miss))
PY
```
Expected: `missing CSVs: []`. `short` may list genuine recent IPOs (acceptable — they drop from early folds). If any are missing, re-run Step 1 for those codes.

- [ ] **Step 3: Commit the new CSVs**

```bash
git add RL/data/*.csv RL/data/meta.yaml
git commit -m "data(rl): fetch 2015-now history for v6 new universe (~44 stocks)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Code tweaks for N=90 — water-fill `n_iters` + gate band

**Files:**
- Modify: `RL/dl_portfolio.py:100` and `:147` (default `n_iters: int = 8` → `16`)
- Modify: `RL/check_gate.py:36` (active band `[6,20]` → `[6,30]`)
- Test: `RL/test_cap_water_fill.py` (append an N=90 convergence test)

- [ ] **Step 1: Write the failing test**

Append to `RL/test_cap_water_fill.py` before `if __name__ == "__main__":`, and add a call `test_n90_cap_converges()` to the `__main__` sequence:

```python
def test_n90_cap_converges():
    # 90 stocks, concentrated softmax -> overflow must redistribute and the
    # cap must hold tightly with the default n_iters at this universe size.
    rng = np.random.default_rng(90)
    logits = rng.normal(size=(8, 91)) * 3.0   # 90 stocks + cash, peaky
    w = np.exp(logits - logits.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    out = cap_water_fill_np(w, 0.10)           # uses default n_iters
    assert (out[:, :90] <= 0.10 + 1e-9).all(), out[:, :90].max()
    assert np.allclose(out.sum(-1), 1.0, atol=1e-9)
    # residual driven to cash should be tiny when headroom is ample (90*0.10=9.0)
    leaked = out[:, -1].mean() - w[:, -1].mean()
    assert leaked < 0.05, leaked   # < 5% unintended cash from non-convergence
```

Run `CUDA_VISIBLE_DEVICES="" uv run python -m RL.test_cap_water_fill` — it may pass or fail at n_iters=8; either way proceed (the bump sharpens it).

- [ ] **Step 2: Bump `n_iters` default to 16**

In `RL/dl_portfolio.py`, the two function signatures are:
```python
def cap_water_fill_np(w: np.ndarray, cap: float, n_iters: int = 8,
def cap_water_fill_torch(w: torch.Tensor, cap: float, n_iters: int = 8,
```
Change both `n_iters: int = 8` to `n_iters: int = 16`.

- [ ] **Step 3: Widen the gate active band**

In `RL/check_gate.py`, line 36 is:
```python
        ("mean active in [6,20]", 6 <= mean_active <= 20, f"{mean_active:.1f}"),
```
Change to:
```python
        ("mean active in [6,30]", 6 <= mean_active <= 30, f"{mean_active:.1f}"),
```

- [ ] **Step 4: Run tests**

Run: `CUDA_VISIBLE_DEVICES="" uv run python -m RL.test_cap_water_fill`
Expected: `PASS cap_water_fill` (all tests incl. `test_n90_cap_converges`).

- [ ] **Step 5: Commit**

```bash
git add RL/dl_portfolio.py RL/check_gate.py RL/test_cap_water_fill.py
git commit -m "feat(rl): n_iters 8->16 + gate active band [6,30] for N=90 universe

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Trainer seed-naming + workflows — daily weekday retrain + v6 ckpts

**Files:**
- Modify: `RL/dl_train_deploy.py:153-154` (derive per-seed filename from `--out`)
- Modify: `.github/workflows/retrain_v5.yml` → rename to `retrain_v6.yml`, cron, timeout, ckpt name, top-N flag
- Modify: `.github/workflows/deploy_rl.yml` → v6 ckpt paths

- [ ] **Step 1: Make the per-seed filename derive from `--out`**

`RL/dl_train_deploy.py` currently HARDCODES the v5 name (lines 153-154):
```python
        out_path = (
            f"models/dl_v5_seed{seed}.pt" if args.seeds else args.out
        )
```
Change it to derive from `--out` (so `--out models/dl_v6_deploy.pt` → `models/dl_v6_seed{N}.pt`, while the default `--out models/dl_v5_deploy.pt` keeps producing `dl_v5_seed{N}.pt` — backward-compatible):
```python
        out_path = (
            args.out.replace("_deploy.pt", f"_seed{seed}.pt") if args.seeds else args.out
        )
```

- [ ] **Step 2: Verify the naming (no training — just the path logic)**

Run: `CUDA_VISIBLE_DEVICES="" uv run python -c "print('models/dl_v6_deploy.pt'.replace('_deploy.pt', '_seed0.pt'))"`
Expected: `models/dl_v6_seed0.pt`.

- [ ] **Step 3: Rename + edit the retrain workflow**

```bash
git mv .github/workflows/retrain_v5.yml .github/workflows/retrain_v6.yml
```
In `retrain_v6.yml`: change cron (line 10) `"0 1 * * 0"` → `"0 1 * * 1-5"` (daily weekdays); `timeout-minutes: 90` (line 22) → `120`; the train command — add `--ensemble-top-n 15` and `--out models/dl_v6_deploy.pt` before `--seeds` so it reads:
```bash
            --max-weight 0.10 \
            --emb-dim 4 \
            --ensemble-top-n 15 \
            --out models/dl_v6_deploy.pt \
            --seeds 0,1,2,3,4
```
and change `git add models/dl_v5_seed*.pt` (line 57) to `git add models/dl_v6_seed*.pt`, and the commit message's `v5` → `v6`.

- [ ] **Step 4: Point deploy workflow at v6 ckpts**

In `.github/workflows/deploy_rl.yml`, lines 57-61 list `models/dl_v5_seed0.pt … dl_v5_seed4.pt`. Change all five to `models/dl_v6_seed0.pt … dl_v6_seed4.pt`.

- [ ] **Step 5: Verify YAML is well-formed**

Run: `CUDA_VISIBLE_DEVICES="" uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/retrain_v6.yml')); yaml.safe_load(open('.github/workflows/deploy_rl.yml')); print('yaml ok')"`
Expected: `yaml ok`.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/
git commit -m "ci(rl): retrain_v6 daily weekdays + 120min timeout; deploy -> v6 ckpts

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Retrain the v6 deploy ensemble

**Files:** none (produces `models/dl_v6_seed{0..4}.pt`)

- [ ] **Step 1: Train 5 seeds on the ~90 universe**

Run (CPU, ~30–40 min at N=90; backgroundable):
```bash
CUDA_VISIBLE_DEVICES="" uv run python -u -m RL.dl_train_deploy \
  --epochs 500 --train-recent 500 --window 50 --hidden 32 \
  --max-weight 0.10 --emb-dim 4 --ensemble-top-n 15 \
  --out models/dl_v6_deploy.pt --seeds 0,1,2,3,4 2>&1 | tee logs/v6_train.log
```
Expected: `train tensor: torch.Size([440, ~90, 50, 14])`, 5 `saved models/dl_v6_seed{N}.pt` lines.

- [ ] **Step 2: Verify the ckpt configs carry the new universe**

Run:
```bash
CUDA_VISIBLE_DEVICES="" uv run python - <<'PY'
import torch
from RL.constant import stock_ids
for s in range(5):
    c = torch.load(f"models/dl_v6_seed{s}.pt", map_location="cpu", weights_only=False)
    cfg = c["config"]
    assert c["stock_ids"] == stock_ids, "stock_ids mismatch!"
    print(f"seed{s}: num_stocks={cfg['num_stocks']} cap_overflow={cfg.get('cap_overflow')} "
          f"ensemble_top_n={cfg.get('ensemble_top_n')} feat={cfg['feat_per_stock']} "
          f"val_sharpe={c['val_sharpe']:+.3f}")
PY
```
Expected: `num_stocks=~90`, `cap_overflow=waterfill`, `ensemble_top_n=15`, `feat=14`, `stock_ids` matches.

- [ ] **Step 3: Commit the v6 ckpts**

```bash
git add models/dl_v6_seed*.pt models/dl_v6_deploy.pt
git commit -m "retrain(rl): v6 5-seed ensemble on ~90 universe

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Walk-forward validation + top-N tuning

**Files:** none (validation; updates `--ensemble-top-n` if the probe says so)

- [ ] **Step 1: Run the v6 walk-forward (instrumented)**

Run (CPU, ~5h at N=90; backgroundable):
```bash
CUDA_VISIBLE_DEVICES="" uv run python -u -m RL.walk_forward_dl_ensemble \
  --tag v6 --seeds 5 --epochs 500 --window 50 --hidden 32 \
  --train-recent 500 --max-weight 0.10 --ensemble-top-n 15 \
  > logs/v6_walkforward.log 2>&1
```
Expected: writes `walk_forward_v6_results.json` + `walk_forward_v6_cache.pkl`.

- [ ] **Step 2: Sweep top-N offline on the cache**

Run: `CUDA_VISIBLE_DEVICES="" uv run python -m RL.probe_ensemble_topn walk_forward_v6_cache.pkl`
(If the probe's `top_n` grid needs higher values for N=90, edit `RL/probe_ensemble_topn.py` line ~57 `for top_n in (None, 25, 20, 15, 12, 10, 8)` to `(None, 40, 33, 26, 20, 15)` first.)
Expected: a table of mean/median alpha + active per top_n. Pick the top_n with the best median alpha and active-count in `[6,30]`.

- [ ] **Step 3: Check the success bar**

Run:
```bash
CUDA_VISIBLE_DEVICES="" uv run python - <<'PY'
import json, numpy as np
r = json.load(open("walk_forward_v6_results.json"))
a = np.array([x["alpha"] for x in r])
print(f"folds={len(r)} mean_alpha={a.mean()*100:+.2f}% median={np.median(a)*100:+.2f}% "
      f"pos={(a>0).sum()}/{len(a)} std={a.std()*100:.2f}%")
print("BEATS v5 median (+2.71%):", np.median(a) > 0.0271)
PY
```
**Pass if median alpha > +2.71%** (v5's robust figure) with the chosen top_n; record the result in the spec under a `## Result` section and commit.

- [ ] **Step 4: If the chosen top_n ≠ 15, update the configs**

If the probe picked a different top_n: update `--ensemble-top-n <N>` in `.github/workflows/retrain_v6.yml`, re-run Task 6 Step 1 with that value (so the deploy ckpts' config carries it), and re-verify (Task 6 Step 2). Commit. (No re-walk-forward needed — the probe already measured that top_n's OOS alpha from the cache.)

- [ ] **Step 5: Record result + commit**

Append a `## Result (2026-06-..)` section to `docs/superpowers/specs/2026-06-14-v6-universe-expansion-design.md` (mean/median alpha, chosen top_n, pass/fail vs +2.71%, active/cash), then:
```bash
git add docs/ walk_forward_v6_results.json
git commit -m "docs(rl): v6 walk-forward result + chosen top-N

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: Gated cutover to v6 (STOP for human approval)

**Files:** (only on explicit user approval) `main` via a promotion commit

- [ ] **Step 1: Present the result + get authorization**

Surface the Task 7 numbers (median alpha vs +2.71%, active, top_n). Do NOT push to main without explicit user approval — this is live (sim) trading on the production default branch. If the success bar failed, recommend keeping v5 and stop.

- [ ] **Step 2: On approval, build the promotion (mirror the v5 cutover)**

On a branch off current `origin/main`: bring v6 `RL/*.py`, `models/dl_v6_seed*.pt`, `docs/`, the v6 workflows from `v6-universe`; write a fresh `deploy_state.json` (halted=false, cash/peak = the real current broker cash, inventory zeroed — the live `--live` reconcile confirms holdings before acting); keep v4/v5 ckpts for rollback. Verify with a local `--paper` run (active in band, no errors), then push to main on user authorization. Next weekday cron runs v6 `--live`.

---

## Notes for the implementer

- Every Python run forces CPU with `CUDA_VISIBLE_DEVICES=""`.
- `stock_ids` order is append-only and load-bearing — never reorder/remove the existing 46 (the `deploy_rl.py:658` guard invalidates all checkpoints otherwise).
- Tasks 3, 6, 7 are long-running (hours) — run backgrounded and monitor logs.
- All work stays on `v6-universe`; `main`/v4 stay untouched until Task 8 approval.
- Exclude ESB stocks throughout (proxy close, missing open → silent NaN features).
