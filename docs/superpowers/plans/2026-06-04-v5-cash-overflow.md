# v5 Cap-Overflow Water-Fill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the cap-overflow→cash rule with conviction-preserving water-fill so the 5-seed walk-forward ensemble beats v4's +0.98% OOS alpha.

**Architecture:** One shared `cap_water_fill` (numpy + torch, parity-tested) in `RL/dl_portfolio.py`, wired into all three cap sites (`PortfolioNetLSTM._cap_renorm`, `walk_forward_dl_ensemble.aggregate_weights`, `deploy_rl.predict_dl_ensemble_weights`), gated by a `config.cap_overflow` flag (absent → legacy cash) for checkpoint backward-compat.

**Tech Stack:** Python 3.13 (`.venv`, uv), numpy, PyTorch 2.11 CPU. No pytest — tests are standalone `uv run python -m RL.<test>` scripts that print `PASS` / raise on failure. All commands prefix `CUDA_VISIBLE_DEVICES=""` to force CPU.

**Branch:** `v5-iteration` only. No changes to `main`, `deploy_rl.yml`, or `retrain_v4.yml`.

---

### Task 1: Core `cap_water_fill_np` / `cap_water_fill_torch` + unit tests

**Files:**
- Modify: `RL/dl_portfolio.py` (add two module-level functions after `sparsemax`/before `PortfolioNetLSTM`, near line 160)
- Test: `RL/test_cap_water_fill.py` (create)

- [ ] **Step 1: Write the failing test**

Create `RL/test_cap_water_fill.py`:

```python
"""Standalone tests for cap_water_fill (no pytest). Run:
    CUDA_VISIBLE_DEVICES="" uv run python -m RL.test_cap_water_fill
"""
import numpy as np
import torch

from RL.dl_portfolio import cap_water_fill_np, cap_water_fill_torch


def _softmax_rows(shape, seed):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=shape)
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _concentrated(n_stocks=20, cash=0.10, spike=0.30, seed=0):
    """Feasible (n_stocks + cash) simplex with one stock above a 0.10 cap.
    n_stocks must be large enough that n_stocks*cap >> invested mass, so
    water-fill has headroom to absorb overflow without falling back to cash."""
    rng = np.random.default_rng(seed)
    rest = rng.dirichlet(np.ones(n_stocks - 1)) * (1.0 - cash - spike)
    stocks = np.concatenate([[spike], rest])
    return np.concatenate([stocks, [cash]])[None, :]  # (1, n_stocks+1)


def test_simplex_and_cap():
    w = _softmax_rows((7, 47), 0)            # 46 stocks + cash
    out = cap_water_fill_np(w, 0.10)
    n = w.shape[-1] - 1
    assert np.allclose(out.sum(-1), 1.0, atol=1e-9), out.sum(-1)
    assert (out[..., :n] <= 0.10 + 1e-9).all(), out[..., :n].max()
    # deliberate cash preserved or grown (never shrunk)
    assert (out[..., -1] + 1e-12 >= w[..., -1]).all()


def test_no_overflow_unchanged():
    # 10 stocks + cash, each 1/11 < cap -> nothing to redistribute
    w = np.full((1, 11), 1.0 / 11.0)
    out = cap_water_fill_np(w, 0.10)
    assert np.allclose(out, w, atol=1e-9), out


def test_overflow_goes_to_stocks_not_cash():
    # one stock at 0.30 over a 0.10 cap, 20 stocks -> ample headroom
    w = _concentrated(n_stocks=20, cash=0.10, spike=0.30, seed=5)
    out = cap_water_fill_np(w, 0.10)
    assert (out[0, :20] <= 0.10 + 1e-9).all(), out[0, :20].max()
    assert abs(out[0, -1] - 0.10) < 1e-9, out[0, -1]   # deliberate cash preserved
    assert abs(out.sum() - 1.0) < 1e-9


def test_torch_numpy_parity():
    w = _softmax_rows((5, 47), 1)
    a = cap_water_fill_np(w, 0.10)
    b = cap_water_fill_torch(torch.from_numpy(w), 0.10).numpy()
    assert np.abs(a - b).max() < 1e-6, np.abs(a - b).max()


def test_torch_differentiable():
    w = torch.tensor(_softmax_rows((3, 47), 2), requires_grad=True)
    out = cap_water_fill_torch(w, 0.10)
    out.sum().backward()
    assert w.grad is not None and torch.isfinite(w.grad).all()


if __name__ == "__main__":
    test_simplex_and_cap()
    test_no_overflow_unchanged()
    test_overflow_goes_to_stocks_not_cash()
    test_torch_numpy_parity()
    test_torch_differentiable()
    print("PASS cap_water_fill")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" uv run python -m RL.test_cap_water_fill`
Expected: FAIL — `ImportError: cannot import name 'cap_water_fill_np'`

- [ ] **Step 3: Write minimal implementation**

In `RL/dl_portfolio.py`, add after the `sparsemax` function (just before `class PortfolioNetLSTM`):

```python
def cap_water_fill_np(w: np.ndarray, cap: float, n_iters: int = 8) -> np.ndarray:
    """Redistribute per-stock weight above `cap` to below-cap stocks,
    proportional to their current weight (conviction-preserving). The cash
    slot at index -1 is preserved; only residual overflow that cannot fit
    under the cap falls to cash.

    w: (..., N+1) rows summing to 1 (stocks + cash). Returns same shape with
    every stock <= cap, rows still summing to 1, output cash >= input cash.
    Masked / zero-weight stocks receive nothing (share ∝ weight = 0).
    """
    w = np.asarray(w, dtype=np.float64).copy()
    stock = w[..., :-1]
    cash = w[..., -1:]
    stock_mass = stock.sum(axis=-1, keepdims=True)
    for _ in range(n_iters):
        excess = np.maximum(stock - cap, 0.0).sum(axis=-1, keepdims=True)
        if not np.any(excess > 1e-12):
            break
        stock = np.minimum(stock, cap)
        below = stock < cap
        pool = (stock * below).sum(axis=-1, keepdims=True)
        safe_pool = np.where(pool > 0, pool, 1.0)
        add = np.where(below & (pool > 0), excess * stock / safe_pool, 0.0)
        stock = stock + add
    residual = np.maximum(stock_mass - stock.sum(axis=-1, keepdims=True), 0.0)
    return np.concatenate([stock, cash + residual], axis=-1)


def cap_water_fill_torch(w: torch.Tensor, cap: float, n_iters: int = 8) -> torch.Tensor:
    """Differentiable torch twin of cap_water_fill_np (fixed iteration count).

    Same rule; backprops via clamp/min/division (subgradient at the cap edge).
    """
    stock = w[..., :-1]
    cash = w[..., -1:]
    stock_mass = stock.sum(dim=-1, keepdim=True)
    for _ in range(n_iters):
        excess = torch.clamp(stock - cap, min=0.0).sum(dim=-1, keepdim=True)
        stock = torch.clamp(stock, max=cap)
        below = (stock < cap).to(stock.dtype)
        pool = (stock * below).sum(dim=-1, keepdim=True)
        safe_pool = torch.where(pool > 0, pool, torch.ones_like(pool))
        add = below * excess * stock / safe_pool
        add = torch.where(pool > 0, add, torch.zeros_like(add))
        stock = stock + add
    residual = torch.clamp(stock_mass - stock.sum(dim=-1, keepdim=True), min=0.0)
    return torch.cat([stock, cash + residual], dim=-1)
```

Note: `np` and `torch` are already imported at the top of `RL/dl_portfolio.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `CUDA_VISIBLE_DEVICES="" uv run python -m RL.test_cap_water_fill`
Expected: `PASS cap_water_fill`

- [ ] **Step 5: Commit**

```bash
git add RL/dl_portfolio.py RL/test_cap_water_fill.py
git commit -m "feat(rl): cap_water_fill (np+torch) conviction-preserving overflow"
```

---

### Task 2: Wire site 1 — `PortfolioNetLSTM._cap_renorm` branch on `cap_overflow`

**Files:**
- Modify: `RL/dl_portfolio.py` — `PortfolioNetLSTM.__init__` (lines 169-186), `_cap_renorm` (lines 207-213)
- Test: `RL/test_cap_water_fill.py` (append a net-level test)

- [ ] **Step 1: Write the failing test**

Append to `RL/test_cap_water_fill.py` (before the `__main__` block, and add its call inside `__main__`):

```python
def test_net_cap_renorm_branch():
    from RL.dl_portfolio import PortfolioNetLSTM
    # 20 stocks + cash; one stock at 0.30 (over cap), deliberate cash = 0.10
    w = torch.tensor(_concentrated(n_stocks=20, cash=0.10, spike=0.30, seed=6))
    net_cash = PortfolioNetLSTM(num_stocks=20, feat_per_stock=14,
                                max_weight=0.10, cap_overflow="cash")
    net_wf = PortfolioNetLSTM(num_stocks=20, feat_per_stock=14,
                              max_weight=0.10, cap_overflow="waterfill")
    out_cash = net_cash._cap_renorm(w)
    out_wf = net_wf._cap_renorm(w)
    # legacy dumps the ~0.20 overflow to cash -> cash > 0.10
    assert out_cash[0, -1].item() > 0.10 + 1e-6, out_cash[0, -1].item()
    # water-fill keeps deliberate cash ~0.10
    assert abs(out_wf[0, -1].item() - 0.10) < 1e-6, out_wf[0, -1].item()
    assert (out_wf[0, :20] <= 0.10 + 1e-9).all()
```

Add `test_net_cap_renorm_branch()` to the `__main__` sequence.

- [ ] **Step 2: Run test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" uv run python -m RL.test_cap_water_fill`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'cap_overflow'`

- [ ] **Step 3: Write minimal implementation**

In `RL/dl_portfolio.py`, change `__init__` signature (line 169-170) from:

```python
    def __init__(self, num_stocks: int, feat_per_stock: int = 14,
                 window_len: int = 50, hidden: int = 64, max_weight: float = 0.10,
                 use_sparsemax: bool = False, emb_dim: int = 4):
```
to:
```python
    def __init__(self, num_stocks: int, feat_per_stock: int = 14,
                 window_len: int = 50, hidden: int = 64, max_weight: float = 0.10,
                 use_sparsemax: bool = False, emb_dim: int = 4,
                 cap_overflow: str = "cash"):
```

Add after `self.use_sparsemax = use_sparsemax` (line 178):

```python
        self.cap_overflow = cap_overflow  # "cash" (legacy) | "waterfill"
```

Replace `_cap_renorm` (lines 207-213) with:

```python
    def _cap_renorm(self, w: torch.Tensor) -> torch.Tensor:
        if self.cap_overflow == "waterfill":
            return cap_water_fill_torch(w, self.max_weight)
        N = self.N
        stock = w[:, :N]
        cash = w[:, N:]
        capped = torch.clamp(stock, max=self.max_weight)
        excess = (stock - capped).sum(dim=1, keepdim=True)
        return torch.cat([capped, cash + excess], dim=1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `CUDA_VISIBLE_DEVICES="" uv run python -m RL.test_cap_water_fill`
Expected: `PASS cap_water_fill`

- [ ] **Step 5: Commit**

```bash
git add RL/dl_portfolio.py RL/test_cap_water_fill.py
git commit -m "feat(rl): PortfolioNetLSTM cap_overflow flag (cash|waterfill)"
```

---

### Task 3: Wire site 2 — `aggregate_weights` branch + `run_fold` passes waterfill

**Files:**
- Modify: `RL/walk_forward_dl_ensemble.py` — `aggregate_weights` (lines 32-78), `run_fold` call site (lines 149-151)
- Test: `RL/test_cap_water_fill.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `RL/test_cap_water_fill.py` (and call it in `__main__`):

```python
def test_aggregate_weights_branch():
    from RL.walk_forward_dl_ensemble import aggregate_weights
    # 2 seeds, 1 timestep, 20 stocks + cash. Both spike stock 0 over cap.
    s0 = _concentrated(n_stocks=20, cash=0.10, spike=0.30, seed=7)  # (1, 21)
    s1 = _concentrated(n_stocks=20, cash=0.10, spike=0.28, seed=8)
    cash_out = aggregate_weights([s0, s1], cap=0.10, cap_overflow="cash")
    wf_out = aggregate_weights([s0, s1], cap=0.10, cap_overflow="waterfill")
    assert (wf_out[:, :20] <= 0.10 + 1e-9).all()
    assert np.allclose(wf_out.sum(-1), 1.0)
    # water-fill parks less in cash than legacy (averaged spike ~0.29 > cap)
    assert wf_out[0, -1] < cash_out[0, -1] - 1e-6, (wf_out[0, -1], cash_out[0, -1])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" uv run python -m RL.test_cap_water_fill`
Expected: FAIL — `TypeError: aggregate_weights() got an unexpected keyword argument 'cap_overflow'`

- [ ] **Step 3: Write minimal implementation**

In `RL/walk_forward_dl_ensemble.py`, add the import at the top (after the existing `from RL.dl_portfolio import ...` line 22):

```python
from RL.dl_portfolio import cap_water_fill_np
```
(Merge into the existing import line: `from RL.dl_portfolio import cap_water_fill_np, realized_returns, train_one_fold_lstm`.)

Change `aggregate_weights` signature (lines 32-36) to add the param:

```python
def aggregate_weights(
    per_seed_weights: list[np.ndarray],
    per_seed_scores: list[float] | None = None,
    cap: float = 0.10,
    cap_overflow: str = "cash",
) -> np.ndarray:
```

Replace the cap tail (lines 73-78, from `N = agg.shape[1] - 1` through `return agg`) with:

```python
    if cap_overflow == "waterfill":
        return cap_water_fill_np(agg, cap)
    N = agg.shape[1] - 1
    stocks = agg[:, :N]
    excess = np.maximum(stocks - cap, 0.0).sum(axis=1)  # (T,)
    agg[:, :N] = np.minimum(stocks, cap)
    agg[:, -1] = agg[:, -1] + excess
    return agg
```

Update the `run_fold` call site (lines 149-151) from:

```python
    agg = aggregate_weights(
        per_seed_w, per_seed_scores=per_seed_val_sharpe, cap=max_weight
    )
```
to:
```python
    agg = aggregate_weights(
        per_seed_w, per_seed_scores=per_seed_val_sharpe, cap=max_weight,
        cap_overflow="waterfill",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `CUDA_VISIBLE_DEVICES="" uv run python -m RL.test_cap_water_fill`
Expected: `PASS cap_water_fill`

- [ ] **Step 5: Commit**

```bash
git add RL/walk_forward_dl_ensemble.py RL/test_cap_water_fill.py
git commit -m "feat(rl): walk-forward aggregate_weights water-fill cap tail"
```

---

### Task 4: Wire site 3 — deploy reads `cap_overflow` from ckpt config + parity test

**Files:**
- Modify: `RL/deploy_rl.py` — `predict_dl_weights` net build (lines 671-680), `predict_dl_ensemble_weights` cap tail (lines 627-632)
- Test: `RL/test_cap_water_fill.py` (append parity test)

- [ ] **Step 1: Write the failing test**

Append to `RL/test_cap_water_fill.py` (and call it in `__main__`). This reimplements the deploy aggregation math locally and asserts it equals `aggregate_weights` on the same fabricated single-timestep input — catching formula drift between sites 2 and 3:

```python
def test_deploy_walkforward_parity():
    from RL.walk_forward_dl_ensemble import aggregate_weights
    rng = np.random.default_rng(3)
    seeds = [rng.dirichlet(np.ones(47))[None, :] for _ in range(5)]  # (1, N+1)
    scores = [0.21, 0.34, 0.29, 0.30, 0.25]
    wf = aggregate_weights(seeds, per_seed_scores=scores, cap=0.10,
                           cap_overflow="waterfill")[0]
    # deploy-side math: same sharpe-weight, clip, renorm, water-fill
    stacked = np.stack([s[0] for s in seeds])
    sc = np.asarray(scores)
    ws = np.exp(sc - sc.max()); ws /= ws.sum()
    agg = (stacked * ws[:, None]).sum(0)
    agg = np.clip(agg, 0.0, None); agg /= agg.sum()
    deploy = cap_water_fill_np(agg, 0.10)
    assert np.abs(wf - deploy).max() < 1e-9, np.abs(wf - deploy).max()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" uv run python -m RL.test_cap_water_fill`
Expected: FAIL — assertion on `np.abs(wf - deploy).max()` (deploy site still uses excess→cash, so the cap tails differ).

- [ ] **Step 3: Write minimal implementation**

In `RL/deploy_rl.py`, `predict_dl_weights` net build — change the `PortfolioNetLSTM(...)` constructor (currently ends with `emb_dim=cfg.get("emb_dim", 4),`) to also pass the flag:

```python
    net = PortfolioNetLSTM(
        num_stocks=N,
        feat_per_stock=F,
        window_len=L,
        hidden=cfg["hidden"],
        max_weight=cfg["max_weight"],
        use_sparsemax=cfg.get("use_sparsemax", False),
        emb_dim=cfg.get("emb_dim", 4),
        cap_overflow=cfg.get("cap_overflow", "cash"),
    )
```

Add the import near the top of `RL/deploy_rl.py` where `PortfolioNetLSTM` is imported:

```python
from RL.dl_portfolio import cap_water_fill_np
```
(Merge with the existing `from RL.dl_portfolio import ...` line if present; otherwise add it.)

In `predict_dl_ensemble_weights`, capture the flag when reading ckpt config. Where it currently does:

```python
        if cap is None:
            cap = float(ckpt["config"].get("max_weight", 0.10))
```
change to:
```python
        if cap is None:
            cap = float(ckpt["config"].get("max_weight", 0.10))
            cap_overflow = ckpt["config"].get("cap_overflow", "cash")
```
and initialise `cap_overflow = "cash"` next to the `cap = None` line above the loop.

Replace the cap tail (lines 627-631, from `N = len(stock_ids)` through `agg[N] += excess`) with:

```python
    N = len(stock_ids)
    if cap_overflow == "waterfill":
        agg = cap_water_fill_np(agg, cap)
    else:
        excess = float(np.sum(np.maximum(agg[:N] - cap, 0.0)))
        agg[:N] = np.minimum(agg[:N], cap)
        agg[N] += excess
```

(The existing `print(...)` and `return agg` after this block stay unchanged.)

- [ ] **Step 4: Run test to verify it passes**

Run: `CUDA_VISIBLE_DEVICES="" uv run python -m RL.test_cap_water_fill`
Expected: `PASS cap_water_fill`

- [ ] **Step 5: Commit**

```bash
git add RL/deploy_rl.py RL/test_cap_water_fill.py
git commit -m "feat(rl): deploy reads cap_overflow from ckpt config + parity test"
```

---

### Task 5: Train writes `cap_overflow="waterfill"` into config + both net builds

**Files:**
- Modify: `RL/dl_train_deploy.py` — net build (lines 112-120), config dict (lines 155-163)
- Modify: `RL/dl_portfolio.py` — `train_one_fold_lstm` net build (lines 418-426)

- [ ] **Step 1: Update `dl_train_deploy.py` net build**

Change the `PortfolioNetLSTM(...)` call (lines 112-120) to add `cap_overflow="waterfill"`:

```python
        net = PortfolioNetLSTM(
            num_stocks=len(stock_ids),
            feat_per_stock=feat_per_stock,
            window_len=args.window,
            hidden=args.hidden,
            max_weight=args.max_weight,
            use_sparsemax=False,
            emb_dim=args.emb_dim,
            cap_overflow="waterfill",
        ).to(device)
```

- [ ] **Step 2: Update `dl_train_deploy.py` config dict**

Add `"cap_overflow": "waterfill",` inside the `"config"` dict (after `"emb_dim": args.emb_dim,`, line 162):

```python
            "config": {
                "num_stocks": len(stock_ids),
                "feat_per_stock": feat_per_stock,
                "window_len": args.window,
                "hidden": args.hidden,
                "max_weight": args.max_weight,
                "use_sparsemax": False,
                "emb_dim": args.emb_dim,
                "cap_overflow": "waterfill",
            },
```

- [ ] **Step 3: Update `train_one_fold_lstm` net build in `dl_portfolio.py`**

Change the `PortfolioNetLSTM(...)` call (lines 418-426) to add `cap_overflow="waterfill"`:

```python
    net = PortfolioNetLSTM(
        num_stocks=len(stock_ids),
        feat_per_stock=tr["feats"].shape[-1],
        window_len=window_len,
        hidden=hidden,
        max_weight=max_weight,
        use_sparsemax=use_sparsemax,
        cap_overflow="waterfill",
    ).to(device)
```

- [ ] **Step 4: Verify everything still imports + tests pass**

Run: `CUDA_VISIBLE_DEVICES="" uv run python -m RL.test_cap_water_fill`
Expected: `PASS cap_water_fill`

Run: `CUDA_VISIBLE_DEVICES="" uv run python -c "import RL.dl_train_deploy, RL.deploy_rl, RL.walk_forward_dl_ensemble; print('imports OK')"`
Expected: `imports OK`

- [ ] **Step 5: Commit**

```bash
git add RL/dl_train_deploy.py RL/dl_portfolio.py
git commit -m "feat(rl): train v5 with cap_overflow=waterfill, persist in config"
```

---

### Task 6: Smoke validation — 1-fold walk-forward, confirm cash drops

**Files:** none (validation run)

- [ ] **Step 1: Run a 1-fold instrumented walk-forward**

Run:
```bash
CUDA_VISIBLE_DEVICES="" uv run python -u -m RL.walk_forward_dl_ensemble \
  --tag v5_wf_smoke --seeds 5 --epochs 500 --window 50 --hidden 32 \
  --train-recent 500 --max-weight 0.10 --fold-start 0 --fold-end 1 \
  2>&1 | tee logs/v5_wf_smoke.log
```
Expected: completes (~12 min), prints `Fold 0 ENSEMBLE: ... cash=XX.X%`, writes `walk_forward_v5_wf_smoke_results.json` + `walk_forward_v5_wf_smoke_cache.pkl`.

- [ ] **Step 2: Confirm the fix moved cash down**

Compare fold-0 `cash=` against the pre-fix v5 fold-0 (`54.7%`, from `logs/v5_smoke.log` earlier). Expected: materially lower (target trend toward [5,40]%). If cash is unchanged, STOP — the wiring didn't take; recheck Tasks 2-5 before the expensive run.

- [ ] **Step 3: Sanity-check active count didn't balloon to ~46**

In the same line, `active=` should stay in a sane band (roughly 6-25), not collapse to near-46 (which would mean drift to EW). If it ballooned, note it for the acceptance review.

---

### Task 7: Acceptance validation — 5-seed × 10-fold, beat +0.98%

**Files:** none (validation run)

- [ ] **Step 1: Run the full instrumented walk-forward (~2.4h CPU)**

Run (background-capable; redirect to a log):
```bash
CUDA_VISIBLE_DEVICES="" uv run python -u -m RL.walk_forward_dl_ensemble \
  --tag v5_waterfill --seeds 5 --epochs 500 --window 50 --hidden 32 \
  --train-recent 500 --max-weight 0.10 > logs/v5_waterfill_wf.log 2>&1
```
Expected: writes `walk_forward_v5_waterfill_results.json` + `walk_forward_v5_waterfill_cache.pkl`.

- [ ] **Step 2: Check the acceptance bar**

Run:
```bash
CUDA_VISIBLE_DEVICES="" uv run python - <<'PY'
import json, numpy as np
r = json.load(open("walk_forward_v5_waterfill_results.json"))
a = np.array([x["alpha"] for x in r])
c = np.array([x["agg_cash_avg"] for x in r])
print(f"folds={len(r)} mean_alpha={a.mean()*100:+.2f}% pos={(a>0).sum()}/{len(a)} "
      f"std={a.std()*100:.2f}% mean_cash={c.mean()*100:.1f}%")
print("BEATS v4 (+0.98%):", a.mean() > 0.0098)
PY
```
Expected: prints summary. **PASS if `mean_alpha > +0.98%`.**

- [ ] **Step 3: Aggregation robustness via offline sweep**

Run: `CUDA_VISIBLE_DEVICES="" uv run python -m RL.sweep_aggregation walk_forward_v5_waterfill_cache.pkl`
Expected: prints sharpe_weighted / uniform / median gates. Note whether the result is robust to the aggregation scheme (it should be, given the tight val_sharpe spread).

- [ ] **Step 4: Record outcome**

Append a one-paragraph result (mean alpha, cash%, pass/fail vs +0.98%) to the spec file `docs/superpowers/specs/2026-06-04-v5-cash-overflow-design.md` under a new `## Result` section, and commit.

- [ ] **Step 5: Decision gate (do NOT auto-ship)**

If PASS: surface the numbers and propose the deploy cutover (retrain 5 v5 seeds via `dl_train_deploy`, then a *separate* reviewed change to `deploy_rl.yml` + `retrain` workflow — pointer flip LAGS the gate, per the outage lesson). If FAIL: report and consider the fallback levers (raise `max_weight`, headroom-proportional redistribution, or revisit features). Either way, stop for human review before touching `main`/CI.

---

## Notes for the implementer

- Every run forces CPU with `CUDA_VISIBLE_DEVICES=""` (an RTX GPU is present but training is specced for CPU; the scripts auto-pick cuda otherwise).
- The three cap sites MUST stay in sync — that drift caused a production outage. Task 4's parity test is the guardrail; do not skip it.
- `max_weight` stays 0.10 throughout — do not raise it this round (would confound attribution of the water-fill effect).
- Work stays on `v5-iteration`. No edits to `main`, `.github/workflows/`, or any `models/*.pt` until Task 7 passes and a human approves the cutover.
