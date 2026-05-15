# CLAUDE.md

Guide Claude Code (claude.ai/code) for this repo.

## Repo layout

Repo bundle **three independent Python subprojects** sharing simulated-trading API (NCKU `ciot.imis.ncku.edu.tw/sim_stock`) but otherwise no dependency:

- `stock_api/` — fresh data-fetch layer scraping price history direct from **official TWSE / TPEX / ESB endpoints**. Has **own `.git/`** (treat as vendored/sibling repo; commits to files inside `stock_api/` go to *that* repo, not outer).
- `stock_backtest_/` — backtest framework pulling historical data from **internal lab API at `http://140.116.86.242:8081`** (not TWSE/TPEX/ESB), plus strategy-execution engine.
- `RL/` — deep-learning portfolio agent + live deploy. v4 LSTM (supervised, Sharpe-loss) is the current production model; v1–v3 PPO (stable_baselines3) is legacy but loadable. Calls `stock_api.Get_User_Stocks / Buy_Stock / Sell_Stock` directly for live trading. Walk-forward backtests + GitHub Actions CI (daily deploy + weekly retrain) live here.
- `main.py` — single-file demo of `stock_api` (hard-codes `ACCOUNT`/`PASSWORD` for trading API; never commit real credentials).
- `models/` — production checkpoints consumed by `RL/deploy_rl.py`: `dl_v4_seed{0..4}.pt` (active 5-seed v4 LSTM ensemble), `dl_v4_deploy.pt` (single-seed v4), `ppo_*.zip` + `vec_normalize_*.pkl` (legacy v3 PPO).

No top-level package — each subproject used independently.

## Commands

Env is **uv-managed** at `./.venv` (not `./venv` — that path is stale). `requirements.txt` at repo root covers `stock_api/` + `RL/`. `stock_backtest_/requirements.txt` is separate.

```bash
# Setup
uv venv --python 3.12
uv pip install -r requirements.txt
uv pip install -r stock_backtest_/requirements.txt  # only if running backtest framework

# stock_api demo
uv run python main.py

# stock_backtest_ example (must run from inside stock_backtest_/
# because it does `from backtest.backtest import *`)
cd stock_backtest_ && uv run python examplebacktest.py

# RL — live deploy (NCKU sim_stock)
uv run python -m RL.deploy_rl --paper --dl-ensemble models/dl_v4_seed{0..4}.pt   # dry-run
uv run python -m RL.deploy_rl --live  --dl-ensemble models/dl_v4_seed{0..4}.pt   # submit orders
uv run python -m RL.deploy_rl --status                                            # ledger snapshot
uv run python -m RL.deploy_rl --reconcile-only                                    # broker inventory sync

# RL — retrain 5-seed v4 LSTM ensemble (matches weekly CI)
uv run python -m RL.dl_train_deploy --epochs 500 --train-recent 500 \
    --window 50 --hidden 32 --max-weight 0.10 --seeds 0,1,2,3,4

# RL — walk-forward eval
uv run python -m RL.walk_forward_dl --lstm --tag v4_w50h32 --window 50 --hidden 32  # single-seed
uv run python -m RL.walk_forward_dl_ensemble --seeds 5 --epochs 500 \
    --window 50 --hidden 32 --train-recent 500 --tag v4_5seed                       # ensemble OOS

# Lint / format — ruff is the *only* configured tool (see .vscode/settings.json
# and .ruff_cache/). Do not introduce black/autopep8/isort/flake8.
ruff check .
ruff format .
```

No test suite, no build step. CI = two GitHub Actions: daily deploy + weekly retrain (see `Deploy & retrain workflow` below). `pylint` referenced in `.vscode/settings.json` only to *disable* missing-docstring warnings.

## Architecture: `stock_api/`

Single public surface re-exported from `stock_api/__init__.py`:
- `get_taiwan_stock_data(stock_code, start, end)` — price history.
- `Get_User_Stocks / Buy_Stock / Sell_Stock` — wrappers around NCKU trading API.
- `get_stock_info / get_stock_market / load_symbol_map` — symbol lookup.

**Market dispatch** (`core.py` → `fetchers.py`): `get_taiwan_stock_data` reads symbol's market from `stock_symbol_map.json` via `symbols.get_stock_market`, normalizes `ETF→TWSE`, `OTC→TPEX`, `ESB→ESB`, dispatches to one of three fetchers. Each fetcher returns richer schema; `to_legacy_schema` strips down to 10 documented columns (`date, capacity, turnover, high, low, close, change, transaction_volume, stock_code_id, open`) before return to callers.

**ESB no official OHLC.** `get_esb_stock_data` synthesizes:
- `close` = weighted-average price (`turnover / capacity`), marked via `close_is_proxy=True` in internal schema (dropped by `to_legacy_schema`).
- `open` = `pd.NA`.
- `change` = `close.diff()` computed *after* date filter.
Keep proxy semantics in mind when extend pipeline — strategies assuming real OHLC silently misbehave on ESB symbols.

**TPEX unit conversion.** TPEX returns volumes in **lots (張)** and turnover in **千元**; fetcher multiplies both by 1000 so all three markets emit shares/元.

**Networking discipline** (`utils.py`):
- Every fetcher sleeps `2.0s` between monthly requests — do not remove; upstream sources rate-limit aggressively.
- `safe_get_json` does exponential backoff + jitter, raises `UpstreamHTMLResponse` when maintenance/block page served as HTML in place of JSON.
- ROC dates (`115/03/02`) converted via `roc_to_ad` (`+1911`).
- `clean_numeric` strips thousands separators and ex-rights/ex-dividend markers (`X`, `除權`, `除息`, `----`) before `pd.to_numeric(errors="coerce")`.

## Architecture: `stock_backtest_/`

`backtest/backtest.py` = one large module containing whole engine. Pieces:

- `Stock_API` (`backtest/Stock_API.py`) — thin REST client. **Different upstream from `stock_api/`**: price history from `http://140.116.86.242:8081`, while buy/sell/inventory still hit NCKU `ciot.imis.ncku.edu.tw` trading API.
- `Stock_Information_Memory` — process-wide cache; `BacktestSystem.__get_all_stock_information` calls `load_all_stock_data(...)` once at run start so per-day strategy callbacks read from memory instead of re-hitting API.
- `Stock_Information` — per-(stock, date) view with `.rolling(n)` to step back `n` trading days. Returns `None` for `price_close` when no data; `BacktestSystem._current_stock_info` walks back up to 20 days to skip non-trading dates.
- `InventoryManager` — FIFO lots, computes realized profit on sell.
- `Transaction_Tool` — accumulates `TransactionRecord`s user's strategy emits; `action` codes `0=noop, 1=sell, 2=buy` (defined in `type.py`).
- `BacktestSystem` — orchestrator. Lifecycle:
  1. `set_backtest_period(start, end)` (dates `YYYYMMDD` strings).
  2. `set_cash_balance(...)`.
  3. `execute_strategy(name, select_stock_func, trade_strategy_func)` — runs day-by-day until `end_date`, advances via `next_day()` using **2330 (TSMC) as trading-calendar reference** to skip closed sessions.
  4. `calculate_performance()` → `(StockPerformance, StockPerformanceDetail)`.
  5. `save_performance_to_xls(...)` then `upload_performance_*_to_web()` push results back to lab API.

**User strategy contract** (see `examplebacktest.py` for working example):
- `select_stock_func(all_stock_info, previous_stock_pool, user_cash) -> List[str]`
- `trade_strategy_func(stock_code_list, user_inventory, user_cash, transaction_tool) -> List[TransactionRecord]` — must call `transaction_tool.buy_stock` / `sell_stock` to record orders.

**Cached dataset.** `stock_backtest_/data/stock_data.csv` (~9 MB) plus `save_data_info.yaml` = on-disk snapshot consumed by `Stock_Information_Memory`; YAML pins `start_date`/`end_date` covered. If backtest period exceeds range, engine refetches from lab API.

## Architecture: `RL/`

**Two model generations coexist** — both still loadable, but only v4 is trained and deployed:

- **v1–v3 PPO (legacy).** `env.py` registers `TradingEnv-v0` (gymnasium); training via `train.py` (`stable_baselines3.PPO`, `SubprocVecEnv`, `VecNormalize`). `policy.py:PerAssetEncoder` is the permutation-equivariant feature extractor. Inference needs the matching `vec_normalize_*.pkl` to undo obs normalization. Artifacts: `models/ppo_*.zip` + `models/vec_normalize_*.pkl`. Re-deployable via `--ensemble model.zip:norm.pkl ...`, but no longer the default.
- **v4 LSTM (active).** `dl_portfolio.py:PortfolioNetLSTM` — per-stock LSTM over `window_len=50` day windows produces a score per stock, softmax with per-stock cap (`max_weight=0.10`, excess → cash slot) yields target weights. Trained end-to-end on negative Sharpe (`sharpe_loss`) with L1-turnover-cost-aware `realized_returns` — no env, no exploration, no VecNormalize. Trained via `dl_train_deploy.py`. Artifacts: `models/dl_v4_seed{0..4}.pt` + `models/dl_v4_deploy.pt` — self-contained checkpoints carry `state_dict`, `config` (num_stocks/feat_per_stock/window_len/hidden/max_weight), `stock_ids`, `feat_cols`, `trained_through`.

**Feature pipeline** (`feature.py:FeatureExtractor`). 12 features per stock: 8 local technicals (`return`, `bias_5`, `bias_20`, `macd_h`, `rsi_14`, `bb_pos`, `atr`, `capacity_change`) + 4 cross-sectional daily ranks (`return_rank`, `rsi_14_rank`, `bias_20_rank`, `capacity_change_rank`). The rank features are what give the model market-relative signal — keep them when extending the feature set, or you'll lose the cross-sectional component.

**Universe** (`constant.py:stock_ids`). 46 Taiwan large/mid-caps (semis 13, financials 10, industrials/transport/retail 10+, telecom 4). Stock-ID order is *load-bearing* — checkpoints store the list at training time; `predict_dl_weights` (`deploy_rl.py:574`) refuses to run if `stock_ids` order has changed since training. Re-train when extending the universe.

**Data cache** (`RL/data/<sid>.csv`). One file per stock, 10-col schema matching `stock_api.get_taiwan_stock_data`. Refresh policy in `deploy_rl.py`: if CSV `last_date < today − CSV_STALE_DAYS=7`, fetch from TWSE via `stock_api`; otherwise reuse. CI commits the refreshed CSVs back daily.

**Walk-forward eval.**
- `walk_forward_dl.py` — 10 fixed 6-month folds from 2019 H1 through 2023 H2 (`gen_rolling_folds`). Trains one model per fold, prints + dumps `walk_forward_<tag>_results.json` (per-fold return, EW, alpha, sharpe, cash%, active stocks).
- `walk_forward_dl_ensemble.py` — same fold schedule but trains N seeds per fold and aggregates daily weights via `aggregate_weights` (per-dim median across seeds → renorm → per-stock cap re-enforced). Mirrors live deploy's `predict_dl_ensemble_weights`, so OOS alpha is apples-to-apples with the deployed decision rule. Long: 5 seeds × 10 folds × 500 epochs ≈ 2.4 h on CPU.
- Strategy baselines for comparison: `strategy_smart.py`, `strategy_ewT.py`, `strategy_ew_rebal.py`, `strategy_jI.py`, `strategy_jIT.py` (rule-based equal-weight + threshold rotators). Results dumped as `strategy_*_results.json` at repo root.

**Empirical ceiling**: v4 w50h32 single-seed walk-forward = **+4.09 % mean alpha vs EW**, 5/10 folds positive (within SE of zero). 5-seed median ensemble shrinks this to +0.98 % — median voting penalizes the conviction trades the model relies on. Don't expect more from this architecture without changing the feature set or universe.

## Deploy & retrain workflow

Two GitHub Actions in `.github/workflows/` drive production. Both use uv + Python 3.12 and commit results back to `main`.

**Daily deploy** (`deploy_rl.yml`) — cron `0 9 * * 1-5` (09:00 UTC = **17:00 Taipei**, Mon–Fri). Time is chosen *after* TW market close + 15:30 settlement because NCKU `sim_stock` accepts orders only **16:01 → next-day 08:59 weekdays**; 09:00–16:00 Taipei is hard-blocked (`is_trading_window` at `deploy_rl.py:93`). Job runs `RL.deploy_rl --live --dl-ensemble models/dl_v4_seed{0..4}.pt`, then commits `deploy_state.json` + refreshed `RL/data/*.csv` with message `deploy: state + data YYYY-MM-DD`. Manual override via `workflow_dispatch` with `mode: paper|live`.

**Weekly retrain** (`retrain_v4.yml`) — cron `0 1 * * 0` (Sunday 01:00 UTC = **09:00 Sunday Taipei**, after Friday's CSVs settled into the repo). Runs `RL.dl_train_deploy --seeds 0,1,2,3,4 --epochs 500 --window 50 --hidden 32 --max-weight 0.10 --train-recent 500`, then commits `models/dl_v4_seed*.pt` with `retrain: v4 LSTM 5-seed ensemble YYYY-MM-DD`. Timeout 90 min. Daily deploy on Monday picks up the new weights automatically.

**Live execution sequence** (`deploy_rl.py:main`):
1. Load `deploy_state.json` (ledger). If `halted=True`, log "still halted" and exit without trading.
2. `--live` only: `reconcile_inventory` calls `Get_User_Stocks` and overwrites `state['inventory']` with broker truth (cash isn't broker-readable). Catches drift from rejected orders. Skip with `--no-reconcile`.
3. Refresh data per-stock as needed (`CSV_STALE_DAYS=7`).
4. Build (1, N=46, L=50, F=12) input tensor, run each seed's `PortfolioNetLSTM` deterministically, aggregate via `predict_dl_ensemble_weights` (per-dim median → renorm → cap at 10 %, excess → cash).
5. Convert weights to lot-sized orders (1000-share lots), apply 0.1425 % buy fee / 0.3 % + 0.1425 % sell tax+fee with 20 TWD minimum, round price to TW tick rules. Buys at `min(open, close)`, sells at `max(open, close)` to maximize fill probability.
6. `--live`: submit via `Buy_Stock` / `Sell_Stock`. Each order's broker response is checked; only successful responses update `state['inventory']`. `--paper` prints orders and updates state as if filled.
7. Compute `port_val_after`, update `peak_value`. If `(peak − port_val) / peak > DRAWDOWN_STOP=0.10`, `liquidate_all()` and set `halted=True` (sticky — clear manually by editing the state file; see the `manual_reconcile_bootstrap` precedent in `deploy_state.json.bak.*`).
8. Append entry to `history` (date, actions list with `resp`/`ok` flags, port val before/after, drawdown). Persist.

**State file (`deploy_state.json`).** Schema: `day_count`, `initial_cash` (`INITIAL_CASH=100_000_000` TWD at `deploy_rl.py:44`), `cash_balance`, `inventory: {sid: shares}`, `peak_value`, `halted` (sticky), `history` (list). Committed back by the workflow daily. Pre-run backup written as `deploy_state.json.bak.YYYYMMDD_HHMMSS` (not committed, not auto-cleaned).

**Secrets** (GitHub Actions): `NCKU_ACCOUNT`, `NCKU_PASSWORD`. Surfaced to `deploy_rl.py` as env vars `ACCOUNT`, `PASSWORD`.

## Conventions

- Python ≥ 3.10 (uses `list[str]`, `dict | None`, etc.). `stock_backtest_/environment.yml` pins `>=3.10,<3.15`. Production CI uses 3.12.
- Code, comments, docstrings, example data predominantly **Traditional Chinese** in `stock_api/` + `stock_backtest_/`; **English** in `RL/`. Preserve language when editing.
- All three subprojects intentionally duplicate trading-API client (`Buy_Stock`/`Sell_Stock`/`Get_User_Stocks`); they evolved separately. Do not unify without explicit ask.
- Long-running RL scripts should be launched with `python -u` (or `PYTHONUNBUFFERED=1`) — without it, stdout block-buffers when redirected to a file and progress prints invisibly until exit.