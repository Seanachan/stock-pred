# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo layout

This repo bundles **two independent Python subprojects** that share the same simulated-trading API (NCKU `ciot.imis.ncku.edu.tw/sim_stock`) but otherwise do not depend on each other:

- `stock_api/` — fresh data-fetching layer that scrapes price history directly from the **official TWSE / TPEX / ESB endpoints**. Has its **own `.git/`** (treat as a vendored or sibling repo; commits to files inside `stock_api/` go to *that* repo, not the outer one).
- `stock_backtest_/` — backtesting framework that pulls historical data from an **internal lab API at `http://140.116.86.242:8081`** (not from TWSE/TPEX/ESB), plus a strategy-execution engine.
- `main.py` — single-file demo of `stock_api` (hard-codes `ACCOUNT`/`PASSWORD` for the trading API; do not commit real credentials).

There is no top-level package — each subproject is used independently.

## Commands

```bash
# Setup (root-level venv already exists at ./venv)
source venv/bin/activate
pip install -r requirements.txt                  # for stock_api / main.py
pip install -r stock_backtest_/requirements.txt  # for the backtest framework

# Run the stock_api demo
python main.py

# Run a backtest example (must be invoked from inside stock_backtest_/
# because it does `from backtest.backtest import *`)
cd stock_backtest_ && python examplebacktest.py

# Lint / format — ruff is the *only* configured tool (see .vscode/settings.json
# and .ruff_cache/). Do not introduce black/autopep8/isort/flake8.
ruff check .
ruff format .
```

There is no test suite, no build step, and no CI. `pylint` is referenced in `.vscode/settings.json` only to *disable* missing-docstring warnings.

## Architecture: `stock_api/`

Single public surface re-exported from `stock_api/__init__.py`:
- `get_taiwan_stock_data(stock_code, start, end)` — price history.
- `Get_User_Stocks / Buy_Stock / Sell_Stock` — wrappers around the NCKU trading API.
- `get_stock_info / get_stock_market / load_symbol_map` — symbol lookup.

**Market dispatch** (`core.py` → `fetchers.py`): `get_taiwan_stock_data` reads the symbol's market from `stock_symbol_map.json` via `symbols.get_stock_market`, which normalizes `ETF→TWSE`, `OTC→TPEX`, `ESB→ESB`, then dispatches to one of three fetchers. Each fetcher returns a richer schema; `to_legacy_schema` strips it down to the 10 documented columns (`date, capacity, turnover, high, low, close, change, transaction_volume, stock_code_id, open`) before returning to callers.

**ESB has no official OHLC.** `get_esb_stock_data` synthesizes:
- `close` = weighted-average price (`turnover / capacity`), marked via `close_is_proxy=True` in the internal schema (dropped by `to_legacy_schema`).
- `open` = `pd.NA`.
- `change` = `close.diff()` computed *after* date filtering.
Keep this proxy semantics in mind whenever extending the pipeline — strategies that assume real OHLC will silently misbehave on ESB symbols.

**TPEX unit conversion.** TPEX returns volumes in **lots (張)** and turnover in **千元**; the fetcher multiplies both by 1000 so all three markets emit shares/元.

**Networking discipline** (`utils.py`):
- Every fetcher sleeps `2.0s` between monthly requests — do not remove; the upstream sources rate-limit aggressively.
- `safe_get_json` does exponential backoff + jitter and raises `UpstreamHTMLResponse` when a maintenance/block page is served as HTML in place of JSON.
- ROC dates (`115/03/02`) are converted via `roc_to_ad` (`+1911`).
- `clean_numeric` strips thousands separators and ex-rights/ex-dividend markers (`X`, `除權`, `除息`, `----`) before `pd.to_numeric(errors="coerce")`.

## Architecture: `stock_backtest_/`

`backtest/backtest.py` is one large module containing the whole engine. The pieces:

- `Stock_API` (`backtest/Stock_API.py`) — thin REST client. **Different upstream from `stock_api/`**: price history comes from `http://140.116.86.242:8081`, while buy/sell/inventory still hit the NCKU `ciot.imis.ncku.edu.tw` trading API.
- `Stock_Information_Memory` — process-wide cache; `BacktestSystem.__get_all_stock_information` calls `load_all_stock_data(...)` once at the start of a run so per-day strategy callbacks read from memory instead of re-hitting the API.
- `Stock_Information` — per-(stock, date) view with `.rolling(n)` to step back `n` trading days. Returns `None` for `price_close` when no data exists; `BacktestSystem._current_stock_info` walks back up to 20 days to skip non-trading dates.
- `InventoryManager` — FIFO lots, computes realized profit on sell.
- `Transaction_Tool` — accumulates `TransactionRecord`s the user's strategy emits; `action` codes are `0=noop, 1=sell, 2=buy` (defined in `type.py`).
- `BacktestSystem` — orchestrator. Lifecycle:
  1. `set_backtest_period(start, end)` (dates are `YYYYMMDD` strings).
  2. `set_cash_balance(...)`.
  3. `execute_strategy(name, select_stock_func, trade_strategy_func)` — runs day-by-day until `end_date`, advancing via `next_day()` which uses **2330 (TSMC) as the trading-calendar reference** to skip closed sessions.
  4. `calculate_performance()` → `(StockPerformance, StockPerformanceDetail)`.
  5. `save_performance_to_xls(...)` then `upload_performance_*_to_web()` push results back to the lab API.

**User strategy contract** (see `examplebacktest.py` for a working example):
- `select_stock_func(all_stock_info, previous_stock_pool, user_cash) -> List[str]`
- `trade_strategy_func(stock_code_list, user_inventory, user_cash, transaction_tool) -> List[TransactionRecord]` — must call `transaction_tool.buy_stock` / `sell_stock` to record orders.

**Cached dataset.** `stock_backtest_/data/stock_data.csv` (~9 MB) plus `save_data_info.yaml` is the on-disk snapshot consumed by `Stock_Information_Memory`; the YAML pins the `start_date`/`end_date` covered. If a backtest period exceeds that range, the engine refetches from the lab API.

## Conventions

- Python ≥ 3.10 (uses `list[str]`, `dict | None`, etc.). `stock_backtest_/environment.yml` pins `>=3.10,<3.15`.
- Code, comments, docstrings, and example data are predominantly **Traditional Chinese** — preserve language when editing.
- The two subprojects intentionally duplicate the trading-API client (`Buy_Stock`/`Sell_Stock`/`Get_User_Stocks`); they evolved separately, so do not attempt to unify them without an explicit ask.
