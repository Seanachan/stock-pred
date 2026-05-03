# Stock Prediction Reinforcement Learning

Multi-stock PPO trading agent for Taiwan stocks (46 large/mid caps), with live deploy via GitHub Actions to NCKU `sim_stock` simulator.

## Repo Layout

| Path | Purpose |
|---|---|
| `stock_api/` | TWSE / TPEX / ESB official price-history fetcher (own git repo, vendored) |
| `stock_backtest_/` | Lab backtest framework using internal API (`140.116.86.242:8081`) |
| `RL/` | PPO trading agent: env, features, train, eval, walk-forward, live deploy |
| `models/` | Production model weights (`ppo_deploy_final.zip`, `vec_normalize_deploy_final.pkl`) |
| `.github/workflows/` | CI cron — daily live deploy at 11:00 TW |

## Installation

```shell
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Data

Per-stock daily CSVs live in `RL/data/<stock_id>.csv`. Schema: `date, capacity, turnover, high, low, close, change, transaction_volume, stock_code_id, open`.

Fetch fresh history (TWSE / TPEX / ESB official endpoints, ~2s rate-limit per request):

```shell
.venv/bin/python -m RL.fetch_data
```

Coverage tracked in `RL/data/meta.yaml`.

## RL Environment

`RL/env.py` registers `TradingEnv-v0` with:

- **Universe**: 46 stocks (`RL/constant.py:stock_ids`).
- **Observation** (`12 × 46 + 1 + 46 = 599`-dim):
  - Per stock × 46:
    - Local (8): `return, bias_5, bias_20, macd_h, rsi_14, bb_pos, atr, capacity_change`
    - Cross-sectional rank (4): `return_rank, rsi_14_rank, bias_20_rank, capacity_change_rank` (per-day percentile across basket)
  - Cash balance (normalized by initial cash)
  - Inventory (lots / 1000)
- **Action**: `MultiDiscrete([7] * 46)` — per-stock category:
  - `0=sell_100% | 1=sell_50% | 2=sell_25% | 3=hold | 4=buy_5%_cash | 5=buy_15%_cash | 6=buy_30%_cash`
- **Reward** (per step): `agent_log_ret − bh_log_ret − 0.05·max(dd−0.15, 0) − 0.1·max(herfindahl−0.3, 0) − turnover_cost − 0.0001·invalid_count`
  - `bh_log_ret`: equal-weight buy-and-hold baseline (forces structural alpha target)
  - Drawdown / Herfindahl: weak penalties (deploy config; tune in `env.py`)
  - Costs: 0.1425% fee + 0.3% sell tax modeled

### Feature Indicators

Computed in `RL/feature.py` via `pandas_ta`:

- **Bias** (均線乖離率) `(close - SMA_N) / SMA_N`. ±large = pullback / rebound pressure.
- **SMA** (簡單移動平均線). 20-day SMA = mid-term trend.
- **MACD-H** (histogram / close). Momentum divergence.
- **RSI-14** (relative strength). Overbought > 0.7, oversold < 0.3.
- **Bollinger Bands position**. Where close sits in 20d band.
- **ATR** (average true range / close). Realized volatility proxy.
- **Capacity change** (volume pct change). Liquidity dynamics.
- **Cross-sectional ranks**: per-day percentile across basket — gives agent relative-strength signal.

## Training

### Walk-forward eval (research)

Rolling 6-month validation folds across 2019–2023, fixed 2015 train start:

```shell
.venv/bin/python -m RL.walk_forward
```

Outputs `walk_forward_<tag>_results.json` with per-fold return / EW / alpha / trades.

Past results (mean alpha across versions, all negative):

| version | mean alpha | std | + folds |
|---|---|---|---|
| v7 (annual) | −14.11% | 13.63% | 1/5 |
| v8 (annual) | −14.26% | 13.81% | 1/5 |
| v10 (5-fold) | −6.57% | 19.31% | 2/5 |
| B0 (10-fold 6mo) | −10.86% | 13.66% | 2/10 |

Conclusion: PPO on daily-frequency TW stocks does not produce structural alpha across regimes; best-of-N folds are sample variance.

### Deploy training

Single-fold training on most recent regime (`RL/train_deploy.py`):

```shell
# Final deploy model (trains on all data through 2026-04-22, no val held out)
.venv/bin/python -m RL.train_deploy --full

# Held-out validation runs (for seed-stability check)
.venv/bin/python -m RL.train_deploy --seed 0
.venv/bin/python -m RL.train_deploy --seed 1
.venv/bin/python -m RL.train_deploy --seed 2
```

Hyperparams: PPO, `pi=[512,256], vf=[512,256]`, ReLU, lr 3e-4→1e-5 linear, n_steps=1024, n_epochs=10, ent_coef=0.02, target_kl=0.1, 8 SubprocVecEnv parallel envs, 3M timesteps.

## Backtesting Strategies

Several non-RL baselines exist for comparison:

| Script | Strategy | 2018–2026 return | Notes |
|---|---|---|---|
| `RL.strategy_jI` | 200MA regime gate × 60d momentum-weight, monthly rebal | +56% | Monthly rebal in TW = mean-reversion tax |
| `RL.strategy_jIT` | J+I + Mahalanobis turbulence hard-stop (Yang et al, AI4Finance) | +72% | Turbulence whipsaws on V-shaped TW dips |
| `RL.strategy_ewT` | BH-EW + turbulence stop | +96% | Same whipsaw issue |
| `RL.strategy_ew_rebal` | Various rebalance frequencies | varies | Quarterly > monthly > biweekly |
| `RL.strategy_smart` | BH-EW + manufactured pad trades | **+225.77%** | Closest to theoretical EW (+229.81%) |

Conclusion: in the 2018–2026 regime, naive buy-and-hold of equal-weighted basket dominates all "smart" rule-based protections.

## Live Deploy (GitHub Actions)

Daily cron pulls latest TWSE data, runs PPO inference, submits Buy/Sell via NCKU `sim_stock` API.

### Setup

1. Push `models/` + `RL/deploy_rl.py` + `.github/workflows/deploy_rl.yml` to GitHub.
2. **GitHub repo → Settings → Secrets and variables → Actions** → add:
   - `NCKU_ACCOUNT` = your NCKU student ID
   - `NCKU_PASSWORD` = your trading-API password
3. Verify with paper run: **Actions** tab → "Deploy RL Daily Trade" → **Run workflow** → choose `paper`.
4. Cron auto-fires 03:00 UTC Mon–Fri (= 11:00 TW). Each run commits updated `deploy_state.json`.

### Local dry-run / debug

```shell
# Print today's actions, no orders submitted
.venv/bin/python -m RL.deploy_rl --paper

# Submit live (needs ACCOUNT/PASSWORD in .env)
.venv/bin/python -m RL.deploy_rl --live

# Show ledger snapshot
.venv/bin/python -m RL.deploy_rl --status
```

### Risk Overlay

`deploy_rl.py` adds a hard-stop overlay on top of the model: if portfolio drawdown exceeds 10% from peak, all positions are liquidated and trading halts for the rest of the run. Override threshold via `DRAWDOWN_STOP` constant.

### State Persistence

`deploy_state.json` (committed by workflow each run) tracks:

- `day_count`, `cash_balance`, `inventory`
- `peak_value` (for drawdown calc)
- `halted` (sticky flag after stop trigger)
- `history` (per-day action log + portfolio value)

## NCKU sim_stock Performance Metrics

The simulator scores strategies on:

- **累積報酬率** = `(總資產 − 初始) / 初始 × 100%`
- **單筆交易報酬率** = `(賣 − 成本) / 成本 × 100%`
- **年化報酬率** = `(報酬率 / 持有交易日) × 252 × 100%`
- **平均獲利 / 虧損報酬率** = sum / count, separately for + / − trades
- **勝率** = winning trades / total trades
- **獲利因子** = total profit / total loss

Rule: must execute ≥ 100 trades over the evaluation period.

## License / Credits

Course project — NCKU `sim_stock` simulator. Strategy research uses techniques from Yang et al. (AI4Finance ensemble) and standard quant baselines.
