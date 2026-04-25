# RL — Multi-stock PPO agent

Reinforcement-learning trading agent trained against `stock_backtest_/backtest/BacktestSystem`.

## Layout

| File | Role |
|------|------|
| `env.py` | `gymnasium.Env` wrapping `BacktestSystem` |
| `features.py` | Observation builder (price window, indicators, account state) |
| `networks.py` | Actor-critic MLP (shared trunk, two heads) |
| `rollout.py` | Trajectory buffer + GAE advantage |
| `ppo.py` | Clipped surrogate objective update |
| `train.py` | Entry point: env + agent + loop |
| `eval.py` | Adapter turning trained policy into `trade_strategy_func` |
