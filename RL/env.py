"""Multi-stock PPO trading agent — continuous portfolio-weight action space.

Action: Box(-3, 3, shape=(N+1,)) raw logits → softmax → target portfolio
weights (46 stocks + cash). Each step rebalances toward target, paying tx
cost on the L1 trade volume. Reward = rolling Sharpe over last K step
returns.
"""

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register

from RL.feature import FeatureExtractor
from .constant import stock_ids

# Register as gyn envronment, so that it can be used with stable-baselines3 and other RL libraries
# Once registered, you can create an instance of the environment using standard interfaces gym.make("StockEnv-v0")
register(
    id="TradingEnv-v0",
    entry_point="RL.env:TradingEnv",
)


class TradingEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        stock_ids=stock_ids,
        stock_data=None,
        backtest_system=None,
        render_mode=None,
        eval_mode=False,
        window_len=240,
    ):
        super(TradingEnv, self).__init__()
        self.eval_mode = eval_mode
        self.window_len = window_len

        if stock_data is None:
            raise ValueError("stock_data cannot be None")

        # Environment State
        self.num_stocks = len(stock_ids)
        self.current_step = 0
        self.initial_cash = 100_000_000
        self.cash_balance = self.initial_cash
        self.inventory = np.zeros(self.num_stocks, dtype=int)
        self.total_trades = 0
        self._invalid_count = 0

        # For reward calculation and performance tracking
        self.asset_history = [self.initial_cash]
        self.stock_ids = stock_ids
        self.backtest_system = backtest_system
        self.render_mode = render_mode

        # Action: raw logits over (N stocks + cash); env softmaxes to weights.
        # Bounds chosen so post-softmax weights span ~ uniform .. concentrated.
        self.action_space = spaces.Box(
            low=-3.0,
            high=3.0,
            shape=(self.num_stocks + 1,),
            dtype=np.float32,
        )
        self.sharpe_window = 20
        self.return_history = deque(maxlen=self.sharpe_window)
        self.feat_per_stock = 12
        self.observation_dim = (
            (self.num_stocks * self.feat_per_stock) + 1 + self.num_stocks
        )  # 12 features per stock + cash balance + current holdings

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        # Trade Settings
        self.initial_cash = 100_000_000
        self.transaction_fee_rate = 0.001425
        self.min_transaction_fee = 20
        self.tax_rate = 0.003

        # Common trading dates
        self.dates = sorted(
            set.intersection(*[set(df.index) for df in stock_data.values()])
        )
        self.max_steps = len(self.dates) - 1

        # Precompute features for all stocks to speed up training
        self.price_memory = np.zeros(
            (self.max_steps + 1, self.num_stocks)
        )  # 1D df for stock prices
        self.features_memory = np.zeros(
            (self.max_steps + 1, self.num_stocks * self.feat_per_stock)
        )  # Flattened features for all stocks
        self.cash_balance = self.initial_cash

        market_dfs = FeatureExtractor(stock_ids).extract_features(stock_data)

        for i, date in enumerate(self.dates):
            day_features = []
            for j, stock_id in enumerate(self.stock_ids):
                df = market_dfs.get(stock_id)
                if df is not None and date in df.index:
                    row = df.loc[date]
                    self.price_memory[i, j] = row["close"]

                    stock_features = [
                        row["return"],
                        row["bias_5"],
                        row["bias_20"],
                        row["macd_h"],
                        row["rsi_14"],
                        row["bb_pos"],
                        row["atr"],
                        row["capacity_change"],
                        row.get("return_rank", 0.5),
                        row.get("rsi_14_rank", 0.5),
                        row.get("bias_20_rank", 0.5),
                        row.get("capacity_change_rank", 0.5),
                    ]
                    day_features.extend(stock_features)
                else:
                    self.price_memory[i, j] = 0.0
                    day_features.extend([0.0] * self.feat_per_stock)

            self.features_memory[i] = np.nan_to_num(
                day_features, nan=0.0, posinf=1.0, neginf=-1.0
            )

    def _get_obs(self):
        """Get current stock features and portfolio state"""
        current_features = self.features_memory[self.current_step]

        cash_norm = np.array([self.cash_balance / self.initial_cash], dtype=np.float32)
        inv_norm = self.inventory.astype(np.float32) / 1000.0

        return np.concatenate((current_features, cash_norm, inv_norm)).astype(
            np.float32
        )

    def reset(self, *, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Reset the environment and return the initial observation"""
        super().reset(seed=seed, options=options)

        if self.eval_mode:
            self.start_step = 0
            self.end_step = self.max_steps
        else:
            max_start = max(self.max_steps - self.window_len - 1, 1)
            self.start_step = int(self.np_random.integers(0, max_start))
            self.end_step = min(self.start_step + self.window_len, self.max_steps)

        self.current_step = self.start_step
        self.cash_balance = self.initial_cash
        self.inventory = np.zeros(self.num_stocks, dtype=int)
        self.total_trades = 0
        self._invalid_count = 0
        self.asset_history = [self.initial_cash]
        self.return_history.clear()

        # BH baseline: invest initial cash equally across valid stocks at start prices
        start_prices = self.price_memory[self.start_step]
        valid_start = (start_prices > 0) & ~np.isnan(start_prices)
        n_valid = int(valid_start.sum())
        self.bh_shares = np.zeros(self.num_stocks, dtype=np.float64)
        if n_valid > 0:
            per_stock_cash = self.initial_cash / n_valid
            self.bh_shares[valid_start] = per_stock_cash / start_prices[valid_start]

        initialize_observation = self._get_obs()
        return initialize_observation, {}

    def step(self, action=None) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Rebalance toward softmax(action) target weights, then advance one day.

        action: Box(N+1,) raw logits — last entry is cash weight.
        Reward: rolling Sharpe of last K step returns (mean / (std+eps)).
        """
        current_prices = self.price_memory[self.current_step]
        previous_total_asset = self.asset_history[-1]

        target_weights = self._softmax(np.asarray(action, dtype=np.float64))

        valid = (current_prices > 0) & ~np.isnan(current_prices)
        if not valid.any():
            self.current_step += 1
            self.asset_history.append(previous_total_asset)
            self.return_history.append(0.0)
            return self._terminate_step(previous_total_asset, current_prices, 0, 0.0)

        port_value = self.cash_balance + float(
            np.sum(self.inventory[valid] * current_prices[valid])
        )

        target_stock_value = port_value * target_weights[: self.num_stocks]
        target_stock_value[~valid] = 0.0
        target_lots = np.zeros(self.num_stocks, dtype=int)
        target_lots[valid] = (
            target_stock_value[valid] / (current_prices[valid] * 1000)
        ).astype(int)
        target_shares = target_lots * 1000

        delta_shares = target_shares - self.inventory
        step_fees = 0.0
        step_trades = 0

        # Sells first (free cash for buys later in the same step).
        for i in np.where(delta_shares < 0)[0]:
            if not valid[i]:
                continue
            shares = int(-delta_shares[i])
            price = float(current_prices[i])
            gross = shares * price
            cost = gross * (self.tax_rate + self.transaction_fee_rate)
            self.cash_balance += gross - cost
            self.inventory[i] -= shares
            step_fees += cost
            step_trades += 1

        for i in np.where(delta_shares > 0)[0]:
            if not valid[i]:
                continue
            shares = int(delta_shares[i])
            price = float(current_prices[i])
            gross = shares * price
            fee = max(gross * self.transaction_fee_rate, self.min_transaction_fee)
            if self.cash_balance < gross + fee:
                # Trim to whatever we can afford in 1000-share lots.
                affordable_lots = int(
                    max(self.cash_balance - self.min_transaction_fee, 0)
                    // (price * 1000)
                )
                if affordable_lots <= 0:
                    self._invalid_count += 1
                    continue
                shares = affordable_lots * 1000
                gross = shares * price
                fee = max(gross * self.transaction_fee_rate, self.min_transaction_fee)
            self.cash_balance -= gross + fee
            self.inventory[i] += shares
            step_fees += fee
            step_trades += 1

        self.total_trades += step_trades

        self.current_step += 1
        terminated = self.current_step >= self.end_step
        truncated = False
        next_prices = (
            self.price_memory[self.current_step] if not terminated else current_prices
        )

        stock_value = float(np.sum(self.inventory * next_prices))
        current_total_asset = self.cash_balance + stock_value

        step_ret = (current_total_asset / max(previous_total_asset, 1.0)) - 1.0
        self.return_history.append(step_ret)

        if len(self.return_history) >= 5:
            arr = np.asarray(self.return_history)
            std = float(arr.std())
            reward = float(arr.mean()) / (std + 1e-6)
        else:
            reward = step_ret * 100.0

        self.asset_history.append(current_total_asset)

        info = {
            "total_asset": current_total_asset,
            "total_trades": self.total_trades,
            "total_fees": step_fees,
            "return_rate": (current_total_asset / self.initial_cash) - 1,
            "step_return": step_ret,
            "inventory": {
                code: int(inv) for code, inv in zip(self.stock_ids, self.inventory)
            },
            "asset_history": self.asset_history,
            "target_weights": target_weights.tolist(),
        }

        obs = self._get_obs()
        assert np.isfinite(obs).all(), (
            f"NaN obs at step {self.current_step}, "
            f"cash={self.cash_balance}, inv={self.inventory}, prices={next_prices}"
        )
        if self.render_mode == "human" and terminated:
            self.render()
        return obs, float(reward), terminated, truncated, info

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        z = logits - np.max(logits)
        e = np.exp(z)
        return e / np.sum(e)

    def _terminate_step(self, total_asset, prices, trades, fees):
        info = {
            "total_asset": total_asset,
            "total_trades": self.total_trades,
            "total_fees": fees,
            "return_rate": (total_asset / self.initial_cash) - 1,
            "step_return": 0.0,
            "inventory": {
                code: int(inv) for code, inv in zip(self.stock_ids, self.inventory)
            },
            "asset_history": self.asset_history,
        }
        return self._get_obs(), 0.0, self.current_step >= self.end_step, False, info

    def render(self):
        """Plot asset history for the episode."""
        if self.render_mode != "human":
            return

        import time

        import matplotlib.pyplot as plt

        fname = f"img/asset_history{time.strftime('%Y-%m-%d_%H-%M-%S')}.png"
        final_asset = self.asset_history[-1]
        return_pct = (final_asset / self.initial_cash - 1) * 100

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(self.asset_history, color="steelblue", linewidth=1.5)
        ax.axhline(self.initial_cash, color="gray", linestyle="--", linewidth=1)
        ax.set_title(
            f"Asset History | Final: {final_asset:,.0f} TWD | "
            f"Return: {return_pct:+.2f}% | Trades: {self.total_trades}"
        )
        ax.set_xlabel("Step")
        ax.set_ylabel("Total Asset (TWD)")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(fname, dpi=100)
        plt.close(fig)
        print(f"Saved {fname}  (final={final_asset:,.0f}, return={return_pct:+.2f}%)")
