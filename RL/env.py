"""Multi-stock PPO trading agent built on top of stock_backtest_.BacktestSystem."""

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

        # Define action and observation space
        # 0=sell_2, 1=sell_1, 2=hold, 3=buy_1, 4=buy_2 per stock
        self.action_space = spaces.MultiDiscrete([5] * self.num_stocks)
        self.observation_dim = (
            (self.num_stocks * 8) + 1 + self.num_stocks
        )  #  8 features per stock + cash balance + current holdings

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
            (self.max_steps + 1, self.num_stocks * 8)
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
                    ]
                    day_features.extend(stock_features)
                else:
                    self.price_memory[i, j] = 0.0
                    day_features.extend([0.0] * 8)

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

        initialize_observation = self._get_obs()
        return initialize_observation, {}

    def step(self, action=None) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one time step within the environment"""
        current_prices = self.price_memory[self.current_step]
        step_trades = 0
        step_fees = 0.0

        # Action mapping: 0=sell_2, 1=sell_1, 2=hold, 3=buy_1, 4=buy_2
        for i, act in enumerate(action):
            price = current_prices[i]
            if price == 0 or np.isnan(price):
                continue

            if act == 4 or act == 3:
                lots = 2 if act == 4 else 1
                cost = lots * price * 1000
                transaction_fee = max(
                    cost * self.transaction_fee_rate, self.min_transaction_fee
                )
                if self.cash_balance >= cost + transaction_fee:
                    self.cash_balance -= cost + transaction_fee
                    self.inventory[i] += 1000 * lots
                    step_fees += transaction_fee
                    step_trades += 1
                else:
                    self._invalid_count += 1

            elif act == 1 or act == 0:
                lots_to_sell = 2 if act == 0 else 1
                available_lots = self.inventory[i] // 1000
                lots_to_sell = min(lots_to_sell, available_lots)
                if lots_to_sell > 0:
                    shares = lots_to_sell * 1000
                    revenue = (
                        shares * price * (1 - self.tax_rate - self.transaction_fee_rate)
                    )
                    sell_cost = (
                        shares * price * (self.tax_rate + self.transaction_fee_rate)
                    )
                    self.cash_balance += revenue
                    self.inventory[i] -= shares
                    step_fees += sell_cost
                    step_trades += 1
                else:
                    self._invalid_count += 1

            elif act == 2:
                pass

        self.total_trades += step_trades

        # Next Day
        self.current_step += 1

        # Recalculate asset
        terminated = self.current_step >= self.end_step
        truncated = False
        next_prices = (
            self.price_memory[self.current_step] if not terminated else current_prices
        )

        # Reward = active return (agent log-return - EW basket log-return)
        stock_value = np.sum(self.inventory * next_prices)
        current_total_asset = self.cash_balance + stock_value
        previous_total_asset = self.asset_history[-1]

        # Equal-weight basket return per step (relative to prior step prices)
        valid_now = (next_prices > 0) & ~np.isnan(next_prices)
        valid_prev = (current_prices > 0) & ~np.isnan(current_prices)
        valid = valid_now & valid_prev
        if valid.any():
            ew_step_return = float(
                np.mean(next_prices[valid] / current_prices[valid]) - 1.0
            )
        else:
            ew_step_return = 0.0
        ew_log_ret = float(np.log(max(1.0 + ew_step_return, 1e-8)))

        agent_log_ret = float(
            np.log(max(current_total_asset, 1) / max(previous_total_asset, 1))
        )
        reward = agent_log_ret - ew_log_ret

        peak = max(self.asset_history)
        drawdown = (peak - current_total_asset) / max(peak, 1)
        reward -= 0.1 * max(drawdown - 0.1, 0)

        turnover_cost = step_fees / max(previous_total_asset, 1)
        reward -= turnover_cost

        if current_total_asset > 0:
            weights = (self.inventory * next_prices) / current_total_asset
            herfindahl = float(np.sum(weights**2))
            reward -= 0.5 * max(herfindahl - 0.3, 0)

        invalid_penalty = self._invalid_count * 0.0001
        reward -= invalid_penalty
        self._invalid_count = 0

        self.asset_history.append(current_total_asset)

        info = {
            "total_asset": current_total_asset,
            "total_trades": self.total_trades,
            "total_fees": step_fees,
            "return_rate": (current_total_asset / self.initial_cash) - 1,
            "inventory": {
                code: int(inventory)
                for code, inventory in zip(self.stock_ids, self.inventory)
            },
            "asset_history": self.asset_history,
        }

        # return self._get_obs(), float(reward), terminated, truncated, info
        obs = self._get_obs()
        assert np.isfinite(obs).all(), (
            f"NaN obs at step {self.current_step}, "
            f"cash={self.cash_balance}, "
            f"inv={self.inventory}, "
            f"prices={next_prices}"
        )
        if self.render_mode == "human" and terminated:
            self.render()
        return obs, float(reward), terminated, truncated, info

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
