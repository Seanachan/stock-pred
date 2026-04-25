"""Multi-stock PPO trading agent built on top of stock_backtest_.BacktestSystem."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register

from RL.feature import FeatureExtractor
from RL.status import TradeStatus

from .constant import stock_ids

# Register as gyn envronment, so that it can be used with stable-baselines3 and other RL libraries
# Once registered, you can create an instance of the environment using standard interfaces gym.make("StockEnv-v0")
register(
    id="TradingEnv-v0",
    entry_point="RL.env:TradingEnv",
)


class TradingEnv(gym.Env):
    metadata = {"render.modes": ["human"], "render_fps": 1}

    def __init__(
        self,
        stock_ids=stock_ids,
        stock_data=None,
        backtest_system=None,
    ):
        super(TradingEnv, self).__init__()

        if stock_data is None:
            raise ValueError("stock_data cannot be None")

        # Environment State
        self.num_stocks = len(stock_ids)
        self.current_step = 0
        self.initial_cash = 100_000_000
        self.cash_balance = self.initial_cash
        self.inventory = np.zeros(self.num_stocks, dtype=int)
        self.total_trades = 0

        # For reward calculation and performance tracking
        self.asset_history = [self.initial_cash]
        self.stock_ids = stock_ids
        self.backtest_system = backtest_system

        # Define action and observation space
        self.action_space = spaces.MultiDiscrete(
            [3] * self.num_stocks  # e.g., Sell, Hold, Buy
        )
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

        self.current_step = 0
        self.cash_balance = self.initial_cash
        self.inventory = np.zeros(self.num_stocks, dtype=int)
        self.total_trades = 0

        self.asset_history = [self.initial_cash]

        initialize_observation = self._get_obs()
        return initialize_observation, {}

    def step(self, action=None) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one time step within the environment"""
        current_prices = self.price_memory[self.current_step]
        step_trades = 0

        for i, act in enumerate(action):
            price = current_prices[i]
            if price == 0 or np.isnan(price):
                continue

            if act == TradeStatus.BUY:
                cost = price * 1000
                transaction_fee = max(
                    cost * self.transaction_fee_rate, self.min_transaction_fee
                )
                if self.cash_balance >= cost + transaction_fee:
                    self.cash_balance -= cost + transaction_fee
                    self.inventory[i] += 1000
                    step_trades += 1

            elif act == TradeStatus.SELL:
                if self.inventory[i] > 0:
                    revenue = (
                        price * 1000 * (1 - self.tax_rate - self.transaction_fee_rate)
                    )
                    self.cash_balance += revenue
                    self.inventory[i] -= 1000
                    step_trades += 1

            elif act == TradeStatus.HOLD:
                pass

        self.total_trades += step_trades

        # Next Day
        self.current_step += 1

        # Recalculate asset
        terminated = self.current_step >= self.max_steps
        truncated = False
        next_prices = (
            self.price_memory[self.current_step] if not terminated else current_prices
        )

        # Reward
        stock_value = np.sum(self.inventory * next_prices)
        current_total_asset = self.cash_balance + stock_value
        previous_total_asset = self.asset_history[-1]
        reward = float(
            np.log(max(current_total_asset, 1) / max(previous_total_asset, 1))
        )

        if terminated and self.total_trades < 100:
            reward -= 1

        self.asset_history.append(current_total_asset)

        info = {
            "total_asset": current_total_asset,
            "total_trades": self.total_trades,
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
        return obs, float(reward), terminated, truncated, info
