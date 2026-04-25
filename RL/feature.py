from typing import Dict, List

import numpy as np
from pandas import DataFrame
from pandas_ta.momentum import macd, rsi
from pandas_ta.overlap import sma
from pandas_ta.volatility import atr, bbands


class FeatureExtractor:
    def __init__(self, stock_ids: List[str]):
        self.stock_ids = stock_ids
        self.feature_dim_per_stock = 8

    def extract_features(
        self, stock_data: Dict[str, DataFrame]
    ) -> Dict[str, DataFrame]:
        """
        Transform history stock data into feature vectors for each stock.
        """
        market_dfs: Dict[str, DataFrame] = {}

        for stock_id, df in stock_data.items():
            # Ensure we calculate 20MA, append 0s if not enough data
            if df is None or len(df) < 30:
                market_dfs[stock_id] = DataFrame(
                    np.zeros((1, self.feature_dim_per_stock), dtype=float)
                )
                continue

            df_copy = df.copy()

            # Daily returns
            df_copy["return"] = df_copy["close"].pct_change()

            # Bias & SMA
            df_copy["sma_5"] = sma(df_copy["close"], length=5)
            df_copy["bias_5"] = (df_copy["close"] / df_copy["sma_5"]) - 1

            df_copy["sma_20"] = sma(df_copy["close"], length=20)
            df_copy["bias_20"] = (df_copy["close"] / df_copy["sma_20"]) - 1

            # MACD
            _macd = macd(df_copy["close"])
            df_copy["macd_h"] = (
                _macd[_macd.columns[1]] / df_copy["close"] if _macd is not None else 0.0
            )

            # RSI
            df_copy["rsi_14"] = rsi(df_copy["close"], length=14) / 100

            # Boolinger Bands
            _bbands = bbands(df_copy["close"], length=20)
            df_copy["bb_pos"] = (
                (df_copy["close"] - _bbands[_bbands.columns[0]])
                / (_bbands[_bbands.columns[2]] - _bbands[_bbands.columns[0]])
                if _bbands is not None
                else 0.5
            )

            # ATR (AverageTrue Range)
            df_copy["atr"] = (
                atr(df_copy["high"], df_copy["low"], df_copy["close"], length=14)
                / df_copy["close"]
            )

            # Capacity Change
            df_copy["capacity_change"] = df_copy["capacity"].pct_change()

            market_dfs[stock_id] = df_copy

        return market_dfs
