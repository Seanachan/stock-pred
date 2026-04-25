from typing import List, Dict


class Error_Status:
    """
    錯誤訊息
    """

    def __init__(self, status, message):

        # 是否成功
        self.status = status
        # 狀態描述
        self.message = message

    def __repr__(self):
        members = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({members})"

    def model_dump(self) -> Dict:
        return {key: self._convert_to_dict(value) for key, value in self.__dict__.items()}

    def _convert_to_dict(self, value):
        if isinstance(value, list):
            return [self._convert_to_dict(item) for item in value]
        elif hasattr(value, "__dict__"):
            return {key: self._convert_to_dict(val) for key, val in value.__dict__.items()}
        else:
            return value


class TransactionRecord:
    """
    委託紀錄
    """

    def __init__(self, stock_code, stock_price, shares, action):
        self.stock_code = stock_code
        self.stock_price = stock_price
        self.shares = shares
        # """交易動作 action:0 不動作"""
        # """action:1 則賣出"""
        # """action:2 則買入"""
        self.action = action

    def __repr__(self):
        members = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({members})"

    def model_dump(self) -> Dict:
        return {key: self._convert_to_dict(value) for key, value in self.__dict__.items()}

    def _convert_to_dict(self, value):
        if isinstance(value, list):
            return [self._convert_to_dict(item) for item in value]
        elif hasattr(value, "__dict__"):
            return {key: self._convert_to_dict(val) for key, val in value.__dict__.items()}
        else:
            return value


class TransactionRecordHistory:
    """
    歷史交易紀錄
    """

    def __init__(
        self,
        stock_code,
        date,
        buy_price,
        sell_price,
        shares,
        profit,
        action,
        error_status: Error_Status,
    ):
        self.stock_code = stock_code
        self.date = date
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.shares = shares
        self.profit = profit
        # """交易動作 action:0 不動作"""
        # """action:1 則賣出"""
        # """action:2 則買入"""
        self.action = action

        self.error_status = error_status

    def __repr__(self):
        members = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({members})"

    def model_dump(self) -> Dict:
        return {key: self._convert_to_dict(value) for key, value in self.__dict__.items()}

    def _convert_to_dict(self, value):
        if isinstance(value, list):
            return [self._convert_to_dict(item) for item in value]
        elif hasattr(value, "__dict__"):
            return {key: self._convert_to_dict(val) for key, val in value.__dict__.items()}
        else:
            return value


class User_Inventory:
    def __init__(self, stock_code, record_date, price, shares):
        self.stock_code = stock_code
        self.record_date = record_date
        self.price = price
        self.shares = shares

    def __repr__(self):
        members = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({members})"

    def model_dump(self) -> Dict:
        return {key: self._convert_to_dict(value) for key, value in self.__dict__.items()}

    def _convert_to_dict(self, value):
        if isinstance(value, list):
            return [self._convert_to_dict(item) for item in value]
        elif hasattr(value, "__dict__"):
            return {key: self._convert_to_dict(val) for key, val in value.__dict__.items()}
        else:
            return value


class Daily_Asset:
    def __init__(self, record_date, stock_assets, cash_assets):
        self.record_date = record_date
        self.stock_assets = stock_assets
        self.cash_assets = cash_assets

    def __repr__(self):
        members = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({members})"

    def model_dump(self) -> Dict:
        return {key: self._convert_to_dict(value) for key, value in self.__dict__.items()}

    def _convert_to_dict(self, value):
        if isinstance(value, list):
            return [self._convert_to_dict(item) for item in value]
        elif hasattr(value, "__dict__"):
            return {key: self._convert_to_dict(val) for key, val in value.__dict__.items()}
        else:
            return value


class StockPerformanceDetail:
    def __init__(
        self,
        historical_transactions: List[TransactionRecordHistory],
        daily_values: List[Daily_Asset],
    ):
        # 歷史成交
        self.historical_transactions_list = historical_transactions
        # 歷史資產變化
        self.daily_values_list = daily_values

    def __repr__(self):
        members = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({members})"

    def model_dump(self) -> Dict:
        return {key: self._convert_to_dict(value) for key, value in self.__dict__.items()}

    def _convert_to_dict(self, value):
        if isinstance(value, list):
            return [self._convert_to_dict(item) for item in value]
        elif hasattr(value, "__dict__"):
            return {key: self._convert_to_dict(val) for key, val in value.__dict__.items()}
        else:
            return value


class StockPerformance:
    def __init__(
        self,
        start_date,
        end_date,
        update_time,
        realized_profit,
        unrealized_profit,
        total_assets,
        cash_assets,
    ):
        self.start_date = start_date
        self.end_date = end_date
        self.update_time = update_time
        self.realized_profit = realized_profit
        self.unrealized_profit = unrealized_profit
        self.total_assets = total_assets
        self.cash_assets = cash_assets

    def __repr__(self):
        members = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({members})"

    def model_dump(self) -> Dict:
        return {key: self._convert_to_dict(value) for key, value in self.__dict__.items()}

    def _convert_to_dict(self, value):
        if isinstance(value, list):
            return [self._convert_to_dict(item) for item in value]
        elif hasattr(value, "__dict__"):
            return {key: self._convert_to_dict(val) for key, val in value.__dict__.items()}
        else:
            return value
