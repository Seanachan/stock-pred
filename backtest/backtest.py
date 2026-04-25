import os
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

import pandas as pd
import requests
import yaml
from tqdm import tqdm

from backtest.Stock_API import Stock_API

from .type import (
    Daily_Asset,
    Error_Status,
    StockPerformance,
    StockPerformanceDetail,
    TransactionRecord,
    TransactionRecordHistory,
)


class InventoryManager:
    def __init__(self):
        self.stocks: Dict[str, pd.DataFrame] = {}

    def get_all_stock_id(self) -> List[str]:
        return list(self.stocks.keys())

    def buy(self, ticker, record_date, price, quantity):
        """
        買入操作，根據股票代碼將買入數量和價格加入庫存。
        :param ticker: 股票代碼
        :param price: 買入價格
        :param quantity: 買入數量
        :param record_date: 買入日期
        """

        # print(self.stocks)
        if ticker not in self.stocks:
            self.stocks[ticker] = pd.DataFrame(
                columns=["buy_price", "quantity", "total_cost", "record_date"]
            )

        total_cost = price * quantity
        new_entry = pd.DataFrame(
            {
                "buy_price": [price],
                "quantity": [quantity],
                "total_cost": [total_cost],
                "record_date": [record_date],
            }
        )
        self.stocks[ticker] = pd.concat(
            [self.stocks[ticker], new_entry], ignore_index=True
        )

        return Error_Status(True, "買入成功")

    def sell(self, ticker, record_date, price, quantity):
        """
        賣出操作，根據股票代碼從庫存中減少相應數量，並計算損益。
        :param ticker: 股票代碼
        :param price: 賣出價格
        :param quantity: 賣出數量
        :param record_date: 賣出日期
        """

        if ticker not in self.stocks or self.stocks[ticker].empty:
            return -1, -1, -1, Error_Status(False, "庫存沒有股票")

        total_sold = quantity
        profit = 0
        total_cost = 0  # 用來計算加權平均價格
        total_shares_sold = 0  # 賣出的總股數

        while total_sold > 0 and not self.stocks[ticker].empty:
            buy_price, buy_quantity, total_cost_of_entry, buy_record_date = self.stocks[
                ticker
            ].iloc[0]

            if buy_quantity <= total_sold:
                # 如果庫存中的數量少於或等於賣出數量
                profit += (price - buy_price) * buy_quantity
                total_shares_sold += buy_quantity
                total_sold -= buy_quantity
                total_cost += buy_price * buy_quantity
                self.stocks[ticker] = (
                    self.stocks[ticker].drop(0).reset_index(drop=True)
                )  # 從庫存中移除這筆買入
            else:
                # 如果庫存中的數量大於賣出數量
                profit += (price - buy_price) * total_sold
                total_shares_sold += total_sold
                total_cost += buy_price * total_sold
                self.stocks[ticker].at[0, "total_cost"] = (
                    self.stocks[ticker].at[0, "total_cost"] - buy_price * total_sold
                )
                self.stocks[ticker].at[0, "quantity"] = (
                    self.stocks[ticker].at[0, "quantity"] - total_sold
                )  # type: ignore
                total_sold = 0  # 完成賣出
        if self.stocks[ticker].empty:
            del self.stocks[ticker]

        if total_shares_sold > 0:
            average_buy_price = total_cost / total_shares_sold
        else:
            average_buy_price = 0  # 若無賣出，則平衡價為 0

        return (
            profit,
            total_shares_sold,
            average_buy_price,
            Error_Status(True, "賣出成功"),
        )

    def get_balance_profit(self, ticker):
        if ticker not in self.stocks or self.stocks[ticker].empty:
            return 0
        return self.stocks[ticker]["total_cost"].sum()

    def get_inventory(self, ticker):
        if ticker not in self.stocks or self.stocks[ticker].empty:
            return pd.DataFrame(
                columns=["buy_price", "quantity", "total_cost", "record_date"]
            )
        return self.stocks[ticker]

    def get_average_cost(self, ticker):
        if ticker not in self.stocks or self.stocks[ticker].empty:
            return 0
        total_shares = self.stocks[ticker]["quantity"].sum()
        total_cost = self.stocks[ticker]["total_cost"].sum()
        return total_cost / total_shares if total_shares > 0 else 0

    def get_shares(self, ticker):
        if ticker not in self.stocks or self.stocks[ticker].empty:
            return 0
        total_shares = self.stocks[ticker]["quantity"].sum()
        return total_shares if total_shares > 0 else 0

    def calculate_stock_assets(self, ticker, today_price):
        """
        計算目前的未實現損益。
        :param ticker: 股票代碼
        :param today_price: 今日的股價
        :return: 總市值
        """
        if ticker not in self.stocks or self.stocks[ticker].empty:
            return None

        inventory = self.stocks[ticker]
        stock_assets = 0

        # 計算每筆買入的損益，並累加到總損益
        for index, row in inventory.iterrows():
            quantity = row["quantity"]
            # 計算每筆買入的損益
            stock_assets = stock_assets + today_price * quantity

        return stock_assets

    def calculate_unrealized_profit(self, ticker, today_price):
        """
        計算目前的未實現損益。
        :param ticker: 股票代碼
        :param today_price: 今日的股價
        :return: 實現損益
        """
        if ticker not in self.stocks or self.stocks[ticker].empty:
            return None

        inventory = self.stocks[ticker]
        unrealized_profit = 0

        # 計算每筆買入的損益，並累加到總損益
        for index, row in inventory.iterrows():
            buy_price = row["buy_price"]
            quantity = row["quantity"]
            # 計算每筆買入的損益
            unrealized_profit = unrealized_profit + (today_price - buy_price) * quantity

        return unrealized_profit


class Stock_Information_Memory:
    stock_data = pd.DataFrame()

    @classmethod
    def load_yaml_to_dict(cls, file_path):
        """
        從 YAML 檔案讀取並轉換為字典。
        :param file_path: YAML 檔案路徑
        :return: 轉換後的字典
        """
        try:
            with open(file_path, "r") as yaml_file:
                # 使用 FullLoader 來加載 YAML 並轉換為字典
                return yaml.load(yaml_file, Loader=yaml.FullLoader)
        except Exception as e:
            print(f"讀取 YAML 檔案時出錯: {e}")
            return {}

    @classmethod
    def load_all_stock_data(
        cls, stock_code_list: List[str], start_date: str, end_date: str
    ):

        if os.path.exists("./data/stock_data.csv") and os.path.exists(
            "./data/save_data_info.yaml"
        ):
            yaml_dict = Stock_Information_Memory.load_yaml_to_dict(
                "./data/save_data_info.yaml"
            )

            yaml_start_date = datetime.strptime(yaml_dict["start_date"], "%Y%m%d")
            yaml_end_date = datetime.strptime(yaml_dict["end_date"], "%Y%m%d")

            _start_date = datetime.strptime(start_date, "%Y%m%d") - timedelta(90)
            _end_date = datetime.strptime(end_date, "%Y%m%d")

            if _start_date >= yaml_start_date and _end_date <= yaml_end_date:
                print("載入現有股票資訊")
                cls.stock_data = pd.read_csv(
                    "./data/stock_data.csv", dtype={"stock_code": str, "date": str}
                )
                cls.stock_data.set_index(["stock_code", "date"], inplace=True)
                cls.stock_data.sort_index(inplace=True)
                return

        print("載入所有股票數據中...")
        all_data = []
        # 提前拿三個月資料以免沒有前幾天資料
        start_date = (datetime.strptime(start_date, "%Y%m%d") - timedelta(90)).strftime(
            "%Y%m%d"
        )
        for stock_code in tqdm(stock_code_list, desc="下載股票資料"):
            result = Stock_API.Get_Stock_Informations(stock_code, start_date, end_date)

            if not isinstance(result, list):
                print(f"警告: {stock_code} API 回傳數據格式錯誤，跳過")
                continue

            for data in result:
                if not isinstance(data, dict):
                    print(f"錯誤: {stock_code} 的資料格式不正確 -> {data}")
                    continue

                if "date" not in data or "close" not in data:
                    print(f"缺少關鍵欄位: {stock_code} - {data}")
                    continue

                # 確保 `stock_code` 也加進 data
                data["stock_code"] = stock_code
                data["date"] = datetime.utcfromtimestamp(data["date"]).strftime(
                    "%Y%m%d"
                )
                all_data.append(data)

        if not all_data:
            print("警告: 沒有可用的股票數據，請檢查 API 回應")
            return

        cls.stock_data = pd.DataFrame(all_data)
        cls.stock_data["stock_code"] = cls.stock_data["stock_code"].astype(str)

        # 檢查 DataFrame 是否包含 `stock_code` 和 `date`
        if (
            "stock_code" not in cls.stock_data.columns
            or "date" not in cls.stock_data.columns
        ):
            print("錯誤: stock_data 缺少 stock_code 或 date 欄位，請檢查數據格式")
            print(cls.stock_data.head())  # Debugging
            return

        cls.stock_data.set_index(["stock_code", "date"], inplace=True)
        cls.stock_data.sort_index(inplace=True)
        print("股票數據載入完成！")  # 這行移到迴圈外部，確保只執行一次

        # 儲存資料
        if not os.path.exists("./data"):
            os.mkdir("./data")
        Stock_Information_Memory.save_dict_to_yaml(
            {"start_date": start_date, "end_date": end_date},
            "./data/save_data_info.yaml",
        )
        cls.stock_data.to_csv("./data/stock_data.csv")

    @classmethod
    def save_dict_to_yaml(cls, data_dict, file_path):
        """
        將字典儲存為 YAML 格式的檔案。
        :param data_dict: 要儲存的字典
        :param file_path: 儲存的檔案路徑
        """
        try:
            with open(file_path, "w") as yaml_file:
                yaml.dump(
                    data_dict, yaml_file, default_flow_style=False, allow_unicode=True
                )
            print(f"字典已成功儲存到 {file_path}")
        except Exception as e:
            print(f"儲存 YAML 檔案時出錯: {e}")

    @classmethod
    def get_stock_information(cls, stock_code: str, date: str):
        """從快取的 DataFrame 查詢股票資料"""
        try:
            row = cls.stock_data.loc[[(stock_code, date)]]
            if row.empty:
                return {}
            return row.iloc[0].to_dict()
        except KeyError:
            return {}


class Stock_Information:
    def __init__(self, stock_code, init_time: datetime):

        self.stock_code = stock_code
        self.init_time = init_time  # 格式 YYYYmmdd

    def rolling(self, bias):
        if bias < 0:
            raise ValueError("rolling bias can not negative")
        rolling_time = self.init_time - timedelta(bias)
        stock_info = Stock_Information(self.stock_code, rolling_time)
        # 以防無資料，向前找近期開盤
        for i in range(1, 20):
            if stock_info.price_close is not None:
                break
            else:
                rolling_time = rolling_time - timedelta(i)
                stock_info = Stock_Information(self.stock_code, rolling_time)

        return stock_info

    def get_price(self, price_type: str):
        # 通用方法查詢價格資訊
        init_time_str = self.init_time.strftime("%Y%m%d")
        stock_info = Stock_Information_Memory.get_stock_information(
            self.stock_code, init_time_str
        )

        price = stock_info.get(price_type)
        if price is None:
            return None
        return price

    @property
    def price_close(self):
        return self.get_price("close")

    @property
    def price_open(self):
        return self.get_price("open")

    @property
    def price_high(self):
        return self.get_price("high")

    @property
    def price_low(self):
        return self.get_price("low")

    @property
    def price_change_amount(self):
        """今日收盤價 - 昨日收盤價"""
        prev_day = self.rolling(1)
        if prev_day.price_close is None or self.price_close is None:
            return None
        return round(self.price_close - prev_day.price_close, 2)

    @property
    def price_change_percent(self):
        """漲跌幅 (%)"""
        prev_day = self.rolling(1)
        if prev_day.price_close is None or self.price_close is None:
            return None
        return round(
            (self.price_close - prev_day.price_close) / prev_day.price_close * 100, 2
        )


class User_Inventory:
    def __init__(self, stock_info: Stock_Information, avg_price, shares):
        self.stock_info = stock_info
        self.avg_price = avg_price
        self.shares = shares

    def __repr__(self):
        members = ", ".join(f"{key}={value}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({members})"

    def model_dump(self) -> Dict:
        return {
            key: self._convert_to_dict(value) for key, value in self.__dict__.items()
        }

    def _convert_to_dict(self, value):
        if isinstance(value, list):
            return [self._convert_to_dict(item) for item in value]
        elif hasattr(value, "__dict__"):
            return {
                key: self._convert_to_dict(val) for key, val in value.__dict__.items()
            }
        else:
            return value


class Transaction_Tool:
    """
    使用者下單工具
    """

    def __init__(self):
        self.transaction_record: List[TransactionRecord] = []

    def buy_stock(self, stock_code, price, shares):
        self.transaction_record.append(TransactionRecord(stock_code, price, shares, 2))

    def sell_stock(self, stock_code, price, shares):
        self.transaction_record.append(TransactionRecord(stock_code, price, shares, 1))

    def __str__(self):
        return "\n".join(str(record) for record in self.transaction_record)

    def __repr__(self):
        return self.__str__()


class BacktestSystem:
    """回測系統"""

    def __init__(self, account, password):
        self.api = Stock_API(account, password)
        self.transaction_history: List[TransactionRecordHistory] = []
        self.current_date: Optional[datetime] = None
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None
        self.strategy_id = None
        self.strategy_name = None
        self.daily_asset_list: List[Daily_Asset] = []  # 用來記錄每日資產變化
        self._init_cash_balance: float = 0
        self.cash_balance: float = 0
        self.select_stock_func: Optional[Callable] = None
        self.trade_strategy_func: Optional[Callable] = None
        # 庫存管理
        self._user_inventory_manager = InventoryManager()

        self.__stock_code_list = Stock_API.get_all_stock_information()

    def _reset(self):
        # 重新設定
        self._user_inventory_manager = InventoryManager()
        self.transaction_history: List[TransactionRecordHistory] = []
        self.current_date = self.start_date
        self.daily_asset_list: List[Daily_Asset] = []
        self.cash_balance = self._init_cash_balance

    def __get_all_stock_information(self):
        """一次性下載所有股票的歷史數據"""
        assert self.start_date is not None and self.end_date is not None, (
            "set_backtest_period must be called before loading stock data"
        )
        Stock_Information_Memory.load_all_stock_data(
            self.__stock_code_list,
            self.start_date.strftime("%Y%m%d"),
            self.end_date.strftime("%Y%m%d"),
        )

    def set_cash_balance(self, cash_balance):
        self._init_cash_balance = cash_balance
        self.cash_balance = cash_balance

    def set_backtest_period(self, start_date, end_date):
        self.start_date = datetime.strptime(start_date, "%Y%m%d")
        self.end_date = datetime.strptime(end_date, "%Y%m%d")
        self.current_date = self.start_date

    def start_backtest(self):
        # 開始回測，確保數據已經下載
        self.__get_all_stock_information()
        print("回測開始！")

    def _current_stock_info(self, stock_id, current_date) -> Stock_Information:
        stock_info = Stock_Information(stock_id, current_date)
        # 以防無資料，向前找近期開盤
        for i in range(1, 20):
            if stock_info.price_close is not None:
                break
            else:
                stock_info = stock_info.rolling(i)
        return stock_info

    def next_day(self, pbar):
        # 前進到下一天，並記錄每日資產變化
        assert self.current_date is not None and self.end_date is not None
        cash_balance = self.cash_balance
        total_stock_assets = 0

        for stock_id in self._user_inventory_manager.get_all_stock_id():
            stock_info = self._current_stock_info(stock_id, self.current_date)
            stock_assets = self._user_inventory_manager.calculate_stock_assets(
                stock_id, stock_info.price_close
            )
            if stock_assets is None:
                continue

            total_stock_assets = total_stock_assets + stock_assets

        # 將當日資產變化加入
        self.daily_asset_list.append(
            Daily_Asset(self.current_date, total_stock_assets, cash_balance)
        )

        # 前進一天 (跳過未開盤)
        while self.current_date <= self.end_date:
            self.current_date += timedelta(days=1)
            pbar.update(1)
            # 只到有開盤日期(以台積電為基準)
            stock_ref = Stock_Information("2330", self.current_date)
            if stock_ref.price_close is not None and stock_ref.price_close != {}:
                break

        return self.current_date <= self.end_date

    def set_select_stock(self, func):
        self.select_stock_func = func

    def select_stock(self, current_date, previous_stock_pool) -> List[str]:
        # 執行選股策略
        stock_info_list = [
            Stock_Information(stock_code, current_date)
            for stock_code in self.__stock_code_list
        ]
        previous_stock_pool_list = [
            Stock_Information(stock_code, current_date)
            for stock_code in previous_stock_pool
        ]
        assert self.select_stock_func is not None, (
            "set_select_stock must be called first"
        )
        return self.select_stock_func(
            stock_info_list, previous_stock_pool_list, self.cash_balance
        )

    def set_trade_strategy(self, func):
        self.trade_strategy_func = func

    def trade_strategy(
        self, stock_code_list: List[str], current_date
    ) -> List[TransactionRecord]:
        stock_info_list = [
            Stock_Information(stock_code, current_date)
            for stock_code in stock_code_list
        ]
        user_inventory = [
            User_Inventory(
                Stock_Information(stock_code, current_date),
                self._user_inventory_manager.get_average_cost(stock_code),
                self._user_inventory_manager.get_shares(stock_code),
            )
            for stock_code in self._user_inventory_manager.get_all_stock_id()
        ]
        assert self.trade_strategy_func is not None, (
            "set_trade_strategy must be called first"
        )
        return self.trade_strategy_func(
            stock_info_list, user_inventory, self.cash_balance, Transaction_Tool()
        )

    def execute_strategy(
        self, strategy_name, select_stock_strategy_func, trade_strategy_func
    ):

        # 通用策略執行器，使用自定義交易策略函式。

        self.set_select_stock(select_stock_strategy_func)  # 設定選股策略
        self.set_trade_strategy(trade_strategy_func)  # 設定交易策略
        self.__get_all_stock_information()

        assert self.current_date is not None and self.end_date is not None, (
            "set_backtest_period must be called before execute_strategy"
        )
        total_steps = (self.end_date - self.current_date).days

        print("執行策略: 使用者定義交易策略")

        stock_code_list = []

        with tqdm(total=total_steps, desc="Loading...") as pbar:
            while self.current_date <= self.end_date:
                transaction_record = self.trade_strategy(
                    stock_code_list, self.current_date
                )

                for unit_transaction_record in transaction_record:
                    if unit_transaction_record.action == 2:
                        self.buy_stock(unit_transaction_record, self.current_date)
                    elif unit_transaction_record.action == 1:
                        self.sell_stock(unit_transaction_record, self.current_date)

                stock_code_list = self.select_stock(self.current_date, stock_code_list)

                self.next_day(pbar)

        # stock_performance , stock_performance_detail = self.calculate_performance()

        # self.save_performance_to_xls(strategy_name, stock_performance , stock_performance_detail)

        # self._reset()

    def buy_stock(self, transaction_record: TransactionRecord, date: datetime):
        """
        回測買入

        transaction_record : 交易紀錄
        date : 交易時間 YYYYmmdd

        """

        stock_code = transaction_record.stock_code
        buy_price = transaction_record.stock_price
        shares = transaction_record.shares

        stock_info = Stock_Information(stock_code, date)
        # 執行買入操作
        if stock_info.price_low <= buy_price and stock_info.price_high >= buy_price:
            total_cost = buy_price * shares
            if self.cash_balance >= total_cost:
                error_status = self._user_inventory_manager.buy(
                    stock_code, date, buy_price, shares
                )
                if error_status.status:
                    self.cash_balance -= total_cost
            else:
                error_status = Error_Status(False, "現金不足")

        else:
            error_status = Error_Status(False, "交易價格不在股價區間")
        self.transaction_history.append(
            TransactionRecordHistory(
                stock_code, date, buy_price, -1, shares, -1, 2, error_status
            )
        )

    def sell_stock(self, transaction_record: TransactionRecord, date: datetime):
        """
        回測賣出

        transaction_record : 交易紀錄
        date : 交易時間 YYYYmmdd

        """
        # 執行賣出操作
        stock_code = transaction_record.stock_code
        sell_price = transaction_record.stock_price
        shares = transaction_record.shares

        average_buy_price = -1
        total_shares_sold = -1
        profit = -1

        stock_info = Stock_Information(stock_code, date)
        # 執行買入操作
        if stock_info.price_low <= sell_price and stock_info.price_high >= sell_price:
            profit, total_shares_sold, average_buy_price, error_status = (
                self._user_inventory_manager.sell(stock_code, date, sell_price, shares)
            )
            if error_status.status and total_shares_sold != -1:
                self.cash_balance = self.cash_balance + sell_price * total_shares_sold

        else:
            error_status = Error_Status(False, "交易價格不在股價區間")

        self.transaction_history.append(
            TransactionRecordHistory(
                stock_code,
                date,
                average_buy_price,
                sell_price,
                total_shares_sold,
                profit,
                1,
                error_status,
            )
        )

    def calculate_performance(self):

        print("開始計算回測績效...")
        # 計算所有成功賣出之股票歷史
        realized_profit = 0
        for unit_transaction_history in self.transaction_history:
            if (
                unit_transaction_history.action == 1
                and unit_transaction_history.error_status.status
            ):
                realized_profit = realized_profit + unit_transaction_history.profit

        # 個別計算股票庫存
        unrealized_profit = 0
        for stock_id in self._user_inventory_manager.get_all_stock_id():
            stock_info = self._current_stock_info(stock_id, self.end_date)
            unit_unrealized_profit = (
                self._user_inventory_manager.calculate_unrealized_profit(
                    stock_id, stock_info.price_close
                )
            )
            if unit_unrealized_profit is None:
                continue
            unrealized_profit = unrealized_profit + unit_unrealized_profit

        total_assets = (
            self.daily_asset_list[-1].cash_assets
            + self.daily_asset_list[-1].stock_assets
        )

        # 獲取當前的日期和時間
        current_time = datetime.today()
        # 以字串形式顯示當前時間
        today = current_time.strftime("%Y-%m-%d %H:%M:%S")
        cash_assets = self.daily_asset_list[-1].cash_assets
        stock_performance = StockPerformance(
            self.start_date,
            self.end_date,
            today,
            realized_profit,
            unrealized_profit,
            total_assets,
            cash_assets,
        )

        historical_transactions = []

        for unit_transaction_history in self.transaction_history:
            if unit_transaction_history.error_status:
                historical_transactions.append(unit_transaction_history)

        stock_performance_detail = StockPerformanceDetail(
            historical_transactions, self.daily_asset_list
        )
        return stock_performance, stock_performance_detail

    def save_performance_to_xls(
        self,
        strategy_name,
        performance: StockPerformance,
        stock_performance_detail: StockPerformanceDetail,
        filename="backtest_performance.xlsx",
    ):
        """將回測績效存入 XLS 檔案"""

        print("儲存回測績效至 Excel...")

        writer = pd.ExcelWriter(filename, engine="xlsxwriter")

        # === (0) 計算總收益率 (Total Return) ===
        if self._init_cash_balance > 0:
            total_return = (
                (performance.total_assets - self._init_cash_balance)
                / self._init_cash_balance
                * 100
            )
        else:
            total_return = 0  # 避免除以零

        # === (1) 儲存「回測績效」(摘要) ===
        summary_data = {
            "策略名稱": [f"{strategy_name}"],
            "起始期間": [f"{performance.start_date}"],
            "結束期間": [f"{performance.end_date}"],
            "更新時間": [performance.update_time],
            "總收益率 (%)": [f"{total_return:.2f}%"],
            "已實現收益": [f"{performance.realized_profit:.4f}"],
            "未實現收益": [f"{performance.unrealized_profit:.4f}"],
            "最終現金餘額": [performance.cash_assets],
            "最終總資產價值": [performance.total_assets],
        }
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_excel(writer, sheet_name="回測績效", index=False)

        # === (2) 儲存「歷史已實現損益」：只保留賣出且成功的交易 ===
        #     新增「買入價格」與「獲利率(%)」欄位
        realized_data = {
            "股票代碼": [],
            "交易日期": [],
            "買入價格": [],
            "賣出價格": [],
            "交易股數": [],
            "獲利": [],
            "獲利率(%)": [],
        }

        for trans in stock_performance_detail.historical_transactions_list:
            if trans.action == 1 and trans.error_status.status:
                # 計算獲利率 = profit / (buy_price * shares) * 100
                buy_price = trans.buy_price
                shares = trans.shares
                profit = trans.profit
                if buy_price > 0 and shares > 0:
                    profit_rate = (profit / (buy_price * shares)) * 100
                else:
                    profit_rate = 0

                realized_data["股票代碼"].append(trans.stock_code)
                realized_data["交易日期"].append(trans.date)
                realized_data["買入價格"].append(
                    buy_price
                )  # 顯示賣出交易時的平均買入價格
                realized_data["賣出價格"].append(trans.sell_price)
                realized_data["交易股數"].append(shares)
                realized_data["獲利"].append(profit)
                realized_data["獲利率(%)"].append(f"{profit_rate:.2f}")

        df_realized = pd.DataFrame(realized_data)
        df_realized.to_excel(writer, sheet_name="歷史已實現損益", index=False)

        # === (3) 儲存「歷史交易委託」：合併買入/賣出價格 => 「成交價格」，移除「獲利」欄位
        order_data = {
            "股票代碼": [],
            "交易日期": [],
            "交易動作": [],
            "成交價格": [],
            "交易股數": [],
        }

        for trans in stock_performance_detail.historical_transactions_list:
            # 判斷交易動作，轉成「買入」或「賣出」
            if trans.action == 2:  # 買入
                action_str = "買入"
                deal_price = trans.buy_price
            elif trans.action == 1:  # 賣出
                action_str = "賣出"
                deal_price = trans.sell_price
            else:
                action_str = "不動作"
                deal_price = None

            order_data["股票代碼"].append(trans.stock_code)
            order_data["交易日期"].append(trans.date)
            order_data["交易動作"].append(action_str)
            order_data["成交價格"].append(deal_price)
            order_data["交易股數"].append(trans.shares)

        df_orders = pd.DataFrame(order_data)
        df_orders.to_excel(writer, sheet_name="歷史交易委託", index=False)

        # === (4) 儲存「每日資產變化」 ===
        daily_data = {
            "記錄日期": [],
            "股票資產價值": [],
            "現金資產": [],
        }
        for unit_daily in stock_performance_detail.daily_values_list:
            daily_data["記錄日期"].append(unit_daily.record_date)
            daily_data["股票資產價值"].append(unit_daily.stock_assets)
            daily_data["現金資產"].append(unit_daily.cash_assets)
        df_daily = pd.DataFrame(daily_data)
        df_daily.to_excel(writer, sheet_name="每日資產變化", index=False)

        # === (5) 儲存「委託紀錄」 ===
        transaction_history_dict = {
            "委託日期": [],
            "交易價錢": [],
            "交易股數": [],
            "交易動作": [],
            "狀態": [],
            "委託說明": [],
        }
        for unit_transaction_history in self.transaction_history:
            transaction_history_dict["委託日期"].append(unit_transaction_history.date)
            transaction_history_dict["交易股數"].append(unit_transaction_history.shares)

            if unit_transaction_history.action == 1:
                transaction_history_dict["交易價錢"].append(
                    unit_transaction_history.sell_price
                )
            elif unit_transaction_history.action == 2:
                transaction_history_dict["交易價錢"].append(
                    unit_transaction_history.buy_price
                )
            else:
                transaction_history_dict["交易價錢"].append(None)

            if unit_transaction_history.action == 1:
                transaction_history_dict["交易動作"].append("賣出")
            elif unit_transaction_history.action == 2:
                transaction_history_dict["交易動作"].append("買入")
            else:
                transaction_history_dict["交易動作"].append("不動作")

            if unit_transaction_history.error_status.status:
                transaction_history_dict["狀態"].append("成功")
            else:
                transaction_history_dict["狀態"].append("失敗")

            transaction_history_dict["委託說明"].append(
                unit_transaction_history.error_status.message
            )

        df_transaction_history = pd.DataFrame(transaction_history_dict)
        df_transaction_history.to_excel(writer, sheet_name="委託歷史紀錄", index=False)

        # === (6) 關閉 Writer ===
        writer.close()
        print(f"回測績效已成功儲存至 {filename}")

    def read_xls_performance(self, file_path):
        # 讀取 Excel 檔案
        file_path = "backtest_performance.xlsx"
        df = pd.read_excel(file_path, sheet_name="回測績效")
        df = df.astype(str)  # 轉換為字串，避免 NaN 問題

        # **將第一筆數據存入 `data` 變數**
        data = df.to_dict(orient="records")[0]  # 取第一筆數據

        # 建立 **中 -> 英** 的 Key 對應字典
        key_mapping = {
            "策略名稱": "strategy_name",
            "起始期間": "start_date",
            "結束期間": "end_date",
            "更新時間": "update_time",
            "總收益率 (%)": "total_return",
            "已實現收益": "realized_profit",
            "未實現收益": "unrealized_profit",
            "最終現金餘額": "cash_balance",
            "最終總資產價值": "total_value",
        }

        # 轉換 Key
        translated_data = {
            db_key: data[excel_key]
            for excel_key, db_key in key_mapping.items()
            if excel_key in data
        }

        # 輸出結果
        translated_data["account"] = self.api.account
        translated_data["password"] = self.api.password
        translated_data["stock_code"] = None

        # 輸出結果
        return translated_data

    def read_xls_performance_detail(self, file_path: str):
        """
        讀取 Excel 檔案中「歷史交易委託」工作表，
        並將其轉換成適合後續使用的結構（只處理委託紀錄）。
        """

        df_orders = pd.read_excel(file_path, sheet_name="歷史交易委託")
        orders_records = df_orders.to_dict(orient="records")

        orders_mapping = {
            "股票代碼": "stock_code",
            "交易日期": "transaction_date",
            "交易動作": "transaction_action",
            "成交價格": "deal_price",  # 先暫存成交價格
            "交易股數": "transaction_shares",
        }

        orders_data_converted = []
        for row in orders_records:
            converted_row = {}
            for excel_col, db_col in orders_mapping.items():
                value = row.get(excel_col, None)
                if isinstance(value, pd.Timestamp):
                    value = value.strftime("%Y-%m-%d %H:%M:%S")
                converted_row[db_col] = value
            # 根據交易動作將成交價格分別放入買入價格或賣出價格
            action = converted_row.get("transaction_action", "")
            deal_price = converted_row.pop("deal_price", None)
            if action == "買入":
                converted_row["deal_price"] = deal_price
            elif action == "賣出":
                converted_row["deal_price"] = deal_price

            # 委託紀錄一般不包含獲利資訊
            converted_row["profit"] = None
            converted_row["profit_rate"] = None
            converted_row["strategy_id"] = None
            converted_row["stock_description"] = None

            orders_data_converted.append(converted_row)

        return orders_data_converted

    def read_xls_performance_record(self, file_path: str):
        """
        讀取 Excel 檔案中「歷史已實現損益」工作表，
        並將其轉換成適合後續使用的結構（僅包含賣出交易的資料）。
        """

        df_realized = pd.read_excel(file_path, sheet_name="歷史已實現損益")
        realized_records = df_realized.to_dict(orient="records")

        realized_mapping = {
            "股票代碼": "stock_code",
            "交易日期": "transaction_date",
            "買入價格": "buy_price",
            "賣出價格": "sell_price",
            "交易股數": "transaction_shares",
            "獲利": "profit",
            "獲利率(%)": "profit_rate",
        }

        realized_data_converted = []
        for row in realized_records:
            converted_row = {}
            for excel_col, db_col in realized_mapping.items():
                value = row.get(excel_col, None)
                if isinstance(value, pd.Timestamp):
                    value = value.strftime("%Y-%m-%d %H:%M:%S")
                converted_row[db_col] = value
            # 已實現損益工作表代表的都是賣出交易
            converted_row["transaction_action"] = "賣出"
            converted_row["strategy_id"] = None
            converted_row["stock_description"] = None
            realized_data_converted.append(converted_row)

        return realized_data_converted

    def upload_performance_to_web(self):
        """使用 `read_xls_performance()` 讀取績效數據並上傳到 API"""
        file_path = "backtest_performance.xlsx"

        # 讀取並轉換 XLS 數據
        data = self.read_xls_performance(file_path)

        # 確保 `stock_code` 不是 None
        if data["stock_code"] is None:
            data["stock_code"] = ""

        # 確保 `total_return` 為數字
        if isinstance(data["total_return"], str) and "%" in data["total_return"]:
            data["total_return"] = float(data["total_return"].replace("%", ""))

        # 確保 `update_time` 格式正確
        if " " in data["update_time"]:
            data["update_time"] = data["update_time"].split(" ")[0] + " 00:00:00"

        # API 端點
        api_url = (
            "http://140.116.86.242:8081/api/stock/api/v1/insert_backtest_performance"
        )
        print("發送數據到 API:")  # Debug: 檢查送出的數據

        # 發送 POST 請求
        response = requests.post(api_url, json=data)

        if response.status_code == 201:
            print("績效數據上傳成功！")
            # 從API取得strategy_id
            self.strategy_id = requests.get(
                f"http://140.116.86.242:8081/api/stock/get_strategy_id?strategy_name={
                    data['strategy_name']
                }"
            ).json()["data"]["strategy_id"]
            print(f"get strategy_id : {self.strategy_id}")
        else:
            print("上傳:", response.status_code, response.text)

    def upload_performance_detail_to_web(self):

        file_path = "backtest_performance.xlsx"
        # 讀取「歷史交易委託」資料，該函式只回傳一個 list（orders_data_converted）
        orders_data = self.read_xls_performance_detail(file_path)

        # API 端點
        api_url = "http://140.116.86.242:8081/api/stock/api/v1/insert_backtest_performance_detail"

        print("發送歷史交易委託數據到 API:")  # Debug: 檢查送出的數據

        # 假設 self.strategy_id 已存在，若不存在可先設定，例如 self.strategy_id = 1
        for line in orders_data:
            line["strategy_id"] = self.strategy_id  # 加入策略編號
            # print(line)
            response = requests.post(api_url, json=line)
            if response.status_code == 200:
                # print("一筆交易委託數據上傳成功！")
                pass
            else:
                print("上傳交易委託數據失敗:", response.status_code, response.text)

    def upload_performance_record_to_web(self):

        file_path = "backtest_performance.xlsx"
        # 讀取「歷史已實現損益」資料，該函式只回傳一個 list（realized_data_converted）
        record_data = self.read_xls_performance_record(file_path)

        # API 端點
        api_url = "http://140.116.86.242:8081/api/stock/api/v1/insert_backtest_performance_record"

        print("發送歷史已實現損益數據到 API:")  # Debug: 檢查送出的數據

        for line in record_data:
            line["strategy_id"] = self.strategy_id
            response = requests.post(api_url, json=line)
            if response.status_code == 200:
                # print("一筆歷史已實現損益數據上傳成功！")
                pass
            else:
                print(
                    "上傳歷史已實現損益數據失敗:", response.status_code, response.text
                )

    def save_result(self, account, password, name):
        pass
