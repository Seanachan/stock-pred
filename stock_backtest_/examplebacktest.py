import os
from typing import List

from dotenv import load_dotenv

from backtest.backtest import *

load_dotenv()


def stupidmethod(
    all_stock_info: List[Stock_Information],
    previous_stock_pool: List[Stock_Information],
    user_cash: float,
) -> List[str]:
    stock_pool = []
    for stock in all_stock_info:
        prev_close = stock.rolling(1).price_close  # 取得前一天收盤價

        if stock.price_close is not None and prev_close is not None and prev_close != 0:
            price_change = (stock.price_close - prev_close) / prev_close * 100

            if price_change > 7:  # 漲幅大於 7%
                stock_pool.append(stock.stock_code)

    return stock_pool


def stupidtrademethod(
    stock_code_list: List[Stock_Information],
    user_inventory: List[User_Inventory],
    user_cash: float,
    transaction_tool: Transaction_Tool,
):

    shares_per_month = 10
    for stock in stock_code_list:
        previous_day = stock.rolling(1)

        if previous_day.price_close is None or stock.price_close is None:
            continue  # 跳過沒有數據的股票

        if previous_day.price_close < stock.price_close:
            transaction_tool.buy_stock(
                stock.stock_code, stock.price_close, shares_per_month
            )

    for user_stock in user_inventory:
        previous_day = user_stock.stock_info.rolling(1)

        if (
            previous_day.price_close is None
            or user_stock.stock_info.price_close is None
        ):
            continue  # 跳過沒有數據的股票
        if previous_day.price_close > user_stock.stock_info.price_close:
            transaction_tool.sell_stock(
                user_stock.stock_info.stock_code,
                user_stock.stock_info.price_close,
                user_stock.shares,
            )

    return transaction_tool.transaction_record


if __name__ == "__main__":
    # 用戶登入
    account = os.getenv("ACCOUNT")
    password = os.getenv("PASSWORD")
    backtest = BacktestSystem(account, password)

    # 讓使用者輸入策略名稱
    strategy_name = "Test BOT"

    # 設定回測期間
    start_date = "20250220"
    end_date = "20250530"
    backtest.set_backtest_period(start_date, end_date)

    # 設定初始資金 (course spec: 100,000,000 虛擬幣)
    cash_balance = 100_000_000
    backtest.set_cash_balance(cash_balance)

    # 執行回測
    backtest.execute_strategy(strategy_name, stupidmethod, stupidtrademethod)

    # 計算回測績效
    performance, performance_detail = backtest.calculate_performance()

    if performance:
        print("回測績效計算完成，開始儲存 XLS 檔案與圖表...")
        backtest.save_performance_to_xls(
            strategy_name, performance, performance_detail, "backtest_performance.xlsx"
        )

        # 新增：讀取回測績效摘要數據
        performance_data = backtest.read_xls_performance("backtest_performance.xlsx")
        print("讀取並轉換回測績效數據：")
        # print(performance_data)

        # 新增：上傳回測績效摘要數據至 API
        print("上傳回測績效數據至 API...")
        backtest.upload_performance_to_web()

        # 如果需要上傳明細數據，也呼叫新的 API
        print("上傳回測績效明細數據至 API...")
        backtest.upload_performance_detail_to_web()

        print("上傳回測績效已實現紀錄數據至 API...")
        backtest.upload_performance_record_to_web()
    else:
        print("無法計算回測績效，請檢查交易數據！")
