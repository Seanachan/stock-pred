import os

from dotenv import load_dotenv

from stock_api import Get_User_Stocks, get_taiwan_stock_data
from stock_api.core import Buy_Stock

load_dotenv()

ACCOUNT = os.getenv("ACCOUNT")
PASSWORD = os.getenv("PASSWORD")
assert ACCOUNT is not None, "Please set ACCOUNT in .env file"
assert PASSWORD is not None, "Please set PASSWORD in .env file"

# 取得股票資訊
df = get_taiwan_stock_data("2330", "2024-01-01", "2024-03-31")

# 取得持有股票
user_stocks = Get_User_Stocks(ACCOUNT, PASSWORD)

print(get_taiwan_stock_data("2330", "2024-01-01", "2024-03-31"))
# 預約購入股票
Buy_Stock(ACCOUNT, PASSWORD, 2330, 1, 1975)

# 預約售出股票
# Sell_Stock(ACCOUNT, PASSWORD, 2330, 1, 1978)
