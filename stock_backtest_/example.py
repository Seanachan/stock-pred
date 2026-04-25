from backtest.Stock_API import Stock_API

# 用戶登入
user = Stock_API("帳號", "密碼")

# 查看用戶庫存
print(user.Get_User_Stocks())

# 查看股票資料
stock_id = "2330"  # 台積電
start_date = "20250217"  # 起始時間 格式YYYYMMDD
end_date = "20250221"  # 結束時間 格式YYYYMMDD
print(Stock_API.Get_Stock_Informations(stock_id, start_date, end_date))

# 購買股票
stock_id = "2454"  # 聯發科
share = 1  # 張數
price = 1353  # 價錢
user.Buy_Stock(stock_id, share, price)

# 售出股票
stock_id = "2330"  # 台積電
share = 1  # 張數
price = 1095  # 價錢
user.Sell_Stock(stock_id, share, price)
