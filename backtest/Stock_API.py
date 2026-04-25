# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 15:46:11 2023

@author: asus
"""

import requests
import time

from typing import List


class Stock_API:
    # 創建證券使用者資訊
    # Input:
    #   account: 使用者帳號
    #   password: 使用者密碼
    def __init__(self, account, password):
        self.account = account
        self.password = password

    # 取得股票資訊
    # Input:
    #   stock_code: 股票ID
    #   start_date: 開始日期，YYYYMMDD
    #   stop_date: 結束日期，YYYYMMDD
    # Output: 持有股票陣列
    @staticmethod
    def Get_Stock_Informations(stock_code, start_date, stop_date):
        # time.sleep(0.5)
        information_url = (
            "http://140.116.86.242:8081/stock/"
            + "api/v1/api_get_stock_info_from_date_json/"
            + str(stock_code)
            + "/"
            + str(start_date)
            + "/"
            + str(stop_date)
        )
        for i in range(5):
            try:
                result = requests.get(information_url).json()
                if result["result"] == "success":
                    return result["data"]
            except:
                print(f" connect fail, retry {i + 1} time")

        return dict([])

    # 取得持有股票
    # Input:
    #   account: 使用者帳號
    #   password: 使用者密碼
    # Output: 持有股票陣列
    def Get_User_Stocks(self):
        data = {"account": self.account, "password": self.password}
        search_url = "https://ciot.imis.ncku.edu.tw/sim_stock/trading_api/get_user_stocks"
        result = requests.post(search_url, data=data).json()
        if result["result"] == "success":
            return result["data"]
        return dict([])

    # 預約購入股票
    # Input:
    #   account: 使用者帳號
    #   password: 使用者密碼
    #   stock_code: 股票ID
    #   stock_shares: 購入張數
    #   stock_price: 購入價格
    # Output: 是否成功預約購入(True/False)
    def Buy_Stock(self, stock_code, stock_shares, stock_price):

        print("Buying stock...")
        data = {
            "account": self.account,
            "password": self.password,
            "stock_code": stock_code,
            "stock_shares": stock_shares,
            "stock_price": stock_price,
        }
        buy_url = "https://ciot.imis.ncku.edu.tw/sim_stock/trading_api/buy"
        result = requests.post(buy_url, data=data).json()
        print("Result: " + result["result"] + "\nStatus: " + result["status"])
        return result["result"] == "success"

    # 預約售出股票
    # Input:
    #   account: 使用者帳號
    #   password: 使用者密碼
    #   stock_code: 股票ID
    #   stock_shares: 售出張數
    #   stock_price: 售出價格
    # Output: 是否成功預約售出(True/False)

    def Sell_Stock(self, stock_code, stock_shares, stock_price):
        print("Selling stock...")
        data = {
            "account": self.account,
            "password": self.password,
            "stock_code": stock_code,
            "stock_shares": stock_shares,
            "stock_price": stock_price,
        }
        sell_url = "https://ciot.imis.ncku.edu.tw/sim_stock/trading_api/sell"
        result = requests.post(sell_url, data=data).json()
        print("Result: " + result["result"] + "\nStatus: " + result["status"])
        return result["result"] == "success"

    @staticmethod
    def get_all_stock_information() -> List[str]:
        url = "http://140.116.86.242:8081/api/stock/get_stock_list"
        result = requests.get(url).json()
        if result["result"] == "success":
            raw_data = result["data"]
            filtered_codes = []
            for item in raw_data:
                # 只要上市(0)或上櫃(1)
                if len(item["code"]) == 4:
                    filtered_codes.append(item["code"])
            return filtered_codes
        return []


if __name__ == "__main__":
    c = Stock_API.Get_Stock_Informations("2330", "20241205", "20241205")
    print(c)
