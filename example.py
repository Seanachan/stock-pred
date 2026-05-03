"""Local sandbox for NCKU sim_stock API + TWSE price fetch.

Reads ACCOUNT/PASSWORD from .env. Will NOT submit any orders unless you pass
--buy or --sell explicitly with all 3 args (sid, lots, price).

Examples:
  uv run python example.py                              # query inventory + price fetch
  uv run python example.py --buy 2002 1 19.55          # test buy 1 lot of 2002 @19.55
  uv run python example.py --sell 2002 1 19.55         # test sell 1 lot of 2002 @19.55

Note: NCKU API uses LOT count (張) for stock_shares parameter, NOT raw share count.
1 lot = 1000 shares.
"""

import argparse
import os
import sys

from dotenv import load_dotenv

from stock_api import (
    Buy_Stock,
    Get_User_Stocks,
    Sell_Stock,
    get_taiwan_stock_data,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", default="2330", help="stock id for price fetch")
    parser.add_argument("--start", default="20260101")
    parser.add_argument("--end", default="20260430")
    parser.add_argument("--buy", nargs=3, metavar=("SID", "LOTS", "PRICE"),
                        help="test buy: --buy 2002 1 19.55")
    parser.add_argument("--sell", nargs=3, metavar=("SID", "LOTS", "PRICE"),
                        help="test sell: --sell 2002 1 19.55")
    args = parser.parse_args()

    load_dotenv()
    account = os.getenv("ACCOUNT")
    password = os.getenv("PASSWORD")
    if not account or not password:
        print("ERROR: ACCOUNT/PASSWORD missing in .env")
        sys.exit(1)
    print(f"account: {account}\n")

    print(f"=== Get_User_Stocks ===")
    try:
        inv = Get_User_Stocks(account, password)
        print(f"raw response: {inv}")
    except Exception as e:
        print(f"err: {e}")

    print(f"\n=== get_taiwan_stock_data({args.stock}, {args.start}..{args.end}) ===")
    try:
        df = get_taiwan_stock_data(args.stock, args.start, args.end)
        if df is None or df.empty:
            print("empty result")
        else:
            print(f"rows: {len(df)}, cols: {list(df.columns)}")
            print(df.head(3))
            if len(df) > 6:
                print("...")
                print(df.tail(3))
    except Exception as e:
        print(f"err: {e}")

    if args.buy:
        sid, lots, price = args.buy[0], int(args.buy[1]), float(args.buy[2])
        print(f"\n=== Buy_Stock({sid}, lots={lots}, price={price}) ===")
        try:
            ok = Buy_Stock(account, password, sid, lots, price)
            print(f"success: {ok}")
        except Exception as e:
            print(f"err: {e}")

    if args.sell:
        sid, lots, price = args.sell[0], int(args.sell[1]), float(args.sell[2])
        print(f"\n=== Sell_Stock({sid}, lots={lots}, price={price}) ===")
        try:
            ok = Sell_Stock(account, password, sid, lots, price)
            print(f"success: {ok}")
        except Exception as e:
            print(f"err: {e}")


if __name__ == "__main__":
    main()
