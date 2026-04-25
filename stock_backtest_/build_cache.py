"""
從 TWSE/TPEX 官方源直接抓資料,寫成 backtest 引擎吃的 cache 格式。
繞過已停用的 lab price-history API。

用法:
    python build_cache.py --codes 2330,2454,2317
    python build_cache.py --top 30                 # 取 lab stock list 前 30 檔
    python build_cache.py --all                    # 全市場 (數小時)
    python build_cache.py --codes 2330 --start 20250101 --end 20250530

行為:
    - 已在 cache 內且日期區間涵蓋者會跳過 (續跑友善)
    - 寫入 data/stock_data.csv + data/save_data_info.yaml
    - 預設區間 20241122 ~ 20250530 (期末評分視窗 + 90 天暖機)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "stock_data.csv"
YAML_PATH = DATA_DIR / "save_data_info.yaml"

CACHE_COLUMNS = [
    "stock_code", "date",
    "capacity", "turnover", "high", "low", "close",
    "change", "transaction_volume", "stock_code_id", "open",
]

# 讓 import stock_api 可以從 repo root 找到
sys.path.insert(0, str(ROOT.parent))
from stock_api import get_taiwan_stock_data  # noqa: E402
from stock_backtest_.backtest.Stock_API import Stock_API  # noqa: E402


def yyyymmdd_to_dash(s: str) -> str:
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


def load_existing_cache() -> pd.DataFrame:
    if not CSV_PATH.exists():
        return pd.DataFrame(columns=CACHE_COLUMNS)
    df = pd.read_csv(CSV_PATH, dtype={"stock_code": str, "date": str, "stock_code_id": str})
    return df


def codes_already_covered(existing: pd.DataFrame, start: str, end: str) -> set[str]:
    """回傳已經完整覆蓋 [start, end] 區間的股票代碼集合。"""
    if existing.empty:
        return set()
    have = set()
    for code, grp in existing.groupby("stock_code"):
        dates = grp["date"]
        if dates.min() <= start and dates.max() >= end:
            have.add(code)
    return have


def fetch_one(code: str, start_dash: str, end_dash: str) -> pd.DataFrame | None:
    """抓單一股票,回傳引擎吃的格式 (or None 表示沒資料/失敗)。"""
    try:
        df = get_taiwan_stock_data(code, start_dash, end_dash)
    except Exception as exc:
        print(f"  ! {code} 抓取失敗: {exc}")
        return None

    if df is None or df.empty:
        return None

    df = df.copy()
    df["stock_code"] = code
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
    return df[CACHE_COLUMNS]


def resolve_codes(args) -> list[str]:
    if args.codes:
        return [c.strip() for c in args.codes.split(",") if c.strip()]

    print("從 lab API 取得股票清單...")
    all_codes = Stock_API.get_all_stock_information()
    if not all_codes:
        raise RuntimeError("get_all_stock_information() 回傳空清單,lab API 可能掛了")
    print(f"  共 {len(all_codes)} 檔")

    if args.all:
        return all_codes
    if args.top:
        return all_codes[: args.top]

    # 預設: 一組常見高流動性權值股,讓 demo 快速跑通
    default = ["2330", "2317", "2454", "2412", "2308", "2882", "2891", "1301", "1303", "0050"]
    return [c for c in default if c in all_codes] or default


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--codes", help="逗號分隔的股票代碼,例如 2330,2454")
    parser.add_argument("--top", type=int, help="從 lab stock list 取前 N 檔")
    parser.add_argument("--all", action="store_true", help="抓全市場 (慢)")
    parser.add_argument("--start", default="20241122", help="起始日 YYYYMMDD (預設 20241122)")
    parser.add_argument("--end", default="20250530", help="結束日 YYYYMMDD (預設 20250530)")
    parser.add_argument("--force", action="store_true", help="忽略現有 cache,重抓")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)

    start_dash = yyyymmdd_to_dash(args.start)
    end_dash = yyyymmdd_to_dash(args.end)

    codes = resolve_codes(args)
    print(f"目標: {len(codes)} 檔, 區間 {args.start} ~ {args.end}")

    existing = load_existing_cache()
    skip = set() if args.force else codes_already_covered(existing, args.start, args.end)
    if skip:
        print(f"略過已涵蓋的 {len(skip)} 檔: {sorted(skip)[:5]}{'...' if len(skip) > 5 else ''}")

    todo = [c for c in codes if c not in skip]
    print(f"要抓: {len(todo)} 檔")

    new_frames = []
    for i, code in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {code} ...", end=" ", flush=True)
        df = fetch_one(code, start_dash, end_dash)
        if df is None or df.empty:
            print("(無資料)")
            continue
        print(f"{len(df)} 筆")
        new_frames.append(df)

    if not new_frames:
        print("沒有新資料寫入。")
        return 0

    new_df = pd.concat(new_frames, ignore_index=True)

    if not args.force and not existing.empty:
        # 移除舊的同 (code, date) 列,以新資料覆蓋
        merged = pd.concat([existing, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["stock_code", "date"], keep="last")
    else:
        merged = new_df

    merged = merged.sort_values(["stock_code", "date"]).reset_index(drop=True)
    merged.to_csv(CSV_PATH, index=False)

    yaml_start = min(args.start, merged["date"].min())
    yaml_end = max(args.end, merged["date"].max())
    with YAML_PATH.open("w") as f:
        yaml.safe_dump({"start_date": yaml_start, "end_date": yaml_end}, f)

    print(f"\n寫入 {CSV_PATH}  ({len(merged)} 列, {merged['stock_code'].nunique()} 檔)")
    print(f"寫入 {YAML_PATH}  ({yaml_start} ~ {yaml_end})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
