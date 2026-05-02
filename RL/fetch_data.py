"""Fetch top-10 TW stock OHLCV for RL training.

Usage:
    python -m RL.fetch_data

Writes one CSV per stock into RL/data/ plus a meta.yaml describing the snapshot.
Safe to re-run: existing CSVs that cover the requested range are skipped.
"""

from __future__ import annotations

import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from stock_api import get_taiwan_stock_data
from stock_api.utils import UpstreamHTMLResponse

STOCKS: list[tuple[str, str]] = [
    # Semis + electronics (15)
    ("2330", "台積電"),
    ("2317", "鴻海"),
    ("2454", "聯發科"),
    ("2308", "台達電"),
    ("2382", "廣達"),
    ("2303", "聯電"),
    ("3711", "日月光投控"),
    ("6669", "緯穎"),
    ("5347", "世界"),
    ("8046", "南電"),
    ("2474", "可成"),
    ("2354", "鴻準"),
    ("2376", "技嘉"),
    ("3034", "聯詠"),
    ("3008", "大立光"),
    # Telecom + media (4)
    ("3045", "台灣大"),
    ("4904", "遠傳"),
    ("2412", "中華電"),
    ("8454", "富邦媒"),
    # Financials (10)
    ("2881", "富邦金"),
    ("2882", "國泰金"),
    ("2891", "中信金"),
    ("2884", "玉山金"),
    ("2885", "元大金"),
    ("2886", "兆豐金"),
    ("2887", "台新金"),
    ("2890", "永豐金"),
    ("2880", "華南金"),
    ("2892", "第一金"),
    # Industrials + materials (10)
    ("1101", "台泥"),
    ("1102", "亞泥"),
    ("1216", "統一"),
    ("1301", "台塑"),
    ("1303", "南亞"),
    ("1326", "台化"),
    ("2002", "中鋼"),
    ("2207", "和泰車"),
    ("6505", "台塑化"),
    ("9910", "豐泰"),
    # Transport + retail + healthcare + tech (11)
    ("2603", "長榮"),
    ("2609", "陽明"),
    ("2615", "萬海"),
    ("2618", "長榮航"),
    ("2912", "統一超"),
    ("5880", "合庫金"),
    ("6415", "矽力"),
    ("6446", "藥華藥"),
    ("9921", "巨大"),
    ("3231", "緯創"),
    ("6770", "力積電"),
]

START_DATE = "20150101"
END_DATE = "20260422"

DATA_DIR = Path(__file__).resolve().parent / "data"


def _coverage_ok(csv_path: Path, start: str, end: str) -> bool:
    if not csv_path.exists():
        return False
    try:
        df = pd.read_csv(csv_path, usecols=["date"], parse_dates=["date"])
    except Exception:
        return False
    if df.empty:
        return False
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    return df["date"].min() <= start_ts + pd.Timedelta(days=35) and df["date"].max() >= end_ts - pd.Timedelta(days=35)


def fetch_one(code: str, name: str) -> pd.DataFrame | None:
    try:
        df = get_taiwan_stock_data(code, START_DATE, END_DATE)
    except UpstreamHTMLResponse as e:
        print(f"[{code} {name}] upstream HTML (maintenance/block): {e}", file=sys.stderr)
        return None
    except Exception:
        print(f"[{code} {name}] unexpected failure:", file=sys.stderr)
        traceback.print_exc()
        return None

    if df is None or df.empty:
        print(f"[{code} {name}] empty result", file=sys.stderr)
        return None

    df = df.sort_values("date").reset_index(drop=True)
    return df


def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    failures: list[str] = []
    t0 = time.time()

    for i, (code, name) in enumerate(STOCKS, 1):
        csv_path = DATA_DIR / f"{code}.csv"
        elapsed = time.time() - t0
        print(f"\n=== [{i}/{len(STOCKS)}] {code} {name}  (elapsed {elapsed / 60:.1f} min) ===")

        if _coverage_ok(csv_path, START_DATE, END_DATE):
            print(f"[{code}] cached CSV covers range; skipping.")
            df = pd.read_csv(csv_path, parse_dates=["date"])
            results[code] = {
                "name": name,
                "rows": len(df),
                "first_date": str(df["date"].min().date()),
                "last_date": str(df["date"].max().date()),
                "status": "cached",
            }
            continue

        df = fetch_one(code, name)
        if df is None:
            failures.append(code)
            results[code] = {"name": name, "status": "failed"}
            continue

        df.to_csv(csv_path, index=False)
        print(f"[{code}] wrote {len(df)} rows -> {csv_path.relative_to(DATA_DIR.parent.parent)}")
        results[code] = {
            "name": name,
            "rows": len(df),
            "first_date": str(df["date"].min().date()),
            "last_date": str(df["date"].max().date()),
            "status": "fetched",
        }

    meta = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "requested_start": START_DATE,
        "requested_end": END_DATE,
        "source": "stock_api (TWSE official)",
        "split_suggestion": {
            "train": "2015-01 .. 2022-12",
            "val": "2023-01 .. 2023-12",
            "test": "2024-01 .. 2026-04",
        },
        "schema": [
            "date", "capacity", "turnover", "high", "low",
            "close", "change", "transaction_volume", "stock_code_id", "open",
        ],
        "stocks": results,
        "failures": failures,
    }
    meta_path = DATA_DIR / "meta.yaml"
    with meta_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)

    total = time.time() - t0
    print(f"\n=== done in {total / 60:.1f} min ===")
    print(f"meta -> {meta_path}")
    if failures:
        print(f"failed codes: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
