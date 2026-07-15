"""数据迁移 / 维护脚本。

用法:
    # 查看本地数据概览
    python scripts/migrate.py --info

    # 将所有 Parquet 汇总导入 DuckDB
    python scripts/migrate.py --sync-duckdb
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zenstock.data import DataStorage
from zenstock.data.types import Freq
from zenstock.logger import setup_logging


def cmd_info(storage: DataStorage) -> None:
    """显示本地数据概览。"""
    symbols = storage.list_symbols(Freq.DAILY)
    print(f"📁 数据目录: {storage.parquet_dir}")
    print(f"📊 已存储股票: {len(symbols)} 只")
    if symbols:
        print(f"   示例: {', '.join(symbols[:10])} ...")

    stock_list = storage.get_stock_list()
    if not stock_list.empty:
        print(f"📋 股票列表: {len(stock_list)} 条")


def cmd_sync_duckdb(storage: DataStorage) -> None:
    """把 Parquet 数据同步到 DuckDB 汇总表。"""
    import pandas as pd

    symbols = storage.list_symbols(Freq.DAILY)
    if not symbols:
        print("⚠️  无数据可同步")
        return

    print(f"🔄 同步 {len(symbols)} 只股票到 DuckDB ...")
    frames = []
    for sym in symbols:
        df = storage.read_klines(sym, Freq.DAILY)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    con = storage._get_duckdb()  # noqa: SLF001
    con.execute("DROP TABLE IF EXISTS klines")
    con.execute("CREATE TABLE klines AS SELECT * FROM all_df")
    con.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON klines(symbol)")
    print(f"✅ DuckDB 同步完成: {len(all_df):,} 条 → klines 表")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ZenStock 数据维护工具")
    parser.add_argument("--info", action="store_true", help="显示数据概览")
    parser.add_argument("--sync-duckdb", action="store_true", help="同步到DuckDB")
    args = parser.parse_args(argv)

    setup_logging()
    storage = DataStorage()

    if args.info:
        cmd_info(storage)
    elif args.sync_duckdb:
        cmd_sync_duckdb(storage)
    else:
        parser.print_help()

    storage.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
