"""命令行入口：数据下载。"""

from __future__ import annotations

import argparse
import sys

from zenstock.logger import setup_logging


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口。"""
    parser = argparse.ArgumentParser(
        prog="zenstock-download",
        description="ZenStock 数据下载工具",
    )
    parser.add_argument("--start", default="2020-01-01", help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="结束日期 (YYYY-MM-DD)，默认今天")
    parser.add_argument(
        "--symbols",
        default=None,
        help="股票代码（逗号分隔），不填则下载全市场",
    )
    parser.add_argument(
        "--source",
        default=None,
        choices=["akshare", "baostock"],
        help="数据源（覆盖配置文件）",
    )
    parser.add_argument(
        "--freq",
        default="D",
        choices=["D", "W", "M", "1", "5", "15", "30", "60"],
        help="K 线周期：D=日线 5=5分钟 1=1分钟（仅 akshare 最近数据）等",
    )
    parser.add_argument(
        "--list-only", action="store_true", help="仅更新股票列表，不下载K线"
    )
    args = parser.parse_args(argv)

    setup_logging()

    # 延迟导入（加速 --help）
    from datetime import datetime

    import pandas as pd

    from zenstock.data import DataStorage, get_downloader
    from zenstock.data.types import Freq

    end = args.end or datetime.now().strftime("%Y-%m-%d")
    downloader = get_downloader(args.source)
    storage = DataStorage()

    # 把字符串频率转为 Freq 枚举
    try:
        freq = Freq(args.freq)
    except ValueError:
        freq = Freq.DAILY

    # 1 分钟只能取最近数据：自动将 end 设为今天，start 不做限制
    if freq == Freq.MIN1 and args.source != "akshare":
        print("⚠️  1 分钟仅 AKShare 数据源支持，自动切换为 akshare")
        downloader = get_downloader("akshare")
        args.source = "akshare"
    if freq == Freq.MIN1:
        # 强制 end 为今天，避免误传历史日期时混淆
        end = datetime.now().strftime("%Y-%m-%d")
        print("💡  1 分钟数据仅保留最近 5~9 个交易日（新浪接口限制）")

    # 1) 确定下载范围（指定 symbols 时跳过全市场列表拉取，避免被限流）
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
        stock_list = pd.DataFrame(
            {"symbol": symbols, "name": symbols, "market": ""}
        )
        print(f"📥 指定下载 {len(symbols)} 只股票的K线 [{args.start} ~ {end}]")
    else:
        print(f"📦 正在获取股票列表 ({downloader.name}) ...")
        try:
            stock_list = downloader.fetch_stock_list()
            storage.save_stock_list(stock_list)
            print(f"   ✓ {len(stock_list)} 只股票")
        except Exception as e:
            print(f"⚠️  获取股票列表失败: {e}")
            print("   请用 --symbols 指定股票代码，或稍后重试。")
            return 1
        symbols = stock_list["symbol"].tolist()
        if args.list_only:
            return 0
        print(f"📥 开始下载 {len(symbols)} 只股票的K线 [{args.start} ~ {end}]")

    # 3) 批量下载
    all_data = downloader.download_many(
        symbols, start_date=args.start, end_date=end, freq=freq
    )

    # 4) 存储
    if all_data:
        total = 0
        for symbol, df in all_data.items():
            total += storage.save_klines(df, freq=freq)
        freq_name = freq.display_name
        print(f"💾 存储完成: {total:,} 条{freq_name}K线 → {storage.parquet_dir}")
    else:
        print("⚠️ 未下载到任何数据")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
