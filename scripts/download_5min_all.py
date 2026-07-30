"""批量下载全市场 A 股 5 分钟 K 线数据。

特点：
  - 自动从 BaoStock 获取全市场 A 股列表（~5200 只）
  - 断点续传：跳过已有数据文件的股票
  - 进度保存：记录已完成和失败的股票
  - 错误重试：每只最多重试 3 次
  - 增量追加：已有文件追加新数据

用法:
    # 下载全市场最近 30 天的 5 分钟数据
    python scripts/download_5min_all.py --days 30

    # 指定日期范围
    python scripts/download_5min_all.py --start 2026-06-01 --end 2026-07-15

    # 限制数量（测试用）
    python scripts/download_5min_all.py --days 30 --limit 50

    # 强制重新下载（不跳过已有）
    python scripts/download_5min_all.py --days 30 --force
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# 将 src 加入 path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import pandas as pd
from tqdm import tqdm

from zenstock.data.types import Freq
from zenstock.logger import get_logger, setup_logging

log = get_logger(__name__)

PARQUET_DIR = _PROJECT_ROOT / "data" / "parquet"
PROGRESS_FILE = _PROJECT_ROOT / "data" / "cache" / "download_5min_progress.json"


def fetch_all_a_stocks() -> list[tuple[str, str]]:
    """从 BaoStock 获取全市场 A 股股票列表。

    Returns:
        [(baostock_code, name), ...] 如 [('sh.600000', '浦发银行'), ...]
    """
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"BaoStock 登录失败: {lg.error_msg}")

    # 找最近有数据的交易日（BaoStock 当天数据可能延迟入库）
    # 用最近 10 个交易日逐一尝试，取第一个返回数据 > 100 的
    latest_day = None
    test_end = datetime.now().strftime("%Y-%m-%d")
    test_start = (datetime.now() - timedelta(days=20)).strftime("%Y-%m-%d")
    rs_dates = bs.query_trade_dates(start_date=test_start, end_date=test_end)
    trade_days = []
    while rs_dates.next():
        d = rs_dates.get_row_data()
        if d[1] == "1":  # is_trading_day
            trade_days.append(d[0])

    # 从最近的交易日往前试
    for day in reversed(trade_days):
        rs_test = bs.query_all_stock(day=day)
        count = 0
        while rs_test.next():
            count += 1
            if count > 100:  # 足够了
                break
        if count > 100:
            latest_day = day
            break

    if not latest_day:
        latest_day = trade_days[-1] if trade_days else "2026-07-16"
    log.info(f"使用交易日: {latest_day}")

    rs = bs.query_all_stock(day=latest_day)
    data = []
    while rs.next():
        data.append(rs.get_row_data())

    def is_a_stock(code: str, name: str) -> bool:
        """精确判断 A 股股票（排除指数、ETF、债券、B股）。"""
        if any(kw in name for kw in ["指数", "ETF", "LOF", "债", "B股", "分级", "联接"]):
            return False
        c = code.replace("sh.", "").replace("sz.", "").replace("bj.", "")
        market = code[:2]
        num = c[:3]
        if market == "sh" and num in ("600", "601", "603", "605", "688"):
            return True
        if market == "sz" and num in ("000", "001", "002", "003", "300", "301"):
            return True
        return False

    stocks = [(d[0], d[2]) for d in data if is_a_stock(d[0], d[2])]
    bs.logout()
    log.info(f"获取 A 股股票列表: {len(stocks)} 只")
    return stocks


def bs_code_to_symbol(bs_code: str) -> str:
    """sh.600000 → 600000。"""
    return bs_code.split(".")[1]


def download_one(
    bs_code: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame | None:
    """下载单只股票的 5 分钟数据。"""
    import baostock as bs

    rs = bs.query_history_k_data_plus(
        bs_code,
        "date,time,open,high,low,close,volume,amount",
        start_date=start_date,
        end_date=end_date,
        frequency="5",
        adjustflag="2",  # 前复权
    )

    data = []
    while rs.next():
        data.append(rs.get_row_data())

    if not data:
        return None

    df = pd.DataFrame(data, columns=rs.fields)
    symbol = bs_code_to_symbol(bs_code)
    df["symbol"] = symbol

    # 转换数据类型
    for c in ["open", "high", "low", "close", "volume", "amount"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 合并 date + time → datetime
    # time 格式: "20260701093500000"
    df["date"] = pd.to_datetime(
        df["time"].str[:14], format="%Y%m%d%H%M%S", errors="coerce"
    )

    keep = ["date", "symbol", "open", "high", "low", "close", "volume", "amount"]
    df = df[[c for c in keep if c in df.columns]]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_progress() -> dict:
    """加载下载进度。"""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "failed": [], "last_run": None}


def save_progress(progress: dict) -> None:
    """保存下载进度。"""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    progress["last_run"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="批量下载全市场 A 股 5 分钟数据")
    parser.add_argument("--days", type=int, default=30, help="下载最近 N 天的数据（默认30）")
    parser.add_argument("--start", default=None, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=0, help="限制下载数量（0=全部）")
    parser.add_argument("--force", action="store_true", help="强制重新下载（不跳过已有）")
    parser.add_argument("--sleep", type=float, default=0.15, help="请求间隔秒（默认0.15）")
    args = parser.parse_args(argv)

    setup_logging()

    # 确定日期范围
    end_date = args.end or datetime.now().strftime("%Y-%m-%d")
    if args.start:
        start_date = args.start
    else:
        start_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")

    print(f"📥 全市场 A 股 5 分钟数据下载")
    print(f"   日期范围: {start_date} ~ {end_date}")
    print(f"   存储目录: {PARQUET_DIR}")

    # 1. 获取股票列表
    stocks = fetch_all_a_stocks()
    if args.limit > 0:
        stocks = stocks[: args.limit]
        print(f"   ⚠️ 限制下载前 {args.limit} 只（测试模式）")

    total = len(stocks)
    print(f"   股票数量: {total}")

    # 2. 加载进度
    progress = load_progress()
    completed_set = set(progress["completed"]) if not args.force else set()

    # 3. 过滤已完成的
    todo = []
    skipped = 0
    for bs_code, name in stocks:
        symbol = bs_code_to_symbol(bs_code)
        parquet_path = PARQUET_DIR / f"{symbol}_5.parquet"
        if symbol in completed_set or (parquet_path.exists() and not args.force):
            skipped += 1
        else:
            todo.append((bs_code, name, symbol))

    print(f"   已跳过: {skipped}，待下载: {len(todo)}")

    if not todo:
        print("✅ 所有股票数据已存在，无需下载")
        return 0

    # 4. 批量下载
    import baostock as bs

    bs.login()

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    success = 0
    failed = 0
    total_bars = 0
    consecutive_fails = 0  # 连续失败计数（用于断连检测）

    pbar = tqdm(todo, desc="下载5分钟数据", unit="只")
    for bs_code, name, symbol in pbar:
        # 重试（含断连自动重连）
        df = None
        for attempt in range(5):
            try:
                df = download_one(bs_code, start_date, end_date)
                consecutive_fails = 0  # 成功则重置
                break
            except Exception as e:
                err_str = str(e)
                # 检测连接断开类错误
                is_conn_err = any(kw in err_str for kw in [
                    "10054", "10053", "Connection", "连接", "接收数据异常",
                    "login", "logout",
                ])
                wait = (2 ** attempt) * (3 if is_conn_err else 1)
                if attempt < 4:
                    log.debug(f"重试 {attempt+1}/5 {bs_code}: {e}, 等待{wait}秒")
                    time.sleep(wait)
                    # 连接错误时尝试重新登录 BaoStock
                    if is_conn_err:
                        try:
                            bs.logout()
                        except Exception:
                            pass
                        time.sleep(2)
                        try:
                            bs.login()
                        except Exception:
                            pass
                        time.sleep(1)
                else:
                    log.debug(f"下载失败 {bs_code}: {e}")

        if df is not None and not df.empty:
            parquet_path = PARQUET_DIR / f"{symbol}_5.parquet"
            # 只在非 force 模式且文件较小时才追加合并
            # 全量下载时直接覆盖（追加合并 6 万行太慢）
            if not args.force and parquet_path.exists() and len(df) < 5000:
                old_df = pd.read_parquet(parquet_path)
                df = pd.concat([old_df, df]).drop_duplicates(
                    subset=["date"], keep="last"
                )
                df = df.sort_values("date").reset_index(drop=True)

            df.to_parquet(parquet_path, index=False, engine="pyarrow")
            total_bars += len(df)
            success += 1
            progress["completed"].append(symbol)
        else:
            failed += 1
            consecutive_fails += 1
            if symbol not in progress["failed"]:
                progress["failed"].append(symbol)

            # 连续失败超过 20 只 → 服务器可能全面断连，长等待
            if consecutive_fails >= 20:
                pbar.set_description("⚠️连续失败，等待60秒后重连")
                time.sleep(60)
                try:
                    bs.logout()
                    time.sleep(3)
                    bs.login()
                except Exception:
                    pass
                consecutive_fails = 0
                pbar.set_description("下载5分钟数据")

        pbar.set_postfix(ok=success, fail=failed, bars=f"{total_bars:,}")

        # 定期保存进度（每 50 只）
        if (success + failed) % 50 == 0:
            save_progress(progress)

        time.sleep(args.sleep)

    bs.logout()

    # 5. 最终保存进度
    save_progress(progress)

    print(f"\n{'='*50}")
    print(f"✅ 下载完成")
    print(f"   成功: {success} 只")
    print(f"   失败: {failed} 只")
    print(f"   总 K 线: {total_bars:,} 条")
    print(f"   进度文件: {PROGRESS_FILE}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
