"""数据层类型定义。"""

from __future__ import annotations

from enum import Enum

from pandas import DataFrame


class Freq(str, Enum):
    """K线周期。"""
    DAILY = "D"        # 日线
    WEEKLY = "W"       # 周线
    MONTHLY = "M"      # 月线
    MIN1 = "1"         # 1 分钟（仅 AKShare 新浪支持，最近约 5~9 个交易日）
    MIN5 = "5"         # 5 分钟（AKShare 新浪 + BaoStock 5+ 日历史）
    MIN15 = "15"       # 15 分钟
    MIN30 = "30"       # 30 分钟
    MIN60 = "60"       # 60 分钟

    @property
    def is_minute(self) -> bool:
        """是否分钟级别。"""
        return self.value in ("1", "5", "15", "30", "60")

    @property
    def display_name(self) -> str:
        """人类可读名称。"""
        return {
            Freq.DAILY: "日线",
            Freq.WEEKLY: "周线",
            Freq.MONTHLY: "月线",
            Freq.MIN1: "1 分钟",
            Freq.MIN5: "5 分钟",
            Freq.MIN15: "15 分钟",
            Freq.MIN30: "30 分钟",
            Freq.MIN60: "60 分钟",
        }[self]


class Adjust(str, Enum):
    """复权类型。"""
    NONE = "none"      # 不复权
    QFQ = "qfq"        # 前复权
    HFQ = "hfq"        # 后复权


# 人类可读的频率选项（前端下拉框用）
FREQ_OPTIONS: list[tuple[str, str]] = [
    # (内部值, 显示名)
    ("D", "日线"),
    ("W", "周线"),
    ("M", "月线"),
    ("1", "1 分钟（仅最近 5~9 个交易日）"),
    ("5", "5 分钟（BaoStock 历史可查）"),
    ("15", "15 分钟"),
    ("30", "30 分钟"),
    ("60", "60 分钟"),
]


# 标准列名（ZenStock 统一内部格式）
STANDARD_COLUMNS = [
    "date",        # 交易日期（日线）或 datetime（分钟线）
    "symbol",      # 股票代码，如 "000001"
    "open",        # 开盘价
    "high",        # 最高价
    "low",         # 最低价
    "close",       # 收盘价
    "volume",      # 成交量（手）
    "amount",      # 成交额（元）
    "turnover",    # 换手率（%）
    "pct_change",  # 涨跌幅（%）
]


def is_valid_klines(df: DataFrame) -> bool:
    """检查 DataFrame 是否为合法的 K 线数据。"""
    if df is None or df.empty:
        return False
    required = {"date", "symbol", "open", "high", "low", "close", "volume"}
    return required.issubset(set(df.columns))
