"""交易日历工具。"""

from __future__ import annotations

import pandas as pd


def trading_days(start: str, end: str) -> list[str]:
    """获取 [start, end] 区间内的交易日（基于 pandas BDay 近似）。

    注意：这是近似值，精确的 A 股交易日历建议从数据源获取。
    """
    dates = pd.bdate_range(start=start, end=end)
    return [d.strftime("%Y-%m-%d") for d in dates]


def date_range(start: str, end: str, freq: str = "D") -> list[str]:
    """通用日期范围。"""
    dates = pd.date_range(start=start, end=end, freq=freq)
    return [d.strftime("%Y-%m-%d") for d in dates]
