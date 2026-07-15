"""A 股市场工具函数。"""

from __future__ import annotations


def normalize_symbol(symbol: str) -> str:
    """规范化股票代码为 6 位字符串。

    >>> normalize_symbol("1")
    '000001'
    >>> normalize_symbol("600519")
    '600519'
    >>> normalize_symbol("sz.000001")
    '000001'
    """
    s = str(symbol).strip().lower()
    # 去掉前缀 sz. sh. bj.
    for prefix in ("sh.", "sz.", "bj.", "sh", "sz", "bj"):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    s = s.split(".")[0]  # 处理 000001.SZ
    return s.zfill(6)


def detect_market(symbol: str) -> str:
    """根据代码判断市场。

    Returns:
        'SH'（沪市）| 'SZ'（深市）| 'BJ'（北交所）| 'UNKNOWN'
    """
    s = normalize_symbol(symbol)
    if s.startswith(("60", "68")):  # 主板、科创板
        return "SH"
    if s.startswith(("00", "30")):  # 主板、创业板
        return "SZ"
    if s.startswith(("43", "83", "87", "88")):  # 北交所
        return "BJ"
    return "UNKNOWN"


def is_st_symbol(name: str) -> bool:
    """判断股票名称是否为 ST/*ST 股。"""
    name = str(name).upper()
    return name.startswith(("ST", "*ST", "S*ST", "SST"))
