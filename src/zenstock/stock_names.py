"""A股股票代码 → 名称映射表。

用于在 Streamlit UI 中显示"代码 + 名称"，方便查看。
当本地 DuckDB 的 stock_list 表有数据时，会自动合并覆盖这里的静态映射。
"""

from __future__ import annotations

# 常用股票代码 → 名称（手工维护的精选池）
# 后续可通过 fetch_stock_list 拉取全市场后自动合并
STOCK_NAMES: dict[str, str] = {
    # ---- 沪市主板 ----
    "600519": "贵州茅台",
    "601318": "中国平安",
    "600036": "招商银行",
    "601398": "工商银行",
    "600276": "恒瑞医药",
    "601899": "紫金矿业",
    "601138": "工业富联",
    "600938": "中国海油",
    "600967": "内蒙一机",
    "603697": "有友食品",
    "600219": "南山铝业",
    "600000": "浦发银行",
    "600028": "中国石化",
    "600030": "中信证券",
    "601857": "中国石油",
    "601288": "农业银行",
    "601988": "中国银行",
    "601628": "中国人寿",
    "601088": "中国神华",
    "601668": "中国建筑",
    # ---- 沪市科创板 ----
    "688981": "中芯国际",
    "688256": "寒武纪",
    # ---- 深市主板 ----
    "000001": "平安银行",
    "000002": "万科A",
    "000333": "美的集团",
    "000651": "格力电器",
    "000858": "五粮液",
    "002120": "韵达股份",
    "002594": "比亚迪",
    "002475": "立讯精密",
    "002241": "歌尔股份",
    "002714": "牧原股份",
    # ---- 深市创业板 ----
    "300750": "宁德时代",
    "300059": "东方财富",
    "300760": "迈瑞医疗",
    "300015": "爱尔眼科",
}


def get_stock_name(symbol: str) -> str:
    """获取股票名称，找不到时返回空字符串。

    Args:
        symbol: 6 位股票代码

    Returns:
        股票名称（如 "贵州茅台"），找不到返回 ""
    """
    s = str(symbol).zfill(6)
    return STOCK_NAMES.get(s, "")


def get_stock_label(symbol: str) -> str:
    """获取"代码 名称"格式的标签，方便 UI 显示。

    >>> get_stock_label("600519")
    '600519 贵州茅台'
    >>> get_stock_label("999999")
    '999999'
    """
    s = str(symbol).zfill(6)
    name = STOCK_NAMES.get(s, "")
    return f"{s} {name}" if name else s


def merge_stock_list(symbol_name_map: dict[str, str]) -> None:
    """把外部传入的 {代码: 名称} 合并到静态映射表（运行时扩充）。

    Args:
        symbol_name_map: 从 DuckDB 或数据源拉取的代码-名称映射
    """
    for code, name in symbol_name_map.items():
        s = str(code).zfill(6)
        if name and s not in STOCK_NAMES:
            STOCK_NAMES[s] = str(name)
