"""缠论分析模块（基于 czsc 库）。

提供 DataFrame → czsc RawBar 的适配、多级别 CZSC 分析、买卖点信号识别。
严格按照缠论层级：K线 → 分型 → 笔 → 线段 → 中枢 → 买卖点。
"""

from zenstock.chanlun.adapter import df_to_bars
from zenstock.chanlun.analyzer import ChanlunAnalyzer, ChanlunResult
from zenstock.chanlun.segments import (
    BuySellPoint,
    LineSegment,
    ZSPyramid,
    detect_buy_sell_points,
    extract_line_segments,
    extract_zhongshu_from_segments,
)

__all__ = [
    "df_to_bars",
    "ChanlunAnalyzer",
    "ChanlunResult",
    "LineSegment",
    "ZSPyramid",
    "BuySellPoint",
    "extract_line_segments",
    "extract_zhongshu_from_segments",
    "detect_buy_sell_points",
]
