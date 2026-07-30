"""缠论分析模块（基于 czsc 库）。

提供 DataFrame → czsc RawBar 的适配、多级别 CZSC 分析、买卖点信号识别。
严格按照缠论层级：K线 → 分型 → 笔 → 线段 → 中枢 → 买卖点。
支持多级别递归分析：1分钟 → 5分钟 → 30分钟 → 日线。
"""

from zenstock.chanlun.adapter import df_to_bars
from zenstock.chanlun.analyzer import ChanlunAnalyzer, ChanlunResult
from zenstock.chanlun.bi_state import (
    BiState,
    build_disease_matrix,
    classify_disease,
    compute_bi_state,
    diagnose_multilevel,
    is_valid_transition,
)
from zenstock.chanlun.multi_level import (
    LevelResult,
    MultiLevelAnalyzer,
    MultiLevelResult,
    TrendType,
)
from zenstock.chanlun.segments import (
    BuySellPoint,
    LineSegment,
    ZSPyramid,
    detect_buy_sell_points,
    extract_line_segments,
    extract_zhongshu_from_segments,
)

__all__ = [
    # adapter
    "df_to_bars",
    # analyzer (single level)
    "ChanlunAnalyzer",
    "ChanlunResult",
    # bi_state (两重表里关系状态机，第91-92课)
    "BiState",
    "compute_bi_state",
    "is_valid_transition",
    "classify_disease",
    "build_disease_matrix",
    "diagnose_multilevel",
    # multi_level (recursive analysis)
    "MultiLevelAnalyzer",
    "MultiLevelResult",
    "LevelResult",
    "TrendType",
    # segments
    "LineSegment",
    "ZSPyramid",
    "BuySellPoint",
    "extract_line_segments",
    "extract_zhongshu_from_segments",
    "detect_buy_sell_points",
]
