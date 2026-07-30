"""缠论多级别递归分析（严格按原文第62-71课递归定义）。

递归链（由低到高）：
  1分钟K线 → 笔 → 线段 → 1分钟中枢 → 1分钟走势类型
      ↓ (1分钟走势类型作为次级别)
  5分钟线段 → 5分钟中枢 → 5分钟走势类型
      ↓ (5分钟走势类型作为次级别)
  30分钟线段 → 30分钟中枢 → 30分钟走势类型
      ↓
  日线线段 → 日线中枢 → 日线走势类型

关键定理（第17课）：
  - 走势中枢由至少三个连续次级别走势类型重叠部分构成
  - 盘整 = 只含一个中枢的走势类型
  - 趋势 = 包含两个以上依次同向、互不重叠的中枢
  - 走势终完美：任何级别的任何走势类型终要完成
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from zenstock.chanlun.adapter import df_to_bars
from zenstock.chanlun.segments import (
    LineSegment,
    ZSPyramid,
    extract_line_segments,
    extract_zhongshu_from_segments,
)
from zenstock.data.types import Freq
from zenstock.logger import get_logger

log = get_logger(__name__)


# ==================== 走势类型定义 ====================

@dataclass
class TrendType:
    """缠论走势类型（盘整或趋势）。

    按原文第17-18课定义：
    - 盘整：只包含一个中枢
    - 上涨趋势：两个以上依次同向、互不重叠的中枢，方向向上
    - 下跌趋势：两个以上依次同向、互不重叠的中枢，方向向下
    """
    freq_name: str              # 所属级别名称，如 "1分钟"、"5分钟"
    trend_class: str            # "盘整" / "上涨趋势" / "下跌趋势"
    start_dt: Any               # 起始时间
    end_dt: Any                 # 结束时间
    start_price: float          # 起始价格
    end_price: float            # 结束价格
    high: float                 # 最高价
    low: float                  # 最低价
    zs_count: int               # 包含的中枢数量
    zs_list: list[ZSPyramid] = field(default_factory=list)  # 中枢列表
    segment_count: int = 0      # 包含的线段数
    bi_count: int = 0           # 包含的笔数

    @property
    def is_pan_zheng(self) -> bool:
        return self.trend_class == "盘整"

    @property
    def is_trend(self) -> bool:
        return "趋势" in self.trend_class

    @property
    def direction(self) -> str:
        if "上涨" in self.trend_class:
            return "up"
        if "下跌" in self.trend_class:
            return "down"
        return "sideways"

    def summary(self) -> str:
        return (
            f"[{self.freq_name}] {self.trend_class} "
            f"{self.start_dt}→{self.end_dt} "
            f"({self.zs_count}中枢, {self.segment_count}线段, {self.bi_count}笔)"
        )


# ==================== 单级别分析结果 ====================

@dataclass
class LevelResult:
    """单级别缠论分析完整结果。"""
    freq_name: str                          # 级别名称，如 "1分钟"、"5分钟"
    freq: Freq                              # 频率枚举
    bars_count: int = 0
    bi_list: list = field(default_factory=list)
    segment_list: list[LineSegment] = field(default_factory=list)
    zs_list: list[ZSPyramid] = field(default_factory=list)
    trend_types: list[TrendType] = field(default_factory=list)

    @property
    def bi_count(self) -> int:
        return len(self.bi_list)

    @property
    def segment_count(self) -> int:
        return len(self.segment_list)

    @property
    def zs_count(self) -> int:
        return len(self.zs_list)

    def summary(self) -> str:
        return (
            f"[{self.freq_name}] bars={self.bars_count}, "
            f"笔={self.bi_count}, 线段={self.segment_count}, "
            f"中枢={self.zs_count}, 走势类型={len(self.trend_types)}"
        )


# ==================== 多级别联立分析结果 ====================

@dataclass
class MultiLevelResult:
    """多级别递归分析结果。

    包含从最低级别到最高级别的完整分析链。
    """
    levels: dict[str, LevelResult] = field(default_factory=dict)

    def get(self, freq_name: str) -> LevelResult | None:
        return self.levels.get(freq_name)

    def all_summaries(self) -> list[str]:
        return [lv.summary() for lv in self.levels.values()]

    def all_trend_types(self) -> list[TrendType]:
        """收集所有级别的走势类型。"""
        all_tt: list[TrendType] = []
        for lv in self.levels.values():
            all_tt.extend(lv.trend_types)
        return sorted(all_tt, key=lambda t: str(t.start_dt or ""))

    @property
    def lowest_level(self) -> LevelResult | None:
        """最低级别（如 1分钟）。"""
        keys = list(self.levels.keys())
        return self.levels.get(keys[0]) if keys else None

    @property
    def highest_level(self) -> LevelResult | None:
        """最高级别（如 日线）。"""
        keys = list(self.levels.keys())
        return self.levels.get(keys[-1]) if keys else None


# ==================== 多级别分析器 ====================

# 级别递归链：从低到高
LEVEL_CHAIN: list[tuple[str, Freq]] = [
    ("1分钟", Freq.MIN1),
    ("5分钟", Freq.MIN5),
    ("30分钟", Freq.MIN30),
    ("日线", Freq.DAILY),
]


class MultiLevelAnalyzer:
    """多级别缠论递归分析器。

    支持两种模式：
    1. **递归模式**：从最低级别逐级向上构建（严格按原文递归定义）
    2. **独立模式**：各级别独立分析（利用 czsc 在各频率 K 线上分别处理）

    递归模式需要最低级别（1分钟）数据完整覆盖，适用于日内的精确分析。
    独立模式适用于跨级别概览，各级别独立但可通过 ZSPyramid 交叉引用。

    Usage::

        analyzer = MultiLevelAnalyzer()
        result = analyzer.analyze({Freq.MIN1: df1, Freq.MIN5: df5, Freq.MIN30: df30, Freq.DAILY: df_d})
        for name, lv in result.levels.items():
            print(lv.summary())
    """

    def analyze(
        self,
        data_by_freq: dict[Freq, pd.DataFrame],
        mode: str = "independent",
    ) -> MultiLevelResult:
        """执行多级别缠论分析。

        Args:
            data_by_freq: {Freq: DataFrame} 各频率的 K 线数据
            mode: "recursive"（递归模式）或 "independent"（独立模式）

        Returns:
            MultiLevelResult
        """
        result = MultiLevelResult()

        # 按级别从低到高排序
        ordered = self._order_by_level(data_by_freq)

        if mode == "recursive" and len(ordered) >= 2:
            log.info("使用递归模式进行多级别分析")
            self._analyze_recursive(ordered, result)
        else:
            log.info("使用独立模式进行多级别分析")
            self._analyze_independent(ordered, result)

        return result

    # ---------- 独立模式 ----------

    def _analyze_independent(
        self,
        ordered: list[tuple[str, Freq, pd.DataFrame]],
        result: MultiLevelResult,
    ) -> None:
        """独立模式：各级别独立运行 czsc 分析 + 线段/中枢识别。"""
        for freq_name, freq, df in ordered:
            lv = self._analyze_single_level(df, freq, freq_name)
            result.levels[freq_name] = lv

    def _analyze_single_level(
        self, df: pd.DataFrame, freq: Freq, freq_name: str
    ) -> LevelResult:
        """对单个级别的 K 线数据执行完整缠论分析。"""
        from czsc import CZSC  # type: ignore

        bars = df_to_bars(df, freq)

        if len(bars) < 10:
            log.warning(f"[{freq_name}] 数据不足（{len(bars)}根）")
            return LevelResult(freq_name=freq_name, freq=freq, bars_count=len(bars))

        # 1. czsc 笔和分型
        try:
            czsc_obj = CZSC(bars)
        except Exception as e:
            log.error(f"[{freq_name}] czsc 分析失败: {e}")
            return LevelResult(freq_name=freq_name, freq=freq, bars_count=len(bars))

        bi_list = list(czsc_obj.bi_list)

        # 2. 线段
        segments = extract_line_segments(bi_list) if len(bi_list) >= 3 else []

        # 3. 中枢（基于线段）
        zs_list = extract_zhongshu_from_segments(segments, bi_list) if segments else []

        # 4. 走势类型分类
        trend_types = _classify_trend_types(segments, zs_list, freq_name)

        return LevelResult(
            freq_name=freq_name,
            freq=freq,
            bars_count=len(bars),
            bi_list=bi_list,
            segment_list=segments,
            zs_list=zs_list,
            trend_types=trend_types,
        )

    # ---------- 递归模式 ----------

    def _analyze_recursive(
        self,
        ordered: list[tuple[str, Freq, pd.DataFrame]],
        result: MultiLevelResult,
    ) -> None:
        """递归模式：最低级别逐级向上构建。

        1. 最低级别（1分钟）：K 线 → 笔 → 线段 → 中枢 → 走势类型
        2. 高级别（5/30/日）：用低级别走势类型作为"线段"，构建本级别中枢和走势类型
        """
        # 第一步：分析最低级别
        base_name, base_freq, base_df = ordered[0]
        base_lv = self._analyze_single_level(base_df, base_freq, base_name)
        result.levels[base_name] = base_lv

        # 第二步：逐级向上递归
        prev_lv = base_lv
        for freq_name, freq, df in ordered[1:]:
            lv = self._build_level_from_lower(prev_lv, df, freq, freq_name)
            result.levels[freq_name] = lv
            prev_lv = lv

    def _build_level_from_lower(
        self,
        lower_lv: LevelResult,
        higher_df: pd.DataFrame,
        higher_freq: Freq,
        higher_name: str,
    ) -> LevelResult:
        """从低级别走势类型构建高级别分析。

        核心思想（原文）：
        - 低级别的"走势类型" = 高级别的"线段"（近似）
        - 三个高级别线段重叠 = 高级别中枢

        实际实现：
        1. 用低级别走势类型的端点构造"高级别线段"
        2. 在这些线段上构建高级别中枢
        3. 同时用高级别 K 线数据做 czsc 笔/线段作为补充参考
        """
        # 同时做独立分析作为参考
        independent = self._analyze_single_level(higher_df, higher_freq, higher_name)

        # 用低级别走势类型端点构造高级别线段
        pseudo_segments = _lower_trends_to_segments(lower_lv.trend_types)

        # 在这些伪线段上构建中枢
        zs_from_lower = extract_zhongshu_from_segments(pseudo_segments, [])
        zs_from_lower = [zs for zs in zs_from_lower if zs.is_valid]

        # 合并：独立中枢 + 递归中枢
        merged_zs = independent.zs_list.copy()
        for zs in zs_from_lower:
            if not any(_zs_overlap(zs, mz) for mz in merged_zs):
                merged_zs.append(zs)

        # 走势类型分类
        trend_types = _classify_trend_types(
            independent.segment_list, merged_zs, higher_name
        )

        return LevelResult(
            freq_name=higher_name,
            freq=higher_freq,
            bars_count=independent.bars_count,
            bi_list=independent.bi_list,
            segment_list=independent.segment_list,
            zs_list=merged_zs,
            trend_types=trend_types,
        )

    # ---------- 工具 ----------

    @staticmethod
    def _order_by_level(
        data_by_freq: dict[Freq, pd.DataFrame],
    ) -> list[tuple[str, Freq, pd.DataFrame]]:
        """按级别从低到高排序。"""
        ordered: list[tuple[str, Freq, pd.DataFrame]] = []
        for name, freq in LEVEL_CHAIN:
            if freq in data_by_freq:
                ordered.append((name, freq, data_by_freq[freq]))
        return ordered


# ==================== 走势类型分类 ====================

def _classify_trend_types(
    segments: list[LineSegment],
    zs_list: list[ZSPyramid],
    freq_name: str,
) -> list[TrendType]:
    """根据中枢列表和线段将走势分解为盘整/趋势。

    核心逻辑：
    1. 遍历中枢列表，将相邻且不重叠的中枢归为趋势
    2. 孤立的中枢归为盘整
    3. 中枢之间的线段是"连接段"

    按原文：
    - 盘整 = 只含一个中枢的完整走势
    - 趋势 = 至少两个同向、互不重叠的中枢
    """
    if not zs_list:
        return []

    results: list[TrendType] = []
    i = 0

    while i < len(zs_list):
        zs = zs_list[i]

        # 收集同向不重叠的中枢
        group_zs: list[ZSPyramid] = [zs]
        j = i + 1
        while j < len(zs_list):
            next_zs = zs_list[j]
            # 中枢不重叠 = 趋势延续
            if next_zs.zd > group_zs[-1].zg or next_zs.zg < group_zs[-1].zd:
                group_zs.append(next_zs)
                j += 1
            else:
                # 重叠 = 中枢延伸或级别扩展，跳过后续重叠的中枢
                # 不再修改 group_zs（避免丢失原始信息），直接跳过
                j += 1

        # 判断方向：看首尾中枢的位置关系
        if len(group_zs) >= 2:
            # 趋势方向：后中枢整体高于/低于前中枢
            if group_zs[-1].zd > group_zs[0].zg:
                direction = "上涨趋势"
            elif group_zs[-1].zg < group_zs[0].zd:
                direction = "下跌趋势"
            else:
                # 中枢之间有重叠但未合并，说明是震荡，仍归为盘整
                direction = "盘整"
                group_zs = [group_zs[0]]
            zs_count = len(group_zs)
        else:
            direction = "盘整"
            zs_count = 1

        # 找到包围这些中枢的线段
        zs_segments = _find_segments_for_zs(segments, group_zs[0], group_zs[-1])

        start_dt = group_zs[0].start_dt
        end_dt = group_zs[-1].end_dt
        start_price = zs_segments[0].start_price if zs_segments else 0.0
        end_price = zs_segments[-1].end_price if zs_segments else 0.0
        high = max((s.high for s in zs_segments), default=0.0)
        low = min((s.low for s in zs_segments), default=0.0)

        results.append(TrendType(
            freq_name=freq_name,
            trend_class=direction,
            start_dt=start_dt,
            end_dt=end_dt,
            start_price=start_price,
            end_price=end_price,
            high=high,
            low=low,
            zs_count=zs_count,
            zs_list=group_zs,
            segment_count=len(zs_segments),
            bi_count=sum(s.bi_count for s in zs_segments) if zs_segments else 0,
        ))

        i = j

    return results


def _lower_trends_to_segments(
    trend_types: list[TrendType],
) -> list[LineSegment]:
    """将低级别走势类型转换为高级别的"伪线段"。

    每个低级别走势类型（盘整或趋势）作为一个高级别线段。
    """
    segments: list[LineSegment] = []
    for i, tt in enumerate(trend_types):
        is_up = tt.end_price >= tt.start_price
        seg = LineSegment(
            direction="up" if is_up else "down",
            start_dt=tt.start_dt,
            end_dt=tt.end_dt,
            start_price=tt.start_price,
            end_price=tt.end_price,
            high=tt.high,
            low=tt.low,
            bi_count=tt.bi_count,
            bi_indices=(i, i),
        )
        segments.append(seg)
    return segments


def _find_segments_for_zs(
    segments: list[LineSegment],
    first_zs: ZSPyramid,
    last_zs: ZSPyramid,
) -> list[LineSegment]:
    """找到包围给定中枢范围的线段。"""
    result = []
    for seg in segments:
        if seg.bi_indices[0] <= last_zs.bi_range[1] and seg.bi_indices[1] >= first_zs.bi_range[0]:
            result.append(seg)
    return result


def _zs_overlap(a: ZSPyramid, b: ZSPyramid) -> bool:
    """两个中枢是否重叠。"""
    return not (a.zd > b.zg or a.zg < b.zd)
