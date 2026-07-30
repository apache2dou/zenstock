"""多级别联立诊断引擎（文档 §4 病情矩阵）。

实现 5分钟→30分钟→日线→周线 四级联立的病情诊断。

核心逻辑（文档 §4）：
    1. 对每个级别分别计算 BiState
    2. 组成病情矩阵（4行 × 1状态）
    3. 对相邻级别做病情诊断（未病/欲病/已病）
    4. 用诊断结果过滤交易信号

病情矩阵示例（文档 §4.4）：
    | 级别   | 状态     | 组合    |
    |--------|---------|---------|
    | 5分钟  | ...     | (1, 0)  |
    | 30分钟 | ...     | (1, 1)  |
    | 日线   | ...     | (-1, 0) |
    | 周线   | ...     | (-1, 1) |

诊断规则（文档 §4.2 §4.3）：
    - 大级别延伸中 → 小级别波动无价值（过滤噪音）
    - 小级别先出现反转信号 → 未病（预警）
    - 大级别确认反转 → 已病（操作）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from zenstock.chanlun.bi_state import (
    BiState,
    classify_disease,
    diagnose_multilevel,
)
from zenstock.logger import get_logger

log = get_logger(__name__)

# 四级联立的标准级别顺序（从低到高）
MULTI_LEVEL_ORDER = ["5分钟", "30分钟", "日线", "周线"]


@dataclass
class DiseaseMatrix:
    """四级病情矩阵快照。

    记录某个时间点上，5分钟/30分钟/日线/周线 四个级别的笔状态，
    以及相邻级别的病情诊断结果。
    """
    states: dict[str, BiState]             # {级别名: BiState}
    diagnoses: list[dict[str, Any]]        # 相邻级别的诊断结果列表
    overall_health: str = "未知"           # 综合病情："健康"/"未病"/"欲病"/"已病"
    action_filter: str = "HOLD"            # 综合操作过滤："BUY"/"SELL"/"HOLD"/"BLOCK"

    def summary(self) -> str:
        """一行摘要。"""
        parts = [f"{lv}{st.value}" for lv, st in self.states.items()]
        return " | ".join(parts) + f" → {self.overall_health}({self.action_filter})"


def compute_multilevel_states(
    df_5min: pd.DataFrame | None = None,
    df_30min: pd.DataFrame | None = None,
    df_daily: pd.DataFrame | None = None,
    df_weekly: pd.DataFrame | None = None,
) -> dict[str, BiState]:
    """计算四个级别的当前笔状态。

    对每个级别的 DataFrame，用 czsc 识别最后一笔方向和分型，
    返回 BiState。

    Args:
        df_5min: 5分钟 K 线
        df_30min: 30分钟 K 线
        df_daily: 日线 K 线
        df_weekly: 周线 K 线

    Returns:
        {"5分钟": BiState, "30分钟": BiState, "日线": BiState, "周线": BiState}
        缺少数据的级别不包含在返回字典中。
    """
    from zenstock.chanlun.adapter import df_to_bars
    from zenstock.data.types import Freq

    level_data = {
        "5分钟": (df_5min, Freq.MIN5),
        "30分钟": (df_30min, Freq.MIN30),
        "日线": (df_daily, Freq.DAILY),
        "周线": (df_weekly, Freq.WEEKLY),
    }

    states: dict[str, BiState] = {}
    for level_name, (df, freq) in level_data.items():
        if df is None or df.empty or len(df) < 10:
            continue
        try:
            state = _compute_single_level_state(df, freq)
            if state is not None:
                states[level_name] = state
        except Exception as e:
            log.debug(f"计算 {level_name} 状态失败: {e}")

    return states


def _compute_single_level_state(df: pd.DataFrame, freq) -> BiState | None:
    """计算单个级别的当前笔状态。"""
    from czsc import CZSC  # type: ignore
    from zenstock.chanlun.adapter import df_to_bars

    bars = df_to_bars(df, freq)
    if len(bars) < 10:
        return None

    czsc_obj = CZSC(bars)
    bi_list = list(czsc_obj.bi_list)

    if not bi_list:
        return None

    last_bi = bi_list[-1]
    bi_direction = getattr(last_bi, "direction", None)
    # 用 codepoint 辅助函数可靠判断（避免 Windows 源码编码问题）
    from zenstock.chanlun.bi_state import (
        compute_bi_state, czsc_direction_is_up, czsc_mark_is_top, czsc_mark_is_bottom,
    )
    is_up = czsc_direction_is_up(bi_direction)

    # 只看当前未完成分型（ubi_fxs），与训练器保持一致
    fx_forming = False
    ubi_fxs = list(getattr(czsc_obj, "ubi_fxs", []))
    if ubi_fxs:
        last_fx = ubi_fxs[-1]
        mark = getattr(last_fx, "mark", "")
        if (is_up and czsc_mark_is_top(mark)) or (not is_up and czsc_mark_is_bottom(mark)):
            fx_forming = True

    return compute_bi_state("up" if is_up else "down", fx_forming)


def diagnose_disease_matrix(states: dict[str, BiState]) -> DiseaseMatrix:
    """对多级别状态做完整病情诊断。

    按文档 §4 的规则：
    1. 对相邻级别做 classify_disease
    2. 汇总得出综合病情（取最严重的诊断结果）
    3. 生成操作过滤信号

    综合病情判断逻辑：
    - 如果任一相邻对诊断"已病" → 综合已病
    - 如果任一相邻对诊断"欲病" → 综合欲病
    - 如果任一相邻对诊断"未病" → 综合未病
    - 否则健康

    操作过滤：
    - 已病 + 方向向下 → BLOCK（禁止买入）
    - 已病 + 方向向上 → SELL
    - 欲病 → 谨慎 HOLD（准备操作）
    - 未病 → HOLD（观察）
    - 健康 → 按概率信号操作
    """
    # 对相邻级别做诊断
    if len(states) < 2:
        return DiseaseMatrix(
            states=states,
            diagnoses=[],
            overall_health="未知",
            action_filter="HOLD",
        )

    diagnoses = diagnose_multilevel(states)

    # 综合病情（取最严重的）
    severity_order = {"已病": 4, "欲病": 3, "未病": 2, "健康": 1, "未知": 0}
    worst_health = "健康"
    worst_rank = None

    for diag in diagnoses:
        health = diag.get("health", "未知")
        if severity_order.get(health, 0) > severity_order.get(worst_health, 0):
            worst_health = health
            worst_rank = diag.get("rank")

    # 操作过滤信号
    # 最高级别的方向决定大方向
    level_order = [lv for lv in MULTI_LEVEL_ORDER if lv in states]
    big_state = states[level_order[-1]] if level_order else None

    if worst_health == "已病":
        if big_state and big_state.direction == -1:
            action_filter = "BLOCK"  # 禁止买入（下跌已病）
        elif big_state and big_state.direction == 1:
            action_filter = "SELL"   # 上涨已病（顶已确认）
        else:
            action_filter = "HOLD"
    elif worst_health == "欲病":
        action_filter = "HOLD"       # 谨慎，暂不操作
    elif worst_health == "未病":
        action_filter = "HOLD"       # 观察预警
    else:
        action_filter = "PASS"       # 健康，按概率信号放行

    return DiseaseMatrix(
        states=states,
        diagnoses=diagnoses,
        overall_health=worst_health,
        action_filter=action_filter,
    )


def resample_for_multilevel(
    df: pd.DataFrame,
    source_freq: str,
) -> dict[str, pd.DataFrame]:
    """从源数据重采样出四个级别的 K 线。

    根据源频率自动推导需要哪些重采样：
    - 源=5分钟 → 直接用，重采样出 30分钟/日线/周线
    - 源=日线 → 直接用，重采样出 周线（无5分钟/30分钟）
    - 源=30分钟 → 直接用，重采样出 日线/周线（无5分钟）

    Args:
        df: 源 K 线 DataFrame
        source_freq: 源频率值（如 "5", "30", "D"）

    Returns:
        {级别名: DataFrame} 字典（只包含成功生成的级别）
    """
    from zenstock.data.resample import resample_klines
    from zenstock.data.types import Freq

    result: dict[str, pd.DataFrame] = {}
    freq_map = {"5": "5分钟", "30": "30分钟", "D": "日线", "W": "周线"}
    source_level = freq_map.get(source_freq, "未知")

    # 源数据直接放入
    if source_level in freq_map.values():
        result[source_level] = df.copy()

    # 按需重采样
    resample_targets: list[tuple[str, Freq]] = []
    if source_freq == "5":
        resample_targets = [
            ("30分钟", Freq.MIN30),
            ("日线", Freq.DAILY),
            ("周线", Freq.WEEKLY),
        ]
    elif source_freq == "30":
        resample_targets = [
            ("日线", Freq.DAILY),
            ("周线", Freq.WEEKLY),
        ]
    elif source_freq in ("D", "日线"):
        resample_targets = [
            ("周线", Freq.WEEKLY),
        ]

    for level_name, target_freq in resample_targets:
        try:
            resampled = resample_klines(df, target_freq)
            if not resampled.empty and len(resampled) >= 5:
                result[level_name] = resampled
        except Exception as e:
            log.debug(f"重采样 {level_name} 失败: {e}")

    return result
