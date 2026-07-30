"""缠论"走势结构的两重表里关系"（第91-92课）。

这是缠论中一套基于**笔状态**的多级别状态机诊断方法。

核心定理（缠中说禅笔定理）：
    任何的当下，在任何时间周期的 K 线图中，走势必然落在一确定的具有明确方向的笔当中。
    在笔当中的位置，必然只有两种情况：
    （1）在分型构造中；（2）在分型构造确认后延伸为笔的过程中。

四态定义（两个变量的数组 (d, s)）：
    d = 方向：  1=向上笔, -1=向下笔
    s = 阶段：  0=分型构造中, 1=延伸中
    组合：
        (1, 1)   向上笔延伸中       UP_EXTENDING
        (1, 0)   向上笔出现顶分型    UP_FX_FORMING
        (-1, 1)  向下笔延伸中       DOWN_EXTENDING
        (-1, 0)  向下笔出现底分型    DOWN_FX_FORMING

状态转移规则（不能随便连接）：
    (1, 1)  ──只能──>  (1, 0)
    (-1, 1) ──只能──>  (-1, 0)
    (1, 0)  ──两种──>  (1, 1) 或 (-1, 1)
    (-1, 0) ──两种──>  (-1, 1) 或 (1, 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class BiState(Enum):
    """笔的四态（对应原文的 (d, s) 数组）。"""

    UP_EXTENDING = "(1, 1)"      # 向上笔延伸中
    UP_FX_FORMING = "(1, 0)"     # 向上笔出现顶分型构造
    DOWN_EXTENDING = "(-1, 1)"   # 向下笔延伸中
    DOWN_FX_FORMING = "(-1, 0)"  # 向下笔出现底分型构造

    @property
    def direction(self) -> int:
        """方向：1=向上，-1=向下。"""
        return 1 if self.value.startswith("(1") else -1

    @property
    def is_extending(self) -> bool:
        """是否处于笔延伸中（s=1）。"""
        return self in (BiState.UP_EXTENDING, BiState.DOWN_EXTENDING)

    @property
    def is_fx_forming(self) -> bool:
        """是否处于分型构造中（s=0）。"""
        return self in (BiState.UP_FX_FORMING, BiState.DOWN_FX_FORMING)


# ==================== 单级别状态计算 ====================

# ==================== CZSC 枚举解析辅助 ====================

# CZSC 的 C 扩展枚举返回中文字符串，但 Windows 下源码中的中文字面量
# 在运行时可能被错误解释。因此用 Unicode codepoint 精确匹配：
#   向上: 上 = U+4E0A       向下: 下 = U+4E0B
#   顶分型: 顶 = U+9876     底分型: 底 = U+5E95
_UP_CODEPOINT = 0x4E0A      # 上
_DOWN_CODEPOINT = 0x4E0B    # 下
_TOP_CODEPOINT = 0x9876     # 顶
_BOTTOM_CODEPOINT = 0x5E95  # 底


def czsc_direction_is_up(direction) -> bool:
    """判断 CZSC Direction 枚举是否为向上。

    CZSC 的 Direction 是 C 扩展枚举，``str()`` 返回中文 "向上"/"向下"，
    但没有 ``.name`` 属性。用最后一个字符的 codepoint 可靠判断。
    """
    d = str(direction)
    if not d:
        return False
    # 向上=0x4E0A（兼容英文 "Up"/"up"）
    return d[-1] == "p" or d[-1] == "P" or ord(d[-1]) == _UP_CODEPOINT


def czsc_mark_is_top(mark) -> bool:
    """判断 CZSC Mark 枚举是否为顶分型。

    ``str()`` 返回中文 "顶分型"/"底分型"，用首字符 codepoint 判断。
    兼容英文 "G"/"g"（czsc 某些版本）。
    """
    m = str(mark)
    if not m:
        return False
    return m[0] in ("G", "g") or ord(m[0]) == _TOP_CODEPOINT


def czsc_mark_is_bottom(mark) -> bool:
    """判断 CZSC Mark 枚举是否为底分型。"""
    m = str(mark)
    if not m:
        return False
    return m[0] in ("D", "d") or ord(m[0]) == _BOTTOM_CODEPOINT


def compute_bi_state(last_bi_direction: str, fx_forming: bool) -> BiState:
    """根据最后一笔的方向和是否在分型构造中，计算当前笔状态。

    Args:
        last_bi_direction: 最后一笔的方向，"up" 或 "down"。
        fx_forming: 是否正在构造分型（True=分型构造中, False=笔延伸中）。

    Returns:
        对应的 BiState 枚举值。
    """
    is_up = (
        last_bi_direction.lower().startswith("up")
        or last_bi_direction == "up"
        or (len(last_bi_direction) > 0 and ord(last_bi_direction[-1]) == _UP_CODEPOINT)
    )
    if is_up:
        return BiState.UP_FX_FORMING if fx_forming else BiState.UP_EXTENDING
    return BiState.DOWN_FX_FORMING if fx_forming else BiState.DOWN_EXTENDING


# ==================== 状态转移校验 ====================

# 合法转移表（原文：四种状态是不能随便连接的）
_VALID_TRANSITIONS: dict[BiState, frozenset[BiState]] = {
    BiState.UP_EXTENDING: frozenset({BiState.UP_FX_FORMING}),
    BiState.DOWN_EXTENDING: frozenset({BiState.DOWN_FX_FORMING}),
    BiState.UP_FX_FORMING: frozenset({BiState.UP_EXTENDING, BiState.DOWN_EXTENDING}),
    BiState.DOWN_FX_FORMING: frozenset({BiState.DOWN_EXTENDING, BiState.UP_EXTENDING}),
}


def is_valid_transition(prev: BiState, curr: BiState) -> bool:
    """判断从前一个状态到当前状态的转移是否合法。

    原文规则：
        (1, 1) 之后绝对不会连接 (-1, 1) 或 (-1, 0)，唯一只能连接 (1, 0)；
        (-1, 1) 只能连接 (-1, 0)；
        (1, 0) 有两种可能：(1, 1) 或 (-1, 1)；
        (-1, 0) 有两种可能：(-1, 1) 或 (1, 1)。
    """
    return curr in _VALID_TRANSITIONS.get(prev, frozenset())


# ==================== fx_count 分桶（文档 §4.6）====================

# 分桶边界：1次 / 2次 / 3+次
_FX_COUNT_BUCKETS = (1, 2, 3)


def bucket_fx_count(count: int) -> int:
    """将分型出现次数分桶（文档 §4.6）。

    文档意图：fx_count 不是精确值，而是分桶统计维度。
    分桶策略：1次 / 2次 / 3+次（3及以上合并，保证样本量充足）。

    Args:
        count: 原始分型出现次数（从 1 开始）。

    Returns:
        分桶后的值：1, 2, 或 3（代表 3+）。
    """
    if count <= 0:
        return 1
    if count > 3:
        return 3
    return count


# ==================== 多级别病情诊断 ====================

@dataclass
class DiseaseResult:
    """多级别病情诊断结果。"""
    rank: int | None         # 恶劣排名（1=最恶劣），上涨行情为 None
    label: str               # 中文标签
    health: str              # "健康" / "未病" / "欲病" / "已病"
    description: str         # 详细说明


def classify_disease(
    big_level_state: BiState, small_level_state: BiState
) -> dict[str, Any]:
    """诊断两个相邻级别的病情（未病/欲病/已病）。

    以原文 2007-12-17 大盘为例，下跌行情的优劣排序：
        第1恶劣：大级别(-1,1) + 小级别(-1,1)  → 同向下跌，最差
        第2恶劣：大级别(-1,1) + 小级别(-1,0)  → 大级别跌 + 小级别底分型
        第3恶劣：大级别(-1,0) + 小级别(-1,1)  → 大级别底分型 + 小级别跌
        第4（转机）：大级别(-1,0) + 小级别(-1,0) → 都在底分型，可能转机

    对于上涨行情，则按未病/欲病/已病的预警逻辑判断：
        健康：大级别(1,1) + 小级别(1,1)
        未病：大级别(1,1) + 小级别(1,0)  → 小级别顶分型是小警告
        欲病：大级别(1,0)形成中 + 小级别(-1,1)
        已病：大级别(-1,1)确认

    Args:
        big_level_state: 大级别（如周线）的笔状态。
        small_level_state: 小级别（如日线）的笔状态。

    Returns:
        {
            "rank": int|None,
            "label": str,
            "health": str,
            "description": str,
        }
    """
    big_down = big_level_state.direction == -1
    big_up = big_level_state.direction == 1
    big_fx = big_level_state.is_fx_forming
    big_ext = big_level_state.is_extending
    small_down = small_level_state.direction == -1
    small_up = small_level_state.direction == 1
    small_fx = small_level_state.is_fx_forming
    small_ext = small_level_state.is_extending

    # ===== 下跌行情的优劣排序（原文 2.3）=====
    if big_down:
        if big_ext and small_level_state == BiState.DOWN_EXTENDING:
            return _result(1, "最恶劣", "已病",
                           "大级别向下笔延伸 + 小级别向下笔延伸，同向下跌最差")
        if big_ext and small_level_state == BiState.DOWN_FX_FORMING:
            return _result(2, "次恶劣", "已病",
                           "大级别向下笔延伸 + 小级别底分型构造")
        if big_fx and small_level_state == BiState.DOWN_EXTENDING:
            return _result(3, "第三恶劣", "欲病",
                           "大级别底分型构造 + 小级别向下笔延伸")
        if big_fx and small_level_state == BiState.DOWN_FX_FORMING:
            return _result(4, "可能出现转机", "未病",
                           "大级别/小级别都在底分型构造，可能出现转机")
        # 大级别下跌 + 小级别向上（反弹）
        if big_ext and small_up:
            return _result(None, "大级别跌+小级别反弹", "未病",
                           "小级别向上是大级别下跌中的反弹，留意是否形成大级别底分型")
        # 大级别下跌延伸 + 小级别顶分型（反弹中出现卖出信号）
        if big_ext and small_level_state == BiState.UP_FX_FORMING:
            return _result(None, "反弹中顶分型", "未病",
                           "大级别下跌延伸中，小级别反弹出现顶分型，反弹可能结束")
        # 大级别底分型 + 小级别顶分型（方向不一致的分型）
        if big_fx and small_level_state == BiState.UP_FX_FORMING:
            return _result(None, "分型方向不一致", "未病",
                           "大级别底分型 + 小级别顶分型，方向矛盾，暂观望")

    # ===== 上涨行情的预警（未病→欲病→已病）=====
    if big_up:
        if big_ext and small_level_state == BiState.UP_EXTENDING:
            return _result(None, "健康", "健康",
                           "大级别向上笔延伸 + 小级别向上笔延伸，走势健康")
        if big_ext and small_level_state == BiState.UP_FX_FORMING:
            return _result(None, "小警告", "未病",
                           "大级别向上延伸 + 小级别出现顶分型，未病（小警告）")
        if big_fx and small_level_state == BiState.UP_FX_FORMING:
            return _result(None, "欲病前兆", "未病",
                           "大级别顶分型构造中 + 小级别顶分型构造")
        if big_fx and small_down:
            return _result(None, "欲病", "欲病",
                           "大级别顶分型构造 + 小级别已向下，欲病向已病发展")
        # 大级别向上 + 小级别已向下延伸
        if big_ext and small_down:
            return _result(None, "调整预警", "未病",
                           "大级别向上延伸但小级别已向下，留意是否演化为大级别顶分型")
        # 大级别顶分型 + 小级别底分型（方向不一致的分型）
        if big_fx and small_level_state == BiState.DOWN_FX_FORMING:
            return _result(None, "分型方向不一致", "未病",
                           "大级别顶分型 + 小级别底分型，方向矛盾，暂观望")

    return _result(None, "未知组合", "未知", f"{big_level_state} + {small_level_state}")


def _result(rank: int | None, label: str, health: str, desc: str) -> dict[str, Any]:
    return {"rank": rank, "label": label, "health": health, "description": desc}


# ==================== 多级别病情矩阵 ====================

@dataclass
class LevelStateRow:
    """病情矩阵的一行：某个级别的笔状态。"""
    level_name: str      # 级别名称，如 "日线"、"周线"
    state: BiState       # 该级别的笔状态


def build_disease_matrix(
    states_by_level: dict[str, BiState],
) -> list[LevelStateRow]:
    """构建多级别病情矩阵。

    原文：这个记录是一个矩阵，按 1分钟、5分钟、30分钟、日、周、月、季、年的级别分类。
    该矩阵有 8 行，每一行就是对应级别的状态数组。

    Args:
        states_by_level: {级别名: BiState} 字典。

    Returns:
        按 1分钟→年 排序的 LevelStateRow 列表。
    """
    order = ["1分钟", "5分钟", "15分钟", "30分钟", "60分钟", "日线", "周线", "月线", "季线", "年线"]
    sorted_levels = sorted(
        states_by_level.items(),
        key=lambda kv: order.index(kv[0]) if kv[0] in order else len(order),
    )
    return [LevelStateRow(level_name=lv, state=st) for lv, st in sorted_levels]


def diagnose_multilevel(
    states_by_level: dict[str, BiState],
) -> list[dict[str, Any]]:
    """对相邻级别的每一对做病情诊断。

    对矩阵中每对相邻级别（大, 小），调用 classify_disease，
    返回所有相邻对的诊断结果列表。

    Args:
        states_by_level: {级别名: BiState} 字典。

    Returns:
        [{"big_level": ..., "small_level": ..., **diagnosis}]
    """
    matrix = build_disease_matrix(states_by_level)
    results: list[dict[str, Any]] = []
    for i in range(len(matrix) - 1):
        big = matrix[i]
        small = matrix[i + 1]
        diag = classify_disease(big.state, small.state)
        results.append({
            "big_level": big.level_name,
            "small_level": small.level_name,
            "big_state": big.state.value,
            "small_state": small.state.value,
            **diag,
        })
    return results
