"""缠论两重表里关系交易策略（第91-92课完整实现）。

合并 v1 + v2，按 `chanlun_two_level_trading.md` 文档规范实现。

核心算法链（文档 §2）：
    大级别状态 + 小级别状态 → 病情诊断 → 分型态决策点 → 中继/反转判断 → 精确买卖

状态定义（BiState）：
    (1, 1)   向上笔延伸中       → 持有/观望
    (1, 0)   向上笔顶分型构造   → 卖决策点
    (-1, 1)  向下笔延伸中       → 空仓/观望
    (-1, 0)  向下笔底分型构造   → 买决策点

第92课三策略完整实现：
    策略1（保守型）：(1,0)后兑现成本，MA5确认卖出
    策略2（震荡型）：围绕分型做短差，MACD背驰判断中继/反转
    策略3（力度比较型）：比较前后两段(1,1)力度，后段<前段→见顶

风险控制（文档 §7）：
    - MA60 趋势过滤（顺趋势交易）
    - MA5 分型确认（过滤假信号）
    - 止损/移动止盈/冷却期/状态确认延迟
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from zenstock.chanlun.bi_state import BiState
from zenstock.strategy.base import Action, BaseStrategy, Signal


# ==================== 盈利矩阵：状态组合→收益分布统计（文档 §9.3） ====================

@dataclass
class ReturnDistribution:
    """某一状态组合的历史收益分布。"""
    returns: list[float] = field(default_factory=list)

    @property
    def sample_size(self) -> int:
        return len(self.returns)

    @property
    def win_rate(self) -> float:
        """P(收益 > 0)。"""
        if not self.returns:
            return 0.0
        return sum(1 for r in self.returns if r > 0) / len(self.returns)

    @property
    def expectancy(self) -> float:
        """期望收益 = 平均收益。"""
        if not self.returns:
            return 0.0
        return sum(self.returns) / len(self.returns)

    @property
    def median_return(self) -> float:
        """中位数收益。"""
        if not self.returns:
            return 0.0
        sorted_r = sorted(self.returns)
        n = len(sorted_r)
        return sorted_r[n // 2] if n % 2 == 1 else (sorted_r[n // 2 - 1] + sorted_r[n // 2]) / 2

    @property
    def payoff_ratio(self) -> float:
        """赔率 = 中位正收益 / 中位负收益绝对值。"""
        wins = [r for r in self.returns if r > 0]
        losses = [abs(r) for r in self.returns if r < 0]
        if not wins or not losses:
            return 0.0
        win_med = sorted(wins)[len(wins) // 2]
        loss_med = sorted(losses)[len(losses) // 2]
        return win_med / loss_med if loss_med > 0 else 0.0


class ProfitMatrix:
    """病情矩阵概率引擎（文档 §9.3 第一部分）。

    键 = (big_state, small_state, trend, has_divergence)
    值 = ReturnDistribution（胜率、赔率、期望收益）

    用法：
        1. 遍历历史 K 线，对每根 K 线计算状态组合 + 远期收益，记录到矩阵
        2. 从矩阵中筛选最大可能盈利的转折状态（白名单）
        3. 实时 on_bar 中查矩阵 → 判断白名单 → 决策
    """

    def __init__(
        self,
        min_samples: int = 10,
        min_win_rate: float = 0.55,
        min_payoff: float = 1.2,
        top_n: int = 20,
    ) -> None:
        self._data: dict[tuple, ReturnDistribution] = defaultdict(ReturnDistribution)
        self.min_samples = min_samples
        self.min_win_rate = min_win_rate
        self.min_payoff = min_payoff
        self.top_n = top_n
        self._whitelist: set[tuple] | None = None

    def record(
        self,
        key: tuple[BiState | None, BiState | None, str, bool],
        forward_return: float,
    ) -> None:
        """记录一次状态组合→远期收益。"""
        self._data[key].returns.append(forward_return)
        self._whitelist = None  # 失效缓存

    def lookup(
        self, key: tuple[BiState | None, BiState | None, str, bool]
    ) -> ReturnDistribution | None:
        """查询某组合的收益分布。"""
        dist = self._data.get(key)
        if dist is not None and dist.sample_size > 0:
            return dist
        return None

    def all_combos(self) -> list[tuple[tuple, ReturnDistribution]]:
        """返回所有组合及其分布。"""
        return list(self._data.items())

    def build_whitelist(self) -> set[tuple]:
        """筛选最大可能盈利的转折状态白名单（文档 §9.3 第二部分）。

        筛选标准：
        - 样本量 >= min_samples
        - 胜率 >= min_win_rate
        - 赔率 >= min_payoff
        按期望收益排序取 Top-N。
        """
        candidates = []
        for key, dist in self._data.items():
            if dist.sample_size < self.min_samples:
                continue
            if dist.win_rate < self.min_win_rate:
                continue
            if dist.payoff_ratio < self.min_payoff and dist.expectancy <= 0:
                continue
            candidates.append((key, dist))

        candidates.sort(key=lambda x: x[1].expectancy, reverse=True)
        self._whitelist = {key for key, _ in candidates[: self.top_n]}
        return self._whitelist

    @property
    def whitelist(self) -> set[tuple]:
        """获取白名单（自动构建）。"""
        if self._whitelist is None:
            self.build_whitelist()
        return self._whitelist  # type: ignore[return-value]

    def is_best_reversal(self, key: tuple) -> bool:
        """判断当前组合是否在白名单内。"""
        return key in self.whitelist

    def save(self, path: str) -> None:
        """保存矩阵到 JSON 文件（供离线预计算后复用）。"""
        import json
        from pathlib import Path

        data = {
            "config": {
                "min_samples": self.min_samples,
                "min_win_rate": self.min_win_rate,
                "min_payoff": self.min_payoff,
                "top_n": self.top_n,
            },
            "combos": {},
        }
        for key, dist in self._data.items():
            # key = (BiState, BiState, str, bool) → 可序列化
            big_s, small_s, trend, has_div = key
            data["combos"][f"{big_s.value}|{small_s.value}|{trend}|{has_div}"] = {
                "returns": dist.returns,
                "win_rate": dist.win_rate,
                "payoff_ratio": dist.payoff_ratio,
                "expectancy": dist.expectancy,
                "sample_size": dist.sample_size,
            }
        data["whitelist"] = [
            f"{k[0].value}|{k[1].value}|{k[2]}|{k[3]}"
            for k in self.whitelist
        ]

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ProfitMatrix":
        """从 JSON 文件加载预计算的矩阵。"""
        import json

        with open(path) as f:
            data = json.load(f)

        cfg = data.get("config", {})
        matrix = cls(
            min_samples=cfg.get("min_samples", 5),
            min_win_rate=cfg.get("min_win_rate", 0.5),
            min_payoff=cfg.get("min_payoff", 0.8),
            top_n=cfg.get("top_n", 30),
        )

        # 反序列化 BiState
        state_map = {s.value: s for s in BiState}
        for key_str, dist_data in data.get("combos", {}).items():
            parts = key_str.split("|")
            big_s = state_map.get(parts[0])
            small_s = state_map.get(parts[1])
            trend = parts[2]
            has_div = parts[3] == "True"
            if big_s is None or small_s is None:
                continue
            key = (big_s, small_s, trend, has_div)
            rd = ReturnDistribution(returns=dist_data["returns"])
            matrix._data[key] = rd

        # 加载白名单
        wl = set()
        for key_str in data.get("whitelist", []):
            parts = key_str.split("|")
            big_s = state_map.get(parts[0])
            small_s = state_map.get(parts[1])
            if big_s and small_s:
                wl.add((big_s, small_s, parts[2], parts[3] == "True"))
        matrix._whitelist = wl

        return matrix


def score_to_grade(dist: ReturnDistribution) -> str:
    """根据收益分布给出 S/A/B/C 等级（文档 §4.7 §9.3）。"""
    if dist.sample_size < 3:
        return "C"
    wr = dist.win_rate
    pr = dist.payoff_ratio
    exp = dist.expectancy
    if wr >= 0.6 and pr >= 1.5 and exp > 1.0:
        return "S"
    if wr >= 0.55 and pr >= 1.0 and exp > 0.5:
        return "A"
    if wr >= 0.5 and exp > 0:
        return "B"
    return "C"


def grade_to_position(grade: str, max_position: float = 1.0) -> float:
    """等级 → 仓位（文档 §4.7）。"""
    table = {"S": 1.0, "A": 0.6, "B": 0.3, "C": 0.0}
    return min(table.get(grade, 0.0), max_position)


# ==================== 概率矩阵（文档 §4.6 更新版） ====================
# 核心变化（相比旧 ProfitMatrix）：
#   1. 胜负 = "是否出现向上笔"（二分类），不再用 forward_return 收益率
#   2. 新增 fx_count 维度：同一分型出现 1/2/3 次后反转概率不同
#   3. 新增 trading_level：30分钟策略 / 日线策略
#   4. 白名单筛选改为纯胜率驱动，不依赖赔率/期望收益

@dataclass
class WinLossStats:
    """某一状态组合的胜负统计（文档 §4.6 二分类胜负）。"""
    wins: int = 0       # 信号验证成功次数（买入→向上笔 / 卖出→向下笔）
    losses: int = 0     # 信号验证失败次数

    @property
    def sample_size(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        """胜率 = P(信号验证成功)。"""
        if self.sample_size == 0:
            return 0.0
        return self.wins / self.sample_size


class ProbabilityMatrix:
    """概率矩阵引擎（文档 §4.6 更新版）。

    键 = (big_state, small_state, fx_count, trading_level, direction)
    值 = WinLossStats（胜率、样本数）

    核心改进：
    - direction："buy"（底分型→向上笔）/ "sell"（顶分型→向下笔）
      买卖信号分开统计，白名单也按方向分别筛选
    - fx_count：分型出现次数（1/2/3...），同状态不同次数胜率不同
    - trading_level："30分钟" / "日线"，决定胜负如何判定
    - 白名单按胜率排序，不依赖赔率/期望收益
    """

    def __init__(
        self,
        min_samples: int = 5,
        min_win_rate: float = 0.55,
        top_n: int = 30,
    ) -> None:
        self._data: dict[tuple, WinLossStats] = defaultdict(WinLossStats)
        self.min_samples = min_samples
        self.min_win_rate = min_win_rate
        self.top_n = top_n
        self._whitelist: set[tuple] | None = None

    def record(
        self,
        big_state: BiState,
        small_state: BiState,
        fx_count: int,
        trading_level: str,
        direction: str,
        is_win: bool,
    ) -> None:
        """记录一次状态组合→胜负。

        Args:
            big_state: 大级别笔状态
            small_state: 小级别笔状态
            fx_count: 当前分型出现次数（分桶后 1/2/3）
            trading_level: "30分钟" / "日线"
            direction: "buy"（底分型→向上笔胜）/ "sell"（顶分型→向下笔胜）
            is_win: True=信号验证成功，False=验证失败
        """
        key = (big_state, small_state, fx_count, trading_level, direction)
        stats = self._data[key]
        if is_win:
            stats.wins += 1
        else:
            stats.losses += 1
        self._whitelist = None

    def lookup(
        self,
        big_state: BiState,
        small_state: BiState,
        fx_count: int,
        trading_level: str,
        direction: str = "",
    ) -> WinLossStats | None:
        """查询某组合的胜负统计。"""
        key = (big_state, small_state, fx_count, trading_level, direction)
        stats = self._data.get(key)
        if stats is not None and stats.sample_size > 0:
            return stats
        return None

    def all_combos(self) -> list[tuple[tuple, WinLossStats]]:
        """返回所有组合及其统计。"""
        return list(self._data.items())

    def build_whitelist(self) -> set[tuple]:
        """筛选高胜率白名单（文档 §4.7）。

        筛选标准：
        - 样本量 >= min_samples
        - 胜率 >= min_win_rate
        按胜率排序取 Top-N。
        """
        candidates = []
        for key, stats in self._data.items():
            if stats.sample_size < self.min_samples:
                continue
            if stats.win_rate < self.min_win_rate:
                continue
            candidates.append((key, stats))

        candidates.sort(key=lambda x: x[1].win_rate, reverse=True)
        self._whitelist = {key for key, _ in candidates[: self.top_n]}
        return self._whitelist

    @property
    def whitelist(self) -> set[tuple]:
        if self._whitelist is None:
            self.build_whitelist()
        return self._whitelist  # type: ignore[return-value]

    def is_high_probability(
        self,
        big_state: BiState,
        small_state: BiState,
        fx_count: int,
        trading_level: str,
        direction: str = "",
    ) -> bool:
        """判断当前组合是否在高胜率白名单内。"""
        key = (big_state, small_state, fx_count, trading_level, direction)
        return key in self.whitelist

    def grade_for(
        self,
        big_state: BiState,
        small_state: BiState,
        fx_count: int,
        trading_level: str,
        direction: str = "",
    ) -> str:
        """根据胜率给出 S/A/B/C 等级（文档 §4.7）。"""
        stats = self.lookup(big_state, small_state, fx_count, trading_level, direction)
        if stats is None or stats.sample_size < self.min_samples:
            return "C"
        key = (big_state, small_state, fx_count, trading_level, direction)
        in_wl = key in self.whitelist
        wr = stats.win_rate
        if in_wl and wr >= 0.55:
            return "S"
        if in_wl and wr >= 0.48:
            return "A"
        if wr >= 0.42:
            return "B"
        return "C"

    def save(self, path: str) -> None:
        """保存到 JSON。"""
        import json
        from pathlib import Path

        data = {
            "config": {
                "min_samples": self.min_samples,
                "min_win_rate": self.min_win_rate,
                "top_n": self.top_n,
            },
            "version": 3,  # v3: 增加 direction 维度（buy/sell）
            "combos": {},
        }
        for key, stats in self._data.items():
            big_s, small_s, fx_count, level, direction = key
            data["combos"][f"{big_s.value}|{small_s.value}|{fx_count}|{level}|{direction}"] = {
                "wins": stats.wins,
                "losses": stats.losses,
                "win_rate": stats.win_rate,
                "sample_size": stats.sample_size,
            }
        data["whitelist"] = [
            f"{k[0].value}|{k[1].value}|{k[2]}|{k[3]}|{k[4]}"
            for k in self.whitelist
        ]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ProbabilityMatrix":
        """从 JSON 加载。"""
        import json

        with open(path) as f:
            data = json.load(f)

        cfg = data.get("config", {})
        matrix = cls(
            min_samples=cfg.get("min_samples", 5),
            min_win_rate=cfg.get("min_win_rate", 0.55),
            top_n=cfg.get("top_n", 30),
        )

        state_map = {s.value: s for s in BiState}
        version = data.get("version", 1)

        for key_str, stats_data in data.get("combos", {}).items():
            parts = key_str.split("|")
            big_s = state_map.get(parts[0])
            small_s = state_map.get(parts[1])
            if big_s is None or small_s is None:
                continue
            fx_count = int(parts[2])
            level = parts[3]
            # v3 格式有 direction（parts[4]），v2 没有 → 默认空字符串兼容
            direction = parts[4] if version >= 3 and len(parts) > 4 else ""
            key = (big_s, small_s, fx_count, level, direction)
            matrix._data[key] = WinLossStats(
                wins=stats_data.get("wins", 0),
                losses=stats_data.get("losses", 0),
            )

        wl = set()
        for key_str in data.get("whitelist", []):
            parts = key_str.split("|")
            big_s = state_map.get(parts[0])
            small_s = state_map.get(parts[1])
            if big_s and small_s:
                direction = parts[4] if version >= 3 and len(parts) > 4 else ""
                wl.add((big_s, small_s, int(parts[2]), parts[3], direction))
        matrix._whitelist = wl
        return matrix




# ==================== 概率统计（趋势上下文） ====================

@dataclass
class EnhancedStats:
    """带趋势上下文的历史状态转移统计。

    transition_counts[(from_state, to_state, trend)] = count

    向后兼容：也支持无 trend 的 v1 调用方式（treat as trend=""）。
    """

    transition_counts: dict[tuple[BiState, BiState, str], int] = field(default_factory=dict)

    def record(self, from_state: BiState, to_state: BiState, trend: str = "") -> None:
        """记录一次状态转移。"""
        key = (from_state, to_state, trend)
        self.transition_counts[key] = self.transition_counts.get(key, 0) + 1

    def total_from(self, state: BiState, trend: str = "") -> int:
        """从某状态出发的总转移次数。

        若 trend 非空，只统计该趋势下的；
        若 trend 为空，统计所有趋势（v1 兼容模式）。

        兼容 v1 的 2-tuple key (from, to) 和 v2 的 3-tuple key (from, to, trend)。
        """
        total = 0
        for key, count in self.transition_counts.items():
            if len(key) == 2:
                frm, _ = key
                if frm == state:
                    total += count
            elif len(key) == 3:
                frm, _, tr = key
                if frm == state and (trend == "" or tr == trend):
                    total += count
        return total

    def probability(self, state: BiState, trend: str = "") -> "SignalProbability":
        """计算某状态在特定趋势下的涨跌概率。"""
        total = self.total_from(state, trend)
        if total == 0 and trend:
            # 回退到不分趋势
            total = self.total_from(state, "")
            trend_filter = ""
        else:
            trend_filter = trend

        if total == 0:
            return SignalProbability(
                state=state, up_prob=0.0, down_prob=0.0,
                signal="HOLD", sample_size=0,
            )

        up_count = 0
        down_count = 0
        for key, count in self.transition_counts.items():
            if len(key) == 2:
                frm, to = key
                if frm == state:
                    if to.direction == 1:
                        up_count += count
                    else:
                        down_count += count
            elif len(key) == 3:
                frm, to, tr = key
                if frm == state and (trend_filter == "" or tr == trend_filter):
                    if to.direction == 1:
                        up_count += count
                    else:
                        down_count += count

        return SignalProbability(
            state=state,
            up_prob=up_count / total,
            down_prob=down_count / total,
            signal="HOLD",
            sample_size=total,
        )


# 向后兼容别名（前端 tab_bi_state.py 使用）
HistoricalStats = EnhancedStats


@dataclass
class SignalProbability:
    """概率信号。"""
    state: BiState
    up_prob: float
    down_prob: float
    signal: str       # "BUY" / "SELL" / "HOLD"
    sample_size: int = 0


def compute_state_signal_probability(
    current_state: BiState,
    stats: EnhancedStats | HistoricalStats,
    buy_threshold: float = 0.6,
    sell_threshold: float = 0.6,
) -> SignalProbability:
    """计算交易信号（兼容 v1 接口，无 trend 参数）。

    向后兼容：前端 tab_bi_state.py 使用此函数做概率矩阵展示。
    """
    prob = stats.probability(current_state, trend="")
    prob.signal = "HOLD"

    if prob.sample_size < 3:
        return prob

    if prob.up_prob >= buy_threshold:
        prob.signal = "BUY"
    elif prob.down_prob >= sell_threshold:
        prob.signal = "SELL"

    return prob


# 增强版别名
compute_enhanced_signal = compute_state_signal_probability


def build_historical_stats(states: list[BiState]) -> EnhancedStats:
    """从状态序列构建历史转移统计（v1 兼容）。"""
    stats = EnhancedStats()
    for i in range(len(states) - 1):
        stats.record(states[i], states[i + 1])
    return stats


# ==================== MA5 确认 ====================

def should_confirm_with_ma5(price: float, ma5: float, is_buy: bool) -> bool:
    """MA5 确认：买入需站上 MA5，卖出需跌破 MA5。

    原文第79课：如果有效跌破5周期均线，分型发展成笔的可能性大大增加。
    """
    if is_buy:
        return price >= ma5 * 0.99
    else:
        return price <= ma5 * 1.01


# ==================== 信号分级系统（文档 §4.7） ====================

# 仓位建议（文档 §4.7 信号分级体系）
GRADE_POSITION: dict[str, float] = {
    "S": 1.0,    # 共振分型 + 背驰 + 顺趋势 → 重仓
    "A": 0.6,    # 共振分型 + 背驰 → 中仓
    "B": 0.3,    # 共振分型 或 欲病确认 → 轻仓
    "C": 0.0,    # 单级别分型 → 观望
}


def is_resonance(big_state: BiState | None, small_state: BiState | None) -> bool:
    """判断大小级别是否共振（同方向分型，文档 §4.5 §4.6）。

    共振买点：大(-1,0) + 小(-1,0)  双底分型
    共振卖点：大(1,0) + 小(1,0)    双顶分型
    """
    if big_state is None or small_state is None:
        return False
    # 双底分型
    if big_state == BiState.DOWN_FX_FORMING and small_state == BiState.DOWN_FX_FORMING:
        return True
    # 双顶分型
    if big_state == BiState.UP_FX_FORMING and small_state == BiState.UP_FX_FORMING:
        return True
    return False


def is_disease_turning_point(disease: Any) -> bool:
    """判断病情矩阵是否显示转机（文档 §4.3 转机：双底分型）。"""
    if disease is None:
        return False
    for d in getattr(disease, "diagnoses", []):
        if "转机" in d.get("label", ""):
            return True
    return False


def signal_grade(
    is_buy: bool,
    resonance: bool,
    divergence: str | None,
    trend: str,
    disease_health: str = "未知",
) -> tuple[str, float]:
    """计算信号等级和建议仓位（文档 §4.7）。

    信号等级判定（文档 §4.7 表）：
    | 等级 | 条件                                   | 仓位 |
    |------|----------------------------------------|------|
    | S    | 共振分型 + 背驰 + 顺趋势               | 重仓 |
    | A    | 共振分型 + 背驰                         | 中仓 |
    | B    | 共振分型 或 欲病确认                    | 轻仓 |
    | C    | 单级别分型                             | 观望 |

    对于买入：共振=双底，背驰=底背驰，顺趋势=trend up/sideways
    对于卖出：共振=双顶，背驰=顶背驰，顺趋势=trend down/sideways

    Args:
        is_buy: True=评估买入信号, False=评估卖出信号
        resonance: 是否大小级别共振
        divergence: "top_divergence" / "bottom_divergence" / None
        trend: "up" / "down" / "sideways"
        disease_health: 病情矩阵综合健康度

    Returns:
        (grade, position_size) 如 ("S", 1.0) 或 ("C", 0.0)
    """
    # 匹配背驰类型
    div_match = False
    if is_buy and divergence == "bottom_divergence":
        div_match = True
    elif not is_buy and divergence == "top_divergence":
        div_match = True

    # 匹配趋势方向
    trend_match = False
    if is_buy and trend in ("up", "sideways"):
        trend_match = True
    elif not is_buy and trend in ("down", "sideways"):
        trend_match = True

    # 信号分级（文档 §4.7）
    if resonance and div_match and trend_match:
        return "S", GRADE_POSITION["S"]
    if resonance and div_match:
        return "A", GRADE_POSITION["A"]
    if resonance or disease_health == "欲病" or disease_health == "未病":
        # 欲病=准备进场（买）或准备离场（卖）
        return "B", GRADE_POSITION["B"]
    return "C", GRADE_POSITION["C"]



# ==================== MACD 背驰检测（策略3 力度比较） ====================

def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """计算 MACD 指标（dif, dea, hist）。

    Returns:
        (dif, dea, hist) 三列 Series
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = (dif - dea) * 2  # MACD 柱子
    return dif, dea, hist


def detect_divergence(
    hist: pd.Series,
    price: pd.Series,
    i: int,
    lookback: int = 60,
    min_interval: int = 10,
) -> str | None:
    """检测 MACD 背驰（策略3 核心算法）。

    原文第92课策略3：比较 (1,0) 区间上下两段 (1,1) 的力度。
    若后段力度（MACD柱子面积）小于前段 → 见顶信号。

    Args:
        hist: MACD 柱子序列
        price: 收盘价序列
        i: 当前 K 线索引
        lookback: 回看窗口
        min_interval: 两个峰之间最小间距

    Returns:
        "top_divergence"（顶背驰，卖出信号）
        "bottom_divergence"（底背驰，买入信号）
        None（无背驰）
    """
    if i < lookback:
        return None

    window = hist.iloc[max(0, i - lookback) : i + 1]
    price_window = price.iloc[max(0, i - lookback) : i + 1]

    # 找正柱子峰（高点）和负柱子谷（低点）
    # 顶背驰：价格创新高但 MACD 柱子面积减小
    # 底背驰：价格创新低但 MACD 负柱子面积减小（绝对值）

    # 简化实现：找最近两个正柱子峰
    pos_peaks = []
    for j in range(1, len(window) - 1):
        if window.iloc[j] > 0 and window.iloc[j] > window.iloc[j - 1] and window.iloc[j] >= window.iloc[j + 1]:
            pos_peaks.append(j)

    if len(pos_peaks) >= 2:
        # 取最后两个峰
        p1, p2 = pos_peaks[-2], pos_peaks[-1]
        if p2 - p1 >= min_interval:
            # 计算两个峰之间的柱子面积
            area1 = window.iloc[p1:p1 + 5].sum()  # 峰附近5根的面积
            area2 = window.iloc[p2:p2 + 5].sum()
            price1 = price_window.iloc[p1]
            price2 = price_window.iloc[p2]
            # 顶背驰：价格创新高但面积减小
            if price2 > price1 and area2 < area1 * 0.8:
                return "top_divergence"

    # 底背驰：找负柱子谷
    neg_valleys = []
    for j in range(1, len(window) - 1):
        if window.iloc[j] < 0 and window.iloc[j] < window.iloc[j - 1] and window.iloc[j] <= window.iloc[j + 1]:
            neg_valleys.append(j)

    if len(neg_valleys) >= 2:
        v1, v2 = neg_valleys[-2], neg_valleys[-1]
        if v2 - v1 >= min_interval:
            area1 = abs(window.iloc[v1:v1 + 5].sum())
            area2 = abs(window.iloc[v2:v2 + 5].sum())
            price1 = price_window.iloc[v1]
            price2 = price_window.iloc[v2]
            # 底背驰：价格创新低但面积减小
            if price2 < price1 and area2 < area1 * 0.8:
                return "bottom_divergence"

    return None


# ==================== 交易策略 ====================

class BiStateStrategy(BaseStrategy):
    """两重表里关系概率交易策略（完整版）。

    实现文档 §5 的第92课三策略 + §7 风险控制规则。

    决策流程（文档 §9.3 伪代码）：
        1. 计算笔状态 (d, s)
        2. 大级别延伸中 → 忽略小级别波动（仅持有/观望）
        3. 病情诊断 → 只在分型态 (1,0)/(-1,0) 决策
        4. 三重过滤：趋势 + MA5 + 背驰
        5. 止损/止盈/冷却期

    参数:
        warmup_bars: 预热期（默认 80）
        buy_threshold: 买入概率阈值（默认 0.6）
        sell_threshold: 卖出概率阈值（默认 0.6）
        ma_trend: 趋势均线周期（默认 60）
        ma_confirm: 确认均线周期（默认 5）
        stop_loss_pct: 止损百分比（默认 5.0）
        cooldown_bars: 冷却期 K 线数（默认 5）
        confirm_bars: 状态确认延迟（默认 1）
        use_divergence: 是否启用 MACD 背驰检测（策略3，默认 True）
        use_multilevel: 是否启用多级别病情矩阵过滤（文档 §4，默认 True）
        multilevel_interval: 多级别诊断间隔（每 N 根 K 线算一次，默认 5）
        position_size: 买入时的目标仓位
    """

    params = (
        ("warmup_bars", 80),
        ("buy_threshold", 0.6),
        ("sell_threshold", 0.6),
        ("ma_trend", 60),
        ("ma_confirm", 5),
        ("stop_loss_pct", 5.0),
        ("take_profit_pct", 15.0),
        ("cooldown_bars", 5),
        ("confirm_bars", 1),
        ("use_divergence", True),
        ("use_multilevel", True),
        ("multilevel_interval", 5),
        ("matrix_path", "data/profit_matrix_multi.json"),  # 预计算矩阵，自动加载缓存
        ("position_size", 1.0),
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._czsc_obj: Any = None
        self._czsc_initialized: bool = False
        self._bi_states: list[BiState] = []
        self._enhanced_stats: EnhancedStats = EnhancedStats()
        self._current_state: BiState | None = None
        self._state_age: int = 0
        self._bars_since_sell: int = 999
        self._entry_price: float = 0.0
        self._highest_since_entry: float = 0.0
        self._ma_trend_series: pd.Series | None = None
        self._ma_confirm_series: pd.Series | None = None
        self._macd_hist: pd.Series | None = None
        # 多级别病情矩阵
        self._multilevel_dfs: dict[str, pd.DataFrame] = {}  # 重采样后的各级别数据
        self._multilevel_initialized: bool = False
        self._last_multilevel_check: int = -999
        self._current_disease: Any = None  # DiseaseMatrix 快照
        # 概率矩阵：优先加载预计算文件，否则在线训练
        self._fx_count: int = 0  # 当前分型出现次数（文档 §4.6 第五维度）
        self._fx_count_dir: int = 0  # 上一次分型的方向（1=up, -1=down）
        self._matrix_signal_state: BiState | None = None
        self._matrix_fx_count: int = 0
        self._matrix_fx_count_dir: int = 0
        self._trading_level: str = "30分钟"  # 交易策略级别：与训练矩阵一致
        if self.p.matrix_path:
            try:
                self._profit_matrix: ProbabilityMatrix = ProbabilityMatrix.load(self.p.matrix_path)
                self._matrix_ready = True
            except Exception:
                self._profit_matrix = ProbabilityMatrix(
                    min_samples=5, min_win_rate=0.45, top_n=30,
                )
                self._matrix_ready = False
        else:
            self._profit_matrix = ProbabilityMatrix(
                min_samples=5, min_win_rate=0.45, top_n=30,
            )
            self._matrix_ready = False

    # ---- v1 兼容属性 ----
    @property
    def _historical_stats(self) -> EnhancedStats:
        """v1 兼容：前端 tab_bi_state.py 通过此属性访问统计。"""
        return self._enhanced_stats

    # ---- czsc 增量更新 ----

    def _init_czsc(self, df: pd.DataFrame, upto_idx: int) -> None:
        if self._czsc_initialized:
            return
        try:
            from czsc import CZSC  # type: ignore
            from zenstock.chanlun.adapter import df_to_bars

            freq = self._infer_freq(df)
            bars = df_to_bars(df.iloc[: upto_idx + 1], freq)
            if len(bars) >= 10:
                self._czsc_obj = CZSC(bars, max_bi_num=len(bars))
                self._czsc_initialized = True
        except Exception:
            self._czsc_obj = None

    @staticmethod
    def _infer_freq(df: pd.DataFrame) -> str:
        # 回测数据通常把频率放在 DataFrame.attrs，而不是逐行存储的列中。
        # 先读 attrs，避免 5 分钟数据被误判成日线，进而查询错误的矩阵级别。
        attrs_freq = getattr(df, "attrs", {}).get("freq")
        if attrs_freq is not None:
            return str(attrs_freq)
        if "freq" in df.columns and len(df) > 0:
            return str(df["freq"].iloc[0])
        return "D"

    def _infer_trading_level(self, df: pd.DataFrame) -> str:
        """将回测输入频率映射到对应的概率矩阵执行级别。"""
        freq = self._infer_freq(df)
        return "日线" if freq in ("D", "日线") else "30分钟"

    def _update_matrix_fx_count(self, state: BiState) -> int:
        """按矩阵交易级别统计分型次数，而不是按原始K线级别统计。"""
        from zenstock.chanlun.bi_state import bucket_fx_count

        if self._matrix_signal_state != state:
            if state.is_fx_forming:
                if self._matrix_signal_state is None or not self._matrix_signal_state.is_fx_forming:
                    if self._matrix_fx_count_dir != state.direction:
                        self._matrix_fx_count = 1
                        self._matrix_fx_count_dir = state.direction
                    else:
                        self._matrix_fx_count += 1
            self._matrix_signal_state = state
        return bucket_fx_count(max(self._matrix_fx_count, 1))

    def _update_czsc_bar(self, df: pd.DataFrame, i: int) -> None:
        if self._czsc_obj is None:
            self._init_czsc(df, i)
            return
        try:
            from zenstock.chanlun.adapter import df_to_bars
            freq = self._infer_freq(df)
            bars = df_to_bars(df.iloc[i : i + 1], freq)
            if bars:
                self._czsc_obj.update(bars[0])
        except Exception:
            pass

    def _compute_bi_state_from_czsc(self, i: int, df: pd.DataFrame) -> BiState | None:
        """根据 czsc 计算当前笔状态。

        必须与训练器 ``compute_level_states`` 使用完全相同的因果状态提取逻辑：
        最后一笔方向 + ``ubi_fxs``（当前未完成分型），而不是完整历史 ``fx_list``。
        否则训练时记录的状态与执行时看到的状态不匹配，会导致零交易。
        """
        if self._czsc_obj is None:
            return None

        bi_list = list(self._czsc_obj.bi_list)
        if not bi_list:
            return None

        last_bi = bi_list[-1]
        # CZSC 的 Direction/Mark 是 C 扩展枚举，str() 返回中文。
        # 用 codepoint 辅助函数可靠判断（避免 Windows 源码编码问题）。
        from zenstock.chanlun.bi_state import (
            compute_bi_state, czsc_direction_is_up, czsc_mark_is_top, czsc_mark_is_bottom,
        )
        bi_direction = getattr(last_bi, "direction", None)
        is_up = czsc_direction_is_up(bi_direction)
        direction = "up" if is_up else "down"

        # 只看当前未完成分型（ubi_fxs），与训练器保持一致
        fx_forming = False
        ubi_fxs = list(getattr(self._czsc_obj, "ubi_fxs", []))
        if ubi_fxs:
            last_fx = ubi_fxs[-1]
            mark = getattr(last_fx, "mark", "")
            if (is_up and czsc_mark_is_top(mark)) or (not is_up and czsc_mark_is_bottom(mark)):
                fx_forming = True

        return compute_bi_state(direction, fx_forming)

    # ---- 均线 + MACD 预计算 ----

    def _ensure_indicators(self, df: pd.DataFrame) -> None:
        """预计算均线和 MACD（一次性）。"""
        if self._ma_trend_series is None:
            self._ma_trend_series = df["close"].rolling(self.p.ma_trend, min_periods=1).mean()
            self._ma_confirm_series = df["close"].rolling(self.p.ma_confirm, min_periods=1).mean()
            if self.p.use_divergence:
                _, _, self._macd_hist = compute_macd(df["close"])

    # ---- 多级别病情矩阵 ----

    def _init_multilevel(self, df: pd.DataFrame) -> None:
        """初始化多级别数据（一次性重采样）。

        根据输入数据的频率，自动推导出可用的更高级别数据。
        """
        if self._multilevel_initialized or not self.p.use_multilevel:
            return
        try:
            from zenstock.chanlun.multi_level_disease import resample_for_multilevel
            source_freq = self._infer_freq(df)
            self._multilevel_dfs = resample_for_multilevel(df, source_freq)
            self._multilevel_initialized = True
        except Exception:
            self._multilevel_initialized = True  # 失败也不重试

    def _train_profit_matrix(self, i: int, df: pd.DataFrame) -> None:
        """在预热期结束时，遍历历史数据构建概率矩阵（文档 §4.6）。

        胜负判定：分型态出现后，该级别是否形成向上笔（胜）或向下笔（败）。
        不再使用远期收益窗口——胜负是二分类的笔方向确认。
        """
        if self._matrix_ready:
            return

        close = df["close"]
        self._ensure_indicators(df)

        states = self._bi_states
        if len(states) < self.p.warmup_bars:
            return

        offset = len(df) - len(states)

        # 遍历状态序列，对每个分型态统计后续是否形成向上笔
        from zenstock.chanlun.bi_state import bucket_fx_count

        # fx_count 追踪：在方向反转前，同方向分型出现的次数（文档 §4.6）
        fx_count_dir: int = 0          # 当前方向的分型计数
        fx_count_cur_dir: int = 0      # 上一次分型的方向（1=up, -1=down）
        prev_state: BiState | None = None  # 检测状态转换，避免同一分型重复记录

        # 胜负判定窗口：分型态结束后，扫描后续 N 根 K 线找延伸态
        win_lookahead = 30

        for j in range(len(states) - 1):
            state = states[j]

            # 延伸态：不统计，但用于检测转换
            if not state.is_fx_forming:
                prev_state = state
                continue

            # 同一分型态延续（多根K线停留在同一分型），不重复记录
            if state == prev_state:
                continue

            # 只在分型态统计（延伸态无预测价值）
            # 分型出现次数：方向反转时重置
            cur_dir = state.direction  # 1 或 -1
            if cur_dir != fx_count_cur_dir:
                fx_count_dir = 1
                fx_count_cur_dir = cur_dir
            else:
                fx_count_dir += 1

            # 分桶：1 / 2 / 3+（文档要求分桶统计）
            fx_count = bucket_fx_count(fx_count_dir)

            # 胜负判定：从当前分型态向后扫描，找第一个延伸态
            # 文档定义：分型出现后，该级别是否形成向上笔
            # 分型态会持续多根K线，所以不能只看 j+1，要找到分型结束后的延伸态
            outcome_state: BiState | None = None
            for k in range(j + 1, min(j + 1 + win_lookahead, len(states))):
                sk = states[k]
                if sk.is_extending:
                    outcome_state = sk
                    break

            if outcome_state is None:
                prev_state = state
                continue

            # 底分型后出现向上笔延伸 = 胜（反转向上）
            # 顶分型后出现向下笔延伸 = 卖出信号胜
            # 胜负判定（文档 §4.6 对称设计）：
            #   买入信号（底分型 -1,0）：后续出现向上笔 = 胜
            #   卖出信号（顶分型 1,0）：后续出现向下笔 = 胜
            if state == BiState.DOWN_FX_FORMING:
                direction = "buy"
                is_win = outcome_state == BiState.UP_EXTENDING
            elif state == BiState.UP_FX_FORMING:
                direction = "sell"
                is_win = outcome_state == BiState.DOWN_EXTENDING
            else:
                prev_state = state
                continue

            # 记录到概率矩阵
            self._profit_matrix.record(
                big_state=state,
                small_state=state,
                fx_count=fx_count,
                trading_level=self._trading_level,
                direction=direction,
                is_win=is_win,
            )
            prev_state = state

        self._profit_matrix.build_whitelist()
        self._matrix_ready = True

    @staticmethod
    def _classify_trend_at(ma_series: pd.Series, i: int, price: float | None = None) -> str:
        """从 MA 序列判断某时刻的趋势（与 _determine_trend 一致）。"""
        if ma_series is None or i >= len(ma_series):
            return "sideways"
        ma = ma_series.iloc[i]
        if pd.isna(ma) or ma <= 0:
            return "sideways"
        # 使用价格 vs MA（与 _determine_trend 一致）
        if price is not None and price > 0:
            if price > ma * 1.01:
                return "up"
            elif price < ma * 0.99:
                return "down"
            return "sideways"
        # 无价格时用 MA 斜率（训练阶段回溯历史时使用）
        if i < 5:
            return "sideways"
        ma_prev = ma_series.iloc[i - 5]
        if pd.isna(ma_prev) or ma_prev <= 0:
            return "sideways"
        slope = (ma - ma_prev) / ma_prev
        if slope > 0.01:
            return "up"
        elif slope < -0.01:
            return "down"
        return "sideways"

    def _update_multilevel_disease(self, i: int, df: pd.DataFrame) -> None:
        """计算当前时刻的多级别病情矩阵（文档 §4）。

        每隔 multilevel_interval 根 K 线重新计算一次。
        """
        if not self.p.use_multilevel:
            return
        if i - self._last_multilevel_check < self.p.multilevel_interval:
            return
        self._last_multilevel_check = i

        try:
            from zenstock.chanlun.multi_level_disease import (
                compute_multilevel_states,
                diagnose_disease_matrix,
                resample_for_multilevel,
            )

            # 截止到当前 bar 的数据
            current_df = df.iloc[: i + 1]
            source_freq = self._infer_freq(df)
            level_dfs = resample_for_multilevel(current_df, source_freq)

            states = compute_multilevel_states(
                df_5min=level_dfs.get("5分钟"),
                df_30min=level_dfs.get("30分钟"),
                df_daily=level_dfs.get("日线"),
                df_weekly=level_dfs.get("周线"),
            )

            if len(states) >= 2:
                self._current_disease = diagnose_disease_matrix(states)
            else:
                self._current_disease = None
        except Exception:
            pass

    def _determine_trend(self, i: int, df: pd.DataFrame) -> str:
        """用 MA60 判断大趋势（文档 §7.1）。"""
        self._ensure_indicators(df)
        price = df["close"].iloc[i]
        ma = self._ma_trend_series.iloc[i]
        if pd.isna(ma) or ma <= 0:
            return "sideways"
        if price > ma * 1.01:
            return "up"
        elif price < ma * 0.99:
            return "down"
        return "sideways"

    # ---- 主信号 ----

    def on_bar(self, i: int, df: pd.DataFrame) -> Signal:
        """每根 K 线产生交易信号。"""
        # 初始化阶段
        if i < 10:
            return Signal.hold()

        # 增量更新 czsc
        self._update_czsc_bar(df, i)

        # 计算笔状态
        state = self._compute_bi_state_from_czsc(i, df)
        if state is None:
            return Signal.hold()

        # 状态追踪 + 分型出现次数计数（文档 §4.6 第五维度）
        if self._current_state == state:
            self._state_age += 1
        else:
            if self._current_state is not None:
                trend = self._determine_trend(i, df)
                self._enhanced_stats.record(self._current_state, state, trend)
            # 分型出现次数（文档 §4.6 第五维度）：
            # 在方向反转前，同方向分型连续出现的次数
            if state.is_fx_forming:
                # 只在刚进入分型态时计数（避免同一分型多根K线重复计数）
                if self._current_state is None or not self._current_state.is_fx_forming:
                    if self._fx_count == 0 or state.direction != self._fx_count_dir:
                        # 方向变了或刚重置，重新从1开始
                        self._fx_count = 1
                        self._fx_count_dir = state.direction
                    else:
                        self._fx_count += 1
            elif state.is_extending:
                # 回到延伸态，不重置（等待方向反转后才重置）
                pass
            self._current_state = state
            self._state_age = 0

        self._bi_states.append(state)
        self._bars_since_sell += 1

        # 预热期
        if i < self.p.warmup_bars:
            return Signal.hold()

        # 确保指标已计算
        self._ensure_indicators(df)
        self._init_multilevel(df)
        self._update_multilevel_disease(i, df)
        self._trading_level = self._infer_trading_level(df)
        close = float(df["close"].iloc[i])
        ma5 = float(self._ma_confirm_series.iloc[i])
        trend = self._determine_trend(i, df)

        # ===== 多级别病情矩阵（文档 §4 核心）=====
        disease = self._current_disease
        disease_health = disease.overall_health if disease else "未知"

        # 大级别和小级别状态：从多级别病情矩阵中提取
        big_state = state
        small_state = state
        if disease is not None and len(disease.states) >= 2:
            level_names = list(disease.states.keys())
            big_state = disease.states[level_names[-1]]
            # 5分钟K线是执行载体，但30分钟才是当前概率矩阵的交易级别。
            # 因此5分钟回测必须使用30分钟状态作为small_state，避免拿
            # 5分钟状态查询“周线+30分钟”训练出来的矩阵。
            signal_level = self._trading_level
            small_state = disease.states.get(signal_level, disease.states[level_names[0]])

        matrix_fx_count = self._update_matrix_fx_count(small_state)

        # 在线训练矩阵（如果未加载预计算文件）
        if not self._matrix_ready and i >= self.p.warmup_bars:
            self._train_profit_matrix(i, df)

        # ==================================================================
        # 持仓管理：卖出优先级链（文档 §7，每根K线检查）
        # 优先级从高到低：止损 > 移动止盈 > 矩阵卖出信号
        # ==================================================================
        if self._entry_price > 0:
            total_profit = (close - self._entry_price) / self._entry_price * 100
            if close > self._highest_since_entry:
                self._highest_since_entry = close

            # 1. 固定止损：跌破买入价 stop_loss_pct%
            #    病情矩阵收紧：欲病/已病时止损线收紧 40%
            effective_stop_loss = self.p.stop_loss_pct
            if disease_health in ("欲病", "已病"):
                effective_stop_loss = self.p.stop_loss_pct * 0.6
            loss_pct = -total_profit
            if loss_pct >= effective_stop_loss:
                self._bars_since_sell = 0
                self._entry_price = 0.0
                self._highest_since_entry = 0.0
                return Signal.sell(
                    size=self.p.position_size,
                    reason=f"止损 {loss_pct:.1f}% ({disease_health})",
                )

            # 2. 移动止盈：从最高点回撤超过利润的 50%
            profit_from_high = (self._highest_since_entry - close) / self._highest_since_entry * 100
            if total_profit > 5 and profit_from_high > total_profit * 0.5:
                self._bars_since_sell = 0
                self._entry_price = 0.0
                self._highest_since_entry = 0.0
                return Signal.sell(
                    size=self.p.position_size,
                    reason=f"移动止盈 回撤{profit_from_high:.1f}% 总盈{total_profit:.1f}%",
                )

            # 3. 矩阵卖出信号：小级别顶分型(1,0) + 方向="sell" + 概率矩阵确认大概率向下
            #    只有等级 S/A 才触发卖出（B 级信号不够强，继续持有等止盈/止损）
            #    文档 §4.1：big_state 仅用于过滤大级别延伸中的小级别噪音
            if small_state == BiState.UP_FX_FORMING:
                sell_stats = self._profit_matrix.lookup(
                    big_state, small_state, matrix_fx_count, self._trading_level, "sell",
                )
                if sell_stats and sell_stats.sample_size >= self._profit_matrix.min_samples:
                    sell_grade = self._profit_matrix.grade_for(
                        big_state, small_state, matrix_fx_count, self._trading_level, "sell",
                    )
                    if sell_grade in ("S", "A"):
                        self._bars_since_sell = 0
                        self._entry_price = 0.0
                        self._highest_since_entry = 0.0
                        return Signal.sell(
                            size=self.p.position_size,
                            reason=f"{sell_grade}级卖出 WR={sell_stats.win_rate:.0%} fx={matrix_fx_count}次",
                        )

            # 持仓中不买入
            return Signal.hold()

        # ==================================================================
        # 空仓：买入条件检查（文档 §4.7 + §6）
        # 全部条件 AND（缺一不可）：
        #   1. 底分型 (-1,0)：转折信号
        #   2. 概率矩阵方向="buy"且等级 ≥ B：统计验证有反转概率
        #   3. 趋势非向下：MA60 趋势为 up 或 sideways
        #   4. MA5 确认：收盘价站上 MA5
        #   5. 冷却期已过
        #   6. 状态确认延迟已过
        # ==================================================================

        # 冷却期检查（文档 §7.3）
        if self._bars_since_sell < self.p.cooldown_bars:
            return Signal.hold()

        # 状态确认延迟（文档 §7.3）
        if self._state_age < self.p.confirm_bars:
            return Signal.hold()

        # 买入：必须是小级别底分型；大级别只负责方向和噪音过滤
        if small_state != BiState.DOWN_FX_FORMING:
            return Signal.hold()

        # 趋势过滤：MA60 向下时不做多
        if trend == "down":
            return Signal.hold()

        # 概率矩阵查询：方向="buy"
        buy_stats = self._profit_matrix.lookup(
            big_state, small_state, matrix_fx_count, self._trading_level, "buy",
        )

        # 无统计支撑 = 不交易
        if buy_stats is None or buy_stats.sample_size < self._profit_matrix.min_samples:
            return Signal.hold()

        buy_grade = self._profit_matrix.grade_for(
            big_state, small_state, matrix_fx_count, self._trading_level, "buy",
        )

        # C 级不交易
        if buy_grade == "C":
            return Signal.hold()

        # MA5 确认：收盘价必须站上 MA5
        if not should_confirm_with_ma5(close, ma5, is_buy=True):
            return Signal.hold()

        # 按信号等级决定仓位
        position = grade_to_position(buy_grade, self.p.position_size)
        if position <= 0:
            return Signal.hold()

        self._entry_price = close
        self._highest_since_entry = close
        return Signal.buy(
            size=position,
            reason=f"{buy_grade}级买入 WR={buy_stats.win_rate:.0%} fx={matrix_fx_count}次 {small_state.value} {self._trading_level} 趋势={trend} n={buy_stats.sample_size}",
        )


def diverge_status(divergence: str | None, is_buy: bool) -> str:
    """格式化背驰状态用于显示。"""
    if divergence is None:
        return "无"
    if is_buy and divergence == "bottom_divergence":
        return "底背驰!"
    if not is_buy and divergence == "top_divergence":
        return "顶背驰!"
    return "无"


# 向后兼容别名（前端 tab_bi_state.py 和测试使用 BiStateStrategyV2）
BiStateStrategyV2 = BiStateStrategy
