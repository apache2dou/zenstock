"""缠论线段（Line Segment / LD）识别。

缠论线段定义：
- 至少由 3 笔组成
- 线段的起点是第一笔的起点
- 线段的终点是最后一笔的终点
- 线段方向由第一笔方向决定
- 线段结束条件：出现与线段方向相反的笔，且该笔的极值超过前一个同向笔的极值

简化实现：
- 向上线段：起点为底分型，终点为顶分型
- 向下线段：起点为顶分型，终点为底分型
- 至少 3 笔，最多包含直到方向反转的所有笔
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LineSegment:
    """缠论线段。"""
    direction: str          # "up" 或 "down"
    start_dt: Any           # 起始时间
    end_dt: Any             # 结束时间
    start_price: float      # 起始价格
    end_price: float        # 结束价格
    high: float             # 线段最高点
    low: float              # 线段最低点
    bi_count: int           # 包含的笔数量
    bi_indices: tuple       # 包含的笔索引范围 (start_idx, end_idx)

    @property
    def is_up(self) -> bool:
        return self.direction == "up"

    @property
    def amplitude(self) -> float:
        """线段幅度。"""
        return abs(self.end_price - self.start_price) if self.start_price > 0 else 0.0

    @property
    def area(self) -> float:
        """线段面积（力度），考虑价格水平。"""
        amp = self.amplitude
        total = self.high + self.low
        if total <= 0:
            return 0.0
        return amp * amp / total


def extract_line_segments(bi_list: list) -> list[LineSegment]:
    """从笔列表提取线段。

    缠论线段的简化识别规则：
    1. 至少 3 笔才能形成线段
    2. 线段方向 = 第一笔方向
    3. 线段在出现"特征序列分型"时结束：
       - 向上线段结束：某向下笔的低点 < 前一个向下笔的低点（下降突破）
       - 向下线段结束：某向上笔的高点 > 前一个向上笔的高点（上升突破）
    4. 每个线段至少包含 3 笔

    Args:
        bi_list: czsc BI 对象列表

    Returns:
        list[LineSegment]
    """
    if len(bi_list) < 3:
        return []

    segments: list[LineSegment] = []
    seg_start_idx = 0

    while seg_start_idx < len(bi_list) - 2:
        first_bi = bi_list[seg_start_idx]
        seg_direction = "up" if _bi_is_up(first_bi) else "down"
        seg_end_idx = seg_start_idx  # 至少要到 seg_start_idx + 2

        # 从第三笔开始检查是否结束
        for i in range(seg_start_idx + 2, len(bi_list)):
            bi = bi_list[i]

            if seg_direction == "up":
                # 向上线段：检查向下笔是否创新低（特征序列顶分型）
                if _bi_is_down(bi) and i >= seg_start_idx + 3:
                    # 找前一个向下笔
                    prev_down = _find_prev_same_direction(bi_list, i, "down")
                    if prev_down is not None:
                        if _bi_low(bi) < _bi_low(prev_down):
                            # 线段结束
                            seg_end_idx = i - 1  # 结束于前一笔
                            break
            else:
                # 向下线段：检查向上笔是否创新高
                if _bi_is_up(bi) and i >= seg_start_idx + 3:
                    prev_up = _find_prev_same_direction(bi_list, i, "up")
                    if prev_up is not None:
                        if _bi_high(bi) > _bi_high(prev_up):
                            seg_end_idx = i - 1
                            break

            seg_end_idx = i

        # 构建线段
        if seg_end_idx >= seg_start_idx + 2:  # 至少 3 笔
            seg_bis = bi_list[seg_start_idx : seg_end_idx + 1]
            start_fx = getattr(first_bi, "fx_a", None)
            end_fx = getattr(seg_bis[-1], "fx_b", None)

            start_price = float(getattr(start_fx, "fx", 0) or 0) if start_fx else 0.0
            end_price = float(getattr(end_fx, "fx", 0) or 0) if end_fx else 0.0
            start_dt = getattr(start_fx, "dt", None) if start_fx else None
            end_dt = getattr(end_fx, "dt", None) if end_fx else None

            high = max(_bi_high(b) for b in seg_bis)
            low = min(_bi_low(b) for b in seg_bis)

            segments.append(LineSegment(
                direction=seg_direction,
                start_dt=start_dt,
                end_dt=end_dt,
                start_price=start_price,
                end_price=end_price,
                high=high,
                low=low,
                bi_count=len(seg_bis),
                bi_indices=(seg_start_idx, seg_end_idx),
            ))

        # 下一个线段从当前线段最后一笔的下一笔开始
        seg_start_idx = seg_end_idx + 1

    return segments


# ==================== 中枢（基于线段） ====================

@dataclass
class ZSPyramid:
    """基于线段的中枢。"""
    zg: float              # 中枢上沿（区间最高低点）
    zd: float              # 中枢下沿（区间最低高点）
    zz: float              # 中枢中轴
    start_dt: Any          # 起始时间
    end_dt: Any            # 结束时间
    seg_count: int         # 参与中枢的线段数
    bi_range: tuple        # 对应的笔索引范围

    @property
    def is_valid(self) -> bool:
        return self.zg > self.zd


def extract_zhongshu_from_segments(
    segments: list[LineSegment], bi_list: list
) -> list[ZSPyramid]:
    """从线段列表识别中枢。

    缠论中枢定义（基于线段）：
    连续三个线段的价格区间有重叠部分，重叠区间即为中枢。
    中枢上沿 zg = min(三个线段各自的最高点)
    中枢下沿 zd = max(三个线段各自的最低点)

    Args:
        segments: 线段列表
        bi_list: 原始笔列表（用于索引范围）

    Returns:
        list[ZSPyramid]
    """
    if len(segments) < 3:
        return []

    zs_list: list[ZSPyramid] = []
    i = 0
    while i <= len(segments) - 3:
        s1, s2, s3 = segments[i], segments[i + 1], segments[i + 2]

        # 三个线段的区间重叠
        zg = min(s1.high, s2.high, s3.high)  # 上沿 = 三线段最高点的最小值
        zd = max(s1.low, s2.low, s3.low)     # 下沿 = 三线段最低点的最大值

        if zg > zd:
            # 有效中枢
            zs_list.append(ZSPyramid(
                zg=zg,
                zd=zd,
                zz=(zg + zd) / 2,
                start_dt=s1.start_dt,
                end_dt=s3.end_dt,
                seg_count=3,
                bi_range=(s1.bi_indices[0], s3.bi_indices[1]),
            ))
            i += 3  # 跳过已处理的线段
        else:
            i += 1

    return zs_list


# ==================== 买卖点（基于线段+中枢） ====================

@dataclass
class BuySellPoint:
    """缠论买卖点。"""
    point_type: str       # "一买"/"二买"/"三买"/"一卖"/"二卖"/"三卖"
    dt: Any               # 时间
    price: float          # 价格
    reason: str           # 原因描述
    is_buy: bool          # True=买点, False=卖点


def detect_buy_sell_points(
    segments: list[LineSegment],
    zs_list: list[ZSPyramid],
) -> list[BuySellPoint]:
    """基于线段和中枢识别买卖点。

    缠论买卖点严格定义：
    - 一买：下跌趋势中，最后一个中枢被向下突破后，出现线段级别背驰
    - 二买：一买后的第一次回调（不破一买低点）
    - 三买：突破中枢上沿后回调不进中枢
    - 一卖/二卖/三卖：对称定义

    Args:
        segments: 线段列表
        zs_list: 中枢列表

    Returns:
        list[BuySellPoint]
    """
    points: list[BuySellPoint] = []
    if len(segments) < 3 or not zs_list:
        return points

    last_zs = zs_list[-1]
    last_seg = segments[-1]

    # ===== 买点 =====
    if last_seg.direction == "down":
        # 最近线段向下，可能是下跌末端

        # 一买：向下线段突破了最后一个中枢下沿，且线段背驰
        if len(segments) >= 2:
            prev_seg = segments[-2] if segments[-2].direction == "down" else (
                segments[-3] if len(segments) >= 3 and segments[-3].direction == "down" else None
            )
            if prev_seg:
                # 价格跌破中枢下沿
                if last_seg.low < last_zs.zd:
                    # 线段背驰：面积递减
                    if last_seg.area < prev_seg.area * 0.8 and last_seg.low <= prev_seg.low:
                        points.append(BuySellPoint(
                            point_type="一买",
                            dt=last_seg.end_dt,
                            price=last_seg.end_price,
                            reason=f"线段背驰(面积{last_seg.area:.0f}<{prev_seg.area:.0f}) 破中枢",
                            is_buy=True,
                        ))

        # 三买不会在向下线段出现
    elif last_seg.direction == "up":
        # 最近线段向上

        # 三买：向上突破中枢上沿后回调不进中枢
        if last_seg.low > last_zs.zg:
            points.append(BuySellPoint(
                point_type="三买",
                dt=last_seg.start_dt,
                price=last_seg.start_price,
                reason=f"回调不进中枢(zg={last_zs.zg:.2f})",
                is_buy=True,
            ))

        # 二买：前一个线段是向下的回调，低点不破更早的低点
        if len(segments) >= 3:
            if (segments[-2].direction == "down"
                    and segments[-2].low > segments[-3].low if segments[-3].direction == "down" else False):
                pass  # 二买条件较复杂，暂简化

    # ===== 卖点 =====
    if last_seg.direction == "up":
        # 一卖：向上线段突破了中枢上沿，且线段背驰
        if len(segments) >= 2:
            prev_seg = segments[-2] if segments[-2].direction == "up" else (
                segments[-3] if len(segments) >= 3 and segments[-3].direction == "up" else None
            )
            if prev_seg:
                if last_seg.high > last_zs.zg:
                    if last_seg.area < prev_seg.area * 0.8 and last_seg.high >= prev_seg.high:
                        points.append(BuySellPoint(
                            point_type="一卖",
                            dt=last_seg.end_dt,
                            price=last_seg.end_price,
                            reason=f"线段背驰(面积{last_seg.area:.0f}<{prev_seg.area:.0f}) 破中枢",
                            is_buy=False,
                        ))

    elif last_seg.direction == "down":
        # 三卖：跌破中枢下沿后反弹不进中枢
        if last_seg.high < last_zs.zd:
            points.append(BuySellPoint(
                point_type="三卖",
                dt=last_seg.start_dt,
                price=last_seg.start_price,
                reason=f"反弹不进中枢(zd={last_zs.zd:.2f})",
                is_buy=False,
            ))

    return points


# ==================== BI 工具函数 ====================
def _bi_is_up(bi) -> bool:
    d = str(getattr(bi, "direction", ""))
    return d.endswith("a") or "up" in d.lower() or "上" in d


def _bi_is_down(bi) -> bool:
    d = str(getattr(bi, "direction", ""))
    return d.endswith("b") or "down" in d.lower() or "下" in d


def _bi_high(bi) -> float:
    return float(getattr(bi, "high", 0) or 0)


def _bi_low(bi) -> float:
    return float(getattr(bi, "low", 0) or 0)


def _find_prev_same_direction(
    bi_list: list, current_idx: int, direction: str
) -> Any | None:
    """在 bi_list 中向前找同方向的笔。"""
    check = _bi_is_up if direction == "up" else _bi_is_down
    for i in range(current_idx - 1, -1, -1):
        if check(bi_list[i]):
            return bi_list[i]
    return None
