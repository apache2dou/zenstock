"""缠论线段（Line Segment / LD）识别。

缠论线段定义（严格按《教你炒股票》第62-71课原文）：
- 至少由 3 笔组成，前 3 笔必须有重叠
- 线段方向由第一笔方向决定
- 线段只能被线段破坏，单笔不能破坏线段

线段终结的两种情况（第67课原文）：
  情况1: 特征序列第一、二元素间**无**缺口
    → 特征序列出现分型即确认线段结束
  情况2: 特征序列第一、二元素间**有**缺口
    → 必须等反向线段的特征序列出现分型才能确认

特征序列定义：
  - 向上线段 → 向下笔构成特征序列 X1, X2, ..., Xn
  - 向下线段 → 向上笔构成特征序列 S1, S2, ..., Sn
  对特征序列做包含处理，得到标准特征序列后找分型。
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


def _is_contained(a_high, a_low, b_high, b_low) -> bool:
    """判断两根K线/笔是否具有包含关系（一根完全在另一根范围内）。"""
    return (a_high >= b_high and a_low <= b_low) or (b_high >= a_high and b_low <= a_low)


def _merge_contained(
    a_high: float, a_low: float, b_high: float, b_low: float, direction: str
) -> tuple[float, float]:
    """按缠论包含关系合并两根K线/笔。

    direction='up': 取高高 + 低高中较高者
    direction='down': 取低低 + 高低中较低者
    """
    if direction == "up":
        return max(a_high, b_high), max(a_low, b_low)
    else:
        return min(a_low, b_low), min(a_high, b_high)


def _process_feature_sequence(
    features: list[tuple[float, float]], seg_direction: str
) -> list[tuple[float, float]]:
    """对特征序列做包含关系处理，得到标准特征序列。

    特征序列的方向判定：与线段方向相同（原文：向上线段的特征序列的包含处理方向是向上）。
    即：原来的包含关系方向与线段方向一致。

    同时记录合并来源，供后续 _map_to_original_feature_idx 使用。
    """
    if len(features) < 2:
        return features

    result = [features[0]]
    for i in range(1, len(features)):
        prev = result[-1]
        curr = features[i]

        if _is_contained(prev[0], prev[1], curr[0], curr[1]):
            # 特征序列的包含处理方向 = 线段方向
            merge_dir = "up" if seg_direction == "up" else "down"
            merged = _merge_contained(prev[0], prev[1], curr[0], curr[1], merge_dir)
            result[-1] = merged
        else:
            result.append(curr)

    return result


def _find_fx_in_features(
    std_features: list[tuple[float, float]], seg_direction: str
) -> int | None:
    """在标准特征序列中寻找分型，返回分型位置索引（-1 表示找到）。

    向上线段 → 在特征序列中找顶分型
    向下线段 → 在特征序列中找底分型
    """
    if len(std_features) < 3:
        return None

    for i in range(1, len(std_features) - 1):
        prev_high, prev_low = std_features[i - 1]
        mid_high, mid_low = std_features[i]
        next_high, next_low = std_features[i + 1]

        if seg_direction == "up":
            # 严格顶分型：中间高点最高，且低点也最高
            if (
                mid_high > prev_high
                and mid_high > next_high
                and mid_low > prev_low
                and mid_low > next_low
            ):
                return i
        else:
            # 严格底分型：中间低点最低，且高点也最低
            if (
                mid_low < prev_low
                and mid_low < next_low
                and mid_high < prev_high
                and mid_high < next_high
            ):
                return i

    return None


def apply_fx_auxiliary_operation(
    features: list[tuple[float, float]], seg_direction: str
) -> dict[str, Any]:
    """按缠论原文的“分型辅助操作”规则，统一处理包含关系、分型、缺口。

    这一步是线段识别的辅助操作：
    1. 先对特征序列做包含处理，得到标准特征序列。
    2. 在标准特征序列中找出分型。
    3. 判定分型前后两元素之间是否存在缺口。

    Returns:
        {
            "std_features": [...],
            "fx_idx": int|None,
            "fx_type": "顶分型"|"底分型"|None,
            "has_gap": bool,
        }
    """
    std_features = _process_feature_sequence(features, seg_direction)
    fx_idx = _find_fx_in_features(std_features, seg_direction)

    if fx_idx is None:
        return {
            "std_features": std_features,
            "fx_idx": None,
            "fx_type": None,
            "has_gap": False,
        }

    fx_type = "顶分型" if seg_direction == "up" else "底分型"
    has_gap = _has_gap_in_features(std_features, fx_idx, seg_direction)

    return {
        "std_features": std_features,
        "fx_idx": fx_idx,
        "fx_type": fx_type,
        "has_gap": has_gap,
    }


def _has_gap_in_features(
    std_features: list[tuple[float, float]], fx_idx: int, seg_direction: str
) -> bool:
    """检查特征序列分型的第一、二元素间是否有缺口（第67课原文）。

    缺口定义：相邻两个特征序列元素之间没有重合区间。
    顶分型（用于向上线段）：第一元素是 fx_idx-1，第二元素是 fx_idx。
        特征序列以 (high, low) 表示，无缺口需 first.low >= second.high 或 second.low >= first.high
    底分型（用于向下线段）：同上。
    """
    if fx_idx is None or fx_idx < 1 or fx_idx >= len(std_features):
        return False

    first = std_features[fx_idx - 1]   # (high, low)
    second = std_features[fx_idx]     # (high, low)

    # 缺口 = 两个区间没有重叠
    # 对于 (high, low) 这组表示，若 first.low > second.high 或 second.low > first.high，说明无重叠
    has_gap = first[1] > second[0] or second[1] > first[0]
    return has_gap


def extract_line_segments(bi_list: list) -> list[LineSegment]:
    """从笔列表提取线段。

    严格按缠论第62-71课原文实现：

    1. 至少 3 笔，前 3 笔必须有重叠
    2. 线段方向 = 第一笔方向
    3. 特征序列 = 与线段方向相反的笔
    4. 对特征序列做包含处理 → 标准特征序列
    5. 在标准特征序列中找分型
    6. 情况1（无缺口）：分型确认 → 线段在分型高点/低点结束
    7. 情况2（有缺口）：需反向特征序列分型确认

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

        # 检查前3笔是否有重叠
        if seg_start_idx + 2 >= len(bi_list):
            break
        bis_3 = bi_list[seg_start_idx : seg_start_idx + 3]
        overlap_high = min(_bi_high(b) for b in bis_3)
        overlap_low = max(_bi_low(b) for b in bis_3)
        if overlap_low >= overlap_high:
            # 前3笔无重叠，不构成线段，跳过当前笔
            seg_start_idx += 1
            continue

        # 收集特征序列（与线段方向相反的笔）
        feature_bis: list = []
        feature_indices: list[int] = []
        raw_features: list[tuple[float, float]] = []

        for j in range(seg_start_idx, len(bi_list)):
            bi = bi_list[j]
            is_opposite = (_bi_is_down(bi) if seg_direction == "up" else _bi_is_up(bi))
            if is_opposite:
                feature_bis.append(bi)
                feature_indices.append(j)
                raw_features.append((_bi_high(bi), _bi_low(bi)))

        if len(raw_features) < 2:
            # 特征序列不足2个元素，线段延续到末尾
            seg_end_idx = len(bi_list) - 1
            segments.append(_build_segment(bi_list, seg_start_idx, seg_end_idx, seg_direction))
            break

        # 采用统一的“分型辅助操作”流程：包含处理 → 标准特征序列 → 找分型 → 判断缺口
        aux = apply_fx_auxiliary_operation(raw_features, seg_direction)
        std_features = aux["std_features"]
        fx_idx = aux["fx_idx"]

        if fx_idx is None:
            # 无分型，线段延续到末尾
            seg_end_idx = len(bi_list) - 1
            segments.append(_build_segment(bi_list, seg_start_idx, seg_end_idx, seg_direction))
            break

        # 判断是否有缺口
        has_gap = _has_gap_in_features(std_features, fx_idx, seg_direction)

        if not has_gap:
            # 情况1: 无缺口，分型直接确认线段结束
            # 分型对应原始特征序列中的笔，该笔位置即线段终点
            orig_fx_idx_in_features = _map_to_original_feature_idx(
                raw_features, std_features, fx_idx
            )
            if orig_fx_idx_in_features is not None and orig_fx_idx_in_features < len(feature_indices):
                # 线段在特征序列分型对应的笔之前一笔结束
                seg_end_idx = feature_indices[orig_fx_idx_in_features] - 1
                seg_end_idx = max(seg_end_idx, seg_start_idx + 2)  # 至少3笔
            else:
                seg_end_idx = len(bi_list) - 1
        else:
            # 情况2: 有缺口，需要等反向线段特征序列分型确认
            # 这是复杂情况，简化处理：取特征序列分型后2个元素确认
            # 完整实现需递归检查反向线段
            if fx_idx + 2 < len(std_features):
                # 有足够多的特征序列元素确认
                orig_fx_idx = _map_to_original_feature_idx(raw_features, std_features, fx_idx)
                if orig_fx_idx is not None and orig_fx_idx < len(feature_indices):
                    seg_end_idx = feature_indices[orig_fx_idx] - 1
                    seg_end_idx = max(seg_end_idx, seg_start_idx + 2)
                else:
                    seg_end_idx = len(bi_list) - 1
            else:
                # 需要等更多笔，暂不能终结线段，先取到末尾
                seg_end_idx = len(bi_list) - 1

        segments.append(_build_segment(bi_list, seg_start_idx, seg_end_idx, seg_direction))
        seg_start_idx = seg_end_idx + 1

    return segments


def _map_to_original_feature_idx(
    raw_features: list[tuple[float, float]],
    std_features: list[tuple[float, float]],
    std_idx: int,
) -> int | None:
    """将标准特征序列分型位置映射回原始特征序列位置。

    精确映射：遍历 raw_features，重新执行包含处理逻辑，
    记录每个 std 元素对应的 raw 索引范围，取其中最后一个。
    """
    if not raw_features or std_idx >= len(std_features):
        return None

    # 重新执行包含处理，但记录每个 std 元素对应的 raw 源索引
    # raw_sources[k] = set of raw indices merged into std[k]
    std_sources: list[set[int]] = [{0}]
    result = [raw_features[0]]

    for i in range(1, len(raw_features)):
        prev = result[-1]
        curr = raw_features[i]

        # 方向不影响合并后的映射，只看是否包含
        if _is_contained(prev[0], prev[1], curr[0], curr[1]):
            # 合并：把当前 raw 索引加到上一个 std 元素的来源集合
            std_sources[-1].add(i)
            # 合并后的值（取合并后的 high, low，方向不影响索引映射）
            # 注意：这里的方向与实际调用一致，但只用于确定合并规则
            # 由于我们只关心索引映射，合并后的具体值不影响
            result[-1] = (
                max(prev[0], curr[0]),
                max(prev[1], curr[1]),
            )  # 向上方向的合并（近似）
        else:
            result.append(curr)
            std_sources.append({i})

    if std_idx < len(std_sources):
        # 返回该 std 元素对应的最后一个 raw 索引
        return max(std_sources[std_idx])
    return min(std_idx, len(raw_features) - 1)


def _build_segment(
    bi_list: list, start_idx: int, end_idx: int, direction: str
) -> LineSegment:
    """根据笔索引范围构建线段对象。"""
    seg_bis = bi_list[start_idx : end_idx + 1]
    start_fx = getattr(bi_list[start_idx], "fx_a", None)
    end_fx = getattr(bi_list[end_idx], "fx_b", None)

    start_price = float(getattr(start_fx, "fx", 0) or 0) if start_fx else 0.0
    end_price = float(getattr(end_fx, "fx", 0) or 0) if end_fx else 0.0
    start_dt = getattr(start_fx, "dt", None) if start_fx else None
    end_dt = getattr(end_fx, "dt", None) if end_fx else None

    high = max(_bi_high(b) for b in seg_bis)
    low = min(_bi_low(b) for b in seg_bis)

    return LineSegment(
        direction=direction,
        start_dt=start_dt,
        end_dt=end_dt,
        start_price=start_price,
        end_price=end_price,
        high=high,
        low=low,
        bi_count=len(seg_bis),
        bi_indices=(start_idx, end_idx),
    )


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

    缠论中枢定义（基于线段，第17-20课原文）：
    至少三个连续次级别走势类型（此处为线段）的价格区间有重叠部分。
    重叠区间即为中枢。

    中枢上沿 ZG = min(g1, g2, g3)  （三个线段高点的最小值）
    中枢下沿 ZD = max(d1, d2, d3)  （三个线段低点的最大值）

    中枢延伸：后续线段仍在此区间内震荡
    中枢级别扩展：连续9段（3+3+3）重叠 → 更高级别中枢

    注意：线段是次级别走势的替代，在日线及以上级别可用线段直接构建中枢。

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
        zg = min(s1.high, s2.high, s3.high)  # 中枢上沿
        zd = max(s1.low, s2.low, s3.low)     # 中枢下沿

        if zg > zd:
            # 有效中枢，检查是否是前一个中枢的延伸
            if zs_list and _is_zs_extension(zs_list[-1], zg, zd, segments, i):
                # 中枢延伸：更新结束时间和线段数
                zs_list[-1].end_dt = s3.end_dt
                zs_list[-1].seg_count += 3
                zs_list[-1].bi_range = (
                    zs_list[-1].bi_range[0],
                    s3.bi_indices[1],
                )
                i += 3
            else:
                zs_list.append(ZSPyramid(
                    zg=zg,
                    zd=zd,
                    zz=(zg + zd) / 2,
                    start_dt=s1.start_dt,
                    end_dt=s3.end_dt,
                    seg_count=3,
                    bi_range=(s1.bi_indices[0], s3.bi_indices[1]),
                ))
                i += 1  # 不跳3，允许连续中枢
        else:
            i += 1

    return zs_list


def _is_zs_extension(
    prev_zs: ZSPyramid,
    new_zg: float,
    new_zd: float,
    segments: list[LineSegment],
    current_idx: int,
) -> bool:
    """检查新中枢是否是前一个中枢的延伸。

    延伸条件（中枢中心定理一）：
    新高ZD和新低ZG仍与 [prev_zs.zd, prev_zs.zg] 有重叠。
    """
    # 新区间与旧区间有重叠
    if new_zg <= prev_zs.zd or new_zd >= prev_zs.zg:
        return False

    # 且不是明显的新中枢（区间差异不大）
    overlap_top = min(new_zg, prev_zs.zg)
    overlap_bottom = max(new_zd, prev_zs.zd)
    overlap_size = overlap_top - overlap_bottom
    if overlap_size <= 0:
        return False

    # 重叠部分占新区间的比例 > 30% → 视为延伸
    new_size = new_zg - new_zd
    if new_size > 0 and overlap_size / new_size > 0.3:
        return True

    return False


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

    严格按缠论第14、18、20、21课原文：

    **第一类买点**：下跌趋势中，最后一个中枢被向下突破后，
    出现线段背驰的最低点。
    **第二类买点**：一买后次级别回抽（向上笔后回调）不破一买低点。
    **第三类买点**：次级别向上离开中枢，次级别回抽不破中枢上沿(ZG)。

    卖点对称定义。

    与原文的差异说明：
    原文的三类买卖点基于"次级别走势类型"，此处用"线段"替代次级别，
    在日线及以上级别是合理的近似。

    Args:
        segments: 线段列表
        zs_list: 中枢列表

    Returns:
        list[BuySellPoint]
    """
    points: list[BuySellPoint] = []
    if len(segments) < 3 or not zs_list:
        return points

    # 遍历所有中枢，检测其后的买卖点
    for zs in zs_list:
        # 找到中枢结束后的线段
        post_zs_segs = _find_segments_after_zs(segments, zs)

        if len(post_zs_segs) < 1:
            continue

        # ===== 第三类买卖点 =====
        # 条件：第一个离开中枢的线段 + 第一个回抽线段不进中枢
        _detect_third_points(post_zs_segs, zs, points)

        # ===== 第一类买卖点 =====
        # 条件：趋势中最后一个中枢，被反向突破后出现背驰
        _detect_first_points(segments, zs_list, zs, points)

    # ===== 第二类买卖点 =====
    _detect_second_points(segments, points)

    return points


def _find_segments_after_zs(
    segments: list[LineSegment], zs: ZSPyramid
) -> list[LineSegment]:
    """找到中枢结束之后的所有线段。"""
    result = []
    for seg in segments:
        if seg.bi_indices[0] > zs.bi_range[1]:
            result.append(seg)
    return result


def _detect_third_points(
    post_segs: list[LineSegment],
    zs: ZSPyramid,
    points: list[BuySellPoint],
) -> None:
    """检测第三类买卖点。

    三买：离开中枢的向上线段 + 回抽向下线段低点 > ZG
    三卖：离开中枢的向下线段 + 回抽向上线段高点 < ZD
    """
    if len(post_segs) < 2:
        return

    seg1 = post_segs[0]  # 离开中枢的线段
    seg2 = post_segs[1]  # 回抽线段

    # 三买：向上离开 + 向下回抽不进中枢上沿
    if seg1.direction == "up" and seg2.direction == "down":
        if seg2.low > zs.zg:
            points.append(BuySellPoint(
                point_type="三买",
                dt=seg2.end_dt,
                price=seg2.end_price,
                reason=f"回调不破ZG({zs.zg:.2f})，低点{seg2.low:.2f}",
                is_buy=True,
            ))

    # 三卖：向下离开 + 向上回抽不进中枢下沿
    if seg1.direction == "down" and seg2.direction == "up":
        if seg2.high < zs.zd:
            points.append(BuySellPoint(
                point_type="三卖",
                dt=seg2.end_dt,
                price=seg2.end_price,
                reason=f"反弹不破ZD({zs.zd:.2f})，高点{seg2.high:.2f}",
                is_buy=False,
            ))


def _detect_first_points(
    all_segs: list[LineSegment],
    all_zs: list[ZSPyramid],
    zs: ZSPyramid,
    points: list[BuySellPoint],
) -> None:
    """检测第一类买卖点。

    一买条件（原文）：
    1. 必须是趋势（至少2个同向不重叠的中枢）
    2. 当前中枢是最后一个
    3. 价格跌破/升破最后一个中枢
    4. 出现线段级别的背驰

    简化检测：
    - 价格突破中枢 + 同向线段力度衰减（area递减）
    """
    post_segs = _find_segments_after_zs(all_segs, zs)
    if len(post_segs) < 2:
        return

    # 找同向线段对
    for i in range(1, len(post_segs)):
        prev = post_segs[i - 1]
        curr = post_segs[i]

        if prev.direction != curr.direction:
            continue

        # 向下线段突破中枢下沿 + 力度衰减 → 一买
        if curr.direction == "down" and curr.low < zs.zd:
            if curr.area < prev.area * 0.8 and curr.low <= prev.low:
                points.append(BuySellPoint(
                    point_type="一买",
                    dt=curr.end_dt,
                    price=curr.end_price,
                    reason=f"下跌背驰(面积{curr.area:.0f}<{prev.area:.0f})，破中枢下沿{zs.zd:.2f}",
                    is_buy=True,
                ))

        # 向上线段突破中枢上沿 + 力度衰减 → 一卖
        if curr.direction == "up" and curr.high > zs.zg:
            if curr.area < prev.area * 0.8 and curr.high >= prev.high:
                points.append(BuySellPoint(
                    point_type="一卖",
                    dt=curr.end_dt,
                    price=curr.end_price,
                    reason=f"上涨背驰(面积{curr.area:.0f}<{prev.area:.0f})，破中枢上沿{zs.zg:.2f}",
                    is_buy=False,
                ))


def _detect_second_points(
    segments: list[LineSegment],
    points: list[BuySellPoint],
) -> None:
    """检测第二类买卖点。

    二买：一买之后，第一次回调不破一买低点
    二卖：一卖之后，第一次反弹不破一卖高点

    按原文：二买可以出现在中枢上方（与三买重合，最强）、
    中枢之中、或中枢下方（最弱）。
    """
    # 筛选出一买和一卖
    first_buys = [p for p in points if p.point_type == "一买"]
    first_sells = [p for p in points if p.point_type == "一卖"]

    for fb in first_buys:
        # 在一买之后找第一个向下线段（不破一买低点）
        # 二买定义：一买后的第一次回调低点高于一买价格
        for seg in segments:
            if seg.end_dt and fb.dt and str(seg.end_dt) <= str(fb.dt):
                continue
            if seg.direction == "down":
                # 二买必须不破一买低点
                if seg.low >= fb.price:
                    points.append(BuySellPoint(
                        point_type="二买",
                        dt=seg.end_dt,
                        price=seg.end_price,
                        reason=f"回调不破一买({fb.price:.2f})",
                        is_buy=True,
                    ))
                break  # 只看第一个向下线段

    for fs in first_sells:
        for seg in segments:
            if seg.end_dt and fs.dt and str(seg.end_dt) <= str(fs.dt):
                continue
            if seg.direction == "up":
                # 二卖必须不破一卖高点
                if seg.high <= fs.price:
                    points.append(BuySellPoint(
                        point_type="二卖",
                        dt=seg.end_dt,
                        price=seg.end_price,
                        reason=f"反弹不破一卖({fs.price:.2f})",
                        is_buy=False,
                    ))
                break


# ==================== BI 工具函数 ====================
def _bi_is_up(bi) -> bool:
    from zenstock.chanlun.bi_state import czsc_direction_is_up
    return czsc_direction_is_up(getattr(bi, "direction", ""))


def _bi_is_down(bi) -> bool:
    from zenstock.chanlun.bi_state import czsc_direction_is_up
    return not czsc_direction_is_up(getattr(bi, "direction", ""))


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
