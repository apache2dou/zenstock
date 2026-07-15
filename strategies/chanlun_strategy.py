"""缠论买卖点策略（基于线段+中枢的完整缠论层级）。

分析链路：K线 → czsc(分型→笔) → 线段 → 中枢 → 买卖点

策略逻辑：
    买入条件：
      - 一买：向下线段突破中枢下沿后，线段背驰（面积递减）
      - 三买：向上线段突破中枢上沿后回调不进中枢

    卖出条件（按优先级）：
      - 止损/止盈/超时
      - 一卖：向上线段突破中枢上沿后，线段背驰
      - 三卖：向下线段跌破中枢下沿后反弹不进中枢

性能优化：
    - czsc 分析按间隔执行
    - 线段和中枢从 czsc 的笔自动构建
    - 分析结果缓存
"""

from __future__ import annotations

import pandas as pd

from zenstock.chanlun.analyzer import ChanlunAnalyzer, ChanlunResult
from zenstock.chanlun.segments import (
    detect_buy_sell_points,
    extract_line_segments,
    extract_zhongshu_from_segments,
)
from zenstock.data.types import Freq
from zenstock.logger import get_logger
from zenstock.strategy.base import BaseStrategy, Signal

log = get_logger(__name__)


class ChanlunStrategy(BaseStrategy):
    """缠论买卖点策略（基于线段+中枢）。

    参数:
        analyze_interval: 每隔多少根 K 线做一次完整缠论分析（默认 5）
        min_bi_for_signal: 信号识别所需的最小笔数量（默认 5）
        freq_hint: 数据频率提示
        stop_loss_pct: 止损百分比（默认 5%）
        take_profit_pct: 止盈百分比（默认 15%）
        max_hold_bars: 最大持仓 K 线数（默认 0=不限）
    """

    params = (
        ("analyze_interval", 5),
        ("min_bi_for_signal", 5),
        ("freq_hint", "D"),
        ("stop_loss_pct", 5.0),
        ("take_profit_pct", 15.0),
        ("max_hold_bars", 0),
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._analyzer = ChanlunAnalyzer()
        self._last_result: ChanlunResult | None = None
        self._last_buy_points: list = []
        self._last_sell_points: list = []
        self._entry_price: float = 0.0
        self._entry_bar: int = 0
        self._has_position: bool = False

    def on_bar(self, i: int, df: pd.DataFrame) -> Signal:
        if i < 30:
            return Signal.hold()

        current_close = float(df["close"].iloc[i])

        # 持仓中 → 先查卖出
        if self._has_position:
            sell_signal = self._check_exit(i, current_close)
            if sell_signal is not None:
                self._has_position = False
                self._entry_price = 0.0
                return sell_signal

        # 缠论分析（按间隔）
        need_analyze = (
            self._last_result is None
            or i % self.p.analyze_interval == 0
        )

        if need_analyze:
            sub_df = df.iloc[: i + 1]
            try:
                freq_hint = self._parse_freq(self.p.freq_hint)
                self._last_result = self._analyzer.analyze_single(
                    sub_df, freq_hint, symbol=str(sub_df.get("symbol", [""])[0])
                )
                # 从笔构建线段和中枢，再识别买卖点
                if self._last_result and self._last_result.bi_count >= 3:
                    segments = extract_line_segments(self._last_result.bi_list)
                    zs_list = extract_zhongshu_from_segments(
                        segments, self._last_result.bi_list
                    )
                    self._last_buy_points = [
                        p for p in detect_buy_sell_points(segments, zs_list) if p.is_buy
                    ]
                    self._last_sell_points = [
                        p for p in detect_buy_sell_points(segments, zs_list) if not p.is_buy
                    ]
                else:
                    self._last_buy_points = []
                    self._last_sell_points = []
            except Exception as e:  # noqa: BLE001
                log.debug(f"缠论分析失败 i={i}: {e}")
                return Signal.hold()

        result = self._last_result
        if result is None or result.bi_count < self.p.min_bi_for_signal:
            return Signal.hold()

        # 未持仓 → 查买入
        if not self._has_position and self._last_buy_points:
            bp = self._last_buy_points[-1]
            self._has_position = True
            self._entry_price = current_close
            self._entry_bar = i
            return Signal.buy(size=1.0, reason=f"缠论{bp.point_type}: {bp.reason}")

        return Signal.hold()

    # ==================== 卖出逻辑 ====================
    def _check_exit(self, i: int, current_price: float) -> Signal | None:
        """检查卖出条件（风控优先）。"""
        if not self._has_position or self._entry_price <= 0:
            return None

        pnl_pct = (current_price - self._entry_price) / self._entry_price * 100

        # 止损
        if pnl_pct <= -abs(self.p.stop_loss_pct):
            return Signal.sell(size=1.0, reason=f"止损 {pnl_pct:.1f}%")

        # 止盈
        if pnl_pct >= self.p.take_profit_pct:
            return Signal.sell(size=1.0, reason=f"止盈 +{pnl_pct:.1f}%")

        # 超时
        if self.p.max_hold_bars > 0:
            held = i - self._entry_bar
            if held >= self.p.max_hold_bars:
                return Signal.sell(size=1.0, reason=f"超时 {held}根 盈亏{pnl_pct:.1f}%")

        # 缠论卖点
        if self._last_sell_points:
            sp = self._last_sell_points[-1]
            return Signal.sell(size=1.0, reason=f"缠论{sp.point_type}: {sp.reason}")

        return None

    # ==================== 工具 ====================
    @staticmethod
    def _parse_freq(freq_str: str) -> Freq:
        try:
            return Freq(freq_str)
        except (ValueError, KeyError):
            return Freq.DAILY
