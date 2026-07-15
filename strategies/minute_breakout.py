"""示例策略：5 分钟放量突破。

买入：5 分钟 K 线收盘价突破近 N 根 K 线的最高价，且成交量放大
卖出：跌破近 N 根 K 线的最低价止损 OR 持仓超过 M 根 K 线强制平仓

适合分钟级数据使用。
"""

from __future__ import annotations

import pandas as pd

from zenstock.strategy.base import BaseStrategy, Signal


class MinuteBreakoutStrategy(BaseStrategy):
    """5 分钟级别的 Donchian 通道突破策略。

    参数:
        lookback: 通道回看周期（默认 16 根，即 80 分钟 = 1 小时 20 分）
        volume_mult: 突破时的成交量倍数阈值（默认 1.5）
        max_hold_bars: 最大持仓 K 线条数（默认 16 = 80 分钟）
    """

    params = (
        ("lookback", 16),
        ("volume_mult", 1.5),
        ("max_hold_bars", 16),
    )

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._entry_bar: int | None = None  # 记录买入 K 线索引

    def on_bar(self, i: int, df: pd.DataFrame) -> Signal:
        n = self.p.lookback
        if i < n + 1:
            return Signal.hold()

        high = df["high"]
        low = df["low"]
        close = df["close"]
        volume = df["volume"]

        # Donchian 通道：最近 n 根的最高/最低（不含当前）
        upper = high.iloc[i - n : i].max()
        lower = low.iloc[i - n : i].min()

        # 平均成交量（最近 n 根）
        avg_vol = volume.iloc[i - n : i].mean()
        curr_close = close.iloc[i]
        curr_vol = volume.iloc[i]

        # 已持仓：止损 OR 超时平仓
        if self._entry_bar is not None:
            held = i - self._entry_bar
            if curr_close < lower:
                self._entry_bar = None
                return Signal.sell(size=1.0, reason="跌破通道下轨")
            if held >= self.p.max_hold_bars:
                self._entry_bar = None
                return Signal.sell(size=1.0, reason=f"超时平仓 {held} 根")
            return Signal.hold()

        # 未持仓：突破上轨 + 放量 → 买入
        if curr_close > upper and curr_vol >= avg_vol * self.p.volume_mult:
            self._entry_bar = i
            return Signal.buy(
                size=1.0,
                reason=f"突破上轨 {upper:.2f}，量 {curr_vol/avg_vol:.1f}x",
            )

        return Signal.hold()