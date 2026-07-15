"""示例策略：均线交叉。"""

from __future__ import annotations

import pandas as pd

from zenstock.strategy.base import BaseStrategy, Signal


class MACrossStrategy(BaseStrategy):
    """短期均线上穿长期均线买入，下穿卖出。

    参数:
        fast: 短期均线周期（默认 5 日）
        slow: 长期均线周期（默认 20 日）
    """

    params = (("fast", 5), ("slow", 20))

    def on_bar(self, i: int, df: pd.DataFrame) -> Signal:
        fast, slow = self.p.fast, self.p.slow
        if i < slow:  # 数据不足，等待
            return Signal.hold()

        close = df["close"]
        ma_fast = close.rolling(fast).mean()
        ma_slow = close.rolling(slow).mean()

        curr_fast = ma_fast.iloc[i]
        curr_slow = ma_slow.iloc[i]
        prev_fast = ma_fast.iloc[i - 1]
        prev_slow = ma_slow.iloc[i - 1]

        # 金叉：短均线从下方穿越长均线
        if curr_fast > curr_slow and prev_fast <= prev_slow:
            return Signal.buy(size=1.0, reason=f"金叉 MA{fast}>MA{slow}")

        # 死叉：短均线从上方跌破长均线
        if curr_fast < curr_slow and prev_fast >= prev_slow:
            return Signal.sell(size=1.0, reason=f"死叉 MA{fast}<MA{slow}")

        return Signal.hold()
