"""示例策略：RSI 超买超卖反转。

买入：RSI < oversold（超卖回升）
卖出：RSI > overbought（超买回落）
"""

from __future__ import annotations

import pandas as pd

from zenstock.strategy.base import BaseStrategy, Signal


class RSIStrategy(BaseStrategy):
    """RSI 均值回归策略。

    参数:
        period: RSI 计算周期（默认 14 日）
        oversold: 超卖阈值（默认 30）
        overbought: 超买阈值（默认 70）
    """

    params = (("period", 14), ("oversold", 30), ("overbought", 70))

    def on_bar(self, i: int, df: pd.DataFrame) -> Signal:
        n = self.p.period
        if i < n + 1:  # 需要足够数据
            return Signal.hold()

        rsi = self._calc_rsi(df["close"], n)
        curr = rsi.iloc[i]
        prev = rsi.iloc[i - 1]

        # 超卖区上穿 oversold 线 → 买入
        if prev < self.p.oversold and curr >= self.p.oversold:
            return Signal.buy(size=1.0, reason=f"RSI超卖回升 {curr:.1f}")

        # 超买区下穿 overbought 线 → 卖出
        if prev > self.p.overbought and curr <= self.p.overbought:
            return Signal.sell(size=1.0, reason=f"RSI超买回落 {curr:.1f}")

        return Signal.hold()

    @staticmethod
    def _calc_rsi(close: pd.Series, period: int) -> pd.Series:
        """计算 RSI 指标。"""
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = -delta.clip(upper=0).rolling(period).mean()
        rs = gain / loss.replace(0, 1e-10)
        return 100 - (100 / (1 + rs))
