"""示例策略：布林带突破。

买入：收盘价突破上轨
卖出：收盘价跌破中轨（均线）
"""

from __future__ import annotations

import pandas as pd

from zenstock.strategy.base import BaseStrategy, Signal


class BollingerStrategy(BaseStrategy):
    """布林带突破策略。

    参数:
        period: 均线周期（默认 20 日）
        num_std: 标准差倍数（默认 2.0）
    """

    params = (("period", 20), ("num_std", 2.0))

    def on_bar(self, i: int, df: pd.DataFrame) -> Signal:
        n = self.p.period
        if i < n:
            return Signal.hold()

        close = df["close"]
        ma = close.rolling(n).mean()
        std = close.rolling(n).std()
        upper = ma + self.p.num_std * std
        lower = ma - self.p.num_std * std

        price = close.iloc[i]
        prev_price = close.iloc[i - 1]
        prev_upper = upper.iloc[i - 1]
        curr_mid = ma.iloc[i]

        # 突破上轨 → 买入
        if prev_price <= prev_upper and price > upper.iloc[i]:
            return Signal.buy(size=1.0, reason=f"突破布林上轨 {price:.2f}")

        # 跌破中轨 → 卖出
        if price < curr_mid and prev_price >= ma.iloc[i - 1]:
            return Signal.sell(size=1.0, reason=f"跌破布林中轨 {price:.2f}")

        return Signal.hold()
