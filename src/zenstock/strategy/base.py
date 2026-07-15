"""策略基类与信号定义。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd


class Action(str, Enum):
    """交易动作。"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """单根 K 线产生的交易信号。"""
    action: Action
    size: float = 0.0      # 目标仓位比例 [0, 1]，如 0.5 = 半仓
    price: float | None = None   # 指定价格（默认用收盘价）
    reason: str = ""        # 策略备注，便于复盘

    @classmethod
    def hold(cls) -> Signal:
        return cls(action=Action.HOLD)

    @classmethod
    def buy(cls, size: float = 1.0, reason: str = "") -> Signal:
        return cls(action=Action.BUY, size=size, reason=reason)

    @classmethod
    def sell(cls, size: float = 1.0, reason: str = "") -> Signal:
        return cls(action=Action.SELL, size=size, reason=reason)


class BaseStrategy:
    """策略抽象基类。

    子类需要实现 :meth:`on_bar` 方法，在每根 K 线上产生信号。

    简单示例::

        class MyStrategy(BaseStrategy):
            params = (("n", 20),)

            def on_bar(self, i: int, df: pd.DataFrame) -> Signal:
                if i < self.p.n:
                    return Signal.hold()
                ma = df["close"].iloc[: i + 1].rolling(self.p.n).mean().iloc[-1]
                if df["close"].iloc[i] > ma:
                    return Signal.buy()
                return Signal.sell()
    """

    #: 参数默认值，子类可覆盖。元组形式可被 kwargs 覆盖。
    params: tuple[tuple[str, Any], ...] = ()

    def __init__(self, **kwargs: Any) -> None:
        self.p = _Params(self.params, kwargs)

    def on_bar(self, i: int, df: pd.DataFrame) -> Signal:  # noqa: ARG002
        """在 index=i 的 K 线上产生信号。

        Args:
            i: 当前 K 线在 df 中的索引
            df: 完整 K 线数据（包含到 i 为止的所有数据）

        Returns:
            Signal 对象
        """
        return Signal.hold()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({dict(self.params)})"


class _Params:
    """把 params 元组转成属性访问，例如 self.p.n。"""

    def __init__(
        self, defaults: tuple[tuple[str, Any], ...], overrides: dict[str, Any]
    ) -> None:
        for key, val in defaults:
            setattr(self, key, overrides.get(key, val))
