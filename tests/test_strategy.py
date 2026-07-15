"""测试策略基类。"""

from __future__ import annotations

import pandas as pd

from zenstock.strategy.base import Action, BaseStrategy, Signal


class DummyStrategy(BaseStrategy):
    """简单测试策略：第 3 根 K 线买入，第 5 根卖出。"""
    params = (("buy_day", 3), ("sell_day", 5))

    def on_bar(self, i: int, df: pd.DataFrame) -> Signal:
        if i == self.p.buy_day:
            return Signal.buy(reason="test_buy")
        if i == self.p.sell_day:
            return Signal.sell(reason="test_sell")
        return Signal.hold()


class TestSignal:
    def test_hold_signal(self):
        s = Signal.hold()
        assert s.action == Action.HOLD
        assert s.size == 0.0

    def test_buy_signal(self):
        s = Signal.buy(size=0.5, reason="cross")
        assert s.action == Action.BUY
        assert s.size == 0.5
        assert s.reason == "cross"

    def test_sell_signal(self):
        s = Signal.sell()
        assert s.action == Action.SELL
        assert s.size == 1.0


class TestBaseStrategy:
    def test_params_defaults(self):
        s = DummyStrategy()
        assert s.p.buy_day == 3
        assert s.p.sell_day == 5

    def test_params_override(self):
        s = DummyStrategy(buy_day=10, sell_day=20)
        assert s.p.buy_day == 10
        assert s.p.sell_day == 20

    def test_on_bar_hold(self):
        s = DummyStrategy()
        df = pd.DataFrame({"close": [10, 11, 12]})
        sig = s.on_bar(0, df)
        assert sig.action == Action.HOLD

    def test_on_bar_buy(self):
        s = DummyStrategy()
        df = pd.DataFrame({"close": list(range(10))})
        sig = s.on_bar(3, df)
        assert sig.action == Action.BUY
        assert sig.reason == "test_buy"

    def test_repr(self):
        s = DummyStrategy()
        assert "DummyStrategy" in repr(s)
