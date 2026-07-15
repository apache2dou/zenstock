"""测试指标计算（使用合成数据）。"""

from __future__ import annotations

import pandas as pd
import pytest

from zenstock.analytics import (
    compute_metrics,
    max_drawdown,
    profit_loss_ratio,
    sharpe_ratio,
    win_rate,
)
from zenstock.backtest.engine import BacktestResult, Trade


@pytest.fixture
def sample_equity_curve() -> pd.DataFrame:
    """合成资金曲线：先涨 10%，再跌 5%，再涨 3%。"""
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=6),
        "close": [10.0, 11.0, 10.5, 10.0, 10.8, 11.1],
        "total_equity": [100000, 110000, 104500, 100000, 108000, 111000],
    })


@pytest.fixture
def sample_result(sample_equity_curve) -> BacktestResult:
    """合成回测结果：2 笔配对交易（1 盈 1 亏）。"""
    trades = [
        # 交易1：盈利（10→12）
        Trade(date=pd.Timestamp("2024-01-01"), symbol="000001", action="BUY",
              price=10.0, shares=1000, amount=10000, cost=5),
        Trade(date=pd.Timestamp("2024-01-10"), symbol="000001", action="SELL",
              price=12.0, shares=1000, amount=12000, cost=17),
        # 交易2：亏损（12→11）
        Trade(date=pd.Timestamp("2024-01-15"), symbol="000001", action="BUY",
              price=12.0, shares=1000, amount=12000, cost=5),
        Trade(date=pd.Timestamp("2024-01-20"), symbol="000001", action="SELL",
              price=11.0, shares=1000, amount=11000, cost=16),
    ]
    return BacktestResult(
        trades=trades,
        equity_curve=sample_equity_curve,
        final_capital=111000,
        initial_capital=100000,
    )


class TestTradeMetrics:
    def test_win_rate(self, sample_result):
        wr = win_rate(sample_result)
        assert wr == pytest.approx(0.5, abs=0.01)

    def test_profit_loss_ratio(self, sample_result):
        # 盈利 2000，亏损 1000 → 2.0
        plr = profit_loss_ratio(sample_result)
        assert plr == pytest.approx(2.0, abs=0.1)

    def test_expectancy_positive(self, sample_result):
        m = compute_metrics(sample_result)
        # 胜率50%×盈2000 - 败率50%×亏1000 = 500
        assert m["expectancy"] == pytest.approx(500, abs=50)


class TestEquityMetrics:
    def test_max_drawdown(self, sample_equity_curve):
        mdd, peak, trough = max_drawdown(sample_equity_curve)
        # 峰值 110000，谷底 100000 → 回撤约 9.09%
        assert mdd == pytest.approx(9.09, abs=0.5)

    def test_sharpe_ratio_not_nan(self, sample_equity_curve):
        sr = sharpe_ratio(sample_equity_curve)
        assert pd.notna(sr)
        assert isinstance(sr, float)

    def test_compute_metrics_keys(self, sample_result):
        m = compute_metrics(sample_result)
        expected_keys = {
            "total_trades", "win_rate", "profit_loss_ratio",
            "max_drawdown_pct", "sharpe_ratio", "is_positive_expectancy",
        }
        assert expected_keys.issubset(m.keys())
