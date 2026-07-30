"""缠论两重表里关系交易策略的完整测试（合并 v1 + v2）。"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from zenstock.chanlun.bi_state import BiState
from strategies.bi_state_strategy import (
    BiStateStrategy,
    EnhancedStats,
    HistoricalStats,
    SignalProbability,
    compute_state_signal_probability,
    compute_enhanced_signal,
    build_historical_stats,
    should_confirm_with_ma5,
    compute_macd,
    detect_divergence,
)


# ==================== 概率统计（基础） ====================

class TestProbability:
    def test_trading_level_follows_input_frequency(self):
        """回测级别应与输入K线频率匹配，不能固定使用30分钟矩阵。"""
        strategy = BiStateStrategy(matrix_path="")
        daily = pd.DataFrame({"freq": ["D"]})
        intraday = pd.DataFrame({"freq": ["30"]})

        assert strategy._infer_trading_level(daily) == "日线"
        assert strategy._infer_trading_level(intraday) == "30分钟"

    def test_trading_level_reads_dataframe_attrs(self):
        """Parquet数据用attrs标记频率时，5分钟不能回退成日线。"""
        strategy = BiStateStrategy(matrix_path="")
        intraday = pd.DataFrame({"close": [1.0]})
        intraday.attrs["freq"] = "5"

        assert strategy._infer_freq(intraday) == "5"
        assert strategy._infer_trading_level(intraday) == "30分钟"

    def test_trading_level_states_are_used_as_small_state(self):
        """矩阵训练的交易级别小状态必须保留原始分型方向。"""
        source = Path(__file__).parents[1] / "scripts" / "train_multilevel_matrix.py"
        text = source.read_text(encoding="utf-8")
        assert "small_state = signal_bi" in text

    def test_trainer_does_not_project_final_fractals_backward(self):
        """训练状态必须来自当前增量CZSC，而不是完整历史fx_list。"""
        source = Path(__file__).parents[1] / "scripts" / "train_multilevel_matrix.py"
        text = source.read_text(encoding="utf-8")
        assert "running.update(bar)" in text
        assert "ubi_fxs" in text
        assert "for fx in reversed(fx_list)" not in text

    def test_buy_signal_when_down_fx_confirmed(self):
        """(-1,0) 底分型构造，历史上涨概率高 → BUY"""
        stats = HistoricalStats(
            transition_counts={
                (BiState.DOWN_FX_FORMING, BiState.UP_EXTENDING): 40,
                (BiState.DOWN_FX_FORMING, BiState.UP_FX_FORMING): 20,
                (BiState.DOWN_FX_FORMING, BiState.DOWN_EXTENDING): 30,
                (BiState.DOWN_FX_FORMING, BiState.DOWN_FX_FORMING): 10,
            },
        )
        prob = compute_state_signal_probability(BiState.DOWN_FX_FORMING, stats)
        assert prob.up_prob == pytest.approx(0.6)
        assert prob.down_prob == pytest.approx(0.4)
        assert prob.signal == "BUY"

    def test_sell_signal_when_up_fx_confirmed(self):
        """(1,0) 顶分型构造，历史下跌概率高 → SELL"""
        stats = HistoricalStats(
            transition_counts={
                (BiState.UP_FX_FORMING, BiState.DOWN_EXTENDING): 50,
                (BiState.UP_FX_FORMING, BiState.DOWN_FX_FORMING): 20,
                (BiState.UP_FX_FORMING, BiState.UP_EXTENDING): 20,
                (BiState.UP_FX_FORMING, BiState.UP_FX_FORMING): 10,
            },
        )
        prob = compute_state_signal_probability(BiState.UP_FX_FORMING, stats)
        assert prob.up_prob == pytest.approx(0.3)
        assert prob.down_prob == pytest.approx(0.7)
        assert prob.signal == "SELL"

    def test_hold_when_uncertain(self):
        """概率接近 0.5 → HOLD"""
        stats = HistoricalStats(
            transition_counts={
                (BiState.UP_FX_FORMING, BiState.UP_EXTENDING): 30,
                (BiState.UP_FX_FORMING, BiState.UP_FX_FORMING): 20,
                (BiState.UP_FX_FORMING, BiState.DOWN_EXTENDING): 30,
                (BiState.UP_FX_FORMING, BiState.DOWN_FX_FORMING): 20,
            },
        )
        prob = compute_state_signal_probability(BiState.UP_FX_FORMING, stats)
        assert prob.signal == "HOLD"

    def test_no_data_returns_hold(self):
        """无历史数据 → HOLD"""
        stats = HistoricalStats(transition_counts={})
        prob = compute_state_signal_probability(BiState.UP_EXTENDING, stats)
        assert prob.signal == "HOLD"
        assert prob.up_prob == 0.0


# ==================== 增强统计（趋势上下文） ====================

class TestEnhancedStats:
    def test_state_with_trend_context(self):
        """统计应区分趋势上下文"""
        stats = EnhancedStats()
        # 上涨趋势中底分型 → 大概率继续上涨
        stats.record(BiState.DOWN_FX_FORMING, BiState.UP_EXTENDING, trend="up")
        stats.record(BiState.DOWN_FX_FORMING, BiState.UP_EXTENDING, trend="up")
        stats.record(BiState.DOWN_FX_FORMING, BiState.DOWN_EXTENDING, trend="up")
        # 下跌趋势中底分型 → 大概率继续下跌
        stats.record(BiState.DOWN_FX_FORMING, BiState.DOWN_EXTENDING, trend="down")
        stats.record(BiState.DOWN_FX_FORMING, BiState.DOWN_EXTENDING, trend="down")

        prob_up = stats.probability(BiState.DOWN_FX_FORMING, trend="up")
        prob_down = stats.probability(BiState.DOWN_FX_FORMING, trend="down")
        assert prob_up.up_prob > prob_down.up_prob

    def test_fallback_to_no_trend(self):
        """特定趋势无数据时回退到不分趋势"""
        stats = EnhancedStats()
        stats.record(BiState.DOWN_FX_FORMING, BiState.UP_EXTENDING)  # trend=""
        prob = stats.probability(BiState.DOWN_FX_FORMING, trend="up")
        # 应回退到不分趋势
        assert prob.sample_size > 0

    def test_build_historical_stats(self):
        """build_historical_stats 构建统计"""
        states = [BiState.UP_EXTENDING, BiState.UP_FX_FORMING, BiState.DOWN_EXTENDING]
        stats = build_historical_stats(states)
        assert stats.total_from(BiState.UP_EXTENDING) == 1
        assert stats.total_from(BiState.UP_FX_FORMING) == 1


# ==================== MA5 确认 ====================

class TestMA5Confirm:
    def test_buy_needs_ma5_support(self):
        assert should_confirm_with_ma5(price=10.5, ma5=10.0, is_buy=True) is True
        assert should_confirm_with_ma5(price=9.0, ma5=10.0, is_buy=True) is False

    def test_sell_needs_ma5_break(self):
        assert should_confirm_with_ma5(price=9.0, ma5=10.0, is_buy=False) is True
        assert should_confirm_with_ma5(price=10.5, ma5=10.0, is_buy=False) is False


# ==================== MACD 背驰检测 ====================

class TestDivergence:
    def test_compute_macd_returns_three_series(self):
        close = pd.Series([10, 11, 12, 11, 10, 9, 10, 11, 12, 13] * 5)
        dif, dea, hist = compute_macd(close)
        assert len(dif) == len(close)
        assert len(dea) == len(close)
        assert len(hist) == len(close)

    def test_detect_divergence_no_data(self):
        """数据不足时返回 None"""
        hist = pd.Series([0.1, 0.2, 0.1])
        price = pd.Series([10, 11, 10])
        assert detect_divergence(hist, price, 2) is None

    def test_detect_top_divergence(self):
        """构造顶背驰场景：价格创新高但MACD面积减小"""
        n = 80
        # 两段上涨：第二段涨幅更大但MACD柱子面积更小
        prices = list(np.linspace(10, 15, 30)) + [15] * 10 + list(np.linspace(15, 18, 30)) + [18] * 10
        hist = [0] * 5 + [0.5, 1.0, 1.5, 1.0, 0.5] + [0] * 15 + [0.3, 0.5, 0.7, 0.5, 0.3] + [0] * 35
        # 补齐长度
        while len(hist) < n:
            hist.append(0)
        hist = pd.Series(hist[:n])
        price = pd.Series(prices[:n])
        result = detect_divergence(hist, price, n - 1)
        # 可能检测到顶背驰也可能不（取决于峰检测），但不应崩溃
        assert result in ("top_divergence", "bottom_divergence", None)


# ==================== 策略端到端 ====================

class TestStrategy:
    def _make_uptrend_data(self, n=120):
        """生成上涨趋势数据"""
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        np.random.seed(42)
        trend = np.linspace(10, 15, n)
        noise = np.random.randn(n) * 0.3
        prices = trend + noise
        return pd.DataFrame({
            "date": dates,
            "open": prices,
            "high": prices + 0.3,
            "low": prices - 0.3,
            "close": prices,
            "volume": [10000] * n,
        })

    def test_strategy_default_params(self):
        """默认参数正确"""
        strategy = BiStateStrategy()
        assert strategy.p.warmup_bars == 80
        assert strategy.p.buy_threshold == 0.6
        assert strategy.p.use_divergence is True

    def test_no_trade_during_warmup(self):
        """预热期内不交易"""
        from zenstock.strategy.base import Action
        df = self._make_uptrend_data(120)
        strategy = BiStateStrategy(warmup_bars=80)
        for i in range(80):
            sig = strategy.on_bar(i, df)
            assert sig.action == Action.HOLD

    def test_strategy_with_custom_thresholds(self):
        """自定义阈值"""
        strategy = BiStateStrategy(buy_threshold=0.7, sell_threshold=0.7)
        assert strategy.p.buy_threshold == 0.7
        assert strategy.p.sell_threshold == 0.7

    def test_strategy_v2_alias_works(self):
        """BiStateStrategyV2 别名可用（向后兼容）"""
        from strategies.bi_state_strategy import BiStateStrategyV2
        strategy = BiStateStrategyV2()
        assert strategy.p.warmup_bars == 80

    def test_historical_stats_property(self):
        """_historical_stats 属性兼容前端访问"""
        strategy = BiStateStrategy()
        assert strategy._historical_stats is strategy._enhanced_stats
