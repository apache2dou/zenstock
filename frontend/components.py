"""前端公共：策略注册表与参数 UI 生成。"""

from __future__ import annotations

import importlib
from typing import Any

import streamlit as st

# 策略注册表：名称 → (模块路径, 类名, 显示名, 描述, 是否适合分钟级)
STRATEGY_REGISTRY: dict[str, dict[str, Any]] = {
    "ma_cross": {
        "module": "strategies.ma_cross",
        "class": "MACrossStrategy",
        "label": "📈 均线交叉",
        "desc": "短均线上穿长均线买入，下穿卖出",
        "minute_friendly": True,
    },
    "rsi": {
        "module": "strategies.rsi",
        "class": "RSIStrategy",
        "label": "📊 RSI 超买超卖",
        "desc": "RSI 超卖回升买入，超买回落卖出",
        "minute_friendly": True,
    },
    "bollinger": {
        "module": "strategies.bollinger",
        "class": "BollingerStrategy",
        "label": "📉 布林带突破",
        "desc": "突破上轨买入，跌破中轨卖出",
        "minute_friendly": True,
    },
    "minute_breakout": {
        "module": "strategies.minute_breakout",
        "class": "MinuteBreakoutStrategy",
        "label": "⚡ 5分钟放量突破",
        "desc": "5 分钟级别 Donchian 通道突破 + 成交量确认",
        "minute_friendly": True,
    },
    "chanlun": {
        "module": "strategies.chanlun_strategy",
        "class": "ChanlunStrategy",
        "label": "🔮 缠论买卖点",
        "desc": "基于 czsc 库的缠论一买/二买/三买信号（计算成本较高）",
        "minute_friendly": False,
    },
    "bi_state": {
        "module": "strategies.bi_state_strategy",
        "class": "BiStateStrategy",
        "label": "🔬 两重表里关系",
        "desc": "缠论第91-92课：笔状态机+概率统计+MA趋势过滤+MACD背驰+止损止盈",
        "minute_friendly": False,
    },
}

# 各策略的参数滑块定义: param_key → (label, min, max, default, step)
PARAM_DEFS: dict[str, dict[str, tuple[str, int | float, int | float, int | float, float]]] = {
    "ma_cross": {
        "fast": ("短期均线", 2, 60, 5, 1),
        "slow": ("长期均线", 5, 250, 20, 1),
    },
    "rsi": {
        "period": ("RSI周期", 2, 60, 14, 1),
        "oversold": ("超卖阈值", 5, 50, 30, 1),
        "overbought": ("超买阈值", 50, 95, 70, 1),
    },
    "bollinger": {
        "period": ("均线周期", 5, 100, 20, 1),
        "num_std": ("标准差倍数", 0.5, 4.0, 2.0, 0.1),
    },
    "minute_breakout": {
        "lookback": ("通道回看K线数", 5, 100, 16, 1),
        "volume_mult": ("放量倍数", 1.0, 3.0, 1.5, 0.1),
        "max_hold_bars": ("最大持仓K线数", 5, 200, 16, 1),
    },
    "chanlun": {
        "analyze_interval": ("分析间隔(K线)", 1, 20, 5, 1),
        "min_bi_for_signal": ("最少笔数", 3, 10, 5, 1),
        "stop_loss_pct": ("止损(%)", 2.0, 20.0, 5.0, 0.5),
        "take_profit_pct": ("止盈(%)", 5.0, 50.0, 15.0, 1.0),
        "max_hold_bars": ("最大持仓K线(0=不限)", 0, 500, 0, 10),
    },
    "bi_state": {
        "warmup_bars": ("预热期(K线)", 30, 250, 80, 10),
        "buy_threshold": ("买入概率阈值", 0.5, 0.9, 0.6, 0.05),
        "sell_threshold": ("卖出概率阈值", 0.5, 0.9, 0.6, 0.05),
        "ma_trend": ("趋势均线周期", 20, 120, 60, 5),
        "stop_loss_pct": ("止损(%)", 2.0, 15.0, 5.0, 0.5),
        "take_profit_pct": ("止盈(%)", 5.0, 50.0, 15.0, 1.0),
        "cooldown_bars": ("冷却期(K线)", 0, 20, 5, 1),
    },
}

# 寻优默认网格
DEFAULT_GRIDS: dict[str, dict[str, list[Any]]] = {
    "ma_cross": {"fast": [3, 5, 10, 15], "slow": [10, 20, 30, 60]},
    "rsi": {"period": [7, 14, 21], "oversold": [20, 25, 30], "overbought": [70, 75, 80]},
    "bollinger": {"period": [10, 20, 30], "num_std": [1.5, 2.0, 2.5]},
    "minute_breakout": {
        "lookback": [8, 16, 32, 48],
        "volume_mult": [1.2, 1.5, 2.0],
        "max_hold_bars": [16, 32, 60],
    },
}


def get_strategy_class(name: str):
    """动态加载策略类。"""
    info = STRATEGY_REGISTRY[name]
    mod = importlib.import_module(info["module"])
    return getattr(mod, info["class"])


def render_strategy_selector(key_prefix: str = "") -> str:
    """渲染策略选择下拉框，返回策略 key。

    Args:
        key_prefix: 用于区分不同 Tab 的 widget key 前缀
    """
    options = {k: v["label"] for k, v in STRATEGY_REGISTRY.items()}
    choice = st.selectbox(
        "选择策略",
        options=list(options.keys()),
        format_func=lambda k: options[k],
        key=f"{key_prefix}_strategy_select",
    )
    st.caption(STRATEGY_REGISTRY[choice]["desc"])
    return choice


def render_param_sliders(strategy_name: str, key_prefix: str = "") -> dict[str, Any]:
    """根据策略渲染参数滑块，返回参数字典。

    Args:
        strategy_name: 策略名称
        key_prefix: 用于区分不同 Tab 的 widget key 前缀
    """
    params: dict[str, Any] = {}
    defs = PARAM_DEFS.get(strategy_name, {})
    for key, (label, lo, hi, default, step) in defs.items():
        # Streamlit 要求 min/max/default/step 类型一致：
        # 如果 step 是 float，则全部转为 float
        if isinstance(step, float) or isinstance(default, float):
            lo_f, hi_f = float(lo), float(hi)
            default_f = float(default)
            params[key] = st.slider(
                label, lo_f, hi_f, default_f, step,
                key=f"{key_prefix}_{strategy_name}_{key}",
            )
        else:
            params[key] = st.slider(
                label, lo, hi, default, step,
                key=f"{key_prefix}_{strategy_name}_{key}",
            )
    return params
