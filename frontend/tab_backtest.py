"""Tab: 单标的回测。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.components import (
    get_strategy_class,
    render_param_sliders,
    render_strategy_selector,
)
from zenstock.analytics import compute_metrics
from zenstock.backtest import run_backtest
from zenstock.data.types import Freq
from zenstock.stock_names import get_stock_name


def render(data: pd.DataFrame, symbol: str, freq: Freq) -> None:
    """渲染单标的回测页。"""
    name = get_stock_name(symbol)
    display = f"{symbol} {name}" if name else symbol
    st.subheader(
        f"🎯 策略与参数 — {freq.display_name}（共 {len(data):,} 根K线）"
    )

    col_strat, col_run = st.columns([3, 1])
    with col_strat:
        strategy_name = render_strategy_selector(key_prefix="bt")
    with col_run:
        st.write("")  # 占位对齐
        st.write("")
        run_btn = st.button("🚀 运行回测", type="primary", width="stretch")

    params = render_param_sliders(strategy_name, key_prefix="bt")

    st.divider()

    # K线走势（专业行情图 - TradingView 内核）
    st.subheader(f"📉 {display} 价格走势")
    st.caption("💡 交互：滚轮缩放、拖动平移、十字光标自动跟随 | 右上角 ⛶ 按钮全屏")
    from frontend.lwc_chart import render_kline_chart
    render_kline_chart(
        data=data, symbol=display,
        title=f"{display} {freq.display_name}",
        height=550,
    )

    if not run_btn:
        st.info("👆 点击「运行回测」查看结果")
        return

    # 执行回测
    with st.spinner("回测中..."):
        cls = get_strategy_class(strategy_name)
        strategy = cls(**params)
        result = run_backtest(strategy, data, symbol=symbol)

    metrics = compute_metrics(result)

    # 核心指标卡片
    flag = "✅" if metrics["is_positive_expectancy"] else "⚠️"
    cols = st.columns(6)
    cols[0].metric("胜率", f"{metrics['win_rate']:.1f}%")
    cols[1].metric("赔率", f"{metrics['profit_loss_ratio']:.2f}")
    cols[2].metric("总收益", f"{metrics['total_return_pct']:.1f}%")
    cols[3].metric("年化", f"{metrics['annual_return_pct']:.1f}%")
    cols[4].metric("最大回撤", f"{metrics['max_drawdown_pct']:.1f}%")
    cols[5].metric("夏普", f"{metrics['sharpe_ratio']:.2f}")

    st.markdown(f"### {flag} 策略评估：{'正期望' if metrics['is_positive_expectancy'] else '负期望'}")

    left, right = st.columns(2)

    with left:
        st.subheader("💰 资金曲线")
        eq = result.equity_curve.set_index("date")
        st.line_chart(eq[["total_equity", "market_value"]], width="stretch")

    with right:
        st.subheader("📋 交易明细")
        if result.trades:
            trades_df = pd.DataFrame(
                [
                    {
                        "日期": t.date,
                        "动作": t.action,
                        "价格": f"{t.price:.2f}",
                        "数量": int(t.shares),
                        "金额": f"{t.amount:,.0f}",
                        "成本": f"{t.cost:.2f}",
                        "原因": t.reason,
                    }
                    for t in result.trades
                ]
            )
            st.dataframe(trades_df, width="stretch", hide_index=True)
        else:
            st.warning("无交易记录")

    # 完整指标
    with st.expander("📊 全部指标"):
        # 格式化展示
        nice = {}
        for k, v in metrics.items():
            if isinstance(v, float):
                nice[k] = round(v, 4)
            else:
                nice[k] = v
        st.json(nice)
