"""Tab: 多股票对比。"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.components import (
    get_strategy_class,
    render_strategy_selector,
)
from zenstock.analytics import compute_metrics
from zenstock.backtest import run_backtest
from zenstock.data import DataStorage
from zenstock.data.types import Freq


@st.cache_data(ttl=60)
def load_multi_data(
    symbols: list[str], freq: str, start: str, end: str
) -> dict[str, pd.DataFrame]:
    """批量加载多只股票数据。"""
    storage = DataStorage()
    result = {}
    for sym in symbols:
        df = storage.read_klines(sym, Freq(freq), start, end)
        if not df.empty:
            result[sym] = df
    return result


def render(all_symbols: list[str], freq_value: str, start, end) -> None:
    """渲染多股票对比页。"""
    freq = Freq(freq_value)
    st.subheader(f"📊 多股票对比回测 — {freq.display_name}")

    # 股票选择
    from zenstock.stock_names import get_stock_label
    selected = st.multiselect(
        "选择要对比的股票（最多 10 只）",
        all_symbols,
        default=all_symbols[:3] if len(all_symbols) >= 3 else all_symbols,
        format_func=get_stock_label,
        max_selections=10,
        key=f"cmp_symbols_{freq_value}",
    )

    if not selected:
        st.warning("请至少选择一只股票")
        return

    # 策略选择 + 参数
    strategy_name = render_strategy_selector(key_prefix="cmp")
    st.markdown("**策略参数**")
    params = _render_simple_params(strategy_name, key_prefix="cmp")

    run_btn = st.button("🚀 批量回测", type="primary")

    if not run_btn:
        st.info("👆 点击「批量回测」对比多只股票")
        return

    # 加载数据
    datasets = load_multi_data(selected, freq_value, str(start), str(end))
    if not datasets:
        st.error("所选股票均无数据")
        return

    # 执行批量回测
    progress = st.progress(0.0, "批量回测中...")
    cls = get_strategy_class(strategy_name)
    rows = []
    equity_curves: dict[str, pd.Series] = {}

    for idx, (sym, df) in enumerate(datasets.items()):
        try:
            strategy = cls(**params)
            result = run_backtest(strategy, df, symbol=sym)
            metrics = compute_metrics(result)
            rows.append({"symbol": sym, **metrics})
            # 归一化资金曲线为百分比收益
            eq = result.equity_curve.set_index("date")["total_equity"]
            equity_curves[sym] = (eq / eq.iloc[0] - 1) * 100
        except Exception:  # noqa: BLE001
            pass
        progress.progress((idx + 1) / len(datasets))
    progress.empty()

    if not rows:
        st.error("全部回测失败")
        return

    summary = pd.DataFrame(rows)

    # ===== 对比表格 =====
    st.subheader("📋 对比汇总")
    display = summary[[
        "symbol", "win_rate", "profit_loss_ratio", "total_return_pct",
        "annual_return_pct", "max_drawdown_pct", "sharpe_ratio", "total_trades",
        "is_positive_expectancy",
    ]].copy()
    display.columns = [
        "股票", "胜率%", "赔率", "总收益%", "年化%", "回撤%",
        "夏普", "交易数", "评估",
    ]
    display["评估"] = display["评估"].map(lambda x: "✅" if x else "⚠️")
    # 数值格式化
    for c in ["胜率%", "赔率", "总收益%", "年化%", "回撤%", "夏普"]:
        display[c] = display[c].map(lambda v: f"{v:.2f}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    # ===== 收益曲线对比 =====
    st.subheader("📈 累计收益对比（%）")
    eq_df = pd.DataFrame(equity_curves)
    fig = px.line(
        eq_df, x=eq_df.index, y=eq_df.columns,
        title="累计收益率对比", labels={"value": "累计收益%", "date": "日期"},
    )
    fig.update_layout(height=450, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # ===== 雷达图对比（前 4 名）=====
    st.subheader("🕸️ 雷达图对比（前 4 名）")
    top4 = summary.nlargest(4, "sharpe_ratio")
    if len(top4) >= 2:
        _render_radar(top4)
    else:
        st.info("至少需要 2 只股票才能生成雷达图")

    # ===== 条形图 =====
    left, right = st.columns(2)
    with left:
        st.markdown("**总收益率%**")
        fig_bar = px.bar(
            summary.sort_values("total_return_pct"),
            x="total_return_pct", y="symbol", orientation="h",
            color="is_positive_expectancy",
            color_discrete_map={True: "#2ecc71", False: "#e74c3c"},
        )
        fig_bar.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with right:
        st.markdown("**夏普比率**")
        fig_bar2 = px.bar(
            summary.sort_values("sharpe_ratio"),
            x="sharpe_ratio", y="symbol", orientation="h",
            color="is_positive_expectancy",
            color_discrete_map={True: "#2ecc71", False: "#e74c3c"},
        )
        fig_bar2.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_bar2, use_container_width=True)


def _render_simple_params(strategy_name: str, key_prefix: str = "") -> dict:
    """渲染简洁的参数输入（数字输入框）。"""
    from frontend.components import PARAM_DEFS

    params = {}
    defs = PARAM_DEFS.get(strategy_name, {})
    cols = st.columns(len(defs))
    for idx, (key, (label, lo, hi, default, step)) in enumerate(defs.items()):
        params[key] = cols[idx].number_input(
            label, lo, hi, default, step=step,
            key=f"{key_prefix}_{strategy_name}_{key}",
        )
    return params


def _render_radar(summary: pd.DataFrame) -> None:
    """渲染雷达图。"""
    # 归一化 5 个维度到 [0, 1]
    dims = {
        "胜率": "win_rate",
        "收益": "total_return_pct",
        "夏普": "sharpe_ratio",
        "赔率": "profit_loss_ratio",
    }
    # 最大回撤取反（回撤越小越好）
    mdd_inv = -summary["max_drawdown_pct"]

    radar_df = pd.DataFrame()
    for label, col in dims.items():
        vals = summary[col]
        vmin, vmax = vals.min(), vals.max()
        radar_df[label] = (vals - vmin) / (vmax - vmin + 1e-9)

    vmin_mdd, vmax_mdd = mdd_inv.min(), mdd_inv.max()
    radar_df["回撤控制"] = (mdd_inv - vmin_mdd) / (vmax - vmin_mdd + 1e-9)
    radar_df["股票"] = summary["symbol"].values

    fig = px.line_polar(
        radar_df.melt(id_vars="股票", var_name="维度", value_name="得分"),
        r="得分", theta="维度", color="股票", line_close=True,
        title="策略表现雷达图",
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)
