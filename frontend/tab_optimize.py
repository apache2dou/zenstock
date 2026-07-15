"""Tab: 参数寻优（含热力图可视化）。"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from frontend.components import (
    DEFAULT_GRIDS,
    PARAM_DEFS,
    STRATEGY_REGISTRY,
    get_strategy_class,
    render_strategy_selector,
)
from zenstock.analytics import compute_metrics
from zenstock.backtest import run_backtest
from zenstock.data.types import Freq


def render(data: pd.DataFrame, symbol: str, freq: Freq) -> None:
    """渲染参数寻优页。"""
    st.subheader(
        f"🔍 参数网格寻优 — {freq.display_name}（共 {len(data):,} 根K线）"
    )

    col_strat, col_sort = st.columns(2)
    with col_strat:
        strategy_name = render_strategy_selector(key_prefix="opt")
    with col_sort:
        sort_by = st.selectbox(
            "排序指标",
            ["sharpe_ratio", "total_return_pct", "annual_return_pct",
             "win_rate", "profit_loss_ratio", "calmar_ratio"],
            format_func=lambda x: {
                "sharpe_ratio": "夏普比率",
                "total_return_pct": "总收益率",
                "annual_return_pct": "年化收益率",
                "win_rate": "胜率",
                "profit_loss_ratio": "赔率",
                "calmar_ratio": "卡玛比率",
            }[x],
            key="opt_sort_by",
        )

    # 参数网格编辑
    st.markdown("**参数网格**（用逗号分隔多个值）")
    defaults = DEFAULT_GRIDS.get(strategy_name, {})
    param_defs = PARAM_DEFS.get(strategy_name, {})

    cols = st.columns(len(defaults))
    grid: dict[str, list] = {}
    for idx, (key, vals) in enumerate(defaults.items()):
        label = param_defs.get(key, (key,))[0] if key in param_defs else key
        default_str = ", ".join(str(v) for v in vals)
        raw = cols[idx].text_input(label, value=default_str, key=f"opt_grid_{key}")
        parsed: list = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                parsed.append(int(part))
            except ValueError:
                try:
                    parsed.append(float(part))
                except ValueError:
                    parsed.append(part)
        grid[key] = parsed

    total_combos = int(np.prod([len(v) for v in grid.values()])) if grid else 0
    st.caption(f"共 {total_combos} 个参数组合")
    run_btn = st.button("🚀 开始寻优", type="primary")

    if not run_btn or total_combos == 0:
        st.info("👆 设置参数范围后点击「开始寻优」")
        return

    # 执行网格搜索
    progress = st.progress(0.0, text="寻优中...")
    cls = get_strategy_class(strategy_name)

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    results: list[dict] = []

    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        try:
            strategy = cls(**params)
            result = run_backtest(strategy, data, symbol=symbol)
            metrics = compute_metrics(result)
            results.append({**params, **metrics})
        except Exception:  # noqa: BLE001
            pass
        progress.progress((idx + 1) / len(combos))

    progress.empty()

    if not results:
        st.error("所有组合均失败，请检查参数范围")
        return

    df = pd.DataFrame(results)
    df = df[df["total_trades"] > 0].reset_index(drop=True)
    df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)

    # ===== Top 结果表 =====
    st.subheader(f"🏆 Top 10（按 {sort_by} 排序）")
    display_cols = keys + [
        "win_rate", "profit_loss_ratio", "total_return_pct",
        "annual_return_pct", "max_drawdown_pct", "sharpe_ratio", "total_trades",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    top_df = df[display_cols].head(10).copy()

    # 格式化列名
    rename = {
        "win_rate": "胜率%", "profit_loss_ratio": "赔率",
        "total_return_pct": "总收益%", "annual_return_pct": "年化%",
        "max_drawdown_pct": "回撤%", "sharpe_ratio": "夏普",
        "total_trades": "交易数",
    }
    top_df = top_df.rename(columns={k: v for k, v in rename.items() if k in top_df.columns})
    st.dataframe(top_df, use_container_width=True, hide_index=True)

    # ===== 热力图（仅当 2 个参数时）=====
    if len(keys) == 2:
        st.subheader("🌡️ 参数热力图")
        metric_options = {
            "夏普比率": "sharpe_ratio",
            "总收益率%": "total_return_pct",
            "胜率%": "win_rate",
            "回撤%": "max_drawdown_pct",
        }
        heat_metric = st.selectbox(
            "热力图指标", list(metric_options.keys()),
            key="heatmap_metric",
        )
        heat_col = metric_options[heat_metric]

        pivot = df.pivot_table(
            index=keys[0], columns=keys[1], values=heat_col, aggfunc="first",
        )
        fig = px.imshow(
            pivot,
            labels=dict(color=heat_metric),
            title=f"{heat_metric} 热力图",
            color_continuous_scale="RdYlGn",
            aspect="auto",
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # ===== 3D 曲面图 =====
    if len(keys) == 2:
        with st.expander("🌐 3D 参数曲面"):
            pivot3d = df.pivot_table(
                index=keys[0], columns=keys[1], values=sort_by, aggfunc="first",
            )
            x = pivot3d.columns.tolist()
            y = pivot3d.index.tolist()
            z = pivot3d.values
            fig3d = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
            fig3d.update_layout(
                title=f"{sort_by} 3D 曲面",
                scene=dict(
                    xaxis_title=keys[1], yaxis_title=keys[0], zaxis_title=sort_by,
                ),
                height=500,
            )
            st.plotly_chart(fig3d, use_container_width=True)

    # ===== 完整结果下载 =====
    st.divider()
    st.subheader("💾 完整结果")
    st.dataframe(df, use_container_width=True, hide_index=True)
    csv = df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "📥 下载 CSV",
        csv,
        file_name=f"optimize_{symbol}_{strategy_name}.csv",
        mime="text/csv",
    )
