"""专业行情图表组件，仿东方财富/同花顺交互。

特性：
- 主图（K线 + MA均线 + 笔/中枢）+ 副图（成交量），共享 x 轴联动
- 十字光标（spikemode=across）
- 快捷时间范围按钮（1月/3月/6月/1年/全部）
- 滑动条缩放（rangeslider）+ 滚轮缩放
- 笔合并为单条 trace（性能优化）
- A 股配色（红涨绿跌）
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# 色弱友好配色（避开红绿依赖）
COLOR_UP = "#FF9800"      # 涨=橙黄（暖色、高亮度）
COLOR_DOWN = "#2196F3"   # 跌=蓝（冷色、高饱和）
COLOR_MA5 = "#FFFFFF"    # MA5 白
COLOR_MA10 = "#CE93D8"   # MA10 紫
COLOR_MA20 = "#80CBC4"   # MA20 青绿
COLOR_BI_UP = "#FFD600"  # 向上笔 亮黄
COLOR_BI_DOWN = "#00B0FF"# 向下笔 亮蓝
COLOR_ZS = "#FFC107"     # 中枢 琥珀
COLOR_VOL_UP = "rgba(255,152,0,0.7)"
COLOR_VOL_DOWN = "rgba(33,150,243,0.7)"


def _calc_ma(series: pd.Series, n: int) -> pd.Series:
    """计算均线。"""
    return series.rolling(n, min_periods=1).mean()


def _bi_direction(bi: Any) -> str:
    """判断笔方向：'up' / 'down'。"""
    d = str(getattr(bi, "direction", ""))
    if "a" == d[-1:] or "up" in d.lower() or "上" in d:
        return "up"
    return "down"


def _bi_endpoints(bi: Any) -> tuple[Any, Any, float, float] | None:
    """提取笔的起止时间和价格。返回 (dt_a, dt_b, price_a, price_b) 或 None。"""
    fx_a = getattr(bi, "fx_a", None)
    fx_b = getattr(bi, "fx_b", None)
    if fx_a is None or fx_b is None:
        return None
    return (
        getattr(fx_a, "dt", None),
        getattr(fx_b, "dt", None),
        float(getattr(fx_a, "fx", 0) or 0),
        float(getattr(fx_b, "fx", 0) or 0),
    )


def plot_kline_chart(
    data: pd.DataFrame,
    symbol: str = "",
    title: str = "",
    show_ma: bool = True,
    ma_periods: tuple[int, ...] = (5, 10, 20),
    show_volume: bool = True,
    bi_list: list | None = None,
    zs_list: list | None = None,
    height: int = 700,
) -> go.Figure:
    """绘制专业行情图表。

    Args:
        data: K 线 DataFrame，需含 date/open/high/low/close/volume
        symbol: 股票代码
        title: 标题
        show_ma: 是否显示均线
        ma_periods: 均线周期
        show_volume: 是否显示成交量副图
        bi_list: 笔列表（可选，缠论用）
        zs_list: 中枢列表（可选，缠论用）
        height: 图表高度

    Returns:
        plotly Figure
    """
    df = data.copy()
    x = df["date"]

    # 成交量颜色（涨红跌绿）
    vol_colors = [
        COLOR_VOL_UP if c >= o else COLOR_VOL_DOWN
        for c, o in zip(df["close"], df["open"])
    ]

    # ===== 创建子图：主图 + 成交量副图 =====
    specs = [[{"secondary_y": False}]]
    if show_volume:
        specs = [[{"secondary_y": False}], [{"secondary_y": False}]]

    fig = make_subplots(
        rows=2 if show_volume else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_width=[0.25, 0.75] if show_volume else [1.0],
        row_heights=[0.75, 0.25] if show_volume else [1.0],
    )

    # ===== 1. 主图：K 线 =====
    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="K线",
            increasing_line_color=COLOR_UP,
            decreasing_line_color=COLOR_DOWN,
            increasing_fillcolor=COLOR_UP,
            decreasing_fillcolor=COLOR_DOWN,
            whiskerwidth=0.5,
        ),
        row=1, col=1,
    )

    # ===== 2. 主图：均线 =====
    if show_ma:
        ma_colors = {5: COLOR_MA5, 10: COLOR_MA10, 20: COLOR_MA20}
        for n in ma_periods:
            ma = _calc_ma(df["close"], n)
            fig.add_trace(
                go.Scatter(
                    x=x, y=ma,
                    mode="lines",
                    line=dict(color=ma_colors.get(n, "#999"), width=1.2),
                    name=f"MA{n}",
                    hovertemplate=f"MA{n}: %{{y:.2f}}<extra></extra>",
                ),
                row=1, col=1,
            )

    # ===== 3. 主图：笔（合并为单条 trace 优化性能） =====
    if bi_list:
        bi_x, bi_y = [], []
        bi_colors = []
        for bi in bi_list:
            ep = _bi_endpoints(bi)
            if ep is None:
                continue
            dt_a, dt_b, p_a, p_b = ep
            if dt_a is None or dt_b is None:
                continue
            # 用 None 分段
            if bi_x:
                bi_x.append(None)
                bi_y.append(None)
            bi_x.extend([dt_a, dt_b])
            bi_y.extend([p_a, p_b])
            direction = _bi_direction(bi)
            bi_colors.append(COLOR_BI_UP if direction == "up" else COLOR_BI_DOWN)

        if bi_x:
            # 向上笔和向下笔分别画
            _add_bi_traces(fig, bi_list, row=1)

    # ===== 4. 主图：中枢（矩形） =====
    if zs_list:
        _add_zs_shapes(fig, zs_list)

    # ===== 5. 副图：成交量 =====
    if show_volume:
        fig.add_trace(
            go.Bar(
                x=x,
                y=df["volume"],
                name="成交量",
                marker_color=vol_colors,
                showlegend=False,
                hovertemplate="量: %{y}<extra></extra>",
            ),
            row=2, col=1,
        )

    # ===== 布局：仿行情软件 =====
    _apply_layout(fig, symbol, title, height, has_volume=show_volume)

    return fig


def _add_bi_traces(fig: go.Figure, bi_list: list, row: int) -> None:
    """把笔分为向上/向下两组，各画一条 trace。"""
    up_x, up_y, down_x, down_y = [], [], [], []

    for bi in bi_list:
        ep = _bi_endpoints(bi)
        if ep is None:
            continue
        dt_a, dt_b, p_a, p_b = ep
        if dt_a is None or dt_b is None:
            continue
        direction = _bi_direction(bi)
        target_x, target_y = (up_x, up_y) if direction == "up" else (down_x, down_y)
        if target_x:
            target_x.append(None)
            target_y.append(None)
        target_x.extend([dt_a, dt_b])
        target_y.extend([p_a, p_b])

    if up_x:
        fig.add_trace(
            go.Scatter(
                x=up_x, y=up_y,
                mode="lines+markers",
                line=dict(color=COLOR_BI_UP, width=1.8),
                marker=dict(size=6, color=COLOR_BI_UP),
                name="向上笔",
                hovertemplate="向上笔<br>%{x} %{y:.2f}<extra></extra>",
            ), row=row, col=1,
        )
    if down_x:
        fig.add_trace(
            go.Scatter(
                x=down_x, y=down_y,
                mode="lines+markers",
                line=dict(color=COLOR_BI_DOWN, width=1.8),
                marker=dict(size=6, color=COLOR_BI_DOWN),
                name="向下笔",
                hovertemplate="向下笔<br>%{x} %{y:.2f}<extra></extra>",
            ), row=row, col=1,
        )


def _add_zs_shapes(fig: go.Figure, zs_list: list) -> None:
    """添加中枢矩形。"""
    for zs in zs_list:
        try:
            zg = float(getattr(zs, "zg", 0) or 0)
            zd = float(getattr(zs, "zd", 0) or 0)
            if zg <= 0 or zd <= 0 or zg <= zd:
                continue
            start_dt = getattr(zs, "sdt", None)
            end_dt = getattr(zs, "edt", None)
            if start_dt is None or end_dt is None:
                bis_in = getattr(zs, "bis", [])
                if bis_in:
                    ep0 = _bi_endpoints(bis_in[0])
                    ep1 = _bi_endpoints(bis_in[-1])
                    if ep0 and ep1:
                        start_dt = ep0[0]
                        end_dt = ep1[1]
            if start_dt is None or end_dt is None:
                continue

            fig.add_shape(
                type="rect",
                x0=start_dt, x1=end_dt, y0=zd, y1=zg,
                line=dict(color=COLOR_ZS, width=1.5),
                fillcolor="rgba(255, 183, 77, 0.12)",
                layer="below",
            )
        except Exception:  # noqa: BLE001
            continue


def _apply_layout(
    fig: go.Figure, symbol: str, title: str, height: int, has_volume: bool
) -> None:
    """应用仿行情软件的专业布局。"""
    # 快捷时间范围按钮
    rangeselector = dict(
        buttons=list([
            dict(count=1, label="1月", step="month", stepmode="backward"),
            dict(count=3, label="3月", step="month", stepmode="backward"),
            dict(count=6, label="6月", step="month", stepmode="backward"),
            dict(count=1, label="1年", step="year", stepmode="backward"),
            dict(step="all", label="全部"),
        ]),
        x=0.0, y=1.0,
        xanchor="left", yanchor="top",
        bgcolor="rgba(240,240,240,0.8)",
        activecolor="rgba(137,180,250,0.5)",
    )

    fig.update_layout(
        title=dict(text=title or f"{symbol} 行情", x=0.5, xanchor="center"),
        height=height,
        template="plotly_dark",
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4", size=11),
        margin=dict(l=50, r=30, t=50, b=30),
        showlegend=True,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0,
            xanchor="right", x=1.0, bgcolor="rgba(0,0,0,0)",
        ),
        # 十字光标（关键：跨轴联动）
        hovermode="x unified",
        spikedistance=1000,
        xaxis=dict(
            rangeselector=rangeselector,
            rangeslider=dict(visible=False),  # 主图不显示滑动条
            spikecolor="#89b4fa",
            spikethickness=1,
            spikemode="across",
            spikesnap="cursor",
            showgrid=True,
            gridcolor="rgba(127,127,127,0.15)",
            showspikes=True,
        ),
        yaxis=dict(
            domain=[0.3, 1.0] if has_volume else [0.0, 1.0],
            side="right",  # 价格轴在右边（A股习惯）
            spikecolor="#89b4fa",
            spikethickness=1,
            spikemode="across",
            spikesnap="cursor",
            showgrid=True,
            gridcolor="rgba(127,127,127,0.15)",
            showspikes=True,
            fixedrange=False,
        ),
    )

    # 成交量副图 y 轴
    if has_volume:
        fig.update_yaxes(
            side="right",
            showgrid=True,
            gridcolor="rgba(127,127,127,0.1)",
            row=2, col=1,
        )
        # 副图的 x 轴也显示十字光标
        fig.update_xaxes(
            spikecolor="#89b4fa",
            spikethickness=1,
            spikemode="across",
            spikesnap="cursor",
            showspikes=True,
            showgrid=True,
            gridcolor="rgba(127,127,127,0.15)",
            row=2, col=1,
        )

    # 关闭 K 线的空心默认（避免鼠标移上去有空白）
    fig.update_traces(
        selector=dict(type="candlestick"),
        increasing_line_width=1,
        decreasing_line_width=1,
    )
