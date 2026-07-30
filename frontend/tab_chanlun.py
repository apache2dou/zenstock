"""Tab: 缠论分析 — 多级别递归分析（1分钟→5分钟→30分钟→日线）。

严格按《教你炒股票》原文递归定义：
  - 1分钟K线 → 笔 → 线段 → 1分钟中枢 → 1分钟走势类型
  - 1分钟走势类型（作为次级别）→ 5分钟线段 → 5分钟中枢 → 5分钟走势类型
  - 5分钟走势类型（作为次级别）→ 30分钟线段 → 30分钟中枢 → 30分钟走势类型
  - 30分钟走势类型（作为次级别）→ 日线线段 → 日线中枢 → 日线走势类型

各级别独立分析 + 递归对照展示。
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from zenstock.data import DataStorage
from zenstock.data.types import Freq
from zenstock.stock_names import get_stock_name

# 延迟导入 czsc 相关模块（czsc 是可选依赖）
# detect_buy_sell_points 在 _render_single_level 中使用，按需导入

# 支持的各级别频率（从低到高）
MULTI_FREQS: list[Freq] = [Freq.MIN1, Freq.MIN5, Freq.MIN30, Freq.DAILY]

def render(data: pd.DataFrame, symbol: str, freq: Freq) -> None:
    """渲染缠论多级别分析页。

    自动加载多频率数据（1分钟/5分钟/30分钟/日线），
    执行多级别递归分析，各级别独立展示图表。

    Args:
        data: 主频率 K 线数据（用于日期范围参考）
        symbol: 股票代码
        freq: 用户选择的主频率
    """
    name = get_stock_name(symbol)
    display = f"{symbol} {name}" if name else symbol

    st.subheader(f"🔮 缠论多级别分析 — {display}")

    # 分析按钮
    col1, col2 = st.columns([3, 1])
    with col2:
        run_btn = st.button("🔮 开始多级别分析", type="primary", width="stretch")

    if not run_btn:
        st.info("👆 点击「开始多级别分析」将自动加载 1分钟/5分钟/30分钟/日线 数据并分析")
        with st.expander("📖 缠论多级别递归关系"):
            st.markdown("""
**递归链（原文第62-71课）**：

```
1分钟K线 → 包含处理 → 分型 → 笔 → 线段 → 1分钟中枢
    → 1分钟走势类型（盘整/趋势）
        ↓ 作为次级别
5分钟线段 → 5分钟中枢 → 5分钟走势类型
        ↓ 作为次级别
30分钟线段 → 30分钟中枢 → 30分钟走势类型
        ↓ 作为次级别
日线线段 → 日线中枢 → 日线走势类型
```

**各级别内涵**：
- **1分钟**：最精细，适合超短线（日内T+0）。笔/线段直接由K线构建
- **5分钟**：短线操作级别。1分钟走势类型 ≈ 5分钟一笔
- **30分钟**：中短线。5分钟走势类型 ≈ 30分钟一笔
- **日线**：中长线。30分钟走势类型 ≈ 日线一笔

**走势类型分类**：
- **盘整**：只含1个中枢
- **上涨趋势**：2个以上依次向上的不重叠中枢
- **下跌趋势**：2个以上依次向下的不重叠中枢
""")
        return

    # ===== 加载多频率数据 =====
    with st.spinner("⏳ 正在加载多级别数据..."):
        data_by_freq = _load_multi_freq_data(symbol, data)

    if not data_by_freq:
        st.error("❌ 没有可用的多级别数据。请先下载 5分钟 或 日线 数据。")
        return

    available_names = [f.display_name for f in data_by_freq]
    st.caption(f"📡 可用级别: {' → '.join(available_names)}")

    # ===== 执行多级别分析 =====
    with st.spinner("🔮 缠论多级别递归分析中（可能需要 10-30 秒）..."):
        try:
            from zenstock.chanlun.multi_level import MultiLevelAnalyzer

            mla = MultiLevelAnalyzer()
            multi_result = mla.analyze(data_by_freq, mode="independent")

        except ImportError:
            st.error("❌ czsc 库未安装。请运行: `pip install czsc -U`")
            return
        except Exception as e:
            st.error(f"❌ 分析失败: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    # ===== 总览面板 =====
    _render_overview(multi_result)

    # ===== 各级别 Tab =====
    level_names = list(multi_result.levels.keys())
    if not level_names:
        st.warning("无可用分析结果")
        return

    tabs = st.tabs(["📊 多级别总览"] + [f"📈 {n}" for n in level_names])

    # 总览 Tab
    with tabs[0]:
        _render_level_overview(multi_result)

    # 各级别详情 Tab
    for i, lv_name in enumerate(level_names):
        lv = multi_result.levels[lv_name]
        df = data_by_freq.get(lv.freq)
        if df is None or df.empty:
            continue
        with tabs[i + 1]:
            _render_single_level(lv, df, lv_name, display)


# ==================== 数据加载 ====================

def _load_multi_freq_data(
    symbol: str, reference_df: pd.DataFrame,
) -> dict[Freq, pd.DataFrame]:
    """加载多频率 K 线数据。

    以 reference_df 的日期范围为参考，尝试加载所有支持的频率。
    """
    storage = DataStorage()
    # 从参考数据获取日期范围
    start = str(reference_df["date"].min())[:10] if not reference_df.empty else "2020-01-01"
    end = str(reference_df["date"].max())[:10] if not reference_df.empty else "2099-01-01"

    result: dict[Freq, pd.DataFrame] = {}
    for f in MULTI_FREQS:
        try:
            df = storage.read_klines(symbol, f, start, end)
            if not df.empty and len(df) >= 10:
                result[f] = df
        except Exception:
            continue

    return result


# ==================== 总览面板 ====================

def _render_overview(multi_result) -> None:
    """渲染多级别总览指标。"""
    lv_names = list(multi_result.levels.keys())
    if not lv_names:
        return

    cols = st.columns(len(lv_names))
    for i, name in enumerate(lv_names):
        lv = multi_result.levels[name]
        with cols[i]:
            st.metric(
                f"**{name}**",
                f"{lv.bars_count} K线",
                delta=f"{lv.bi_count}笔 {lv.segment_count}段 {lv.zs_count}中枢",
            )


def _render_level_overview(multi_result) -> None:
    """总览 Tab: 走势类型对比 + 级别递归关系。"""
    st.subheader("📊 各级别走势类型对比")

    # 收集所有走势类型
    all_tt = multi_result.all_trend_types()
    if not all_tt:
        st.info("暂未识别出走势类型")
        return

    # 走势类型表格
    rows = []
    for tt in all_tt:
        emoji = {"盘整": "🟡", "上涨趋势": "🟢", "下跌趋势": "🔴"}.get(tt.trend_class, "⚪")
        rows.append({
            "级别": tt.freq_name,
            "类型": f"{emoji} {tt.trend_class}",
            "起始": str(tt.start_dt or "")[:16],
            "结束": str(tt.end_dt or "")[:16],
            "中枢数": tt.zs_count,
            "段数": tt.segment_count,
            "笔数": tt.bi_count,
            "起价": f"{tt.start_price:.2f}",
            "终价": f"{tt.end_price:.2f}",
            "涨跌": f"{((tt.end_price / tt.start_price - 1) * 100):.1f}%" if tt.start_price > 0 else "-",
        })
    overview_df = pd.DataFrame(rows)
    st.dataframe(overview_df, width="stretch", hide_index=True)

    # 递归关系说明
    st.subheader("🔗 级别递归关系")
    st.markdown("""
| 低级别 | → 高级别 | 对应关系 |
|--------|----------|----------|
| 1分钟走势类型 | → 5分钟线段 | 约 3~5 个 1分钟走势 ≈ 1 个 5分钟线段 |
| 5分钟走势类型 | → 30分钟线段 | 约 3~5 个 5分钟走势 ≈ 1 个 30分钟线段 |
| 30分钟走势类型 | → 日线线段 | 约 3~5 个 30分钟走势 ≈ 1 个日线线段 |

**注意**：各级别独立分析时，此对应关系为近似。严格递归需从最低级别逐级构建（需要完整 1分钟数据覆盖）。
""")


# ==================== 单级别详情 ====================

def _render_single_level(
    lv,
    df: pd.DataFrame,
    lv_name: str,
    symbol_display: str,
) -> None:
    """渲染单级别分析详情：图表 + 指标 + 买卖点 + 明细表。"""
    st.subheader(f"📈 {lv_name} 级别分析")

    # 核心指标
    cols = st.columns(7)
    cols[0].metric("K线", f"{lv.bars_count}")
    cols[1].metric("笔", f"{lv.bi_count}")
    cols[2].metric("线段", f"{lv.segment_count}")
    cols[3].metric("中枢", f"{lv.zs_count}")
    cols[4].metric("走势类型", f"{len(lv.trend_types)}")
    cols[5].metric("高", f"{df['high'].max():.2f}")
    cols[6].metric("低", f"{df['low'].min():.2f}")

    # 走势类型
    if lv.trend_types:
        tt_labels = []
        for tt in lv.trend_types:
            emoji = {"盘整": "🟡", "上涨趋势": "🟢", "下跌趋势": "🔴"}.get(tt.trend_class, "")
            tt_labels.append(f"{emoji} {tt.trend_class}")
        st.caption("走势类型: " + " → ".join(tt_labels))

    # 买卖点（局部导入，czsc 是可选依赖）
    from zenstock.chanlun.segments import detect_buy_sell_points
    bsp_list = detect_buy_sell_points(lv.segment_list, lv.zs_list)
    buy_points = [p for p in bsp_list if p.is_buy]
    sell_points = [p for p in bsp_list if not p.is_buy]

    if buy_points or sell_points:
        sig_cols = st.columns(2)
        with sig_cols[0]:
            for p in buy_points[-3:]:  # 最近3个
                st.success(f"**{p.point_type}** @ {str(p.dt)[:16]}  ¥{p.price:.2f}")
        with sig_cols[1]:
            for p in sell_points[-3:]:
                st.error(f"**{p.point_type}** @ {str(p.dt)[:16]}  ¥{p.price:.2f}")

    # K 线图表（含笔、线段、中枢）
    st.caption("💡 交互：滚轮缩放、拖动平移、十字光标")
    from frontend.lwc_chart import render_kline_chart

    render_kline_chart(
        data=df,
        symbol=symbol_display,
        title=f"{symbol_display} {lv_name} 缠论分析",
        bi_list=lv.bi_list,
        zs_list=lv.zs_list,
        segment_list=lv.segment_list,
        height=550,
    )

    # 明细表
    col_a, col_b = st.columns(2)
    with col_a:
        if lv.segment_list:
            with st.expander(f"📋 线段明细（{lv.segment_count} 段）"):
                seg_df = _segments_to_dataframe(lv.segment_list)
                if not seg_df.empty:
                    st.dataframe(seg_df, width="stretch", hide_index=True)
        if lv.zs_list:
            with st.expander(f"📋 中枢明细（{lv.zs_count} 个）"):
                zs_df = _seg_zs_to_dataframe(lv.zs_list)
                if not zs_df.empty:
                    st.dataframe(zs_df, width="stretch", hide_index=True)

    with col_b:
        if lv.trend_types:
            with st.expander(f"📋 走势类型（{len(lv.trend_types)} 个）"):
                tt_df = _trends_to_dataframe(lv.trend_types)
                if not tt_df.empty:
                    st.dataframe(tt_df, width="stretch", hide_index=True)
        if lv.bi_list:
            with st.expander(f"📋 笔明细（{lv.bi_count} 笔）"):
                bi_df = _bi_to_dataframe(lv.bi_list)
                if not bi_df.empty:
                    st.dataframe(bi_df, width="stretch", hide_index=True)


# ==================== DataFrame 转换 ====================

def _bi_to_dataframe(bi_list) -> pd.DataFrame:
    rows = []
    for i, bi in enumerate(bi_list):
        try:
            fx_a = getattr(bi, "fx_a", None)
            fx_b = getattr(bi, "fx_b", None)
            direction = getattr(bi, "direction", "")
            rows.append({
                "序号": i + 1,
                "方向": str(direction),
                "起点": str(getattr(fx_a, "dt", "") if fx_a else ""),
                "终点": str(getattr(fx_b, "dt", "") if fx_b else ""),
                "起价": float(getattr(fx_a, "fx", 0) or 0) if fx_a else 0,
                "终价": float(getattr(fx_b, "fx", 0) or 0) if fx_b else 0,
                "幅度": float(getattr(bi, "power_price", 0) or 0),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


def _segments_to_dataframe(segments) -> pd.DataFrame:
    rows = []
    for i, seg in enumerate(segments):
        rows.append({
            "序号": i + 1,
            "方向": "↑向上" if seg.is_up else "↓向下",
            "起点": str(seg.start_dt or ""),
            "终点": str(seg.end_dt or ""),
            "起价": seg.start_price,
            "终价": seg.end_price,
            "幅度": seg.amplitude,
            "面积": seg.area,
            "笔数": seg.bi_count,
        })
    return pd.DataFrame(rows)


def _seg_zs_to_dataframe(zs_list) -> pd.DataFrame:
    rows = []
    for i, zs in enumerate(zs_list):
        rows.append({
            "序号": i + 1,
            "上沿(zg)": zs.zg,
            "下沿(zd)": zs.zd,
            "中轨(zz)": zs.zz,
            "起始": str(zs.start_dt or ""),
            "结束": str(zs.end_dt or ""),
            "线段数": zs.seg_count,
        })
    return pd.DataFrame(rows)


def _trends_to_dataframe(trend_types) -> pd.DataFrame:
    rows = []
    for i, tt in enumerate(trend_types):
        emoji = {"盘整": "🟡", "上涨趋势": "🟢", "下跌趋势": "🔴"}.get(tt.trend_class, "")
        rows.append({
            "序号": i + 1,
            "类型": f"{emoji} {tt.trend_class}",
            "起始": str(tt.start_dt or "")[:16],
            "结束": str(tt.end_dt or "")[:16],
            "起价": tt.start_price,
            "终价": tt.end_price,
            "涨跌幅": f"{((tt.end_price / tt.start_price - 1) * 100):.1f}%" if tt.start_price > 0 else "-",
            "中枢数": tt.zs_count,
            "线段数": tt.segment_count,
            "笔数": tt.bi_count,
        })
    return pd.DataFrame(rows)
