"""Tab: 缠论分析（笔、线段、中枢、买卖点）。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from zenstock.data.types import Freq
from zenstock.stock_names import get_stock_name


def render(data: pd.DataFrame, symbol: str, freq: Freq) -> None:
    """渲染缠论分析页。"""
    name = get_stock_name(symbol)
    display = f"{symbol} {name}" if name else symbol
    st.subheader(
        f"🔮 缠论分析 — {freq.display_name}（共 {len(data):,} 根K线）"
    )

    if len(data) < 30:
        st.warning("数据不足 30 根，无法进行缠论分析（至少需要 30 根 K 线）")
        return

    # 分析按钮（缠论分析较慢，按需触发）
    col1, col2 = st.columns([3, 1])
    with col2:
        run_btn = st.button("🔮 开始分析", type="primary", use_container_width=True)

    if not run_btn:
        st.info("👆 点击「开始分析」识别笔、中枢、买卖点")
        # 展示说明
        with st.expander("📖 缠论分析说明"):
            st.markdown("""
**缠论核心概念**：
- **分型（FX）**：连续三根 K 线形成的顶分型 / 底分型
- **笔（BI）**：由相邻的顶底分型连接而成的最小走势单元
- **中枢（ZS）**：至少三笔重叠形成的成交密集区（支撑/阻力）
- **买卖点**：
  - 一买/一卖：趋势背驰后的反转点
  - 二买/二卖：反转后的第一次回调
  - 三买/三卖：突破中枢后的回踩确认

**数据建议**：
- 日线：至少 60 个交易日
- 5 分钟：至少 200 根（约 4 个交易日）
- 1 分钟：至少 500 根（约 2 个交易日）
""")
        return

    # 执行分析
    with st.spinner("缠论分析中（可能需要数秒）..."):
        try:
            from zenstock.chanlun.analyzer import ChanlunAnalyzer
            from zenstock.chanlun.segments import (
                detect_buy_sell_points,
                extract_line_segments,
                extract_zhongshu_from_segments,
            )

            analyzer = ChanlunAnalyzer()
            result = analyzer.analyze_single(data, freq, symbol=symbol)

            # 从笔构建线段和中枢，再识别买卖点
            segments = []
            seg_zs_list = []
            bsp_list = []
            if result.bi_count >= 3:
                segments = extract_line_segments(result.bi_list)
                seg_zs_list = extract_zhongshu_from_segments(
                    segments, result.bi_list
                )
                bsp_list = detect_buy_sell_points(segments, seg_zs_list)

        except ImportError:
            st.error("❌ czsc 库未安装。请运行: `pip install czsc -U`")
            return
        except Exception as e:  # noqa: BLE001
            st.error(f"❌ 分析失败: {e}")
            return

    # ===== 核心指标 =====
    cols = st.columns(6)
    cols[0].metric("K 线数", f"{result.bars_count:,}")
    cols[1].metric("笔", f"{result.bi_count}")
    cols[2].metric("线段", f"{len(segments)}")
    cols[3].metric("中枢(笔)", f"{result.zs_count}")
    cols[4].metric("中枢(线段)", f"{len(seg_zs_list)}")
    cols[5].metric("买卖点", f"{len(bsp_list)}")

    # ===== 买卖点信号（基于线段+中枢）=====
    buy_points = [p for p in bsp_list if p.is_buy]
    sell_points = [p for p in bsp_list if not p.is_buy]

    sig_cols = st.columns(2)
    with sig_cols[0]:
        st.markdown(f"### 🟢 买入信号（{len(buy_points)}）")
        if buy_points:
            for p in buy_points:
                st.success(f"**{p.point_type}** @ {p.dt}  价格 {p.price:.2f}  ({p.reason})")
        else:
            st.info("暂无买入信号")

    with sig_cols[1]:
        st.markdown(f"### 🔴 卖出信号（{len(sell_points)}）")
        if sell_points:
            for p in sell_points:
                st.error(f"**{p.point_type}** @ {p.dt}  价格 {p.price:.2f}  ({p.reason})")
        else:
            st.info("暂无卖出信号")

    # ===== K 线 + 笔 + 线段 + 中枢 可视化 =====
    st.subheader("📈 K 线 + 笔 + 线段 + 中枢")
    st.caption("💡 交互：滚轮缩放、拖动平移、十字光标自动跟随 | 右上角 ⛶ 按钮全屏")
    from frontend.lwc_chart import render_kline_chart
    render_kline_chart(
        data=data,
        symbol=display,
        title=f"{display} {freq.display_name} 缠论分析（线段+中枢）",
        bi_list=result.bi_list,
        zs_list=result.zs_list,
        segment_list=segments,
        height=650,
    )

    # ===== 线段明细表 =====
    if segments:
        with st.expander(f"📋 线段明细（{len(segments)} 段）"):
            seg_df = _segments_to_dataframe(segments)
            if not seg_df.empty:
                st.dataframe(seg_df, use_container_width=True, hide_index=True)

    # ===== 中枢明细表（基于线段）=====
    if seg_zs_list:
        with st.expander(f"📋 中枢明细-线段级（{len(seg_zs_list)} 个）"):
            zs2_df = _seg_zs_to_dataframe(seg_zs_list)
            if not zs2_df.empty:
                st.dataframe(zs2_df, use_container_width=True, hide_index=True)

    # ===== 笔明细表 =====
    if result.bi_list:
        with st.expander(f"📋 笔明细（{result.bi_count} 笔）"):
            bi_df = _bi_to_dataframe(result.bi_list)
            if not bi_df.empty:
                st.dataframe(bi_df, use_container_width=True, hide_index=True)

    # ===== 中枢明细表（基于笔）=====
    if result.zs_list:
        with st.expander(f"📋 中枢明细-笔级（{result.zs_count} 个）"):
            zs_df = _zs_to_dataframe(result.zs_list)
            if not zs_df.empty:
                st.dataframe(zs_df, use_container_width=True, hide_index=True)


def _bi_to_dataframe(bi_list) -> pd.DataFrame:
    """把笔列表转为 DataFrame。"""
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
        except Exception:  # noqa: BLE001
            continue
    return pd.DataFrame(rows)


def _zs_to_dataframe(zs_list) -> pd.DataFrame:
    """把中枢列表转为 DataFrame。"""
    rows = []
    for i, zs in enumerate(zs_list):
        try:
            rows.append({
                "序号": i + 1,
                "上沿(zg)": float(getattr(zs, "zg", 0) or 0),
                "下沿(zd)": float(getattr(zs, "zd", 0) or 0),
                "中轨(zz)": float(getattr(zs, "zz", 0) or 0),
                "起始": str(getattr(zs, "sdt", "") or ""),
                "结束": str(getattr(zs, "edt", "") or ""),
                "笔数": len(getattr(zs, "bis", []) or []),
            })
        except Exception:  # noqa: BLE001
            continue
    return pd.DataFrame(rows)


def _segments_to_dataframe(segments) -> pd.DataFrame:
    """把线段列表转为 DataFrame。"""
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
    """把线段级中枢列表转为 DataFrame。"""
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
