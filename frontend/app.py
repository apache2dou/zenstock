"""ZenStock 交互式可视化前端。

启动方式:
    streamlit run frontend/app.py

功能:
    - 单标的回测：选股票、选策略、调参数、看胜率赔率与资金曲线
    - 参数寻优：网格搜索 + 热力图 + 3D 曲面
    - 多股票对比：批量回测、累计收益曲线、雷达图
    - 数据管理：查看本地数据、下载新数据
"""

from __future__ import annotations

import sys
from pathlib import Path

# 将项目根和 src 加入 path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import pandas as pd
import streamlit as st

from zenstock.data import DataStorage
from zenstock.data.types import FREQ_OPTIONS, Freq
from zenstock.stock_names import get_stock_label, merge_stock_list


@st.cache_data(ttl=300)
def load_symbols(freq: str) -> list[str]:
    storage = DataStorage()
    # 尝试从 DuckDB 合并股票名称
    try:
        stock_list = storage.get_stock_list()
        if not stock_list.empty and "symbol" in stock_list.columns:
            name_col = "name" if "name" in stock_list.columns else None
            if name_col:
                merge_stock_list(dict(zip(stock_list["symbol"], stock_list[name_col])))
    except Exception:  # noqa: BLE001
        pass

    symbols = storage.list_symbols(Freq(freq))
    if symbols:
        return symbols

    # 没有原始数据时，检查是否可以从更细粒度重采样
    freq_enum = Freq(freq)
    resolvable_sources = {
        Freq.MIN15.value: [Freq.MIN5.value, Freq.MIN1.value],
        Freq.MIN30.value: [Freq.MIN5.value, Freq.MIN15.value, Freq.MIN1.value],
        Freq.MIN60.value: [Freq.MIN5.value, Freq.MIN30.value, Freq.MIN15.value, Freq.MIN1.value],
        Freq.DAILY.value: [Freq.MIN5.value, Freq.MIN15.value, Freq.MIN30.value, Freq.MIN60.value],
        Freq.WEEKLY.value: [Freq.DAILY.value],
        Freq.MONTHLY.value: [Freq.DAILY.value],
    }
    sources = resolvable_sources.get(freq_enum.value, [])
    for src in sources:
        src_symbols = storage.list_symbols(Freq(src))
        if src_symbols:
            return src_symbols

    return []


@st.cache_data(ttl=60)
def load_data(symbol: str, freq: str, start: str, end: str) -> pd.DataFrame:
    """加载 K 线数据，如果目标频率没有原始数据，尝试从更细粒度重采样。"""
    from zenstock.data.resample import RESAMPLE_MAP, get_or_resample

    storage = DataStorage()
    freq_enum = Freq(freq)
    df = storage.read_klines(symbol, freq_enum, start, end)

    if not df.empty:
        # read_klines 返回的 Parquet 不包含频率元数据，补回给策略使用。
        df.attrs["freq"] = freq_enum.value
        return df

    # 尝试从更细粒度重采样
    # 按优先级查找可用的源频率
    source_candidates = []
    if freq_enum == Freq.DAILY:
        source_candidates = [Freq.MIN5, Freq.MIN15, Freq.MIN30, Freq.MIN60]
    elif freq_enum == Freq.WEEKLY:
        source_candidates = [Freq.DAILY, Freq.MIN5]
    elif freq_enum == Freq.MONTHLY:
        source_candidates = [Freq.DAILY, Freq.MIN5]
    elif freq_enum == Freq.MIN15:
        source_candidates = [Freq.MIN5, Freq.MIN1]
    elif freq_enum == Freq.MIN30:
        source_candidates = [Freq.MIN5, Freq.MIN15, Freq.MIN1]
    elif freq_enum == Freq.MIN60:
        source_candidates = [Freq.MIN5, Freq.MIN30, Freq.MIN15, Freq.MIN1]

    for src_freq in source_candidates:
        src_df = storage.read_klines(symbol, src_freq, start, end)
        if not src_df.empty and len(src_df) >= 10:
            st.info(f"💡 {freq_enum.display_name}无原始数据，从{src_freq.display_name}合成（缓存中）")
            result = get_or_resample(symbol, src_df, freq_enum)
            result.attrs["freq"] = freq_enum.value
            return result

    return pd.DataFrame()


def main() -> None:
    st.set_page_config(
        page_title="ZenStock - A股量化分析",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📊 ZenStock")
    st.caption("A股量化策略研究与回测平台")

    # ===== 侧边栏全局设置 =====
    with st.sidebar:
        st.header("⚙️ 全局设置")
        # 默认日线，让用户切换到分钟级别
        freq_value = st.selectbox(
            "K 线周期",
            options=[f[0] for f in FREQ_OPTIONS],
            format_func=lambda v: dict(FREQ_OPTIONS)[v],
            index=0,
            key="global_freq",
        )
        freq = Freq(freq_value)

        symbols = load_symbols(freq_value)
        if not symbols:
            st.warning(f"本地无 {freq.display_name} 数据")
            st.info(
                "在「数据管理」页下载，或运行:\n"
                f"`python scripts/download_data.py --symbols 000001 "
                f"--freq {freq_value}`"
            )
            _render_tabs_empty()
            return

        # 选股：用固定 key（不随频率变化），切换频率时尽量保持选中的股票
        # 如果当前选中的股票在新频率下没有数据，则回退到第一个
        symbol = st.selectbox(
            "选择股票",
            symbols,
            format_func=get_stock_label,
            key="global_symbol",
        )
        # 分钟级默认起始日期需要灵活些
        default_start = pd.Timestamp("2024-06-01") if freq.is_minute else pd.Timestamp("2023-01-01")
        col1, col2 = st.columns(2)
        # 增量更新后通过标记强制刷新日期（在 widget 创建前设置）
        if st.session_state.pop("_force_end_today", False):
            st.session_state["global_end"] = pd.Timestamp.today().date()
        if "global_start" not in st.session_state:
            st.session_state["global_start"] = default_start
        if "global_end" not in st.session_state:
            st.session_state["global_end"] = pd.Timestamp.today()
        start = col1.date_input(
            "开始日期", key="global_start"
        )
        end = col2.date_input(
            "结束日期", key="global_end"
        )

        st.divider()
        st.caption("💡 提示：参数与策略在各 Tab 内单独设置")
        if freq == Freq.MIN1:
            st.warning("⚠️ 1 分钟数据仅最近 5~9 个交易日，适合日内策略验证")

    # ===== 主区域 Tabs =====
    tab_bt, tab_opt, tab_cmp, tab_cl, tab_bs, tab_data = st.tabs([
        "🎯 单标的回测",
        "🔍 参数寻优",
        "📊 多股票对比",
        "🔮 缠论分析",
        "🔬 两重表里关系",
        "📂 数据管理",
    ])

    # 加载选中股票数据（回测和寻优共用）
    data = load_data(symbol, freq_value, str(start), str(end))

    with tab_bt:
        if data.empty:
            st.warning(f"无 {symbol} 的 {freq.display_name} 数据")
        else:
            from frontend import tab_backtest
            tab_backtest.render(data, symbol, freq)

    with tab_opt:
        if data.empty:
            st.warning(f"无 {symbol} 的 {freq.display_name} 数据")
        else:
            from frontend import tab_optimize
            tab_optimize.render(data, symbol, freq)

    with tab_cmp:
        from frontend import tab_compare
        tab_compare.render(symbols, freq_value, start, end)

    with tab_cl:
        if data.empty:
            st.warning(f"无 {symbol} 的 {freq.display_name} 数据")
        else:
            from frontend import tab_chanlun
            tab_chanlun.render(data, symbol, freq)

    with tab_bs:
        if data.empty:
            st.warning(f"无 {symbol} 的 {freq.display_name} 数据")
        else:
            from frontend import tab_bi_state
            tab_bi_state.render(data, symbol, freq)

    with tab_data:
        from frontend import tab_data
        tab_data.render()


def _render_tabs_empty() -> None:
    """无数据时仅显示数据管理 Tab。"""
    tab_data = st.tabs(["📂 数据管理"])[0]
    with tab_data:
        from frontend import tab_data
        tab_data.render()


if __name__ == "__main__":
    main()
