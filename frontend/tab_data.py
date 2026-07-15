"""Tab: 数据管理（查看本地数据、下载新数据）。"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from zenstock.data import DataStorage, get_downloader
from zenstock.data.types import FREQ_OPTIONS, Freq
from zenstock.stock_names import get_stock_label, get_stock_name


def render() -> None:
    """渲染数据管理页。"""
    storage = DataStorage()

    st.subheader("📂 本地数据概览")

    # 按频率汇总
    rows = []
    for f in [Freq.DAILY, Freq.MIN1, Freq.MIN5, Freq.MIN15, Freq.MIN30, Freq.MIN60]:
        syms = storage.list_symbols(f)
        if syms:
            # 显示代码 + 名称
            labels = [get_stock_label(s) for s in syms[:10]]
            rows.append({
                "频率": f.display_name,
                "股票数": len(syms),
                "代码": ", ".join(labels) + (" ..." if len(syms) > 10 else ""),
            })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("暂无任何本地数据")

    # 默认显示日线明细
    st.divider()
    st.subheader("📋 数据明细")
    _freq_options = [f.value for f in Freq]
    freq_for_detail = st.selectbox(
        "查看哪个频率的明细",
        options=_freq_options,
        format_func=lambda v: Freq(v).display_name,
        index=_freq_options.index(Freq.MIN5.value),
        key="detail_freq_select_5",
    )
    freq_to_show = Freq(freq_for_detail)
    symbols = storage.list_symbols(freq_to_show)

    # 增量更新控件（与明细表同区域）
    if symbols:
        # 控件行：数据源 + 更新范围 + 按钮
        col_src, col_scope, col_btn = st.columns([2, 2, 2])
        with col_src:
            update_source = st.selectbox(
                "数据源", ["akshare", "baostock"],
                key=f"update_src2_{freq_for_detail}",
            )
        with col_scope:
            update_scope = st.radio(
                "更新范围",
                ["全部", "指定"],
                horizontal=True,
                key=f"update_scope_{freq_for_detail}",
            )
        with col_btn:
            st.write("")  # 对齐
            update_clicked = st.button(
                "🔄 增量更新", type="primary",
                key=f"update_btn_{freq_for_detail}",
            )

        if update_scope == "指定":
            update_symbols = st.multiselect(
                "选择要更新的股票",
                symbols,
                default=symbols[:3] if len(symbols) >= 3 else symbols,
                format_func=get_stock_label,
                key=f"update_syms_{freq_for_detail}",
            )
        else:
            update_symbols = symbols

        if freq_to_show == Freq.MIN1:
            st.warning("1 分钟仅 AKShare 支持")

        # 明细表 + 每行更新按钮
        rows = []
        for sym in symbols:
            df = storage.read_klines(sym, freq_to_show)
            if not df.empty:
                rows.append({
                    "股票": get_stock_label(sym),
                    "代码": sym,
                    "记录数": len(df),
                    "起始": str(df["date"].min()),
                    "截止": str(df["date"].max()),
                    "最新价": f"{df['close'].iloc[-1]:.2f}",
                })
        detail_df = pd.DataFrame(rows)

        if not detail_df.empty:
            # 表头
            hdr = st.columns([3, 1.5, 2.5, 2.5, 1.5, 1])
            hdr[0].markdown("**股票**")
            hdr[1].markdown("**记录数**")
            hdr[2].markdown("**起始日期**")
            hdr[3].markdown("**截止日期**")
            hdr[4].markdown("**最新价**")
            hdr[5].markdown("**更新**")
            st.divider()

            # 数据行（每行：股票名 + 数据 + 更新按钮）
            for _, row in detail_df.iterrows():
                cols = st.columns([3, 1.5, 2.5, 2.5, 1.5, 1])
                cols[0].write(row["股票"])
                cols[1].write(f"{row['记录数']:,}")
                cols[2].write(str(row["起始"])[:10])
                cols[3].write(str(row["截止"])[:10])
                cols[4].write(row["最新价"])
                with cols[5]:
                    if st.button("🔄", key=f"row_upd_{row['代码']}_{freq_for_detail}",
                                 help=f"更新 {row['股票']}"):
                        _do_incremental_update(
                            storage, [row["代码"]], freq_to_show, update_source
                        )
                        st.rerun()

            # 全量更新按钮触发
            if update_clicked and update_symbols:
                _do_incremental_update(
                    storage, update_symbols, freq_to_show, update_source
                )
                st.rerun()

            # 更新结果显示区（固定在表格下方）
            if "update_results" in st.session_state and st.session_state["update_results"]:
                st.divider()
                summary = st.session_state["update_results"][0]
                st.markdown(f"**{summary}**")
                for line in st.session_state["update_results"][1:]:
                    st.text(line)
                if st.button("✖ 关闭", key="close_update_results"):
                    st.session_state["update_results"] = []
                    st.rerun()
        else:
            st.info(f"本地无 {freq_to_show.display_name} 数据")

        # 更新结果显示区（表格无数据时也显示）
        if "update_results" in st.session_state and st.session_state["update_results"]:
            if detail_df.empty:
                st.divider()
                summary = st.session_state["update_results"][0]
                st.markdown(f"**{summary}**")
                for line in st.session_state["update_results"][1:]:
                    st.text(line)
                if st.button("✖ 关闭", key="close_update_results2"):
                    st.session_state["update_results"] = []
                    st.rerun()
    else:
        st.info(f"本地无 {freq_to_show.display_name} 数据")

    stock_list = storage.get_stock_list()
    if not stock_list.empty:
        st.caption(f"已下载股票列表 {len(stock_list)} 条")

    # 下载新数据
    st.divider()
    st.subheader("📥 下载新数据")
    with st.form("download_form"):
        dl_symbols = st.text_input(
            "股票代码（逗号分隔）",
            placeholder="000001,600519,000858",
            key="dl_symbols",
        )
        _dl_freq_options = [f[0] for f in FREQ_OPTIONS]
        dl_freq_value = st.selectbox(
            "K线周期",
            options=_dl_freq_options,
            format_func=lambda v: dict(FREQ_OPTIONS)[v],
            index=_dl_freq_options.index("5"),
            key="dl_freq_5",
        )
        dl_start = st.date_input(
            "开始日期", value=pd.Timestamp("2023-01-01"), key="dl_start"
        )
        dl_source = st.selectbox(
            "数据源", ["akshare", "baostock"], key="dl_source2"
        )
        # 分钟级别提示
        dl_freq = Freq(dl_freq_value)
        if dl_freq == Freq.MIN1:
            st.warning("1 分钟仅 AKShare 支持，且只取最近 5~9 个交易日")
        elif dl_freq.is_minute:
            st.info("分钟级数据：AKShare 5~9 个交易日；BaoStock 完整历史")
        submitted = st.form_submit_button("🚀 开始下载", type="primary")

    if submitted and dl_symbols:
        sym_list = [s.strip() for s in dl_symbols.split(",") if s.strip()]
        # 1 分钟强制使用 akshare
        actual_source = "akshare" if dl_freq == Freq.MIN1 else dl_source
        with st.spinner(f"下载 {len(sym_list)} 只股票的 {dl_freq.display_name} ..."):
            try:
                from datetime import datetime as _dt
                downloader = get_downloader(actual_source)
                all_data = downloader.download_many(
                    sym_list,
                    start_date=str(dl_start),
                    end_date=_dt.now().strftime("%Y-%m-%d"),
                    freq=dl_freq,
                )
                total = 0
                for sym, df in all_data.items():
                    total += storage.save_klines(df, freq=dl_freq)

                st.success(
                    f"✅ 下载完成：{len(all_data)}/{len(sym_list)} 只，"
                    f"共 {total:,} 条{dl_freq.display_name}K线"
                )
                st.cache_data.clear()  # 清除缓存让新数据生效
            except Exception as e:  # noqa: BLE001
                st.error(f"❌ 下载失败：{e}")

    storage.close()


def _do_incremental_update(
    storage: DataStorage,
    symbols: list[str],
    freq: Freq,
    source: str,
) -> None:
    """执行增量更新：从已有数据的末端开始获取新数据并追加。

    注意：此函数不直接渲染 Streamlit 内容，只把结果存入 session_state。
    过程提示和最终结果由调用方在表格下方统一渲染。
    """
    from datetime import datetime, timedelta

    actual_source = "akshare" if freq == Freq.MIN1 else source
    today = datetime.now().strftime("%Y-%m-%d")
    today_dt = datetime.now()
    updated = 0
    skipped = 0
    failed = 0
    results: list[str] = []

    # 标记正在更新（用于表格下方显示进度）
    st.session_state["update_results"] = [f"⏳ 正在更新 {len(symbols)} 只股票..."]

    try:
        downloader = get_downloader(actual_source)
    except Exception as e:  # noqa: BLE001
        st.session_state["update_results"] = [f"❌ 数据源初始化失败: {e}"]
        return

    for idx, sym in enumerate(symbols):
        label = get_stock_label(sym)
        last_date = storage.get_last_date(sym, freq)

        # 实时更新进度提示（存入 session_state，由表格下方渲染区显示）
        st.session_state["update_results"] = [
            f"⏳ 更新中... {idx + 1}/{len(symbols)}：{label}"
        ]

        if last_date is None:
            skipped += 1
            results.append(f"⏭️ {label}：本地无数据，跳过")
        else:
            last_dt = pd.Timestamp(last_date)
            start_dt = last_dt.normalize() + timedelta(days=1)
            start_str = start_dt.strftime("%Y-%m-%d")

            if start_dt > today_dt:
                skipped += 1
                results.append(f"⏭️ {label}：已是最新（{str(last_date)[:10]}）")
            else:
                try:
                    new_data = downloader.download_many(
                        [sym], start_date=start_str, end_date=today, freq=freq,
                    )
                    if sym in new_data and not new_data[sym].empty:
                        storage.save_klines(new_data[sym], freq=freq, mode="merge")
                        new_last = storage.get_last_date(sym, freq)
                        updated += 1
                        results.append(
                            f"✅ {label}：{str(last_date)[:10]} → {str(new_last)[:10]}"
                        )
                    else:
                        results.append(
                            f"ℹ️ {label}：无新数据（{start_str} ~ {today}）"
                        )
                except Exception as e:  # noqa: BLE001
                    failed += 1
                    results.append(f"❌ {label}：{e}")

    # 最终结果存入 session_state
    st.session_state["update_results"] = [
        f"更新完成：✅ {updated} 已更新 · ⏭️ {skipped} 跳过 · ❌ {failed} 失败"
    ] + results

    # 标记需要刷新日期（在下一次 rerun 时由 app.py 在 widget 创建前处理）
    if updated > 0:
        st.session_state["_force_end_today"] = True

    if updated > 0:
        st.cache_data.clear()
        _clear_derived_caches()

    storage.close()


def _clear_derived_caches() -> None:
    """清理重采样和缠论缓存（因为底层数据已更新）。"""
    from pathlib import Path
    from zenstock.config import get_config

    cache_dir = get_config().data.cache_path
    cleared = 0
    for subdir in ["resample", "chanlun"]:
        d = cache_dir / subdir
        if d.exists():
            for f in d.glob("*"):
                f.unlink(missing_ok=True)
                cleared += 1
    if cleared:
        st.info(f"🧹 已清理 {cleared} 个派生缓存文件（重采样/缠论）")
