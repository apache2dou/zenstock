"""Tab: 缠论两重表里关系分析（第91-92课）。

展示内容：
1. 笔状态时间线 — 在 K 线图上标注每根 K 线的 (d,s) 状态
2. 状态转移概率矩阵 — 历史统计的各状态后续涨跌概率
3. 策略回测 — 概率阈值可调，实时对比不同阈值的效果
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保 strategies 包可导入
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

from zenstock.backtest.engine import Backtest
from zenstock.chanlun.bi_state import BiState
from zenstock.stock_names import get_stock_name
from strategies.bi_state_strategy import compute_state_signal_probability


def render(data: pd.DataFrame, symbol: str, freq) -> None:
    """渲染两重表里关系分析页。"""
    name = get_stock_name(symbol)
    display = f"{symbol} {name}" if name else symbol

    st.subheader(f"🔬 两重表里关系分析 — {display}")

    st.caption(
        "基于缠论第91-92课：将走势映射为四态 `(1,1)/(1,0)/(-1,1)/(-1,0)`，"
        "统计历史转移概率，按概率阈值交易。"
    )

    if len(data) < 30:
        st.warning("数据不足 30 根，无法分析")
        return

    # 分析按钮
    col1, col2 = st.columns([3, 1])
    with col2:
        run_btn = st.button("🔬 开始分析", type="primary", width="stretch")

    if not run_btn:
        st.info("👆 点击「开始分析」，计算笔状态时间线和概率统计")
        _render_explanation()
        # 即使不点分析，也可以训练矩阵
        _render_matrix_training()
        return

    # ===== 执行分析 =====
    with st.spinner("正在计算笔状态（增量 czsc 分析，约 5-15 秒）..."):
        try:
            from strategies.bi_state_strategy import (
                BiStateStrategy,
                compute_state_signal_probability,
            )

            # 用低阈值跑一次，收集完整的状态序列和统计
            collector = BiStateStrategy(warmup_bars=10, buy_threshold=0.0, sell_threshold=0.0)
            bt = Backtest(collector, data, symbol=symbol)
            bt.run()

            states_series = collector._bi_states
            stats = collector._historical_stats

        except Exception as e:
            st.error(f"分析失败: {e}")
            import traceback
            st.code(traceback.format_exc())
            return

    if not states_series:
        st.warning("未能识别出笔状态")
        return

    # ===== 1. 总览指标 =====
    _render_overview(data, states_series, stats)

    # ===== 2. 笔状态时间线 =====
    _render_state_timeline(data, states_series, display)

    # ===== 3. 概率矩阵 =====
    _render_probability_matrix(stats)

    # ===== 4. 阈值对比回测 =====
    _render_threshold_backtest(data, symbol)

    # ===== 5. 多级别病情矩阵预统计 =====
    _render_matrix_training()


# ==================== 说明面板 ====================

def _render_explanation() -> None:
    with st.expander("📖 两重表里关系算法说明"):
        st.markdown("""
**笔定理**：任何当下走势必然落在某个具有明确方向的笔中。

**四态定义** `(方向 d, 阶段 s)`：

| 状态 | 含义 | 交易含义 |
|------|------|----------|
| `(1, 1)` 向上延伸 | 上涨趋势进行中 | 持有 |
| `(1, 0)` 顶分型构造 | 上涨出现顶部信号 | 可能卖出 |
| `(-1, 1)` 向下延伸 | 下跌趋势进行中 | 观望 |
| `(-1, 0)` 底分型构造 | 下跌出现底部信号 | 可能买入 |

**状态转移规则**（不能随便连接）：
```
(1, 1)  → 只能 → (1, 0)
(-1, 1) → 只能 → (-1, 0)
(1, 0)  → 两种 → (1, 1) 或 (-1, 1)
(-1, 0) → 两种 → (-1, 1) 或 (1, 1)
```

**概率策略**：统计每个状态历史转移的涨跌概率，
- P(上涨) > 买入阈值 → 买入
- P(下跌) > 卖出阈值 → 卖出
""")


# ==================== 总览指标 ====================

def _render_overview(data, states_series, stats) -> None:
    from collections import Counter
    state_counts = Counter(states_series)
    total = len(states_series)

    cols = st.columns(5)
    cols[0].metric("K 线数", f"{len(data)}")
    cols[1].metric("状态记录", f"{total}")
    cols[2].metric("转移种类", f"{len(stats.transition_counts)}")
    cols[3].metric("当前状态", states_series[-1].value if states_series else "-")

    # 计算最后状态的信号
    if states_series:
        prob = compute_state_signal_probability(states_series[-1], stats)
        signal_emoji = {"BUY": "🟢买入", "SELL": "🔴卖出", "HOLD": "⚪持有"}.get(
            prob.signal, "?"
        )
        cols[4].metric("概率信号", signal_emoji)

    # 各状态分布
    st.markdown("**状态分布：**")
    dist_cols = st.columns(4)
    state_labels = {
        BiState.UP_EXTENDING: "📈 (1,1) 向上延伸",
        BiState.UP_FX_FORMING: "⚠️ (1,0) 顶分型",
        BiState.DOWN_EXTENDING: "📉 (-1,1) 向下延伸",
        BiState.DOWN_FX_FORMING: "🔍 (-1,0) 底分型",
    }
    state_colors = {
        BiState.UP_EXTENDING: "#2ecc71",
        BiState.UP_FX_FORMING: "#f39c12",
        BiState.DOWN_EXTENDING: "#e74c3c",
        BiState.DOWN_FX_FORMING: "#3498db",
    }
    for i, state in enumerate([BiState.UP_EXTENDING, BiState.UP_FX_FORMING,
                                BiState.DOWN_EXTENDING, BiState.DOWN_FX_FORMING]):
        count = state_counts.get(state, 0)
        pct = count / total * 100 if total > 0 else 0
        dist_cols[i].metric(state_labels[state], f"{count} ({pct:.0f}%)")


# ==================== 状态时间线 ====================

def _render_state_timeline(data, states_series, display) -> None:
    st.subheader("📊 笔状态时间线")

    # 将状态序列对齐到 K 线（前 10 根无状态）
    offset = len(data) - len(states_series)

    # 构建带状态的 DataFrame（只取有状态的部分）
    timeline_data = data.iloc[offset:].copy() if offset > 0 else data.copy()
    if len(timeline_data) > len(states_series):
        timeline_data = timeline_data.iloc[: len(states_series)].copy()
    timeline_data["state"] = [s.value for s in states_series[: len(timeline_data)]]
    timeline_data["direction"] = [s.direction for s in states_series[: len(timeline_data)]]

    # 状态变化点（买卖信号参考）
    state_changes = []
    for i in range(1, len(states_series)):
        if states_series[i] != states_series[i - 1]:
            state_changes.append({
                "日期": str(timeline_data.iloc[i]["date"])[:10] if i < len(timeline_data) else "?",
                "从": states_series[i - 1].value,
                "到": states_series[i].value,
                "收盘价": float(timeline_data.iloc[i]["close"]) if i < len(timeline_data) else 0,
            })

    st.caption(f"💡 共 {len(state_changes)} 次状态转移")

    if state_changes:
        changes_df = pd.DataFrame(state_changes[-30:])  # 最近30次
        st.dataframe(changes_df, width="stretch", hide_index=True)

    # K 线图上用背景色标注状态区间
    from frontend.lwc_chart import render_kline_chart

    # 构建买卖点标记（底分型→买，顶分型→卖）
    bi_list = _extract_czsc_bi(data)

    render_kline_chart(
        data=data,
        symbol=display,
        title=f"{display} 笔状态时间线",
        bi_list=bi_list,
        height=500,
    )


def _extract_czsc_bi(data):
    """从 czsc 提取笔列表用于绘图。"""
    try:
        from zenstock.chanlun.adapter import df_to_bars
        from zenstock.data.types import Freq
        from czsc import CZSC

        freq_map = {"日线": "D", "5 分钟": "F5", "30 分钟": "F30"}
        freq_str = getattr(data, "attrs", {}).get("freq", "D")
        bars = df_to_bars(data, freq_str)
        if len(bars) >= 10:
            czsc_obj = CZSC(bars)
            return list(czsc_obj.bi_list)
    except Exception:
        return []
    return []


# ==================== 概率矩阵 ====================

def _render_probability_matrix(stats) -> None:
    from strategies.bi_state_strategy import compute_state_signal_probability

    st.subheader("🎲 状态转移概率矩阵")

    st.caption("每个状态后续转向『向上』和『向下』的历史概率，用于指导交易决策。")

    rows = []
    for from_state in [BiState.UP_EXTENDING, BiState.UP_FX_FORMING,
                       BiState.DOWN_EXTENDING, BiState.DOWN_FX_FORMING]:
        total = stats.total_from(from_state)
        if total == 0:
            continue

        prob = compute_state_signal_probability(from_state, stats, 0.55, 0.55)

        # 各转移目标计数
        transitions = {}
        for to_state in BiState:
            count = stats.transition_counts.get((from_state, to_state), 0)
            if count > 0:
                transitions[f"→{to_state.value}"] = count

        rows.append({
            "当前状态": from_state.value,
            "样本数": total,
            "P(上涨)": f"{prob.up_probability:.1%}",
            "P(下跌)": f"{prob.down_probability:.1%}",
            "信号": {"BUY": "🟢买入", "SELL": "🔴卖出", "HOLD": "⚪持有"}.get(prob.signal, "-"),
            "转移详情": " ".join(f"{k}={v}" for k, v in transitions.items()),
        })

    if rows:
        prob_df = pd.DataFrame(rows)
        st.dataframe(prob_df, width="stretch", hide_index=True)
    else:
        st.info("统计样本不足")


# ==================== 阈值对比回测 ====================

def _render_threshold_backtest(data, symbol) -> None:
    st.subheader("⚖️ 概率阈值对比回测")
    st.caption("不同买入/卖出概率阈值下的回测表现对比。")

    col1, col2 = st.columns(2)
    with col1:
        warmup = st.slider("预热期(K线)", 30, 250, 120, 10, key="bi_warmup")
    with col2:
        run_bt = st.button("🔄 运行对比回测", key="bi_run_bt")

    if not run_bt:
        st.info("点击「运行对比回测」查看不同阈值的效果")
        return

    thresholds = [0.5, 0.6, 0.7, 0.8]

    results = []
    progress = st.progress(0, "正在回测各阈值...")
    for idx, th in enumerate(thresholds):
        try:
            from strategies.bi_state_strategy import BiStateStrategy
            strategy = BiStateStrategy(warmup_bars=warmup, buy_threshold=th, sell_threshold=th)
            bt = Backtest(strategy, data, symbol=symbol)
            result = bt.run()
            s = result.to_summary()
            results.append({
                "阈值": f"({th}, {th})",
                "交易次数": s.get("total_trades", 0),
                "总收益%": s.get("total_return_pct", 0),
                "胜率%": s.get("win_rate", 0),
                "赔率": s.get("profit_loss_ratio", 0),
                "最大回撤%": s.get("max_drawdown_pct", 0),
            })
        except Exception as e:
            results.append({"阈值": f"({th}, {th})", "交易次数": 0, "总收益%": 0,
                            "胜率%": 0, "赔率": 0, "最大回撤%": 0})
        progress.progress((idx + 1) / len(thresholds))

    progress.empty()

    if results:
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, width="stretch", hide_index=True)

        # 高亮最佳行
        best = max(results, key=lambda r: r["总收益%"])
        st.success(
            f"最佳阈值 **{best['阈值']}**："
            f"收益 {best['总收益%']:.2f}%，胜率 {best['胜率%']:.1f}%"
        )


# ==================== 多级别病情矩阵预统计 ====================

def _render_matrix_training() -> None:
    """病情矩阵预统计面板：训练多级别矩阵并展示白名单。"""
    st.subheader("🧮 多级别病情矩阵预统计")
    st.caption(
        "从 5 分钟数据重采样出 30 分钟/日线/周线，对四级联立状态统计历史收益分布，"
        "筛选出最大可能盈利的转折状态白名单。"
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        limit = st.slider("股票数量(0=全部)", 0, 500, 50, 10, key="mt_limit")
    with col2:
        run_train = st.button("训练矩阵", type="primary", key="mt_run")

    # 检查已有矩阵和缓存
    from pathlib import Path as _Path
    import json as _json
    existing = sorted(_Path("data").glob("profit_matrix*.json"))
    cache_dir = _Path("data/cache/matrix")
    cache_count = len(list(cache_dir.glob("*.parquet"))) if cache_dir.exists() else 0
    if existing:
        st.caption(f"已有矩阵: {', '.join(f.name for f in existing)} | 缓存: {cache_count} 只股票")

    if not run_train:
        st.info("多进程并行训练 + 自动缓存，统计分型→向上笔胜率")

        # 如果已有缓存矩阵，直接展示（优先 v3 格式）
        if existing:
            # 找到最新的 v3 格式矩阵（含方向维度）
            v3_matrix_path = None
            for mf in reversed(existing):
                try:
                    with open(mf) as _f:
                        _meta = _json.load(_f)
                    if _meta.get("version") == 3:
                        v3_matrix_path = mf
                        break
                except Exception:
                    continue

            if v3_matrix_path:
                try:
                    from strategies.bi_state_strategy import ProbabilityMatrix
                    matrix = ProbabilityMatrix.load(str(v3_matrix_path))
                    combos = matrix.all_combos()
                    wl = matrix.whitelist

                    st.markdown(f"#### 当前缓存矩阵: `{v3_matrix_path.name}`")
                    st.caption(
                        f"{len(combos)} 种状态组合 | {len(wl)} 个白名单 | "
                        f"{sum(d.sample_size for _, d in combos):,} 条样本"
                    )

                    if wl:
                        wl_rows = []
                        for key in sorted(wl, key=lambda k: -matrix.lookup(*k).win_rate if matrix.lookup(*k) else 0):
                            stats = matrix.lookup(*key)
                            if stats:
                                big_s, small_s, fx_count, level, direction = key
                                wl_rows.append({
                                    "方向": "买入" if direction == "buy" else "卖出",
                                    "大级别(周/日)": big_s.value,
                                    "小级别(30分/5分)": small_s.value,
                                    "分型次数": fx_count,
                                    "策略级别": level,
                                    "胜率": f"{stats.win_rate:.0%}",
                                    "样本数": stats.sample_size,
                                })
                        if wl_rows:
                            st.dataframe(pd.DataFrame(wl_rows), width="stretch", hide_index=True)
                    else:
                        st.warning("白名单为空（样本不足或胜率不达标）")

                    with st.expander(f"全部状态组合（{len(combos)} 种）"):
                        all_rows = []
                        for key, stats in sorted(combos, key=lambda x: -x[1].win_rate):
                            all_rows.append({
                                "方向": "买入" if key[4] == "buy" else "卖出",
                                "大级别": key[0].value,
                                "小级别": key[1].value,
                                "分型次数": key[2],
                                "策略级别": key[3],
                                "胜率": f"{stats.win_rate:.0%}",
                                "样本": stats.sample_size,
                            })
                        st.dataframe(pd.DataFrame(all_rows), width="stretch", hide_index=True)

                except Exception as e:
                    st.error(f"加载缓存矩阵失败: {e}")
            else:
                st.warning("已有旧格式矩阵文件，请重新训练以生成 v3 格式（含买卖方向）")

        return

    # 执行训练
    import multiprocessing as _mp
    workers = _mp.cpu_count()
    with st.spinner(f"训练中（{workers} 进程并行，首次较慢，后续走缓存）..."):
        try:
            import subprocess
            cmd = [
                str(_PROJECT_ROOT / ".venv/Scripts/python.exe"),
                str(_PROJECT_ROOT / "scripts/train_multilevel_matrix.py"),
                "--output", "data/profit_matrix_multi.json",
                "--workers", str(workers),
            ]
            if limit > 0:
                cmd.extend(["--limit", str(limit)])

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=7200,
                cwd=str(_PROJECT_ROOT),
            )
            output = result.stdout + result.stderr

            if result.returncode != 0:
                st.error(f"训练失败: {output[-500:]}")
                return

        except Exception as e:
            st.error(f"训练异常: {e}")
            return

    # 加载并展示训练结果
    try:
        from strategies.bi_state_strategy import ProbabilityMatrix
        matrix = ProbabilityMatrix.load("data/profit_matrix_multi.json")

        # 统计
        combos = matrix.all_combos()
        wl = matrix.whitelist

        st.success(
            f"训练完成: {len(combos)} 种状态组合, {len(wl)} 个白名单, "
            f"{sum(d.sample_size for _, d in combos):,} 条样本"
        )

        # 白名单详情表
        st.markdown("### 白名单（高胜率转折状态）")

        wl_rows = []
        for key in sorted(wl, key=lambda k: -matrix.lookup(*k).win_rate if matrix.lookup(*k) else 0):
            stats = matrix.lookup(*key)
            if stats:
                big_s, small_s, fx_count, level, direction = key
                wl_rows.append({
                    "方向": "买入" if direction == "buy" else "卖出",
                    "大级别(周/日)": big_s.value,
                    "小级别(30分/5分)": small_s.value,
                    "分型次数": fx_count,
                    "策略级别": level,
                    "胜率": f"{stats.win_rate:.0%}",
                    "样本数": stats.sample_size,
                })

        if wl_rows:
            wl_df = pd.DataFrame(wl_rows)
            st.dataframe(wl_df, width="stretch", hide_index=True)
        else:
            st.warning("白名单为空（样本不足或胜率不达标）")

        # 所有组合概览
        with st.expander(f"全部状态组合（{len(combos)} 种）"):
            all_rows = []
            for key, stats in sorted(combos, key=lambda x: -x[1].win_rate):
                all_rows.append({
                    "方向": "买入" if key[4] == "buy" else "卖出",
                    "大级别": key[0].value,
                    "小级别": key[1].value,
                    "分型次数": key[2],
                    "策略级别": key[3],
                    "胜率": f"{stats.win_rate:.0%}",
                    "样本": stats.sample_size,
                })
            st.dataframe(pd.DataFrame(all_rows), width="stretch", hide_index=True)

    except Exception as e:
        st.error(f"加载矩阵失败: {e}")
