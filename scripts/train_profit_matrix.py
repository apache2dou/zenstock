"""离线预计算病情矩阵概率表（文档 §9.3 第一部分）。

用法:
    # 用全部日线数据训练
    python scripts/train_profit_matrix.py --freq D

    # 用 5 分钟数据训练
    python scripts/train_profit_matrix.py --freq 5

    # 指定股票和输出路径
    python scripts/train_profit_matrix.py --symbols 000001,600519 --output data/profit_matrix.json

流程:
    1. 遍历所有指定股票的历史 K 线
    2. 对每根 K 线计算笔状态 + 趋势 + 背驰
    3. 计算持有 N 根后的收益率
    4. 记录到 ProfitMatrix
    5. 构建白名单
    6. 保存到 JSON 文件

回测时加载:
    strategy = BiStateStrategy(matrix_path="data/profit_matrix.json")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm

from zenstock.data.types import Freq


def compute_states_for_df(df: pd.DataFrame, freq_str: str) -> list[tuple]:
    """对单只股票的全部 K 线计算状态序列。

    Returns:
        [(state_value, trend, has_divergence, close_price), ...]
    """
    from czsc import CZSC
    from zenstock.chanlun.adapter import df_to_bars
    from zenstock.chanlun.bi_state import compute_bi_state
    from strategies.bi_state_strategy import compute_macd, detect_divergence

    freq_map = {"D": "D", "5": "F5", "30": "F30", "W": "W"}
    czsc_freq = freq_map.get(freq_str, "D")

    try:
        bars = df_to_bars(df, freq_str)
        if len(bars) < 30:
            return []
    except Exception:
        return []

    try:
        czsc_obj = CZSC(bars)
    except Exception:
        return []

    bi_list = list(czsc_obj.bi_list)
    fx_list = list(getattr(czsc_obj, "fx_list", []))
    if not bi_list:
        return []

    # 均线
    close = df["close"]
    ma60 = close.rolling(60, min_periods=1).mean()
    ma5 = close.rolling(5, min_periods=1).mean()

    # MACD
    _, _, macd_hist = compute_macd(close)

    # 对每根 K 线计算状态
    results = []
    for i in range(len(df)):
        # 找最后一笔（截止到第 i 根）
        # 简化：用全量 czsc 的最后一笔方向（近似，因为 czsc 不支持增量）
        # 实际应逐根增量更新，这里用全量结果映射到时间
        pass

    # 改用更简单的方法：直接用全量 czsc 结果
    # 把 czsc 的笔映射到时间轴上
    last_bi = bi_list[-1]
    bi_direction = getattr(last_bi, "direction", None)
    from zenstock.chanlun.bi_state import czsc_direction_is_up
    is_up = czsc_direction_is_up(bi_direction)

    fx_forming = False
    if fx_list:
        last_fx = fx_list[-1]
        fx_mark = getattr(last_fx, "mark", getattr(last_fx, "type", ""))
        from zenstock.chanlun.bi_state import czsc_mark_is_top, czsc_mark_is_bottom
        if (is_up and czsc_mark_is_top(fx_mark)) or (not is_up and czsc_mark_is_bottom(fx_mark)):
            fx_forming = True

    state = compute_bi_state("up" if is_up else "down", fx_forming)

    # 这种方法只能给出一个状态（全量的最后状态）
    # 对于训练矩阵，需要对每个历史时点计算状态
    # 用更实际的方法：分段映射 czsc 笔到 K 线
    return _compute_states_segmented(df, bi_list, fx_list, ma60, macd_hist)


def _compute_states_segmented(
    df: pd.DataFrame,
    bi_list: list,
    fx_list: list,
    ma60: pd.Series,
    macd_hist: pd.Series,
) -> list[tuple]:
    """把 czsc 的笔/分型映射到 K 线时间轴上，计算每根 K 线的状态。

    简化方法：用最后一笔方向 + 最后一分型判断是否在分型构造中。
    对每根 K 线，检查其时间是否在最后一笔/分型之后。
    """
    from zenstock.chanlun.bi_state import compute_bi_state, BiState
    from strategies.bi_state_strategy import detect_divergence

    close = df["close"]
    results = []

    if not bi_list:
        return []

    # 遍历每根 K 线，找到该时点对应的最后一笔
    for i in range(len(df)):
        dt = df["date"].iloc[i]

        # 找截止到该时点的最后一笔
        last_bi = None
        for bi in bi_list:
            bi_end_dt = getattr(getattr(bi, "fx_b", None), "dt", None)
            if bi_end_dt is not None and bi_end_dt <= dt:
                last_bi = bi
            else:
                break

        if last_bi is None:
            # 找第一笔
            last_bi = bi_list[0]

        bi_direction = getattr(last_bi, "direction", None)
        from zenstock.chanlun.bi_state import czsc_direction_is_up
        is_up = czsc_direction_is_up(bi_direction)

        # 检查该时点是否在分型构造中
        fx_forming = False
        from zenstock.chanlun.bi_state import czsc_mark_is_top, czsc_mark_is_bottom
        for fx in reversed(fx_list):
            fx_dt = getattr(fx, "dt", None)
            if fx_dt is not None and fx_dt <= dt:
                fx_mark = getattr(fx, "mark", getattr(fx, "type", ""))
                if (is_up and czsc_mark_is_top(fx_mark)) or (not is_up and czsc_mark_is_bottom(fx_mark)):
                    fx_forming = True
                break

        state = compute_bi_state("up" if is_up else "down", fx_forming)

        # 趋势
        price = float(close.iloc[i])
        ma_val = float(ma60.iloc[i]) if i < len(ma60) else 0
        if ma_val > 0:
            if price > ma_val * 1.01:
                trend = "up"
            elif price < ma_val * 0.99:
                trend = "down"
            else:
                trend = "sideways"
        else:
            trend = "sideways"

        # 背驰
        div = False
        if macd_hist is not None:
            d = detect_divergence(macd_hist, close, i)
            div = d is not None

        results.append((state, trend, div, price))

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="离线预计算病情矩阵概率表")
    parser.add_argument("--freq", default="D", choices=["D", "5", "30", "W"], help="K 线频率")
    parser.add_argument("--symbols", default=None, help="股票代码（逗号分隔），空=全部")
    parser.add_argument("--output", default="data/profit_matrix.json", help="输出路径")
    parser.add_argument("--limit", type=int, default=0, help="限制股票数（0=全部）")
    parser.add_argument("--trading-level", default="日线", choices=["日线", "30分钟"], help="交易策略级别")
    args = parser.parse_args(argv)

    from strategies.bi_state_strategy import ProbabilityMatrix

    # 找到数据文件
    freq_suffix = {"D": "D", "5": "5", "30": "30", "W": "W"}
    suffix = freq_suffix.get(args.freq, "D")
    parquet_dir = _PROJECT_ROOT / "data" / "parquet"

    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
        files = [parquet_dir / f"{s}_{suffix}.parquet" for s in symbols]
    else:
        files = sorted(parquet_dir.glob(f"*_{suffix}.parquet"))

    if args.limit > 0:
        files = files[: args.limit]

    print(f"训练概率矩阵: {len(files)} 只股票, freq={args.freq}, level={args.trading_level}")
    print(f"   输出: {args.output}")

    matrix = ProbabilityMatrix(min_samples=5, min_win_rate=0.55, top_n=30)

    total_records = 0
    for f in tqdm(files, desc="训练矩阵"):
        try:
            df = pd.read_parquet(f)
            if len(df) < 50:
                continue

            states_data = _compute_states_segmented_for_file(df, args.freq)
            if not states_data:
                continue

            # 按状态序列统计胜负（文档 §4.6）
            fx_count_map: dict[str, int] = {}
            for j in range(len(states_data) - 1):
                state, trend, div, price = states_data[j]
                next_state, _, _, _ = states_data[j + 1]

                if not state.is_fx_forming:  # 只统计分型态
                    fx_count_map = {}  # 延伸态重置
                    continue

                dir_key = str(state.direction)
                fx_count = fx_count_map.get(dir_key, 0) + 1
                fx_count_map[dir_key] = fx_count

                # 胜负判定
                if state.value == "(-1, 0)":  # 底分型
                    is_win = next_state.value == "(1, 1)"  # 后续出现向上笔=胜
                elif state.value == "(1, 0)":  # 顶分型
                    is_win = next_state.value == "(-1, 1)"  # 后续出现向下笔=卖出胜
                else:
                    continue

                matrix.record(state, state, fx_count, args.trading_level, is_win)
                total_records += 1

        except Exception as e:
            print(f"  {f.name}: {e}")
            continue

    wl = matrix.build_whitelist()
    matrix.save(args.output)

    print(f"\n{'='*50}")
    print(f"训练完成")
    print(f"   总记录: {total_records:,}")
    print(f"   组合数: {len(matrix._data)}")
    print(f"   白名单: {len(wl)} 个")
    print(f"   文件: {args.output}")
    print(f"\n   白名单组合:")
    for key in sorted(wl, key=lambda k: -matrix.lookup(*k).win_rate if matrix.lookup(*k) else 0):
        stats = matrix.lookup(*key)
        if stats:
            print(f"   {key[0].value} {key[1].value} fx={key[2]}次 {key[3]}: "
                  f"WR={stats.win_rate:.0%} n={stats.sample_size}")

    return 0


def _compute_states_segmented_for_file(df: pd.DataFrame, freq_str: str):
    """对单只股票计算每根 K 线的状态序列。"""
    from czsc import CZSC
    from zenstock.chanlun.adapter import df_to_bars
    from strategies.bi_state_strategy import compute_macd, detect_divergence
    from zenstock.chanlun.bi_state import compute_bi_state

    try:
        bars = df_to_bars(df, freq_str)
        if len(bars) < 30:
            return []
        czsc_obj = CZSC(bars)
    except Exception:
        return []

    bi_list = list(czsc_obj.bi_list)
    fx_list = list(getattr(czsc_obj, "fx_list", []))
    if not bi_list:
        return []

    close = df["close"]
    ma60 = close.rolling(60, min_periods=1).mean()
    _, _, macd_hist = compute_macd(close)

    return _compute_states_segmented(df, bi_list, fx_list, ma60, macd_hist)


if __name__ == "__main__":
    sys.exit(main())
