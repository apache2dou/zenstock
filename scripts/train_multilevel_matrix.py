"""离线训练多级别病情矩阵（5分钟/30分钟/日线/周线 四级联立）。

核心算法（文档 §9.3 第一部分）：
    1. 从 5 分钟数据重采样出 30 分钟/日线/周线
    2. 对每个级别用 czsc 计算笔状态（max_bi_num=len(bars)，避免默认50上限）
    3. 对每个时间点，构建四级状态组合 key
    4. 统计分型出现后是否形成向上笔（二分类胜负）
    5. 记录到 ProbabilityMatrix

支持多进程并行 + 股票级缓存（data/cache/matrix/）。

用法:
    # 用全部 5 分钟数据训练（自动多进程 + 缓存）
    python scripts/train_multilevel_matrix.py

    # 指定股票数和输出路径
    python scripts/train_multilevel_matrix.py --limit 100 --output data/profit_matrix_multi.json

    # 强制重新计算（忽略缓存）
    python scripts/train_multilevel_matrix.py --no-cache

    # 指定进程数
    python scripts/train_multilevel_matrix.py --workers 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
from tqdm import tqdm


def compute_level_states(df: pd.DataFrame, freq_str: str) -> list[tuple[int, str, bool]]:
    """对单个级别的 K 线计算每根的状态序列。

    Returns:
        [(bar_index, direction, fx_forming), ...]
        direction: "up" / "down"
        fx_forming: True / False
    """
    from czsc import CZSC
    from zenstock.chanlun.adapter import df_to_bars

    try:
        bars = df_to_bars(df, freq_str)
        if len(bars) < 30:
            return []
        czsc_obj = CZSC(bars, max_bi_num=len(bars))
    except Exception:
        return []

    # ``fx_list`` is the final, full-history fractal list.  Projecting it back
    # onto old bars makes a fractal that is confirmed much later appear to be
    # forming for the whole history.  Build CZSC incrementally instead and use
    # ``ubi_fxs`` (the unfinished fractals at the current bar).
    results: list[tuple[int, str, bool]] = []
    running = None
    for i, bar in enumerate(bars):
        try:
            if running is None:
                running = CZSC([bar], max_bi_num=len(bars))
            else:
                running.update(bar)
        except Exception:
            # A malformed bar should not invalidate all later observations.
            results.append((i, results[-1][1], results[-1][2]) if results else (i, "up", False))
            continue

        bi_list = list(running.bi_list)
        if not bi_list:
            results.append((i, "up", False))
            continue

        direction = getattr(bi_list[-1], "direction", None)
        # 用 codepoint 辅助函数可靠判断（避免 Windows 源码编码问题）
        from zenstock.chanlun.bi_state import (
            czsc_direction_is_up, czsc_mark_is_top, czsc_mark_is_bottom,
        )
        is_up = czsc_direction_is_up(direction)

        # ubi_fxs contains only the currently unfinished fractal candidates.
        fx_forming = False
        ubi_fxs = list(getattr(running, "ubi_fxs", []))
        if ubi_fxs:
            mark = getattr(ubi_fxs[-1], "mark", "")
            if (is_up and czsc_mark_is_top(mark)) or (not is_up and czsc_mark_is_bottom(mark)):
                fx_forming = True

        results.append((i, "up" if is_up else "down", fx_forming))

    return results


def state_tuple(direction: str, fx_forming: bool) -> str:
    """方向+分型 → 状态字符串 "(1,1)" 等。"""
    if direction == "up":
        return "(1, 0)" if fx_forming else "(1, 1)"
    else:
        return "(-1, 0)" if fx_forming else "(-1, 1)"


# 胜负判定窗口：分型态结束后，扫描后续 N 根 K 线找延伸态
WIN_LOOKAHEAD = 30


def process_single_stock(
    parquet_path: str,
    trading_level: str,
) -> list[dict]:
    """处理单只股票，返回记录列表。

    每条记录 = {big_state, small_state, fx_count, trading_level, is_win}

    此函数设计为可在子进程中独立运行（无共享状态）。
    """
    import bisect
    from zenstock.chanlun.bi_state import BiState, bucket_fx_count
    from zenstock.data.resample import resample_klines
    from zenstock.data.types import Freq

    state_map = {s.value: s for s in BiState}

    f = Path(parquet_path)
    symbol = f.stem.replace("_5", "")

    try:
        df_5 = pd.read_parquet(f)
        if len(df_5) < 500:
            return []

        # 重采样出更高级别
        df_30 = resample_klines(df_5, Freq.MIN30)
        df_d = resample_klines(df_5, Freq.DAILY)
        df_w = resample_klines(df_5, Freq.WEEKLY)

        if len(df_30) < 50:
            df_30 = None
        if len(df_d) < 30:
            df_d = None
        if len(df_w) < 20:
            df_w = None

        # 计算各级别状态序列（max_bi_num=len 保证处理全部K线）
        states_5 = compute_level_states(df_5, "5")
        states_30 = compute_level_states(df_30, "30") if df_30 is not None else []
        states_d = compute_level_states(df_d, "D") if df_d is not None else []
        states_w = compute_level_states(df_w, "W") if df_w is not None else []

        if not states_5:
            return []

        # 高级别日期列表用于二分查找
        dates_30 = df_30["date"].tolist() if df_30 is not None else []
        dates_d = df_d["date"].tolist() if df_d is not None else []
        dates_w = df_w["date"].tolist() if df_w is not None else []

        def find_higher_state(higher_dates, higher_states, dt):
            """在高级别中找到覆盖该时间点的状态。"""
            if not higher_dates or not higher_states:
                return None
            idx = bisect.bisect_right(higher_dates, dt) - 1
            if idx < 0 or idx >= len(higher_states):
                return None
            _, direction, fx_forming = higher_states[idx]
            return state_tuple(direction, fx_forming)

        # 交易级别决定信号和胜负的观察级别；不能用 5 分钟状态冒充日线/30分钟。
        if trading_level == "日线":
            signal_states = states_d
            signal_dates = dates_d
        else:
            signal_states = states_30
            signal_dates = dates_30
        if not signal_states or not signal_dates:
            return []

        # fx_count 追踪：在方向反转前，同方向分型出现的次数
        fx_count_dir = 0
        fx_count_cur_direction = 0
        prev_signal_bi: BiState | None = None

        records: list[dict] = []
        dt_list = signal_dates

        for i in range(len(signal_states) - 1):
            dt = dt_list[i]

            # 交易级别状态
            _, signal_dir, signal_fx = signal_states[i]
            signal_value = state_tuple(signal_dir, signal_fx)
            signal_bi = state_map.get(signal_value)

            if signal_bi is None:
                continue

            # 只在分型态统计，且只在状态刚转换到分型态时记录一次
            if not signal_bi.is_fx_forming:
                prev_signal_bi = signal_bi
                continue
            if signal_bi == prev_signal_bi:
                continue

            # 分型出现次数：方向反转时重置
            cur_dir = signal_bi.direction
            if cur_dir != fx_count_cur_direction:
                fx_count_dir = 1
                fx_count_cur_direction = cur_dir
            else:
                fx_count_dir += 1
            fx_count = bucket_fx_count(fx_count_dir)

            # 高级别状态
            s30 = find_higher_state(dates_30, states_30, dt)
            sd = find_higher_state(dates_d, states_d, dt)
            sw = find_higher_state(dates_w, states_w, dt)

            if trading_level == "日线":
                # 日线策略：周线为大级别，日线为交易/小级别。
                big_state = state_map.get(sw, state_map.get(sd))
                # signal_bi 已经是当前交易级别（日线）的状态。
                # 不要再按时间戳回查 states_d；重采样边界和分型确认时间
                # 可能错位，导致底分型被记录成顶分型。
                small_state = signal_bi
            else:
                # 30分钟策略：周线为大级别，30分钟为交易/小级别。
                big_state = state_map.get(sw, state_map.get(sd, state_map.get(signal_value)))
                # signal_bi 已经是当前交易级别（30分钟）的状态。
                small_state = signal_bi

            if big_state is None or small_state is None:
                prev_signal_bi = signal_bi
                continue

            # 胜负判定：从当前分型态向后扫描，找第一个延伸态
            outcome_state: BiState | None = None
            for k in range(i + 1, min(i + 1 + WIN_LOOKAHEAD, len(signal_states))):
                _, dir_k, fx_k = signal_states[k]
                s_k = state_tuple(dir_k, fx_k)
                s_k_bi = state_map.get(s_k)
                if s_k_bi is not None and s_k_bi.is_extending:
                    outcome_state = s_k_bi
                    break

            if outcome_state is None:
                prev_signal_bi = signal_bi
                continue

            # 胜负判定（文档 §4.6 对称设计）：
            #   买入信号（底分型 -1,0）：后续出现向上笔 = 胜
            #   卖出信号（顶分型 1,0）：后续出现向下笔 = 胜
            if signal_bi == BiState.DOWN_FX_FORMING:
                direction = "buy"
                is_win = outcome_state == BiState.UP_EXTENDING
            elif signal_bi == BiState.UP_FX_FORMING:
                direction = "sell"
                is_win = outcome_state == BiState.DOWN_EXTENDING
            else:
                prev_signal_bi = signal_bi
                continue

            records.append({
                "symbol": symbol,
                "big_state": big_state.value,
                "small_state": small_state.value,
                "fx_count": fx_count,
                "trading_level": trading_level,
                "direction": direction,
                "is_win": is_win,
            })
            prev_signal_bi = signal_bi

        return records

    except Exception:
        return []


def get_cache_path(symbol: str, trading_level: str = "30分钟") -> Path:
    """获取单只股票的缓存路径。"""
    level_tag = "30m" if trading_level == "30分钟" else "daily"
    return _PROJECT_ROOT / "data" / "cache" / "matrix" / f"{symbol}_{level_tag}.parquet"


def load_cache(symbol: str, trading_level: str = "30分钟") -> list[dict] | None:
    """从缓存加载单只股票的记录。"""
    cache_file = get_cache_path(symbol, trading_level)
    if not cache_file.exists():
        return None
    try:
        df = pd.read_parquet(cache_file)
        return df.to_dict("records")
    except Exception:
        return None


def save_cache(symbol: str, records: list[dict], trading_level: str = "30分钟") -> None:
    """保存单只股票的记录到缓存。"""
    cache_file = get_cache_path(symbol, trading_level)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(cache_file, index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="训练多级别病情矩阵（多进程 + 缓存）")
    parser.add_argument("--limit", type=int, default=0, help="限制股票数（0=全部）")
    parser.add_argument("--output", default="data/profit_matrix_multi.json", help="输出路径")
    parser.add_argument("--min-samples", type=int, default=5)
    parser.add_argument("--trading-level", default="30分钟", choices=["30分钟", "日线"], help="交易策略级别")
    parser.add_argument("--workers", type=int, default=0, help="进程数（0=CPU核心数）")
    parser.add_argument("--no-cache", action="store_true", help="忽略缓存，重新计算")
    args = parser.parse_args(argv)

    from strategies.bi_state_strategy import ProbabilityMatrix

    parquet_dir = _PROJECT_ROOT / "data" / "parquet"
    files = sorted(parquet_dir.glob("*_5.parquet"))
    if args.limit > 0:
        files = files[: args.limit]

    num_workers = args.workers if args.workers > 0 else (os.cpu_count() or 4)

    print(f"训练多级别概率矩阵: {len(files)} 只股票, level={args.trading_level}")
    print(f"进程数: {num_workers}, 缓存: {'禁用' if args.no_cache else '启用'}")
    print(f"输出: {args.output}")

    # 第一阶段：收集每只股票的记录（带缓存）
    all_records: list[dict] = []
    tasks_to_compute: list[tuple[str, str]] = []  # (parquet_path, symbol)

    # 先尝试从缓存加载
    for f in files:
        symbol = f.stem.replace("_5", "")
        if not args.no_cache:
            cached = load_cache(symbol, args.trading_level)
            if cached is not None:
                all_records.extend(cached)
                continue
        tasks_to_compute.append((str(f), symbol))

    cached_count = len(all_records)
    print(f"缓存命中: {len(files) - len(tasks_to_compute)} 只, "
          f"待计算: {len(tasks_to_compute)} 只, "
          f"缓存记录: {cached_count:,} 条")

    # 第二阶段：多进程处理未缓存的股票
    if tasks_to_compute:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(process_single_stock, path, args.trading_level): symbol
                for path, symbol in tasks_to_compute
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="计算状态序列",
            ):
                symbol = futures[future]
                try:
                    records = future.result()
                    all_records.extend(records)
                    # 保存到缓存
                    if not args.no_cache and records:
                        save_cache(symbol, records, args.trading_level)
                except Exception:
                    pass

    # 第三阶段：汇总到 ProbabilityMatrix
    from zenstock.chanlun.bi_state import BiState
    state_map = {s.value: s for s in BiState}

    matrix = ProbabilityMatrix(
        min_samples=args.min_samples,
        min_win_rate=0.45,
        top_n=30,
    )

    for rec in all_records:
        big_s = state_map.get(rec["big_state"])
        small_s = state_map.get(rec["small_state"])
        if big_s is None or small_s is None:
            continue
        matrix.record(
            big_state=big_s,
            small_state=small_s,
            fx_count=rec["fx_count"],
            trading_level=rec["trading_level"],
            direction=rec.get("direction", ""),
            is_win=rec["is_win"],
        )

    # 构建白名单
    wl = matrix.build_whitelist()

    # 保存
    matrix.save(args.output)

    print(f"\n{'='*60}")
    print(f"训练完成")
    print(f"   总记录: {len(all_records):,}")
    print(f"   组合数: {len(matrix._data)}")
    print(f"   白名单: {len(wl)} 个")
    print(f"   文件: {args.output}")

    # 打印白名单 Top-10（按方向分组）
    print(f"\n   白名单 Top-10:")
    sorted_wl = sorted(wl, key=lambda k: -matrix.lookup(*k).win_rate if matrix.lookup(*k) else 0)
    for key in sorted_wl[:10]:
        stats = matrix.lookup(*key)
        if stats:
            dir_label = "买入" if key[4] == "buy" else "卖出" if key[4] == "sell" else key[4]
            print(f"   [{dir_label}] 周/日={key[0].value} 30分/5分={key[1].value} "
                  f"fx={key[2]}次 {key[3]}: "
                  f"WR={stats.win_rate:.0%} n={stats.sample_size}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
