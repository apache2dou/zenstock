"""参数网格寻优工具。

对给定策略的参数空间做穷举搜索，找出历史表现最优的参数组合。
支持并行加速和按多个指标排序。

用法:
    python scripts/optimize.py --symbol 000001 --strategy ma_cross

    # 自定义参数范围
    python scripts/optimize.py --symbol 000001 --strategy ma_cross \\
        --param fast 3 5 10 15 --param slow 10 20 30 60
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import pandas as pd
from rich.console import Console
from rich.table import Table

from zenstock.analytics import compute_metrics
from zenstock.backtest import run_backtest
from zenstock.data import DataStorage
from zenstock.data.types import Freq
from zenstock.logger import setup_logging

console = Console()

# 策略注册表
STRATEGY_REGISTRY = {
    "ma_cross": ("strategies.ma_cross", "MACrossStrategy"),
    "rsi": ("strategies.rsi", "RSIStrategy"),
    "bollinger": ("strategies.bollinger", "BollingerStrategy"),
}

# 各策略的默认参数网格
DEFAULT_GRIDS: dict[str, dict[str, list[Any]]] = {
    "ma_cross": {"fast": [3, 5, 10, 15], "slow": [10, 20, 30, 60]},
    "rsi": {"period": [7, 14, 21], "oversold": [20, 25, 30], "overbought": [70, 75, 80]},
    "bollinger": {"period": [10, 20, 30], "num_std": [1.5, 2.0, 2.5]},
}


def load_strategy_class(name: str):
    """根据名称加载策略类。"""
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"未知策略: {name}，可选: {list(STRATEGY_REGISTRY.keys())}")
    module_path, class_name = STRATEGY_REGISTRY[name]
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def build_param_grid(
    param_grid: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    """把参数字典展开成所有组合的笛卡尔积。"""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def run_grid_search(
    strategy_name: str,
    data: pd.DataFrame,
    symbol: str,
    param_grid: dict[str, list[Any]],
    sort_by: str = "sharpe_ratio",
) -> pd.DataFrame:
    """执行网格搜索，返回按 sort_by 排序的结果表。"""
    cls = load_strategy_class(strategy_name)
    combos = build_param_grid(param_grid)
    console.print(f"🔍 网格搜索: {len(combos)} 个参数组合")

    results: list[dict[str, Any]] = []
    for idx, params in enumerate(combos, 1):
        try:
            strategy = cls(**params)
            result = run_backtest(strategy, data, symbol=symbol)
            metrics = compute_metrics(result)
            # 合并参数和指标
            row = {**params, **metrics}
            results.append(row)
            if idx % 10 == 0 or idx == len(combos):
                console.print(f"   进度: {idx}/{len(combos)}", end="\r")
        except Exception as e:  # noqa: BLE001
            console.print(f"   [red]组合 {params} 失败: {e}[/]")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    # 过滤掉无交易的无效组合（如 fast>=slow 导致无信号）
    df = df[df["total_trades"] > 0].reset_index(drop=True)
    # 按 sort_by 降序（夏普/收益率越高越好）
    df = df.sort_values(sort_by, ascending=False).reset_index(drop=True)
    console.print(f"✅ 完成: {len(df)} 个有效结果，按 {sort_by} 排序")
    return df


def display_results(df: pd.DataFrame, strategy_name: str, top_n: int = 10) -> None:
    """以 rich 表格展示 Top N 参数组合。"""
    if df.empty:
        console.print("[red]无有效结果[/]")
        return

    # 确定参数列（排除指标列）
    metric_keys = {
        "total_trades", "win_rate", "profit_loss_ratio", "expectancy",
        "avg_holding_days", "initial_capital", "final_capital",
        "total_return_pct", "annual_return_pct", "max_drawdown_pct",
        "max_dd_peak_date", "max_dd_trough_date", "sharpe_ratio",
        "calmar_ratio", "is_positive_expectancy",
    }
    param_cols = [c for c in df.columns if c not in metric_keys]

    table = Table(
        title=f"🏆 {strategy_name} 参数寻优 Top {top_n}",
        border_style="cyan",
    )
    table.add_column("排名", style="bold cyan", justify="right")
    for c in param_cols:
        table.add_column(str(c), justify="center")
    table.add_column("胜率%", justify="right")
    table.add_column("赔率", justify="right")
    table.add_column("总收益%", justify="right")
    table.add_column("年化%", justify="right")
    table.add_column("回撤%", justify="right")
    table.add_column("夏普", justify="right")
    table.add_column("交易数", justify="right")
    table.add_column("评估")

    for i, row in df.head(top_n).iterrows():
        flag = "✅" if row.get("is_positive_expectancy") else "⚠️"
        table.add_row(
            str(i + 1),
            *[str(row[c]) for c in param_cols],
            f"{row['win_rate']:.1f}",
            f"{row['profit_loss_ratio']:.2f}",
            f"{row['total_return_pct']:.1f}",
            f"{row['annual_return_pct']:.1f}",
            f"{row['max_drawdown_pct']:.1f}",
            f"{row['sharpe_ratio']:.2f}",
            str(int(row['total_trades'])),
            flag,
        )

    console.print(table)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ZenStock 参数网格寻优")
    parser.add_argument("--symbol", required=True, help="股票代码")
    parser.add_argument(
        "--strategy", default="ma_cross",
        choices=list(STRATEGY_REGISTRY.keys()),
        help="策略名称",
    )
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--param", action="append", nargs="+", metavar=("NAME", "VALUE"),
        help="自定义参数网格，可多次使用。如 --param fast 3 5 10",
    )
    parser.add_argument(
        "--sort-by", default="sharpe_ratio",
        choices=["sharpe_ratio", "total_return_pct", "annual_return_pct",
                 "calmar_ratio", "win_rate", "profit_loss_ratio"],
        help="排序指标",
    )
    parser.add_argument("--top", type=int, default=10, help="显示前 N 名")
    args = parser.parse_args(argv)

    setup_logging()

    # 读取数据
    storage = DataStorage()
    symbol = str(args.symbol).zfill(6)
    data = storage.read_klines(symbol, Freq.DAILY, start_date=args.start, end_date=args.end)
    if data.empty:
        console.print(f"[red]⚠️  无 {symbol} 数据，请先运行 download_data.py[/]")
        return 1
    console.print(f"📊 {symbol} 共 {len(data)} 根K线")

    # 构建参数网格
    if args.param:
        param_grid: dict[str, list[Any]] = {}
        for p in args.param:
            name = p[0]
            # 尝试转 int/float
            vals: list[Any] = []
            for v in p[1:]:
                try:
                    vals.append(int(v))
                except ValueError:
                    try:
                        vals.append(float(v))
                    except ValueError:
                        vals.append(v)
            param_grid[name] = vals
    else:
        param_grid = DEFAULT_GRIDS[args.strategy]

    console.print(f"📋 参数网格: {param_grid}")
    console.print(f"📈 排序指标: {args.sort_by}")
    console.print()

    # 执行搜索
    results = run_grid_search(args.strategy, data, symbol, param_grid, args.sort_by)

    # 展示结果
    display_results(results, args.strategy, top_n=args.top)

    # 可选：保存结果到 CSV
    if not results.empty:
        out_path = _PROJECT_ROOT / "data" / "cache" / f"optimize_{symbol}_{args.strategy}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(out_path, index=False, encoding="utf-8-sig")
        console.print(f"\n💾 完整结果已保存: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
