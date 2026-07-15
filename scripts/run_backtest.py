"""回测运行脚本。

用法:
    # 对单只股票运行均线交叉策略
    python scripts/run_backtest.py --symbol 000001 --strategy ma_cross

    # 指定日期范围与策略参数
    python scripts/run_backtest.py --symbol 600519 --start 2023-01-01 \\
        --strategy ma_cross --fast 10 --slow 30
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))         # 项目根（含 strategies 包）
sys.path.insert(0, str(_PROJECT_ROOT / "src")) # 核心源码

from zenstock.analytics import print_report
from zenstock.backtest import run_backtest
from zenstock.data import DataStorage
from zenstock.data.types import Freq
from zenstock.logger import setup_logging


# 策略注册表：名称 → (模块路径, 类名)
STRATEGY_REGISTRY = {
    "ma_cross": ("strategies.ma_cross", "MACrossStrategy"),
}


def load_strategy(name: str, params: dict) -> object:
    """动态加载策略。"""
    if name not in STRATEGY_REGISTRY:
        # 也支持直接导入任意模块
        try:
            mod = importlib.import_module(name)
            cls = getattr(mod, "Strategy", None) or getattr(mod, name, None)
            if cls is None:
                raise ImportError(f"策略模块 {name} 未找到 Strategy 类")
            return cls(**params)
        except ImportError:
            raise ValueError(f"未知策略: {name}，可选: {list(STRATEGY_REGISTRY.keys())}")

    module_path, class_name = STRATEGY_REGISTRY[name]
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(**params)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ZenStock 回测运行器")
    parser.add_argument("--symbol", required=True, help="股票代码，如 000001")
    parser.add_argument("--strategy", default="ma_cross", help="策略名称")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--freq", default="D", help="K线周期")
    # 策略参数（透传）
    parser.add_argument("--fast", type=int, default=None)
    parser.add_argument("--slow", type=int, default=None)
    args = parser.parse_args(argv)

    setup_logging()

    # 收集策略参数
    params: dict = {}
    if args.fast is not None:
        params["fast"] = args.fast
    if args.slow is not None:
        params["slow"] = args.slow

    print(f"⚙️  加载策略: {args.strategy} {params or ''}")
    strategy = load_strategy(args.strategy, params)

    # 读取数据
    storage = DataStorage()
    symbol = str(args.symbol).zfill(6)
    data = storage.read_klines(symbol, freq=Freq(args.freq), start_date=args.start, end_date=args.end)

    if data.empty:
        print(f"⚠️  本地无 {symbol} 的数据，请先运行 download_data.py")
        return 1

    print(f"📊 读取数据: {symbol} 共 {len(data)} 根K线 [{data['date'].min().date()} ~ {data['date'].max().date()}]")

    # 运行回测
    result = run_backtest(strategy, data, symbol=symbol)

    # 输出报告
    metrics = result.to_summary()
    print()
    print_report(metrics)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
