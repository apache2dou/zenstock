"""回测报告格式化输出。"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()


def format_report(metrics: dict[str, Any]) -> str:
    """将指标字典格式化为多行文本。"""
    lines = [
        "╔══════════════════════════════════════════════╗",
        "║          ZenStock 回测绩效报告               ║",
        "╚══════════════════════════════════════════════╝",
        "",
        "【交易统计】",
        f"  总交易次数      : {metrics['total_trades']}",
        f"  胜率            : {metrics['win_rate']:.2f}%",
        f"  赔率(盈亏比)    : {metrics['profit_loss_ratio']:.2f}",
        f"  期望收益/笔     : {metrics['expectancy']:.2f} 元",
        f"  平均持仓天数    : {metrics['avg_holding_days']:.1f} 天",
        "",
        "【资金曲线】",
        f"  初始资金        : {metrics['initial_capital']:>14,.2f}",
        f"  最终资金        : {metrics['final_capital']:>14,.2f}",
        f"  总收益率        : {metrics['total_return_pct']:>13.2f}%",
        f"  年化收益率      : {metrics['annual_return_pct']:>13.2f}%",
        f"  最大回撤        : {metrics['max_drawdown_pct']:>13.2f}%",
        f"  最大回撤区间    : {metrics['max_dd_peak_date']} → {metrics['max_dd_trough_date']}",
        f"  夏普比率        : {metrics['sharpe_ratio']:>13.2f}",
        f"  卡玛比率        : {metrics['calmar_ratio']:>13.2f}",
        "",
    ]
    flag = "✅ 正期望" if metrics["is_positive_expectancy"] else "⚠️ 负期望"
    lines.append(f"  策略评估        : {flag}")
    return "\n".join(lines)


def print_report(metrics: dict[str, Any]) -> None:
    """使用 rich 表格打印报告到终端。"""
    title_table = Table(title="📊 回测绩效报告", show_header=False, border_style="cyan")
    title_table.add_column("指标", style="bold")
    title_table.add_column("值", justify="right")

    # 交易统计
    trade_section = [
        ("总交易次数", f"{metrics['total_trades']}"),
        ("胜率", f"{metrics['win_rate']:.2f}%"),
        ("赔率(盈亏比)", f"{metrics['profit_loss_ratio']:.2f}"),
        ("期望收益/笔", f"{metrics['expectancy']:,.2f} 元"),
        ("平均持仓天数", f"{metrics['avg_holding_days']:.1f} 天"),
    ]
    for k, v in trade_section:
        title_table.add_row(k, v)

    title_table.add_section()

    # 资金指标
    finance_section = [
        ("初始资金", f"{metrics['initial_capital']:,.2f}"),
        ("最终资金", f"{metrics['final_capital']:,.2f}"),
        ("总收益率", f"{metrics['total_return_pct']:.2f}%"),
        ("年化收益率", f"{metrics['annual_return_pct']:.2f}%"),
        ("最大回撤", f"{metrics['max_drawdown_pct']:.2f}%"),
        (
            "最大回撤区间",
            f"{metrics['max_dd_peak_date']} → {metrics['max_dd_trough_date']}",
        ),
        ("夏普比率", f"{metrics['sharpe_ratio']:.2f}"),
        ("卡玛比率", f"{metrics['calmar_ratio']:.2f}"),
    ]
    for k, v in finance_section:
        title_table.add_row(k, v)

    title_table.add_section()

    flag = "✅ 正期望（可深入研究）" if metrics["is_positive_expectancy"] else "⚠️ 负期望（需优化）"
    title_table.add_row("策略评估", flag)

    console.print(title_table)
