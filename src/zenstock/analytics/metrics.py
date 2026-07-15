"""核心绩效指标计算：胜率、赔率、夏普、最大回撤等。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from zenstock.backtest.engine import BacktestResult


# ============================================================
# 单笔交易维度指标
# ============================================================
def _pair_trades(result: BacktestResult) -> pd.DataFrame:
    """把买卖配对，生成每笔完整交易的盈亏。"""
    pairs = result._pair_trades()  # noqa: SLF001
    if not pairs:
        return pd.DataFrame()
    records = []
    for buy, sell in pairs:
        pnl = (sell.price - buy.price) * sell.shares - buy.cost - sell.cost
        pnl_pct = (sell.price - buy.price) / buy.price * 100
        holding_days = (sell.date - buy.date).days
        records.append({
            "buy_date": buy.date,
            "sell_date": sell.date,
            "buy_price": buy.price,
            "sell_price": sell.price,
            "shares": sell.shares,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "holding_days": holding_days,
            "reason": buy.reason,
        })
    return pd.DataFrame(records)


def win_rate(result: BacktestResult) -> float:
    """胜率 = 盈利交易数 / 总交易数。"""
    df = _pair_trades(result)
    if df.empty:
        return 0.0
    wins = (df["pnl"] > 0).sum()
    return wins / len(df)


def profit_loss_ratio(result: BacktestResult) -> float:
    """赔率（盈亏比）= 平均盈利 / 平均亏损。"""
    df = _pair_trades(result)
    if df.empty:
        return 0.0
    gains = df.loc[df["pnl"] > 0, "pnl"]
    losses = df.loc[df["pnl"] < 0, "pnl"].abs()
    if gains.empty or losses.empty:
        return float("inf") if losses.empty else 0.0
    return gains.mean() / losses.mean()


def expectancy(result: BacktestResult) -> float:
    """每笔交易期望收益。

    E = 胜率 × 平均盈利 − 败率 × 平均亏损
    """
    df = _pair_trades(result)
    if df.empty:
        return 0.0
    wr = win_rate(result)
    lr = 1 - wr
    avg_gain = df.loc[df["pnl"] > 0, "pnl"].mean() if wr > 0 else 0
    avg_loss = df.loc[df["pnl"] < 0, "pnl"].abs().mean() if lr > 0 else 0
    return wr * avg_gain - lr * avg_loss


# ============================================================
# 资金曲线维度指标
# ============================================================
def max_drawdown(equity_curve: pd.DataFrame | pd.Series) -> tuple[float, str, str]:
    """最大回撤。

    Returns:
        (回撤百分比, 峰值日期, 谷底日期)
    """
    if isinstance(equity_curve, pd.DataFrame):
        if "total_equity" in equity_curve.columns:
            values = equity_curve["total_equity"]
        else:
            values = equity_curve.iloc[:, 0]
    else:
        values = equity_curve

    if values.empty:
        return 0.0, "", ""

    running_max = values.cummax()
    drawdown = (values - running_max) / running_max
    peak_idx = drawdown.idxmin()
    # 找到对应的峰值
    peak_value = running_max.iloc[peak_idx] if isinstance(peak_idx, int) else running_max.loc[peak_idx]

    # 找峰值日期
    dates = equity_curve["date"] if isinstance(equity_curve, pd.DataFrame) else values.index

    trough_date = dates.iloc[peak_idx] if isinstance(peak_idx, int) else dates.loc[peak_idx]
    peak_mask = values == peak_value
    peak_date = dates[peak_mask.values].iloc[0] if peak_mask.any() else trough_date

    return abs(drawdown.min()) * 100, str(peak_date), str(trough_date)


def sharpe_ratio(
    equity_curve: pd.DataFrame | pd.Series,
    risk_free_rate: float = 0.025,
    periods_per_year: int = 252,
) -> float:
    """年化夏普比率。"""
    if isinstance(equity_curve, pd.DataFrame):
        values = equity_curve["total_equity"]
    else:
        values = equity_curve
    if len(values) < 2:
        return 0.0
    daily_returns = values.pct_change().dropna()
    if daily_returns.std() == 0:
        return 0.0
    excess = daily_returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess.mean() / daily_returns.std()


def annual_return(
    equity_curve: pd.DataFrame | pd.Series, periods_per_year: int = 252
) -> float:
    """年化收益率（%）。"""
    if isinstance(equity_curve, pd.DataFrame):
        values = equity_curve["total_equity"]
    else:
        values = equity_curve
    if len(values) < 2:
        return 0.0
    total_ret = values.iloc[-1] / values.iloc[0] - 1
    n_years = len(values) / periods_per_year
    if n_years <= 0:
        return 0.0
    return ((1 + total_ret) ** (1 / n_years) - 1) * 100


def calmar_ratio(
    equity_curve: pd.DataFrame, annual_ret: float | None = None
) -> float:
    """卡玛比率 = 年化收益 / 最大回撤。"""
    mdd, _, _ = max_drawdown(equity_curve)
    if mdd == 0:
        return float("inf")
    ar = annual_ret if annual_ret is not None else annual_return(equity_curve)
    return ar / mdd


# ============================================================
# 汇总
# ============================================================
def compute_metrics(result: BacktestResult) -> dict[str, Any]:
    """计算全部核心指标。"""
    ec = result.equity_curve
    df_trades = _pair_trades(result)

    from zenstock.config import get_config
    rf = get_config().backtest.risk_free_rate

    wr = win_rate(result)
    plr = profit_loss_ratio(result)
    mdd, peak_d, trough_d = max_drawdown(ec)
    ar = annual_return(ec)
    sharpe = sharpe_ratio(ec, risk_free_rate=rf)

    return {
        # 交易维度
        "total_trades": len(df_trades),
        "win_rate": wr * 100,
        "profit_loss_ratio": plr,
        "expectancy": expectancy(result),
        "avg_holding_days": df_trades["holding_days"].mean() if not df_trades.empty else 0,
        # 资金维度
        "initial_capital": result.initial_capital,
        "final_capital": result.final_capital,
        "total_return_pct": (result.final_capital / result.initial_capital - 1) * 100,
        "annual_return_pct": ar,
        "max_drawdown_pct": mdd,
        "max_dd_peak_date": peak_d,
        "max_dd_trough_date": trough_d,
        "sharpe_ratio": sharpe,
        "calmar_ratio": ar / mdd if mdd > 0 else float("inf"),
        # 策略正期望判定
        "is_positive_expectancy": wr * plr > 1 if plr != float("inf") else True,
    }
