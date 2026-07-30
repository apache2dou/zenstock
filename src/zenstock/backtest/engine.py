"""事件驱动回测引擎，内置 A 股交易规则。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from zenstock.config import get_config
from zenstock.logger import get_logger
from zenstock.strategy.base import Action, BaseStrategy, Signal

log = get_logger(__name__)


@dataclass
class Trade:
    """单笔交易记录。"""
    date: pd.Timestamp
    symbol: str
    action: str
    price: float
    shares: float
    amount: float
    cost: float        # 交易成本
    reason: str = ""


@dataclass
class Position:
    """持仓状态。"""
    shares: float = 0.0
    avg_cost: float = 0.0     # 持仓均价
    available: float = 0.0    # 可卖数量（T+1：当日买入不可卖）

    @property
    def is_empty(self) -> bool:
        return self.shares <= 0


@dataclass
class BacktestResult:
    """回测结果。"""
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    positions_history: list[dict] = field(default_factory=list)
    final_capital: float = 0.0
    initial_capital: float = 0.0

    @property
    def n_trades(self) -> int:
        """配对后的完整交易次数（一买一卖算一次）。"""
        return len(self._pair_trades())

    def _pair_trades(self) -> list[tuple[Trade, Trade]]:
        """把买卖配对成完整交易。"""
        pairs: list[tuple[Trade, Trade]] = []
        open_buy: Trade | None = None
        for t in self.trades:
            if t.action == "BUY":
                if open_buy is None:
                    open_buy = t
            elif t.action == "SELL":
                if open_buy is not None:
                    pairs.append((open_buy, t))
                    open_buy = None
        return pairs

    def to_summary(self) -> dict[str, Any]:
        """生成可读的回测摘要。"""
        from zenstock.analytics import compute_metrics
        return compute_metrics(self)


class Backtest:
    """单标的回测引擎。

    支持的 A 股规则：
      - T+1：当日买入次日才能卖出
      - 涨跌停限制：涨停买不进，跌停卖不出
      - 佣金（双边）、印花税（卖出）、过户费、最低佣金
      - 滑点
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
        symbol: str = "",
        config: Any | None = None,
    ) -> None:
        self.strategy = strategy
        self.data = data.sort_values("date").reset_index(drop=True)
        self.symbol = symbol or (data["symbol"].iloc[0] if not data.empty else "")
        self.cfg = config or get_config().backtest

        # 状态
        self.cash: float = self.cfg.initial_capital
        self.position = Position()
        self.trades: list[Trade] = []

    def run(self) -> BacktestResult:
        """执行回测，返回结果。"""
        if self.data.empty:
            log.warning("数据为空，回测中止")
            return BacktestResult(initial_capital=self.cfg.initial_capital)

        log.info(f"开始回测 {self.symbol}: {len(self.data)} 根K线")

        equity_records: list[dict] = []
        prev_close = self.data["close"].iloc[0]
        # T+1：记录买入所在交易日，次日（即不同交易日）才解锁可卖
        prev_trade_date = self._trade_date(self.data["date"].iloc[0])

        for i in range(len(self.data)):
            row = self.data.iloc[i]
            date = row["date"]
            close = float(row["close"])
            pct = float(row.get("pct_change", 0.0) or 0.0)

            # T+1 规则：只在换日时才把持仓解锁为可卖
            cur_trade_date = self._trade_date(date)
            if cur_trade_date != prev_trade_date:
                self.position.available = self.position.shares
                prev_trade_date = cur_trade_date

            # 产生信号
            signal = self.strategy.on_bar(i, self.data)
            if isinstance(signal, Action):  # 兼容直接返回 Action
                signal = Signal(action=signal)

            # 执行交易
            self._execute_signal(signal, date, close, pct, prev_close)

            # 记录每日净值
            market_value = self.position.shares * close
            total = self.cash + market_value
            equity_records.append({
                "date": date,
                "close": close,
                "cash": self.cash,
                "position": self.position.shares,
                "market_value": market_value,
                "total_equity": total,
                "return_pct": (total / self.cfg.initial_capital - 1) * 100,
            })

            prev_close = close

        equity_curve = pd.DataFrame(equity_records)
        log.info(
            f"回测完成: 最终资金 {self.cash:.2f}, "
            f"剩余持仓 {self.position.shares:.0f} 股"
        )
        return BacktestResult(
            trades=self.trades,
            equity_curve=equity_curve,
            final_capital=self.cash + self.position.shares * prev_close,
            initial_capital=self.cfg.initial_capital,
        )

    # ==================== 交易执行 ====================
    def _execute_signal(
        self,
        signal: Signal,
        date: pd.Timestamp,
        close: float,
        pct: float,
        prev_close: float,
    ) -> None:
        if signal.action == Action.BUY and self.position.is_empty:
            self._buy(signal, date, close, pct)
        elif signal.action == Action.SELL and not self.position.is_empty:
            self._sell(signal, date, close, pct)

    def _buy(self, signal: Signal, date: pd.Timestamp, close: float, pct: float) -> None:
        # 涨停限制（涨幅 ≥ price_limit_pct）不可买入
        if pct >= self.cfg.price_limit_pct:
            return

        # 目标金额
        use_cash = self.cash * signal.size
        exec_price = self._apply_slippage(close, is_buy=True)
        max_shares = use_cash / (exec_price * (1 + self.cfg.commission))
        shares = int(max_shares / 100) * 100  # A 股 100 股一手
        if shares <= 0:
            return

        amount = shares * exec_price
        cost = self._calc_cost(amount, is_buy=True)
        self.cash -= amount + cost

        self.position.shares += shares
        self.position.avg_cost = (
            (self.position.avg_cost * (self.position.shares - shares) + amount)
            / self.position.shares
        )

        self.trades.append(Trade(
            date=date, symbol=self.symbol, action="BUY",
            price=exec_price, shares=shares, amount=amount, cost=cost,
            reason=signal.reason,
        ))

    def _sell(self, signal: Signal, date: pd.Timestamp, close: float, pct: float) -> None:
        # 跌停限制不可卖出
        if pct <= -self.cfg.price_limit_pct:
            return
        # T+1：无可用仓位
        if self.position.available <= 0:
            return

        shares = self.position.shares * signal.size
        shares = int(shares / 100) * 100
        if shares <= 0:
            return

        exec_price = self._apply_slippage(close, is_buy=False)
        amount = shares * exec_price
        cost = self._calc_cost(amount, is_buy=False)
        self.cash += amount - cost

        self.position.shares -= shares
        self.position.available -= shares
        if self.position.shares <= 0:
            self.position.shares = 0
            self.position.avg_cost = 0.0

        self.trades.append(Trade(
            date=date, symbol=self.symbol, action="SELL",
            price=exec_price, shares=shares, amount=amount, cost=cost,
            reason=signal.reason,
        ))

    # ==================== 费用计算 ====================
    def _calc_cost(self, amount: float, is_buy: bool) -> float:
        """计算交易成本。"""
        # 佣金
        commission = max(amount * self.cfg.commission, self.cfg.min_commission)
        cost = commission
        # 印花税（仅卖出）
        if not is_buy:
            cost += amount * self.cfg.stamp_duty
        # 过户费（双边）
        cost += amount * self.cfg.transfer_fee
        return cost

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """模拟滑点。买入价上浮，卖出价下浮。"""
        if is_buy:
            return price * (1 + self.cfg.slippage)
        return price * (1 - self.cfg.slippage)

    @staticmethod
    def _trade_date(date: pd.Timestamp) -> pd.Timestamp:
        """从 datetime 中提取交易日（仅日期部分），用于 T+1 判断。"""
        return pd.Timestamp(date).normalize()


# ============================================================
# 便捷函数
# ============================================================
def run_backtest(
    strategy: BaseStrategy,
    data: pd.DataFrame,
    symbol: str = "",
) -> BacktestResult:
    """一键运行回测。"""
    bt = Backtest(strategy, data, symbol)
    return bt.run()
