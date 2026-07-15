"""数据采集抽象基类与具体实现（AKShare / BaoStock）。"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Iterable

import pandas as pd
from tqdm import tqdm

from zenstock.config import get_config
from zenstock.data.types import Adjust, Freq, STANDARD_COLUMNS, is_valid_klines
from zenstock.logger import get_logger

log = get_logger(__name__)


class BaseDownloader(ABC):
    """数据下载器抽象基类。"""

    name: str = "base"

    def __init__(self) -> None:
        cfg = get_config().data
        self.adjust = Adjust(cfg.adjust)
        self.sleep = cfg.request_sleep
        self.max_retries = cfg.max_retries

    @abstractmethod
    def fetch_stock_list(self) -> pd.DataFrame:
        """获取全部 A 股股票列表。返回列：symbol, name, market, list_date。"""

    @abstractmethod
    def fetch_klines(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        freq: Freq = Freq.DAILY,
    ) -> pd.DataFrame:
        """下载单只股票 K 线。返回标准列 DataFrame。"""

    # ---- 通用批量下载 ----
    def download_many(
        self,
        symbols: Iterable[str],
        start_date: str,
        end_date: str,
        freq: Freq = Freq.DAILY,
    ) -> dict[str, pd.DataFrame]:
        """批量下载多只股票，返回 {symbol: df}。"""
        results: dict[str, pd.DataFrame] = {}
        symbols = list(symbols)
        for sym in tqdm(symbols, desc=f"[{self.name}] 下载K线"):
            try:
                df = self._fetch_with_retry(sym, start_date, end_date, freq)
                if is_valid_klines(df):
                    results[sym] = df
                else:
                    log.warning(f"数据为空或格式错误: {sym}")
            except Exception as e:  # noqa: BLE001
                log.error(f"下载失败 {sym}: {e}")
            time.sleep(self.sleep)
        log.info(f"下载完成: {len(results)}/{len(symbols)} 只股票")
        return results

    def _fetch_with_retry(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        freq: Freq,
    ) -> pd.DataFrame:
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                df = self.fetch_klines(symbol, start_date, end_date, freq)
                if self.sleep > 0:
                    time.sleep(self.sleep)
                return df
            except Exception as e:  # noqa: BLE001
                last_err = e
                wait = self.sleep * attempt * 2
                log.debug(f"重试 {attempt}/{self.max_retries} {symbol}: {e}")
                time.sleep(wait)
        raise RuntimeError(f"超过最大重试次数 {symbol}: {last_err}")


# ============================================================
# AKShare 实现
# ============================================================
class AKShareDownloader(BaseDownloader):
    """基于 AKShare 的数据下载器（免费、无需 token）。"""

    name = "akshare"

    def fetch_stock_list(self) -> pd.DataFrame:
        import akshare as ak

        # 沪深 A 股一览
        df = ak.stock_zh_a_spot_em()
        df = df.rename(columns={
            "代码": "symbol",
            "名称": "name",
            "最新价": "close",
        })
        df["symbol"] = df["symbol"].astype(str).str.zfill(6)
        df["market"] = df["symbol"].apply(self._detect_market)
        df = df[["symbol", "name", "market"]].copy()
        df = df[~df["name"].str.contains("指数|ETF|债", na=False)]
        log.info(f"获取股票列表: {len(df)} 只")
        return df

    def fetch_klines(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        freq: Freq = Freq.DAILY,
    ) -> pd.DataFrame:
        import akshare as ak

        symbol = str(symbol).zfill(6)
        adj = "" if self.adjust == Adjust.NONE else self.adjust.value

        if freq == Freq.DAILY:
            raw = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust=adj,
            )
            return self._normalize(raw, symbol, freq)

        # ----- 分钟级：统一走新浪接口（更稳定） -----
        # BaoStock 不支持 1 分钟；东方财富 stock_zh_a_hist_min_em 易被限流。
        # 新浪 stock_zh_a_minute 支持 1/5/15/30/60 分钟，仅返回最近 5~9 个交易日。
        sina_symbol = self._to_sina_symbol(symbol)
        period = freq.value  # "1" / "5" / "15" / "30" / "60"
        raw = ak.stock_zh_a_minute(
            symbol=sina_symbol, period=period, adjust=adj or "qfq"
        )
        # 新浪接口不返回日期范围参数，只能从结果中按 start/end 过滤
        raw = self._filter_by_date(raw, start_date, end_date)
        return self._normalize_minute(raw, symbol, freq)

    # ---- 内部工具 ----
    @staticmethod
    def _detect_market(symbol: str) -> str:
        """根据代码判断市场。"""
        s = str(symbol).zfill(6)
        if s.startswith(("60", "68", "11", "13")):
            return "SH"  # 沪市
        if s.startswith(("00", "30", "12")):
            return "SZ"  # 深市
        if s.startswith(("43", "83", "87", "88")):
            return "BJ"  # 北交所
        return "UNKNOWN"

    @staticmethod
    def _normalize(df: pd.DataFrame, symbol: str, freq: Freq) -> pd.DataFrame:
        """将 AKShare 返回的中文列名标准化。"""
        if df is None or df.empty:
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        col_map = {
            "日期": "date",
            "时间": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover",
            "涨跌幅": "pct_change",
            "涨跌额": "price_change",
        }
        df = df.rename(columns=col_map)
        df["symbol"] = symbol

        # 日期解析
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        # 仅保留标准列（存在的话）
        keep = [c for c in STANDARD_COLUMNS if c in df.columns]
        df = df[keep].copy()
        df = df.sort_values("date").reset_index(drop=True)
        return df

    @staticmethod
    def _filter_by_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        if "date" not in df.columns or df.empty:
            return df
        dt = pd.to_datetime(df["date"])
        mask = (dt >= start) & (dt <= end)
        return df.loc[mask].reset_index(drop=True)

    @staticmethod
    def _to_sina_symbol(symbol: str) -> str:
        """6 位代码转新浪格式：'000001' → 'sz000001'，'600519' → 'sh600519'。"""
        s = str(symbol).zfill(6)
        if s.startswith(("60", "68")):
            return f"sh{s}"
        if s.startswith(("00", "30")):
            return f"sz{s}"
        if s.startswith(("43", "83", "87", "88")):
            return f"bj{s}"
        return f"sh{s}"  # 默认

    @staticmethod
    def _normalize_minute(
        df: pd.DataFrame, symbol: str, freq: Freq
    ) -> pd.DataFrame:
        """标准化新浪分钟 K 线数据。

        新浪返回列：day(YYYY-MM-DD HH:MM:SS) / open / high / low / close / volume(字符串) / amount(字符串)
        """
        if df is None or df.empty:
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        # 重要：AKShare 新浪接口 volume 是字符串"1970"，要转数值
        rename = {
            "day": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "amount": "amount",
        }
        df = df.rename(columns=rename)
        for c in ["open", "high", "low", "close", "volume", "amount"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["symbol"] = symbol

        # 分钟级数据没有 pct_change / turnover，从前后 close 计算
        if "pct_change" not in df.columns and "close" in df.columns:
            df["pct_change"] = df["close"].pct_change() * 100

        keep = [c for c in STANDARD_COLUMNS if c in df.columns]
        return df[keep].sort_values("date").reset_index(drop=True)


# ============================================================
# BaoStock 实现
# ============================================================
class BaoStockDownloader(BaseDownloader):
    """基于 BaoStock 的数据下载器（免费，需 login）。"""

    name = "baostock"

    def __init__(self) -> None:
        super().__init__()
        self._logged_in = False

    def _ensure_login(self) -> None:
        if not self._logged_in:
            import baostock as bs
            lg = bs.login()
            if lg.error_code != "0":
                raise RuntimeError(f"BaoStock 登录失败: {lg.error_msg}")
            self._logged_in = True
            log.debug("BaoStock 登录成功")

    def fetch_stock_list(self) -> pd.DataFrame:
        import baostock as bs
        self._ensure_login()
        rs = bs.query_all_stock(day=bs.query_trade_dates().get_data().iloc[-1]["calendar_date"])
        data = []
        while rs.next():
            data.append(rs.get_row_data())
        df = pd.DataFrame(data, columns=rs.fields)
        df = df.rename(columns={"code": "symbol", "code_name": "name"})
        df = df[df["type"] == "1"].copy()  # 仅股票
        return df[["symbol", "name"]].assign(market="")

    def fetch_klines(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        freq: Freq = Freq.DAILY,
    ) -> pd.DataFrame:
        import baostock as bs
        self._ensure_login()

        symbol = str(symbol).zfill(6)
        bs_code = self._to_bs_code(symbol)
        adj = {"qfq": "2", "hfq": "1", "none": "3"}[self.adjust.value]

        if freq == Freq.DAILY:
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount,turn,pctChg",
                start_date=start_date, end_date=end_date,
                frequency="d", adjustflag=adj,
            )
        elif freq.is_minute:
            # BaoStock 最低仅 5 分钟；1 分钟会被服务返回空数据，要拒绝
            if freq == Freq.MIN1:
                raise ValueError(
                    "BaoStock 不支持 1 分钟数据，请使用 akshare 数据源 "
                    "(stock_zh_a_minute 接口仅返回最近 5~9 个交易日)"
                )
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,time,open,high,low,close,volume,amount",
                start_date=start_date, end_date=end_date,
                frequency=freq.value, adjustflag=adj,
            )
        else:
            raise ValueError(f"不支持的 BaoStock 周期: {freq}")

        data = []
        while rs.next():
            data.append(rs.get_row_data())

        df = pd.DataFrame(data, columns=rs.fields)
        df["symbol"] = symbol
        return self._normalize(df, freq)

    @staticmethod
    def _to_bs_code(symbol: str) -> str:
        """000001 → sh.000001 / sz.000001。"""
        s = str(symbol).zfill(6)
        if s.startswith(("60", "68")):
            return f"sh.{s}"
        return f"sz.{s}"

    @staticmethod
    def _normalize(df: pd.DataFrame, freq: Freq) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=STANDARD_COLUMNS)

        col_map = {
            "turn": "turnover",
            "pctChg": "pct_change",
        }
        df = df.rename(columns=col_map)
        # 强制将字符串数值列转为 float（BaoStock 返回字符串）
        for c in ["open", "high", "low", "close", "volume", "amount",
                  "turnover", "pct_change"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # 处理日期：日线用 date 列，分钟级合并 date + time
        if freq.is_minute and "time" in df.columns:
            # BaoStock time 形如 "20240102093500000"
            df["date"] = pd.to_datetime(
                df["time"], format="%Y%m%d%H%M%S%f", errors="coerce"
            )
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        keep = [c for c in STANDARD_COLUMNS if c in df.columns]
        return df[keep].sort_values("date").reset_index(drop=True)


# ============================================================
# 工厂函数
# ============================================================
_DOWNLOADERS: dict[str, type[BaseDownloader]] = {
    "akshare": AKShareDownloader,
    "baostock": BaoStockDownloader,
}


def get_downloader(name: str | None = None) -> BaseDownloader:
    """根据配置获取数据下载器实例。"""
    name = name or get_config().data.source
    cls = _DOWNLOADERS.get(name)
    if cls is None:
        raise ValueError(
            f"未知数据源 '{name}'，可选: {list(_DOWNLOADERS.keys())}"
        )
    return cls()
