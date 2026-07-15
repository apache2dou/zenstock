"""数据存储层：支持 DuckDB + Parquet 双存储。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from zenstock.config import get_config
from zenstock.data.types import Freq
from zenstock.logger import get_logger

log = get_logger(__name__)


class DataStorage:
    """统一的本地数据存储（DuckDB 分析 + Parquet 落地）。

    - Parquet：原始 K 线按 symbol 分文件存储，追加友好
    - DuckDB：汇总表，支持快速 SQL 查询与分析
    """

    def __init__(self) -> None:
        cfg = get_config().data
        self.parquet_dir: Path = cfg.parquet_path
        self.duckdb_path: Path = cfg.duckdb_path
        self._duck: "object | None" = None

    # ==================== Parquet 存储 ====================
    def save_klines(
        self,
        df: pd.DataFrame,
        freq: Freq | str = Freq.DAILY,
        mode: str = "merge",
    ) -> int:
        """保存 K 线数据到 Parquet。

        Args:
            df: 标准 K 线 DataFrame（含 symbol 列）
            freq: K 线周期
            mode: merge(去重合并) | append | overwrite

        Returns:
            写入的记录数
        """
        if df.empty:
            return 0
        freq = Freq(freq) if isinstance(freq, str) else freq
        count = 0
        for symbol, group in df.groupby("symbol"):
            count += self._save_single(str(symbol), group, freq, mode)
        return count

    def _save_single(
        self, symbol: str, df: pd.DataFrame, freq: Freq, mode: str
    ) -> int:
        path = self._parquet_path(symbol, freq)
        path.parent.mkdir(parents=True, exist_ok=True)

        if mode == "overwrite" or not path.exists():
            new_df = df
        elif mode == "append":
            new_df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
        else:  # merge
            if path.exists():
                old = pd.read_parquet(path)
                new_df = pd.concat([old, df], ignore_index=True)
                new_df = new_df.drop_duplicates(subset=["date", "symbol"], keep="last")
                new_df = new_df.sort_values("date").reset_index(drop=True)
            else:
                new_df = df.copy()

        # 使用 PyArrow 写入（snappy 压缩）
        table = pa.Table.from_pandas(new_df, preserve_index=False)
        pq.write_table(table, path, compression="snappy")
        return len(new_df)

    def read_klines(
        self,
        symbol: str,
        freq: Freq | str = Freq.DAILY,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """读取单只股票的 K 线数据。"""
        freq = Freq(freq) if isinstance(freq, str) else freq
        path = self._parquet_path(symbol, freq)
        if not path.exists():
            log.warning(f"无数据文件: {path}")
            return pd.DataFrame()
        df = pd.read_parquet(path)
        # 确保数值列为 float（防止历史数据类型不一致）
        for c in ["open", "high", "low", "close", "volume", "amount",
                  "turnover", "pct_change"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if start_date:
            df = df[df["date"] >= start_date]
        if end_date:
            df = df[df["date"] <= end_date]
        return df.reset_index(drop=True)

    def get_last_date(self, symbol: str, freq: Freq | str = Freq.DAILY) -> str | None:
        """获取某股票已存储的最新日期（用于增量更新）。

        日线返回 "YYYY-MM-DD"，分钟线返回完整时间戳。
        """
        df = self.read_klines(symbol, freq)
        if df.empty:
            return None
        freq_enum = Freq(freq) if isinstance(freq, str) else freq
        last = df["date"].max()
        if freq_enum.is_minute:
            return str(last)  # 完整时间戳
        return str(last.date())

    def list_symbols(self, freq: Freq | str = Freq.DAILY) -> list[str]:
        """列出已存储的所有股票代码。"""
        freq = Freq(freq) if isinstance(freq, str) else freq
        pattern = f"*_{freq.value}.parquet"
        files = self.parquet_dir.glob(pattern)
        return sorted(p.stem.split("_")[0] for p in files)

    def _parquet_path(self, symbol: str, freq: Freq) -> Path:
        return self.parquet_dir / f"{symbol}_{freq.value}.parquet"

    # ==================== DuckDB 存储 ====================
    def save_stock_list(self, df: pd.DataFrame) -> None:
        """保存股票列表到 DuckDB。"""
        con = self._get_duckdb()
        con.execute("CREATE TABLE IF NOT EXISTS stock_list AS SELECT * FROM df")
        con.execute("DELETE FROM stock_list")
        con.execute("INSERT INTO stock_list SELECT * FROM df")
        log.info(f"stock_list 表已更新: {len(df)} 条")

    def get_stock_list(self) -> pd.DataFrame:
        """读取股票列表。"""
        con = self._get_duckdb()
        try:
            return con.execute("SELECT * FROM stock_list").fetchdf()
        except Exception:
            return pd.DataFrame()

    # ==================== DuckDB 查询接口 ====================
    def query(self, sql: str) -> pd.DataFrame:
        """直接执行 SQL 查询（高级用法）。

        示例::

            storage.query('''
                SELECT symbol, AVG(close) as avg_close
                FROM read_parquet('data/parquet/*.parquet')
                GROUP BY symbol
            ''')
        """
        con = self._get_duckdb()
        return con.execute(sql).fetchdf()

    def _get_duckdb(self):
        """惰性获取 DuckDB 连接。"""
        if self._duck is None:
            import duckdb
            self._duck = duckdb.connect(str(self.duckdb_path))
            log.debug(f"DuckDB 连接: {self.duckdb_path}")
        return self._duck

    def close(self) -> None:
        if self._duck is not None:
            self._duck.close()
            self._duck = None
