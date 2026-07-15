"""K 线重采样：从低粒度 K 线合成高粒度 K 线。

例如：5 分钟 → 15/30/60/120 分钟、日线、周线、月线。
结果缓存到 data/cache/ 下，避免重复计算。

支持的合成规则：
    open  = 每组第一根的 open
    high  = 每组的最高 high
    low   = 每组的最低 low
    close = 每组最后一根的 close
    volume = 每组的 volume 之和
    amount = 每组的 amount 之和
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from zenstock.config import get_config
from zenstock.data.types import Freq
from zenstock.logger import get_logger

log = get_logger(__name__)

# 合成关系：目标频率 → (基础频率, pandas resample 规则)
# pandas offset aliases: https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases
RESAMPLE_MAP: dict[str, tuple[str, str]] = {
    # 目标 freq → (所需的基础 freq, pandas rule)
    Freq.MIN15.value: (Freq.MIN5.value, "15min"),
    Freq.MIN30.value: (Freq.MIN5.value, "30min"),
    Freq.MIN60.value: (Freq.MIN5.value, "60min"),
    Freq.DAILY.value: (Freq.MIN5.value, "B"),       # B = business day
    Freq.WEEKLY.value: (Freq.DAILY.value, "W-FRI"),  # 周五为周末
    Freq.MONTHLY.value: (Freq.DAILY.value, "BM"),    # 月末
}

# 额外的合成目标（120 分钟，不是标准 Freq 枚举）
EXTRA_TARGETS: dict[str, tuple[str, str]] = {
    "120": (Freq.MIN5.value, "120min"),
}


def _cache_dir() -> Path:
    """缓存目录。"""
    d = get_config().data.cache_path / "resample"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(symbol: str, target_freq: str, source_hash: str) -> Path:
    """生成缓存文件路径。"""
    return _cache_dir() / f"{symbol}_{target_freq}_{source_hash[:8]}.parquet"


def _df_hash(df: pd.DataFrame) -> str:
    """计算 DataFrame 内容的 hash（用于缓存失效判断）。"""
    h = hashlib.md5()
    # 用日期范围 + 行数 + 最后一根 close 作为快速 hash
    if df.empty:
        return "empty"
    h.update(str(df["date"].min()).encode())
    h.update(str(df["date"].max()).encode())
    h.update(str(len(df)).encode())
    h.update(str(df["close"].iloc[-1]).encode())
    return h.hexdigest()


def resample_klines(
    df: pd.DataFrame,
    target_freq: Freq | str,
) -> pd.DataFrame:
    """将低粒度 K 线重采样为目标粒度。

    Args:
        df: 源 K 线 DataFrame（需含 date/open/high/low/close/volume/symbol）
        target_freq: 目标频率

    Returns:
        重采样后的 DataFrame（标准列），或空 DataFrame（无法合成）
    """
    freq_enum = Freq(target_freq) if isinstance(target_freq, str) else target_freq
    target_key = freq_enum.value

    # 检查是否有合成规则
    rule_entry = RESAMPLE_MAP.get(target_key) or EXTRA_TARGETS.get(target_key)
    if rule_entry is None:
        log.warning(f"不支持合成 {freq_enum.display_name}（无合成规则）")
        return pd.DataFrame()

    if df.empty:
        return df

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # 按交易日分组（分钟→日线时，不能跨日合并）
    source_freq_key, rule = rule_entry
    source_freq = Freq(source_freq_key)

    # 如果源是分钟级、目标是日/周/月，需要先按交易日分组再 resample
    if source_freq.is_minute and not freq_enum.is_minute:
        # 分钟 → 日/周/月：按交易日分组
        df["trading_day"] = df.index.normalize()
        grouped = df.groupby("trading_day")
        agg = grouped.agg(
            symbol=("symbol", "first"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            amount=("amount", "sum"),
        ).reset_index().rename(columns={"trading_day": "date"})

        # 如果目标是周线/月线，再从日线做 resample
        if freq_enum == Freq.WEEKLY:
            agg = _do_resample(agg, "W-FRI")
        elif freq_enum == Freq.MONTHLY:
            agg = _do_resample(agg, "BM")
        # 日线直接用
    else:
        # 分钟 → 分钟：直接 resample
        agg = _do_resample(df.reset_index(), rule)

    if agg.empty:
        return agg

    # 确保列完整
    for c in ["turnover", "pct_change"]:
        if c not in agg.columns:
            if c == "pct_change" and "close" in agg.columns:
                agg[c] = agg["close"].pct_change() * 100
            else:
                agg[c] = 0.0

    # 只保留标准列
    keep = [c for c in ["date", "symbol", "open", "high", "low", "close",
                        "volume", "amount", "turnover", "pct_change"]
            if c in agg.columns]
    agg = agg[keep]

    # 数值类型
    for c in ["open", "high", "low", "close", "volume", "amount", "turnover", "pct_change"]:
        if c in agg.columns:
            agg[c] = pd.to_numeric(agg[c], errors="coerce")

    agg["date"] = pd.to_datetime(agg["date"])
    agg = agg.sort_values("date").reset_index(drop=True)
    return agg


def _do_resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """执行 pandas resample。"""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    agg = df.resample(rule, label="left", closed="left").agg(
        symbol=("symbol", "first"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
    ).dropna(subset=["open", "close"])
    agg = agg.reset_index()
    return agg


def get_or_resample(
    symbol: str,
    source_df: pd.DataFrame,
    target_freq: Freq | str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """获取目标粒度的 K 线数据（带缓存）。

    Args:
        symbol: 股票代码
        source_df: 源 K 线 DataFrame
        target_freq: 目标频率
        use_cache: 是否使用缓存

    Returns:
        目标粒度的 K 线 DataFrame
    """
    freq_enum = Freq(target_freq) if isinstance(target_freq, str) else target_freq
    src_hash = _df_hash(source_df)
    cache_path = _cache_key(symbol, freq_enum.value, src_hash)

    if use_cache and cache_path.exists():
        log.debug(f"命中缓存: {cache_path.name}")
        return pd.read_parquet(cache_path)

    result = resample_klines(source_df, freq_enum)

    if use_cache and not result.empty:
        # 写入缓存
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pandas(result, preserve_index=False)
        pq.write_table(table, cache_path, compression="snappy")
        # 清理同 symbol 同 freq 的旧缓存（只保留最新的）
        for old in _cache_dir().glob(f"{symbol}_{freq_enum.value}_*.parquet"):
            if old != cache_path:
                old.unlink(missing_ok=True)
        log.debug(f"缓存已写入: {cache_path.name}")

    return result
