"""适配层：zenstock DataFrame → czsc RawBar 列表。

使用 czsc 官方的 format_standard_kline 函数进行转换，保证兼容性。
czsc 期望的 DataFrame 列：dt, symbol, open, close, high, low, vol, amount
"""

from __future__ import annotations

import pandas as pd

from zenstock.data.types import Freq
from zenstock.logger import get_logger

log = get_logger(__name__)

# zenstock Freq → czsc Freq 枚举成员的映射
# czsc.Freq 枚举：D=日线, F1=1分钟, F5=5分钟, F15/F30/F60, W=周线, M=月线
ZEN_TO_CZSC_FREQ: dict[str, str] = {
    Freq.DAILY.value: "D",
    Freq.WEEKLY.value: "W",
    Freq.MONTHLY.value: "M",
    Freq.MIN1.value: "F1",
    Freq.MIN5.value: "F5",
    Freq.MIN15.value: "F15",
    Freq.MIN30.value: "F30",
    Freq.MIN60.value: "F60",
}


def df_to_bars(df: pd.DataFrame, freq: Freq | str) -> list:
    """将 zenstock 标准 DataFrame 转为 czsc RawBar 列表。

    使用 czsc 官方的 ``format_standard_kline`` 函数，自动处理列名映射。

    Args:
        df: zenstock K 线 DataFrame，需含 date/open/high/low/close/volume
        freq: zenstock Freq

    Returns:
        list[czsc.RawBar]
    """
    if df is None or df.empty:
        return []

    from czsc import Freq as CzscFreq  # type: ignore
    from czsc import format_standard_kline  # type: ignore

    freq_enum = Freq(freq) if isinstance(freq, str) else freq
    czsc_freq_key = ZEN_TO_CZSC_FREQ.get(freq_enum.value, "D")

    # 把字符串键转为 czsc.Freq 枚举实例
    czsc_freq = getattr(CzscFreq, czsc_freq_key, CzscFreq.D)

    # zenstock 列名 → czsc 期望的列名
    rename_map = {
        "date": "dt",
        "volume": "vol",
    }
    std_df = df.rename(columns=rename_map).copy()

    # 确保必要的列存在
    for col in ["dt", "symbol", "open", "close", "high", "low", "vol", "amount"]:
        if col not in std_df.columns:
            if col == "amount":
                std_df[col] = 0.0
            elif col == "symbol":
                std_df[col] = ""
            else:
                log.warning(f"DataFrame 缺少列: {col}")
                return []

    try:
        bars = format_standard_kline(std_df, freq=czsc_freq)
        return bars
    except Exception as e:  # noqa: BLE001
        log.error(f"format_standard_kline 转换失败: {e}")
        return []
