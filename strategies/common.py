"""策略公共工具。"""

from __future__ import annotations

import pandas as pd


def ensure_column(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.DataFrame:
    """确保 DataFrame 中存在指定列，缺失则填充默认值。"""
    if name not in df.columns:
        df[name] = default
    return df
