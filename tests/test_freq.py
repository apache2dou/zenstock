"""测试 Freq 枚举和数据时间戳解析。"""

from __future__ import annotations

import pandas as pd
import pytest

from zenstock.data.types import FREQ_OPTIONS, Freq


class TestFreq:
    def test_is_minute(self):
        assert Freq.MIN1.is_minute
        assert Freq.MIN5.is_minute
        assert Freq.MIN15.is_minute
        assert not Freq.DAILY.is_minute
        assert not Freq.WEEKLY.is_minute

    def test_display_name(self):
        assert Freq.MIN1.display_name == "1 分钟"
        assert Freq.MIN5.display_name == "5 分钟"
        assert Freq.DAILY.display_name == "日线"

    def test_minute_count(self):
        """Freq 枚举目前应有 5 个分钟级别 + 3 个非分钟。"""
        minutes = [f for f in Freq if f.is_minute]
        assert len(minutes) == 5

    def test_freq_options_for_ui(self):
        """FREQ_OPTIONS 必须涵盖所有非分钟 + 分钟级别。"""
        keys = [k for k, _ in FREQ_OPTIONS]
        for f in Freq:
            assert f.value in keys


class TestTimeParsing:
    """测试 BaoStock 分钟数据 time 列解析逻辑（独立验证算法不依赖真实下载）。"""

    def test_baostock_time_string_parses(self):
        """BaoStock 时间字符串 '20240102093500000' 应解析为正确 datetime。"""
        time_str = "20240102093500000"
        dt = pd.to_datetime(time_str, format="%Y%m%d%H%M%S%f", errors="coerce")
        assert pd.notna(dt)
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 2
        assert dt.hour == 9
        assert dt.minute == 35
        assert dt.second == 0

    def test_sina_day_format_parses(self):
        """新浪 'YYYY-MM-DD HH:MM:SS' 应能被 pd.to_datetime 识别。"""
        s = "2024-01-02 09:35:00"
        dt = pd.to_datetime(s, errors="coerce")
        assert dt.year == 2024
        assert dt.hour == 9
        assert dt.minute == 35