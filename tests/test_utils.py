"""测试市场工具函数。"""

from __future__ import annotations

import pytest

from zenstock.utils.market import detect_market, is_st_symbol, normalize_symbol


class TestNormalizeSymbol:
    @pytest.mark.parametrize("inp,expected", [
        ("1", "000001"),
        ("600519", "600519"),
        ("sz.000001", "000001"),
        ("sh.600519", "600519"),
        ("000001.SZ", "000001"),
        ("600519.SH", "600519"),
    ])
    def test_normalize(self, inp, expected):
        assert normalize_symbol(inp) == expected


class TestDetectMarket:
    @pytest.mark.parametrize("symbol,expected", [
        ("600519", "SH"),    # 沪市主板
        ("688981", "SH"),    # 科创板
        ("000001", "SZ"),    # 深市主板
        ("300750", "SZ"),    # 创业板
        ("830799", "BJ"),    # 北交所
    ])
    def test_market(self, symbol, expected):
        assert detect_market(symbol) == expected


class TestIsST:
    @pytest.mark.parametrize("name,expected", [
        ("ST天宝", True),
        ("*ST盐湖", True),
        ("贵州茅台", False),
        ("中国平安", False),
    ])
    def test_st(self, name, expected):
        assert is_st_symbol(name) == expected
