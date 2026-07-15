"""多级别缠论分析器：笔、线段、中枢、买卖点。"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from zenstock.chanlun.adapter import df_to_bars
from zenstock.config import get_config
from zenstock.data.types import Freq
from zenstock.logger import get_logger

log = get_logger(__name__)


@dataclass
class ChanlunResult:
    """单级别缠论分析结果。"""
    freq: str                          # 频率名称
    bars_count: int = 0                # K 线数量
    bi_list: list = field(default_factory=list)      # 笔列表
    fx_list: list = field(default_factory=list)      # 分型列表
    zs_list: list = field(default_factory=list)      # 中枢列表
    finished_bis: list = field(default_factory=list) # 已完成的笔

    @property
    def bi_count(self) -> int:
        return len(self.bi_list)

    @property
    def zs_count(self) -> int:
        return len(self.zs_list)

    @property
    def fx_count(self) -> int:
        return len(self.fx_list)

    def latest_bi(self) -> Any | None:
        """获取最新一笔。"""
        return self.bi_list[-1] if self.bi_list else None

    def latest_zs(self) -> Any | None:
        """获取最新中枢。"""
        return self.zs_list[-1] if self.zs_list else None

    def summary(self) -> str:
        return (
            f"[{self.freq}] bars={self.bars_count}, "
            f"笔={self.bi_count}, 中枢={self.zs_count}"
        )


@dataclass
class MultiLevelResult:
    """多级别（1/5/30/日）联立分析结果。"""
    results: dict[str, ChanlunResult] = field(default_factory=dict)

    def get(self, freq: str) -> ChanlunResult | None:
        return self.results.get(freq)

    def all_summaries(self) -> list[str]:
        return [r.summary() for r in self.results.values()]


class ChanlunAnalyzer:
    """缠论分析器：单级别 + 多级别联立分析。

    使用 czsc 库的核心算法（分型识别、笔、中枢）。
    """

    def analyze_single(
        self, df: pd.DataFrame, freq: Freq | str,
        symbol: str = "", use_cache: bool = True,
    ) -> ChanlunResult:
        """对单级别 K 线数据做缠论分析。

        Args:
            df: zenstock 标准 K 线 DataFrame
            freq: 频率
            symbol: 股票代码（用于缓存）
            use_cache: 是否使用缓存

        Returns:
            ChanlunResult
        """
        from czsc import CZSC  # type: ignore

        freq_enum = Freq(freq) if isinstance(freq, str) else freq

        # 尝试从缓存加载
        if use_cache and symbol:
            cached = _load_cached_result(symbol, freq_enum, df)
            if cached is not None:
                log.debug(f"缠论结果命中缓存: {symbol} {freq_enum.display_name}")
                return cached

        bars = df_to_bars(df, freq_enum)

        if len(bars) < 10:
            log.warning(f"[{freq_enum.display_name}] 数据不足（{len(bars)} 根），至少需要 10 根")
            return ChanlunResult(freq=freq_enum.display_name, bars_count=len(bars))

        try:
            # czsc 0.10: CZSC 接受整个 bars 列表作为位置参数
            czsc_obj = CZSC(bars)
        except Exception as e:  # noqa: BLE001
            log.error(f"czsc 分析失败 [{freq_enum.display_name}]: {e}")
            return ChanlunResult(freq=freq_enum.display_name, bars_count=len(bars))

        result = ChanlunResult(
            freq=freq_enum.display_name,
            bars_count=len(bars),
            bi_list=list(czsc_obj.bi_list),
            fx_list=list(getattr(czsc_obj, "fx_list", [])),
            zs_list=_extract_zhongshu(list(czsc_obj.bi_list)),
            finished_bis=list(getattr(czsc_obj, "finished_bis", [])),
        )

        # 缓存结果
        if use_cache and symbol:
            _save_cached_result(symbol, freq_enum, df, result)

        return result

    def analyze_multi_level(
        self,
        data_by_freq: dict[Freq, pd.DataFrame],
    ) -> MultiLevelResult:
        """多级别联立分析。

        Args:
            data_by_freq: {Freq: DataFrame} 多个频率的 K 线数据

        Returns:
            MultiLevelResult
        """
        result = MultiLevelResult()
        for freq, df in data_by_freq.items():
            single = self.analyze_single(df, freq)
            result.results[freq.display_name] = single
        return result

    # ==================== 买卖点识别 ====================
    @staticmethod
    def detect_buy_signals(result: ChanlunResult) -> list[dict]:
        """识别一买/二买/三买信号。

        缠论买卖点简化判定：
        - 一买：下跌趋势中最后一个中枢被向下突破后的背驰段结束
        - 二买：一买后的反弹回调不破一买低点
        - 三买：中枢上沿被突破后回调不进中枢

        这里使用 czsc 提供的信号函数（如果可用），否则用简化启发式。
        """
        signals: list[dict] = []
        if result.bi_count < 3:
            return signals

        # 简化启发式：基于最后几笔判断
        bis = result.bi_list
        try:
            # 最后三笔
            b1, b2, b3 = bis[-3], bis[-2], bis[-1]

            # 三买启发式：b3 向上且未跌破中枢
            zs = result.latest_zs()
            if zs is not None:
                zg = getattr(zs, "zg", 0)  # 中枢上沿
                # 最后一笔向上且低点高于中枢上沿 → 三买
                if _is_up_bi(b3) and _bi_low(b3) > zg:
                    signals.append({
                        "type": "三买",
                        "dt": _bi_end_dt(b3),
                        "price": _bi_end_price(b3),
                        "reason": "回调不进中枢",
                    })

            # 一买启发式：连续3笔下移且最后笔力度减弱
            if (_is_down_bi(b1) and _is_down_bi(b3)
                    and _bi_amp(b3) < _bi_amp(b1)):
                signals.append({
                    "type": "一买(背驰)",
                    "dt": _bi_end_dt(b3),
                    "price": _bi_end_price(b3),
                    "reason": "下跌力度减弱（背驰）",
                })
        except Exception as e:  # noqa: BLE001
            log.debug(f"信号识别异常: {e}")

        return signals

    @staticmethod
    def detect_sell_signals(result: ChanlunResult) -> list[dict]:
        """识别一卖/二卖/三卖信号。"""
        signals: list[dict] = []
        if result.bi_count < 3:
            return signals

        bis = result.bi_list
        try:
            b1, b2, b3 = bis[-3], bis[-2], bis[-1]

            # 三卖启发式：向下突破中枢后反弹不进中枢
            zs = result.latest_zs()
            if zs is not None:
                zd = getattr(zs, "zd", 0)  # 中枢下沿
                if _is_down_bi(b3) and _bi_high(b3) < zd:
                    signals.append({
                        "type": "三卖",
                        "dt": _bi_end_dt(b3),
                        "price": _bi_end_price(b3),
                        "reason": "反弹不进中枢",
                    })

            # 一卖启发式：连续3笔上移且最后笔力度减弱
            if (_is_up_bi(b1) and _is_up_bi(b3)
                    and _bi_amp(b3) < _bi_amp(b1)):
                signals.append({
                    "type": "一卖(背驰)",
                    "dt": _bi_end_dt(b3),
                    "price": _bi_end_price(b3),
                    "reason": "上涨力度减弱（背驰）",
                })
        except Exception as e:  # noqa: BLE001
            log.debug(f"信号识别异常: {e}")

        return signals


# ==================== 缠论结果缓存 ====================
def _chanlun_cache_dir() -> Path:
    """缠论分析缓存目录。"""
    d = get_config().data.cache_path / "chanlun"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _data_fingerprint(df: pd.DataFrame) -> str:
    """计算 K 线数据的指纹（用于缓存失效）。"""
    if df.empty:
        return "empty"
    h = hashlib.md5()
    h.update(str(df["date"].min()).encode())
    h.update(str(df["date"].max()).encode())
    h.update(str(len(df)).encode())
    h.update(str(round(float(df["close"].iloc[-1]), 4)).encode())
    return h.hexdigest()[:12]


def _load_cached_result(
    symbol: str, freq: Freq, df: pd.DataFrame
) -> ChanlunResult | None:
    """从缓存加载缠论分析结果。"""
    fp = _data_fingerprint(df)
    cache_path = _chanlun_cache_dir() / f"{symbol}_{freq.value}_{fp}.pkl"
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    except Exception:  # noqa: BLE001
        return None


def _save_cached_result(
    symbol: str, freq: Freq, df: pd.DataFrame, result: ChanlunResult
) -> None:
    """保存缠论分析结果到缓存。"""
    if result.bi_count == 0:
        return
    fp = _data_fingerprint(df)
    cache_path = _chanlun_cache_dir() / f"{symbol}_{freq.value}_{fp}.pkl"
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)
        # 清理同 symbol 同 freq 的旧缓存
        for old in _chanlun_cache_dir().glob(f"{symbol}_{freq.value}_*.pkl"):
            if old != cache_path:
                old.unlink(missing_ok=True)
    except Exception as e:  # noqa: BLE001
        log.debug(f"缓存写入失败: {e}")


# ==================== BI 工具函数 ====================
def _extract_zhongshu(bi_list: list) -> list:
    """从笔列表中识别中枢。

    缠论中枢定义：连续三笔（同向相邻）的价格区间有重叠部分，重叠区间即为中枢。
    中枢上沿 zg = min(三笔的最高点)
    中枢下沿 zd = max(三笔的最低点)
    若 zg > zd，则形成有效中枢。

    Args:
        bi_list: czsc BI 对象列表

    Returns:
        list[czsc.ZS]
    """
    if len(bi_list) < 3:
        return []

    from czsc import ZS  # type: ignore

    zs_list: list = []
    i = 0
    while i <= len(bi_list) - 3:
        b1, b2, b3 = bi_list[i], bi_list[i + 1], bi_list[i + 2]
        # 三笔的高点、低点
        g1, g2, g3 = _bi_high(b1), _bi_high(b2), _bi_high(b3)
        d1, d2, d3 = _bi_low(b1), _bi_low(b2), _bi_low(b3)

        # 中枢区间：重叠部分
        zg = min(max(g1, d1), max(g2, d2), max(g3, d3))  # 上沿 = 三笔各自高点的最小值
        zd = max(min(g1, d1), min(g2, d2), min(g3, d3))  # 下沿 = 三笔各自低点的最大值

        if zg > zd:
            # 形成有效中枢
            try:
                zs = ZS(bis=[b1, b2, b3])
                zs_list.append(zs)
            except Exception:  # noqa: BLE001
                # 如果 ZS 构造失败，用自定义对象兜底
                zs_list.append(_SimpleZS(zg=zg, zd=zd, bis=[b1, b2, b3]))
            i += 3  # 跳过已处理的笔
        else:
            i += 1

    return zs_list


@dataclass
class _SimpleZS:
    """中枢的兜底实现（当 czsc.ZS 构造失败时使用）。"""
    zg: float = 0.0
    zd: float = 0.0
    zz: float = 0.0
    bis: list = field(default_factory=list)
    sdt: Any = None
    edt: Any = None


def _is_up_bi(bi: Any) -> bool:
    """判断笔方向是否向上。"""
    direction = str(getattr(bi, "direction", ""))
    return "up" in direction.lower() or "上" in direction


def _is_down_bi(bi: Any) -> bool:
    direction = str(getattr(bi, "direction", ""))
    return "down" in direction.lower() or "下" in direction


def _bi_high(bi: Any) -> float:
    return float(getattr(bi, "high", 0) or getattr(bi, "fx_b", {}).get("high", 0) or 0)


def _bi_low(bi: Any) -> float:
    return float(getattr(bi, "low", 0) or getattr(bi, "fx_b", {}).get("low", 0) or 0)


def _bi_amp(bi: Any) -> float:
    """笔的幅度。"""
    hi, lo = _bi_high(bi), _bi_low(bi)
    return abs(hi - lo) if lo > 0 else 0


def _bi_end_dt(bi: Any):
    """笔结束时间。"""
    # czsc BI 对象通常有 fx_b (结束分型)
    fx_b = getattr(bi, "fx_b", None)
    if fx_b is not None:
        return getattr(fx_b, "dt", None)
    return getattr(bi, "dt", None)


def _bi_end_price(bi: Any) -> float:
    """笔结束价格。"""
    fx_b = getattr(bi, "fx_b", None)
    if fx_b is not None:
        return float(getattr(fx_b, "fx", 0) or 0)
    if _is_up_bi(bi):
        return _bi_high(bi)
    return _bi_low(bi)
