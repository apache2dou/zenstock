"""全局配置管理：加载 YAML 配置并提供单例访问。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# 项目根目录：pyproject.toml 所在目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "settings.yaml"


@dataclass
class DataConfig:
    """数据层配置。"""
    source: str = "akshare"
    storage: str = "duckdb"
    data_dir: str = "data"
    adjust: str = "qfq"
    default_freq: str = "D"
    incremental: bool = True
    request_sleep: float = 0.3
    max_retries: int = 3

    @property
    def data_path(self) -> Path:
        return PROJECT_ROOT / self.data_dir

    @property
    def parquet_path(self) -> Path:
        p = self.data_path / "parquet"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def duckdb_path(self) -> Path:
        return self.data_path / "zenstock.duckdb"

    @property
    def cache_path(self) -> Path:
        p = self.data_path / "cache"
        p.mkdir(parents=True, exist_ok=True)
        return p


@dataclass
class BacktestConfig:
    """回测引擎配置。"""
    initial_capital: float = 100_000.0
    commission: float = 0.00025
    stamp_duty: float = 0.001
    transfer_fee: float = 0.00001
    min_commission: float = 5.0
    slippage: float = 0.001
    t_plus_1: bool = True
    price_limit_pct: float = 10.0
    risk_free_rate: float = 0.025
    benchmark: str = "000300"


@dataclass
class LoggingConfig:
    """日志配置。"""
    level: str = "INFO"
    log_dir: str = "logs"
    rotation: str = "10 MB"
    retention: str = "30 days"
    console: bool = True


@dataclass
class TushareConfig:
    """Tushare 配置（可选）。"""
    token: str = ""


@dataclass
class Config:
    """全局配置聚合。"""
    data: DataConfig = field(default_factory=DataConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tushare: TushareConfig = field(default_factory=TushareConfig)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        return cls(
            data=DataConfig(**raw.get("data", {})),
            backtest=BacktestConfig(**raw.get("backtest", {})),
            logging=LoggingConfig(**raw.get("logging", {})),
            tushare=TushareConfig(**raw.get("tushare", {})),
        )


# 单例缓存
_config: Config | None = None


def load_config(path: str | os.PathLike | None = None) -> Config:
    """加载 YAML 配置文件。"""
    global _config
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if path.exists():
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}
    _config = Config.from_dict(raw)
    return _config


def get_config() -> Config:
    """获取全局配置单例（惰性加载）。"""
    global _config
    if _config is None:
        load_config()
    return _config


def reload_config(path: str | os.PathLike | None = None) -> Config:
    """重新加载配置（用于热更新）。"""
    return load_config(path)
