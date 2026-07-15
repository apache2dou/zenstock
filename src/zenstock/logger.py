"""统一日志模块，基于 loguru。"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from zenstock.config import get_config

_initialized = False


def setup_logging() -> None:
    """初始化全局日志（幂等）。"""
    global _initialized
    if _initialized:
        return

    cfg = get_config().logging
    logger.remove()  # 清除默认 handler

    # 控制台输出
    if cfg.console:
        logger.add(
            sys.stderr,
            level=cfg.level,
            format="<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan> - <level>{message}</level>",
            colorize=True,
        )

    # 文件输出
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_dir / "zenstock.log",
        level="DEBUG",
        rotation=cfg.rotation,
        retention=cfg.retention,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )

    _initialized = True


def get_logger(name: str = "zenstock"):
    """获取命名 logger。"""
    if not _initialized:
        setup_logging()
    return logger.bind(name=name)
