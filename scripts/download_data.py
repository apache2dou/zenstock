"""数据下载脚本。

用法:
    # 下载全市场日线（默认 2020 至今）
    python scripts/download_data.py

    # 指定股票和日期范围
    python scripts/download_data.py --symbols 000001,600519 --start 2023-01-01

    # 仅更新股票列表
    python scripts/download_data.py --list-only
"""

from __future__ import annotations

import sys
from pathlib import Path

# 将 src 加入 path（支持非 pip install -e 场景）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zenstock.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
