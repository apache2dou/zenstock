"""pytest 配置：自动将 src 加入 sys.path，无需 pip install -e。"""

import sys
from pathlib import Path

src = Path(__file__).resolve().parent / "src"
if src.exists() and str(src) not in sys.path:
    sys.path.insert(0, str(src))
