# 开发指南

## 开发环境搭建

```bash
# 1. 克隆项目
git clone <repo-url>
cd zenstock

# 2. 创建虚拟环境
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac

# 3. 安装开发依赖
pip install -e ".[dev]"

# 4. 安装 pre-commit hooks（可选）
pre-commit install
```

## 代码结构

```
src/zenstock/
├── config.py        # 全局配置（dataclass + YAML）
├── logger.py        # loguru 日志
├── cli.py           # 命令行入口
├── data/            # 数据层
│   ├── types.py     # 类型定义（Freq, Adjust）
│   ├── downloaders.py  # 数据采集（AKShare/BaoStock）
│   └── storage.py   # 存储（DuckDB + Parquet）
├── strategy/        # 策略层
│   └── base.py      # BaseStrategy + Signal
├── backtest/        # 回测引擎
│   └── engine.py    # 事件驱动引擎
├── analytics/       # 绩效分析
│   ├── metrics.py   # 核心指标
│   └── report.py    # 报告输出
└── utils/           # 工具函数
    ├── market.py    # 股票代码工具
    └── time.py      # 时间工具
```

## 添加自定义策略

1. 在 `strategies/` 目录新建文件，例如 `my_strategy.py`
2. 继承 `BaseStrategy`，实现 `on_bar` 方法

```python
from zenstock.strategy import BaseStrategy, Signal
import pandas as pd

class Strategy(BaseStrategy):  # 类名必须叫 Strategy
    params = (("threshold", 0.05),)

    def on_bar(self, i: int, df: pd.DataFrame) -> Signal:
        if i < 5:
            return Signal.hold()
        ret = df["close"].pct_change(5).iloc[i]
        if ret > self.p.threshold:
            return Signal.sell(reason="涨多了")
        if ret < -self.p.threshold:
            return Signal.buy(reason="跌多了")
        return Signal.hold()
```

3. 运行：

```bash
python scripts/run_backtest.py --symbol 000001 --strategy strategies.my_strategy
```

## 添加自定义数据源

1. 继承 `BaseDownloader`，实现 `fetch_stock_list` 和 `fetch_klines`
2. 在 `get_downloader` 工厂函数中注册

## 运行测试

```bash
pytest                          # 全部测试
pytest tests/test_analytics.py  # 指定文件
pytest -v --cov=zenstock        # 带覆盖率
```

## 代码规范

- 使用 `ruff` 进行 lint
- 使用 `black` 格式化
- 使用 `mypy` 类型检查

```bash
ruff check src/ tests/
black src/ tests/
mypy src/
```

## 常见问题

### Q: TA-Lib 安装失败？

TA-Lib 是 C 库，需要先安装底层依赖：

**Windows**：下载 [预编译包](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)

**Mac**：`brew install ta-lib && pip install TA-Lib`

**Linux**：`sudo apt install ta-lib && pip install TA-Lib`

如果安装困难，项目默认使用 `pandas-ta` 作为替代（纯 Python）。

### Q: AKShare 被限流怎么办？

编辑 `config/settings.yaml`，增大 `request_sleep`：

```yaml
data:
  request_sleep: 1.0   # 增大到 1 秒
```
