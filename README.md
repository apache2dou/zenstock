# ZenStock - A股量化分析软件

一个专注于 A 股市场的量化策略研究与回测框架。

## ✨ 核心功能

- 📥 **数据采集**：基于 AKShare/BaoStock，全市场 K 线数据本地化存储
- 💾 **本地存储**：DuckDB + Parquet 高性能列式存储
- 📈 **策略开发**：模块化策略基类，支持自定义技术指标
- 🔄 **回测引擎**：内置 A 股交易规则（T+1、涨跌停、手续费、滑点）
- 📊 **绩效分析**：胜率、赔率、夏普比率、最大回撤等核心指标
- 🖥️ **可视化**：Streamlit 交互式仪表盘

## 📁 项目结构

```
zenstock/
├── config/                  # 配置文件
│   ├── settings.yaml        # 全局设置
│   └── logging.yaml         # 日志配置
├── data/                    # 数据存储（git忽略）
│   ├── cache/               # 缓存
│   ├── parquet/             # Parquet 数据文件
│   └── zenstock.duckdb      # DuckDB 数据库
├── src/zenstock/            # 核心源码
│   ├── data/                # 数据层（采集、存储、读取）
│   ├── strategy/            # 策略层
│   ├── backtest/            # 回测引擎
│   ├── analytics/           # 绩效分析
│   ├── viz/                 # 可视化
│   └── utils/               # 工具函数
├── scripts/                 # 可执行脚本
│   ├── download_data.py     # 数据下载
│   ├── run_backtest.py      # 运行回测
│   └── migrate.py           # 数据迁移
├── strategies/              # 用户策略目录
│   └── ma_cross.py          # 示例：均线交叉策略
├── notebooks/               # Jupyter 研究
├── tests/                   # 单元测试
├── frontend/                # Streamlit 前端
│   └── app.py
├── pyproject.toml           # 项目依赖配置
└── README.md
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

# 安装依赖（开发模式）
pip install -e ".[dev]"
```

### 2. 下载数据

```bash
# 下载全 A 股日线数据（首次需要较长时间）
python scripts/download_data.py --start 2020-01-01 --end 2026-07-09

# 指定股票
python scripts/download_data.py --symbols 000001,600519 --start 2023-01-01
```

### 3. 运行回测

```bash
python scripts/run_backtest.py --strategy ma_cross --symbol 000001 --start 2023-01-01 --end 2026-07-09
```

### 4. 启动可视化

```bash
streamlit run frontend/app.py
```

## 📊 数据说明

| 数据类型 | 来源 | 频率 | 存储 |
|----------|------|------|------|
| 日K线 | AKShare | 收盘后增量 | Parquet + DuckDB |
| 分钟K线 | AKShare | 收盘后增量 | Parquet |
| 股票列表 | AKShare | 每日更新 | DuckDB |
| 复权因子 | AKShare | 每日更新 | DuckDB |
| 财务数据 | AKShare/Tushare | 季度更新 | DuckDB |

## 🔧 配置

编辑 `config/settings.yaml`：

```yaml
data:
  source: akshare       # akshare | baostock | tushare
  storage: duckdb       # duckdb | parquet | sqlite
  adjust: qfq           # qfq(前复权) | hfq(后复权) | none

backtest:
  commission: 0.00025   # 佣金费率 万2.5
  stamp_duty: 0.001     # 印花税 千1（仅卖出）
  slippage: 0.001       # 滑点 0.1%
  min_commission: 5.0   # 最低佣金 5 元
```

## 📈 示例策略

### 均线交叉策略（strategies/ma_cross.py）

```python
from zenstock.strategy import BaseStrategy
import pandas as pd

class MACrossStrategy(BaseStrategy):
    """短期均线上穿长期均线买入，下穿卖出。"""
    params = (("fast", 5), ("slow", 20))

    def on_bar(self, bar, data):
        fast_ma = data["close"].rolling(self.p.fast).mean()
        slow_ma = data["close"].rolling(self.p.slow).mean()
        if fast_ma[-1] > slow_ma[-1] and fast_ma[-2] <= slow_ma[-2]:
            return ("BUY", 1.0)   # 满仓买入
        if fast_ma[-1] < slow_ma[-1] and fast_ma[-2] >= slow_ma[-2]:
            return ("SELL", 1.0)  # 全部卖出
        return ("HOLD", 0.0)
```

## 📐 核心指标定义

| 指标 | 公式 |
|------|------|
| 胜率 | 盈利交易数 / 总交易数 |
| 赔率（盈亏比） | 平均盈利额 / 平均亏损额 |
| 期望收益 | 胜率 × 平均盈利 − 败率 × 平均亏损 |
| 夏普比率 | (年化收益 − 无风险利率) / 年化波动 |
| 最大回撤 | max((peak − trough) / peak) |
| 卡玛比率 | 年化收益 / 最大回撤 |

## 📜 许可证

MIT License - 仅供学习研究使用，不构成投资建议。
