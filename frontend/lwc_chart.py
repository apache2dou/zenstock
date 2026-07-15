"""专业行情图表组件（基于 lightweight-charts JS / TradingView 内核）。

通过 st.components.v1.html 嵌入，提供真正的行情软件级交互：
- 鼠标滚轮直接缩放（无需切换模式）
- 拖拽平移（按住鼠标拖动）
- 十字光标自动跟随
- K线+成交量+均线 一体化
- 笔/中枢叠加
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit.components.v1 as components

# lightweight-charts JS CDN（TradingView 开源版）
# 必须用 v4.x（v5 API 不兼容：v5 用 addSeries(CandlestickSeries) 而非 addCandlestickSeries）
_LWC_JS = "https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"


def render_kline_chart(
    data: pd.DataFrame,
    symbol: str = "",
    title: str = "",
    bi_list: list | None = None,
    zs_list: list | None = None,
    segment_list: list | None = None,
    height: int = 600,
    show_volume: bool = True,
    show_ma: bool = True,
    ma_periods: tuple[int, ...] = (5, 10, 20),
    fullscreen: bool = False,
) -> None:
    """渲染专业行情图表（TradingView 级别交互）。

    Args:
        data: K 线 DataFrame（date/open/high/low/close/volume）
        symbol: 股票代码
        title: 标题
        bi_list: 缠论笔列表（可选）
        zs_list: 缠论中枢列表（可选）
        segment_list: 缠论线段列表（可选，会在图上画粗线）
        height: 图表高度（正常模式）
        show_volume: 是否显示成交量
        show_ma: 是否显示均线
        ma_periods: 均线周期
        fullscreen: 是否大图模式（高度撑满屏幕）
    """
    df = data.copy()
    # 确保 date 是字符串格式（JS 需要 timestamp）
    df["date"] = pd.to_datetime(df["date"])

    # 生成 JS 数据
    candles = []
    volumes = []
    for _, row in df.iterrows():
        ts = int(row["date"].timestamp())
        candles.append({
            "time": ts,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        })
        vol = float(row.get("volume", 0) or 0)
        # 成交量颜色：涨红跌绿
        color = "rgba(255,152,0,0.8)" if row["close"] >= row["open"] else "rgba(33,150,243,0.8)"
        volumes.append({"time": ts, "value": vol, "color": color})

    # 均线数据
    ma_lines = {}
    if show_ma:
        for n in ma_periods:
            ma = df["close"].rolling(n, min_periods=1).mean()
            ma_data = []
            for idx in range(len(df)):
                ts = int(df["date"].iloc[idx].timestamp())
                val = ma.iloc[idx]
                ma_data.append({"time": ts, "value": float(val) if pd.notna(val) else None})
            ma_lines[f"MA{n}"] = ma_data

    # 笔数据（折线）
    bi_series_up = []
    bi_series_down = []
    if bi_list:
        _build_bi_series(bi_list, bi_series_up, bi_series_down)

    # 中枢数据（用矩形区间标注）
    zs_markers = []
    if zs_list:
        _build_zs_markers(zs_list, zs_markers)

    # 线段数据（粗折线，比笔更醒目）
    seg_series_up = []
    seg_series_down = []
    if segment_list:
        _build_segment_series(segment_list, seg_series_up, seg_series_down)

    # 构建 HTML
    import json
    actual_height = height if not fullscreen else 1000
    html = _build_html(
        symbol=symbol,
        title=title,
        candles=candles,
        volumes=volumes,
        ma_lines=ma_lines,
        bi_series_up=bi_series_up,
        bi_series_down=bi_series_down,
        seg_series_up=seg_series_up,
        seg_series_down=seg_series_down,
        zs_markers=zs_markers,
        height=actual_height,
        show_volume=show_volume,
    )

    components.html(html, height=actual_height + 40, scrolling=False)


def _build_bi_series(bi_list, up_list, down_list):
    """从笔列表构建向上/向下折线数据。"""
    for bi in bi_list:
        try:
            fx_a = getattr(bi, "fx_a", None)
            fx_b = getattr(bi, "fx_b", None)
            if fx_a is None or fx_b is None:
                continue
            dt_a = getattr(fx_a, "dt", None)
            dt_b = getattr(fx_b, "dt", None)
            if dt_a is None or dt_b is None:
                continue
            import pandas as _pd
            ts_a = int(_pd.Timestamp(dt_a).timestamp())
            ts_b = int(_pd.Timestamp(dt_b).timestamp())
            p_a = float(getattr(fx_a, "fx", 0) or 0)
            p_b = float(getattr(fx_b, "fx", 0) or 0)
            direction = str(getattr(bi, "direction", ""))
            is_up = direction.endswith("a") or "up" in direction.lower() or "上" in direction
            target = up_list if is_up else down_list
            target.append({"time": ts_a, "value": p_a})
            target.append({"time": ts_b, "value": p_b})
        except Exception:  # noqa: BLE001
            continue


def _build_zs_markers(zs_list, markers):
    """构建中枢标注数据。"""
    import pandas as _pd
    for zs in zs_list:
        try:
            zg = float(getattr(zs, "zg", 0) or 0)
            zd = float(getattr(zs, "zd", 0) or 0)
            if zg <= 0 or zd <= 0 or zg <= zd:
                continue
            sdt = getattr(zs, "sdt", None)
            edt = getattr(zs, "edt", None)
            if sdt is None or edt is None:
                bis = getattr(zs, "bis", [])
                if len(bis) >= 2:
                    sdt = getattr(getattr(bis[0], "fx_a", None), "dt", None)
                    edt = getattr(getattr(bis[-1], "fx_b", None), "dt", None)
            if sdt and edt:
                markers.append({
                    "time": int(_pd.Timestamp(sdt).timestamp()),
                    "endTime": int(_pd.Timestamp(edt).timestamp()),
                    "top": zg,
                    "bottom": zd,
                })
        except Exception:  # noqa: BLE001
            continue


def _build_segment_series(segment_list, up_list, down_list):
    """从线段列表构建向上/向下折线数据（比笔更粗）。"""
    import pandas as _pd
    for seg in segment_list:
        try:
            start_dt = getattr(seg, "start_dt", None) or getattr(seg, "sdt", None)
            end_dt = getattr(seg, "end_dt", None) or getattr(seg, "edt", None)
            if start_dt is None or end_dt is None:
                continue
            start_price = float(getattr(seg, "start_price", 0) or 0)
            end_price = float(getattr(seg, "end_price", 0) or 0)
            if start_price <= 0 or end_price <= 0:
                continue
            ts_a = int(_pd.Timestamp(start_dt).timestamp())
            ts_b = int(_pd.Timestamp(end_dt).timestamp())
            is_up = bool(getattr(seg, "is_up", False))
            target = up_list if is_up else down_list
            target.append({"time": ts_a, "value": start_price})
            target.append({"time": ts_b, "value": end_price})
        except Exception:  # noqa: BLE001
            continue


def _build_html(
    symbol, title, candles, volumes, ma_lines,
    bi_series_up, bi_series_down,
    seg_series_up, seg_series_down,
    zs_markers,
    height, show_volume,
):
    """构建完整的 HTML 页面。"""
    import json

    candles_json = json.dumps(candles)
    volumes_json = json.dumps(volumes)
    ma_json = json.dumps(ma_lines)
    bi_up_json = json.dumps(bi_series_up)
    bi_down_json = json.dumps(bi_series_down)
    seg_up_json = json.dumps(seg_series_up)
    seg_down_json = json.dumps(seg_series_down)
    zs_json = json.dumps(zs_markers)

    # MA 颜色
    # 色弱友好均线配色：白/紫/青绿（避开红绿依赖）
    ma_colors = {"MA5": "#FFFFFF", "MA10": "#CE93D8", "MA20": "#80CBC4"}

    # 生成创建 MA 线的 JS 代码
    ma_js_lines = []
    for name, color in ma_colors.items():
        if name in ma_lines:
            ma_js_lines.append(f"""
            var {name.lower()}_series = chart.addLineSeries({{
                color: '{color}', lineWidth: 1, priceLineVisible: false,
                lastValueVisible: false, crosshairMarkerVisible: false,
            }});
            {name.lower()}_series.setData({name}_data);""")
    ma_js = "\n".join(ma_js_lines)

    # 生成笔的 JS
    bi_js = ""
    if bi_series_up:
        bi_js += f"""
        var bi_up = chart.addLineSeries({{
            color: '#FFD600', lineWidth: 2, priceLineVisible: false,
            lastValueVisible: false, crosshairMarkerVisible: true,
        }});
        bi_up.setData(bi_up_data);"""
    if bi_series_down:
        bi_js += f"""
        var bi_down = chart.addLineSeries({{
            color: '#00B0FF', lineWidth: 2, priceLineVisible: false,
            lastValueVisible: false, crosshairMarkerVisible: true,
        }});
        bi_down.setData(bi_down_data);"""

    # 生成线段的 JS（比笔更粗，更显眼）
    seg_js = ""
    if seg_series_up:
        seg_js += f"""
        var seg_up = chart.addLineSeries({{
            color: '#76FF03', lineWidth: 4, priceLineVisible: false,
            lastValueVisible: false, crosshairMarkerVisible: true,
        }});
        seg_up.setData(seg_up_data);"""
    if seg_series_down:
        seg_js += f"""
        var seg_down = chart.addLineSeries({{
            color: '#FF1744', lineWidth: 4, priceLineVisible: false,
            lastValueVisible: false, crosshairMarkerVisible: true,
        }});
        seg_down.setData(seg_down_data);"""

    # 中枢矩形（用 area series 模拟半透明色块）
    zs_js = ""
    if zs_markers:
        zs_js = """
        zs_markers.forEach(function(zs, idx) {
            // 上沿面积线（从 top 到 bottom 填充，形成矩形色块）
            var zsArea = chart.addAreaSeries({
                topColor: 'rgba(255,193,7,0.25)',
                bottomColor: 'rgba(255,193,7,0.25)',
                lineWidth: 1,
                lineColor: 'rgba(255,193,7,0.8)',
                priceLineVisible: false,
                lastValueVisible: false,
                crosshairMarkerVisible: false,
            });
            // 用4个点构成矩形：左上→右上→右下→左下（area 填充到底部基线）
            // 分成上下两条 area：上沿线 + 下沿线，中间填充
            var topLine = chart.addLineSeries({
                color: 'rgba(255,193,7,0.9)', lineWidth: 2,
                priceLineVisible: false, lastValueVisible: false,
                crosshairMarkerVisible: false,
            });
            topLine.setData([
                {time: zs.time, value: zs.top},
                {time: zs.endTime, value: zs.top},
            ]);
            var botLine = chart.addLineSeries({
                color: 'rgba(255,193,7,0.9)', lineWidth: 2,
                priceLineVisible: false, lastValueVisible: false,
                crosshairMarkerVisible: false,
            });
            botLine.setData([
                {time: zs.time, value: zs.bottom},
                {time: zs.endTime, value: zs.bottom},
            ]);
            // 左右竖线
            var leftLine = chart.addLineSeries({
                color: 'rgba(255,193,7,0.5)', lineWidth: 1,
                priceLineVisible: false, lastValueVisible: false,
                crosshairMarkerVisible: false,
            });
            leftLine.setData([
                {time: zs.time, value: zs.bottom},
                {time: zs.time, value: zs.top},
            ]);
            var rightLine = chart.addLineSeries({
                color: 'rgba(255,193,7,0.5)', lineWidth: 1,
                priceLineVisible: false, lastValueVisible: false,
                crosshairMarkerVisible: false,
            });
            rightLine.setData([
                {time: zs.endTime, value: zs.bottom},
                {time: zs.endTime, value: zs.top},
            ]);
        });"""

    vol_height = 80 if show_volume else 0
    main_height = height - vol_height - 30

    # 成交量副图 JS（单独构建避免 f-string 花括号冲突）
    if show_volume:
        vol_js = """var volumeSeries = chart.addHistogramSeries({
            color: "#2196F3",
            priceFormat: {type: "volume"},
            priceScaleId: "vol",
        });
        chart.priceScale("vol").applyOptions({
            scaleMargins: {top: 0.8, bottom: 0},
        });
        volumeSeries.setData(volumeData);"""
    else:
        vol_js = ""

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    body {{ margin: 0; padding: 0; background: #1e1e2e; }}
    #chart-container {{ position: relative; }}
    #chart {{ width: 100%; }}
    #title {{
        color: #cdd6f4; font-size: 14px; font-weight: bold;
        padding: 8px 12px; background: #1e1e2e;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
</style>
</head>
<body>
<div id="chart-container">
    <div id="title">{title or symbol}</div>
    <div id="chart"></div>
</div>
<script src="{_LWC_JS}"></script>
<script>
var candles = {candles_json};
var volumeData = {volumes_json};
var MA5_data = (function(){{var d={ma_json};return d["MA5"]||[];}})();
var MA10_data = (function(){{var d={ma_json};return d["MA10"]||[];}})();
var MA20_data = (function(){{var d={ma_json};return d["MA20"]||[];}})();
var bi_up_data = {bi_up_json};
var bi_down_data = {bi_down_json};
var seg_up_data = {seg_up_json};
var seg_down_data = {seg_down_json};
var zs_markers = {zs_json};

var chart = LightweightCharts.createChart(document.getElementById('chart'), {{
    layout: {{
        background: {{ type: 'solid', color: '#1e1e2e' }},
        textColor: '#cdd6f4',
        fontSize: 11,
    }},
    grid: {{
        vertLines: {{ color: 'rgba(127,127,127,0.1)' }},
        horzLines: {{ color: 'rgba(127,127,127,0.1)' }},
    }},
    crosshair: {{
        mode: LightweightCharts.CrosshairMode.Normal,
        vertLine: {{
            color: '#FFD54F', width: 1, style: 0,
            labelBackgroundColor: '#FF9800',
        }},
        horzLine: {{
            color: '#FFD54F', width: 1, style: 0,
            labelBackgroundColor: '#2196F3',
        }},
    }},
    rightPriceScale: {{
        borderColor: 'rgba(127,127,127,0.3)',
        scaleMargins: {{ top: 0.05, bottom: 0.25 }},
    }},
    timeScale: {{
        borderColor: 'rgba(127,127,127,0.3)',
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 5,
    }},
    width: 600,
    height: {height},
}});

// K 线主图
var candleSeries = chart.addCandlestickSeries({{
    // 色弱友好配色：涨=橙黄（暖色高亮），跌=蓝（冷色高饱和）
    // 橙与蓝是色弱用户最容易区分的互补色对
    upColor: '#FF9800',
    downColor: '#2196F3',
    borderUpColor: '#FFB300',
    borderDownColor: '#1976D2',
    wickUpColor: '#FFB300',
    wickDownColor: '#1976D2',
}});
candleSeries.setData(candles);

// 均线
{ma_js}

// 笔
{bi_js}

// 线段
{seg_js}

// 中枢
{zs_js}

// 成交量副图
{vol_js}

// 自动适应容器宽度
new ResizeObserver(function() {{
    chart.applyOptions({{ width: document.getElementById('chart').clientWidth }});
}}).observe(document.getElementById('chart'));

// 默认显示最近一段
chart.timeScale().fitContent();

// ===== 全屏功能（突破 iframe 限制）=====
// 利用 allow-same-origin 访问 window.top，把 iframe 自身撑满父窗口
var isFs = false;
// 保存原始样式以便恢复
var origIframeStyle = '';

function toggleFullscreen() {{
    isFs = !isFs;
    try {{
        // 找到自身所在的 iframe 元素（从父窗口看）
        var topDoc = window.top.document;
        var myIframe = null;
        // 遍历父窗口的所有 iframe，找到自己
        var allIframes = topDoc.querySelectorAll('iframe');
        for (var i = 0; i < allIframes.length; i++) {{
            if (allIframes[i].contentWindow === window) {{
                myIframe = allIframes[i];
                break;
            }}
        }}
        if (!myIframe) return;

        if (isFs) {{
            // 保存原始样式
            origIframeStyle = myIframe.style.cssText;
            // 撑满父窗口
            myIframe.style.cssText = 'position:fixed !important;top:0 !important;left:0 !important;width:100vw !important;height:100vh !important;z-index:2147483647 !important;background:#1e1e2e !important;border:none !important;';
            // 调整图表尺寸
            setTimeout(function() {{
                chart.applyOptions({{
                    width: window.innerWidth,
                    height: window.innerHeight - 35,
                }});
                chart.timeScale().fitContent();
            }}, 80);
        }} else {{
            // 恢复
            myIframe.style.cssText = origIframeStyle;
            setTimeout(function() {{
                chart.applyOptions({{
                    width: document.getElementById('chart').clientWidth,
                    height: {height},
                }});
                chart.timeScale().fitContent();
            }}, 80);
        }}
    }} catch(e) {{
        // 跨域时回退到 CSS 全屏
        var container = document.getElementById('chart-container');
        if (isFs) {{
            container.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;z-index:99999;background:#1e1e2e;';
            setTimeout(function() {{
                chart.applyOptions({{ width: window.innerWidth, height: window.innerHeight - 35 }});
                chart.timeScale().fitContent();
            }}, 80);
        }} else {{
            container.style.cssText = 'position:relative;';
            setTimeout(function() {{
                chart.applyOptions({{ width: document.getElementById('chart').clientWidth, height: {height} }});
                chart.timeScale().fitContent();
            }}, 80);
        }}
    }}
}}

// 添加全屏按钮（在 iframe 内部）
var fsBtn = document.createElement('button');
fsBtn.innerHTML = '&#x26F6;';
fsBtn.style.cssText = 'position:fixed;top:8px;right:12px;z-index:999;background:rgba(255,152,0,0.9);color:#1e1e2e;border:none;border-radius:4px;padding:6px 12px;font-size:16px;font-weight:bold;cursor:pointer;font-family:sans-serif;';
fsBtn.title = '全屏 / 退出全屏';
fsBtn.onclick = toggleFullscreen;
document.body.appendChild(fsBtn);
</script>
</body>
</html>"""
