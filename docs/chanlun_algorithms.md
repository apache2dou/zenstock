# 缠论核心算法概要

> 提炼自《缠中说禅：教你炒股票》原文，涵盖从 K 线到买卖点的完整算法体系。

---

## 一、K 线包含关系处理

### 定义
相邻两根 K 线，一根的高低点全在另一根范围内，形成包含关系。

### 算法

```
输入: K线序列 k[0..n]
输出: 无包含关系的标准化K线序列

function 处理包含关系(k_lines):
    result = []
    i = 0
    while i < len(k_lines):
        if i == 0:
            result.append(k_lines[i]); i++; continue
        
        if 有包含关系(result[-1], k_lines[i]):
            # 确定方向：用前一根非包含K线判断
            direction = 向上 if result[-1].high >= result[-2].high else 向下
            
            if direction == 向上:
                # 取高高、低高中较高者
                new_high = max(result[-1].high, k_lines[i].high)
                new_low  = max(result[-1].low,  k_lines[i].low)
            else:  # 向下
                # 取低低、高低中较低者
                new_low  = min(result[-1].low,  k_lines[i].low)
                new_high = min(result[-1].high, k_lines[i].high)
            
            result[-1] = K线(new_high, new_low)  # 合并替换
        else:
            result.append(k_lines[i])
        i += 1
    
    return result
```

### 关键规则
- **顺序处理**：必须从左到右依次处理，不跳级
- **方向判定**：用第 n 根与第 n-1 根（非包含）比较：
  - $g_n \ge g_{n-1}$ → 向上
  - $d_n \le d_{n-1}$ → 向下
- 包含关系不具有传递性

---

## 二、分型 (Fractal)

### 定义

**顶分型**：3 根连续 K 线，中间 K 线的高点最高，低点也最高。
**底分型**：3 根连续 K 线，中间 K 线的低点最低，高点也最低。

```
顶分型:  K[i-1].high < K[i].high > K[i+1].high
         K[i-1].low  < K[i].low  > K[i+1].low

底分型:  K[i-1].low  > K[i].low  < K[i+1].low
         K[i-1].high > K[i].high < K[i+1].high
```

### 算法

```
function 识别分型(k_lines):
    fx_list = []
    for i in 1..len(k_lines)-2:
        if 是顶分型(k_lines, i):
            fx_list.append(('顶', i, k_lines[i].high))
        elif 是底分型(k_lines, i):
            fx_list.append(('底', i, k_lines[i].low))
    return fx_list
```

---

## 三、笔 (Bi / Stroke)

### 定义
两个相邻的顶和底之间构成一笔。笔内部的波动忽略不计。

### 基本规则
1. 必须是**相邻**的顶和底
2. 顶和底之间至少有一根独立 K 线（"共用K线"违反结合律）
3. 顶分型 + 下降K线 + 底分型 = 下降笔
4. 底分型 + 上升K线 + 顶分型 = 上升笔

### 算法

```
function 识别笔(fx_list):
    bi_list = []
    已处理 = set()
    
    for i in 0..len(fx_list)-1:
        if i in 已处理: continue
        
        # 找下一个相反类型的分型
        for j in i+1..len(fx_list)-1:
            if fx_list[i].type != fx_list[j].type:
                # 检查中间是否有K线
                if fx_list[j].index - fx_list[i].index >= 3:
                    direction = '向下' if fx_list[i].type == '顶' else '向上'
                    bi_list.append({
                        'direction': direction,
                        'start': fx_list[i],
                        'end': fx_list[j]
                    })
                    已处理.add(i); 已处理.add(j)
                    break  # 找下一对
    return bi_list
```

---

## 四、线段 (Line Segment)

### 定义
至少由 3 笔组成，前 3 笔必须有重叠部分。

### 关键定义

**特征序列**：以向上笔开始的线段，向下笔构成特征序列 $X_1, X_2, ..., X_n$。

**线段被笔破坏**：
- 向上线段：某向下笔的低点 $\le$ 第 i 个顶的底 ($j \ge i+2$)
- 向下线段：某向上笔的高点 $\ge$ 第 i 个底的顶 ($j \ge i+2$)

**线段终结定理**：线段被破坏的充要条件——被另一个线段破坏。

### 线段划分标准

用特征序列分型判断线段结束：

```
情况1: 特征序列第一、二元素间无缺口
  顶分型的第三元素确认 → 线段在顶分型高点结束
  底分型的第三元素确认 → 线段在底分型低点结束

情况2: 特征序列第一、二元素间有缺口
  必须等从该分型高点/低点开始的下一序列出现反向分型，才能确认
  (这是"线段破坏第二种情况"，需要更多确认)
```

### 算法

```
function 提取线段(bi_list):
    if len(bi_list) < 3: return []
    
    segments = []
    seg_start = 0
    
    while seg_start < len(bi_list) - 2:
        direction = bi_list[seg_start].direction
        # 前3笔必须有重叠
        if not 前3笔有重叠(bi_list, seg_start):
            seg_start += 1; continue
        
        # 构建特征序列 (与线段方向相反的笔)
        特征序列 = [bi for bi in bi_list[seg_start:] if bi.direction != direction]
        
        # 对特征序列做包含处理，找分型
        标准特征序列 = 包含处理(特征序列)
        分型 = 找分型(标准特征序列)
        
        if 分型 and 确认线段结束(分型, 特征序列):
            end_idx = 分型位置
            segments.append(线段(start, end_idx))
            seg_start = end_idx
        else:
            end_idx = len(bi_list) - 1
            segments.append(线段(start, end_idx))
            break
    
    return segments
```

---

## 五、中枢 (Zhongshu / Pivot)

### 核心定义

**缠中说禅走势中枢**：某级别走势类型中，被至少三个连续次级别走势类型所重叠的部分。

### 中枢区间计算

对于次级别走势 $Z_1, Z_2, Z_3$，区间为：
$$ZG = \min(g_1, g_2, g_3)$$
$$ZD = \max(d_1, d_2, d_3)$$
其中 $g_i$ 为高点的最低者，$d_i$ 为低点的最高者。

记 `GG = max(all gn)`, `DD = min(all dn)`，则有：
- **中枢延伸**：任何 Zn 的 [dn, gn] 与 [ZD, ZG] 有重叠
- **中枢新生**（趋势）：后 DD > 前 GG（上涨）或后 GG < 前 DD（下跌）
- **中枢级别扩展**：后 ZG < 前 ZD 且后 GG ≥ 前 DD（产生更高级别中枢）

### 盘整 vs 趋势
- **盘整**：只包含一个中枢的走势类型
- **趋势**：包含两个以上依次同向、互不重叠的中枢

### 算法

```
function 提取中枢(segments):
    zs_list = []
    i = 0
    while i < len(segments) - 2:
        seg1, seg2, seg3 = segments[i], segments[i+1], segments[i+2]
        # 三段必须有重叠
        zs_high = min(seg1.high, seg2.high, seg3.high)
        zs_low  = max(seg1.low,  seg2.low,  seg3.low)
        if zs_low < zs_high:  # 有重叠区间
            zs_list.append(中枢(zs_low, zs_high, i, i+2))
        i += 1
    return zs_list
```

---

## 六、背驰 (Divergence)

### 核心原理
**没有趋势，就没有背驰**。趋势中，对比同向的两个中枢连接段（离开段），后一段力度小于前一段则为背驰。

### MACD 辅助判断

标准两中枢趋势在 MACD 上的表现：
1. **第一段**：黄白线上穿 0 轴，在 0 轴上方形成第一中枢 + 第二类买点
2. **突破段**（b段）：黄白线快速拉起，力度最大
3. **第二中枢形成**：黄白线回抽 0 轴附近
4. **背驰段**（c段）：黄白线不能创新高（或柱子面积/高度不能突破），出现背驰

### 判断标准

```
function 判断背驰(c段, b段):
    # 方法1: MACD红绿柱面积比较
    if c段.MACD面积 < b段.MACD面积:
        return 背驰
    
    # 方法2: MACD黄白线高低比较
    if c段.MACD黄白线高点 < b段.MACD黄白线高点:
        return 背驰
    
    # 方法3: MACD柱子伸长高度
    if c段.MACD柱高 < b段.MACD柱高:
        return 背驰
    
    # 辅助: 价格比较
    if c段价格创新高但力度不足:
        return 顶背驰
    if c段价格创新低但力度不足:
        return 底背驰
    
    return 无背驰
```

### 盘整背驰
- 盘整中比较同向的两段力度
- 判断标准相同，但盘整背驰后不必然趋势反转，可能继续盘整

---

## 七、区间套 (Interval Nested)

### 原理
在大级别背驰段内，通过次级别背驰精确定位转折点。

### 算法（递归下降）
```
function 区间套精确定位(级别):
    if 当前级别没有背驰:
        return None
    
    背驰段 = 找背驰段(级别)
    次级别背驰 = 区间套精确定位(级别 - 1)
    
    if 次级别背驰:
        return 次级别背驰.终点  # 最精确的买/卖点
    else:
        return 背驰段.终点       # 本级背驰结束点
```

---

## 八、三类买卖点

### 第一类买卖点
- **第一类买点**：下跌趋势背驰后，最后一个中枢下方的最低点
- **第一类卖点**：上涨趋势背驰后，最后一个中枢上方的最高点

### 第二类买卖点
- 第一类买点后，次级别回抽的低点（不创新低）
- 第一类卖点后，次级别反弹的高点（不创新高）
- 位置可以在中枢上方/之中/下方（力度依次减弱）

### 第三类买卖点
- **第三类买点**：次级别离开中枢后，次级别回抽不破中枢上沿(ZG)
- **第三类卖点**：次级别离开中枢后，次级别回抽不破中枢下沿(ZD)

### 重要关系
- 第二类 + 第三类买点重合 → 最强走势，往往有一波大行情
- 三类买卖点是唯一被理论保证的 100% 安全买卖点

---

## 九、背驰-转折定理

**某级别趋势的背驰将导致：**
1. 该趋势最后一个中枢的**级别扩展**
2. 该级别**更大级别的盘整**
3. 该级别以上级别的**反趋势**

---

## 十、核心定理汇总

| 定理 | 内容 |
|------|------|
| 走势终完美 | 任何级别的任何走势类型终要完成 |
| 中枢定理一 | 趋势中连接两个中枢的必然是次级别以下走势 |
| 中枢定理二 | 盘整中离开和返回中枢的必然是次级别以下 |
| 中枢定理三 | 中枢被破坏 = 次级别离开 + 次级别回抽不返回中枢 |
| 买卖点完备性 | 只有第一、二、三类买卖点 |
| 线段分解定理 | 线段破坏的充要条件是至少被有重叠的连续 3 笔的其中一笔破坏 |
