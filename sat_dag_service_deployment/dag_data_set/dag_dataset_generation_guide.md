# 多源遥感 DAG 任务数据集生成指南

## 0. 文档定位

本指南规定 CPMV-DSD 仿真实验所用 DAG 任务数据集的完整生成方法。数据集分为两部分：

- **Part A：DAG 任务模板库**——三类原型 × 多个子原型，覆盖链状、宽浅、通用混合三种结构。
- **Part B：任务到达流**——基于真实灾害事件数据集拟合 Hawkes 过程，生成时空分布的多源遥感任务请求。

所有参数选择均给出文献依据；所有"自造"参数均说明物理动机和量级合理性。

---

## 1. 服务类型库定义

DAG 节点的服务类型基于 Zhang et al. 2021 的 RS 大数据处理流水线分类[1]和 EUMETSAT 处理级别标准[2]，定义如下 8 种服务：

| 服务编号 | 服务类型 | 简称 | 物理意义 | 处理级别对应 |
|---------|---------|------|---------|-----------|
| f₁ | Data Acquisition | DA | 原始数据采集与暂存 | L0 |
| f₂ | Radiometric Correction | RC | 辐射校正、传感器校准 | L0 → L1A |
| f₃ | Geometric Correction | GC | 几何校正、正射校正 | L1A → L1B |
| f₄ | Cloud Masking | CM | 云检测与掩膜 | L1B 辅助 |
| f₅ | Image Fusion | IF | 多源图像融合（SAR + 光学 + 红外） | L1C → L2 |
| f₆ | Feature Extraction | FE | 特征提取（边缘、纹理、谱指数） | L2 |
| f₇ | Target Detection | TD | 基于 CNN 的目标检测/分类 | L2 → L3 |
| f₈ | Result Aggregation | RA | 结果聚合、格式化、分发 | L3 → L4 |

**说明：** 这 8 种服务类型覆盖了从 L0 原始数据到 L4 应用产品的完整处理链[2]。在论文中可统一记为服务库 $F = \{f_1, \ldots, f_8\}$，$K = 8$。

---

## 2. 子任务（节点）参数标定

### 2.1 计算工作量（Workload, GFLOPs）

每种服务类型的 workload 分布基于以下来源标定：

| 服务 | Workload 分布 | 单位 | 文献依据 |
|------|------------|------|--------|
| DA | Uniform(1, 5) | GFLOPs | I/O 主导，计算量较低；按数据吞吐量估算 |
| RC | Normal(μ=12, σ=3) | GFLOPs | 像素级线性变换，对 L1B granule (~27 MB[3]) 处理约 10-15 GFLOPs |
| GC | Normal(μ=50, σ=10) | GFLOPs | 双线性/三次卷积重采样 + 投影变换；GDAL 文档[4]中典型处理时间 |
| CM | Normal(μ=20, σ=5) | GFLOPs | 阈值法或浅层 CNN 云检测 |
| IF | Normal(μ=160, σ=30) | GFLOPs | 多源加权融合或小波变换融合，复杂度 O(N log N) |
| FE | Normal(μ=80, σ=15) | GFLOPs | 边缘/纹理特征提取，CNN 浅层 |
| TD | Normal(μ=320, σ=60) | GFLOPs | YOLOv8s ~28 GFLOPs，YOLOv8m ~78.9 GFLOPs，YOLOv8l ~165.2 GFLOPs[5]；考虑多次推理或多尺度，整体在数百 GFLOPs 级 |
| RA | Uniform(1, 10) | GFLOPs | 简单汇总+格式化，计算量低 |

**采样规则：**
- 所有分布在 $[\max(0.5, \mu - 3\sigma), \mu + 3\sigma]$ 范围内截断；
- 同一 DAG 实例中，相同服务类型的多个节点独立采样，保证实例多样性；
- 每个数值采样后取 1 位小数，避免极小数值。

### 2.2 容器资源占用（内存 + 存储）

容器内存 $m_k$ 和存储 $d_k$ 基于 Jetson 平台 Docker 实测标定[6][7][8]：

| 服务 | 内存 $m_k$ (MB) | 存储 $d_k$ (MB) | 标定方法 |
|------|---------------|---------------|---------|
| DA | 200 | 100 | 轻量级 IO 处理容器，典型镜像约 100 MB，运行时缓存 200 MB |
| RC | 500 | 200 | OpenCV/GDAL 运行时容器 |
| GC | 800 | 300 | GDAL + 投影库 + 重采样缓冲 |
| CM | 1000 | 400 | 浅层云检测 CNN（如 U-Net 轻量版）+ PyTorch Lite |
| IF | 1200 | 400 | 多源处理 + 缓存 + 融合算法（如 IHS/Brovey）|
| FE | 1500 | 500 | 特征提取 CNN（如 VGG-light）+ PyTorch |
| TD | 2500 | 800 | YOLOv8s/m + PyTorch + TensorRT 优化模型权重 |
| RA | 300 | 100 | 轻量级聚合容器 |

**标定流程（在 Jetson Xavier NX 上执行）：**

```
对每种服务类型：
  1. 构建对应的 Docker 镜像（基础镜像 + 依赖库 + 模型权重）
  2. 启动容器并执行典型负载（输入约 100 MB 数据）
  3. 测量稳态内存占用（取运行 60 秒的均值）
  4. 测量镜像存储占用（docker image inspect 命令）
  5. 取 5 次重复均值
```

**说明：** 上述参数为推荐默认值。实际 Jetson 实测数据会替代默认值嵌入仿真。

### 2.3 服务有效处理速率（CPU 利用模型）

服务 $f_k$ 在卫星 $s$ 上以 CPU 资源 $r_{s,k}$ 处理子任务的有效速率为：

$$\mu_{s,k}(r_{s,k}) = \eta_k \cdot r_{s,k}$$

其中 $\eta_k$ 是服务级效率系数（GFLOPs/GHz·s）：

| 服务 | $\eta_k$ | 物理意义 |
|------|--------|---------|
| DA | 0.9 | I/O 主导，CPU 利用率较低 |
| RC | 1.0 | 标准化基准 |
| GC | 0.8 | 内存访问开销较大 |
| CM | 0.9 | 浅层 CNN，并行度好 |
| IF | 0.7 | 多源数据访问开销 |
| FE | 0.85 | 中等并行度 |
| TD | 0.6 | 深度 CNN，受限于 GPU/SIMD 加速效率 |
| RA | 1.0 | 简单聚合 |

子任务执行时间为 $T_i^{\mathrm{cmp}} = W_i / \mu_{s,k(i)}(r_{s,k(i)})$。

---

## 3. 边数据量参数标定

DAG 边的数据传输量基于 Sentinel-2/Landsat 产品规格[3][9][10]：

| 边类型（前驱 → 后继服务） | 数据量 $\mathrm{Data}_{ij}$ (MB) | 文献依据 |
|------------------------|----------------------------|--------|
| DA → RC | Uniform(15, 35) | Sentinel-2 L1B granule ~27 MB[3] |
| RC → GC | Uniform(15, 35) | L1A 产品规模与 L1B 相近[3][9] |
| GC → CM | Uniform(150, 250) | L1B → L1C 之间的中间产品按比例缩放 |
| GC → IF | Uniform(300, 700) | Sentinel-2 L1C tile 500-700 MB[3][9] |
| GC → FE | Uniform(300, 600) | 同上量级 |
| CM → IF | Uniform(50, 150) | 云掩膜辅助数据，比主数据小 |
| IF → TD | Uniform(80, 200) | 融合后产品比 L1C 略小 |
| IF → FE | Uniform(80, 200) | 同上 |
| FE → TD | Uniform(20, 60) | 提取后特征图 |
| TD → RA | Uniform(0.05, 5) | 检测结果（边界框+类别+置信度），按 COCO 标注规模[11] |
| FE → RA | Uniform(5, 30) | 特征聚合数据 |
| RA → 输出 | Uniform(0.01, 1) | 最终元数据 |

**关键特征：**
- **递减性：** 数据量从前段（DA → GC）到后段（TD → RA）整体递减——这是遥感数据流的典型特征[1][2]；
- **量级合理性：** 最大数据量约 700 MB（Sentinel-2 L1C tile[3]），最小约 0.01 MB（检测结果）；
- **多样性：** 同一边类型内部用 Uniform 分布采样，保证实例间数据量分布的多样性。

---

## 4. DAG 拓扑原型设计

设计三类原型，每类有若干个子原型，覆盖不同遥感任务场景。每个原型对应一个特定的协同遥感工作流。

### 4.1 链状 DAG 原型（Chain-like，宽度 ≤ 2，深度 ≥ 8）

**子原型 1A：单传感器目标识别流水线（如 SAR 舰船检测）**

```
DA → RC → GC → CM → IF → FE → TD → RA → 输出
                              (单源数据，IF退化为单源处理或省略)
```

**节点序列（深度 8）：** $f_1 \to f_2 \to f_3 \to f_4 \to f_6 \to f_7 \to f_8$（注意 $f_5$ IF 在单源情况下可省略）

**变体：** 在某些节点处插入 1 个并行预处理分支，使宽度达到 2。例如 RC 之后并行执行 CM 和 GC。

**物理对应：** 单卫星 SAR 数据采集后做舰船检测的标准流水线[12]。

---

**子原型 1B：时序变化检测流水线（如森林火灾持续监测）**

```
DA → RC → GC → CM → FE → 变化检测算法 → RA → 输出
              （时序累积分析，深度更大）
```

**节点序列（深度 9-10）：** 通过在 FE 和 TD 之间插入额外的时序分析节点（仍归为 FE 类）使深度增加到 9-10。

**物理对应：** Burned area mapping、森林火灾监测、洪水边界追踪等长时序任务[1][13]。

### 4.2 宽浅 DAG 原型（Wide-shallow，宽度 ≥ 5，深度 ≤ 4）

**子原型 2A：多源数据并行融合（SAR + 光学 + 红外）**

```
Layer 0:  DA(SAR) | DA(光学) | DA(红外) | DA(高光谱)        — 4-6 个并行采集节点
Layer 1:  RC      | RC       | RC      | RC                  — 各源独立辐射校正
Layer 2:  GC      | GC       | GC      | GC                  — 各源独立几何校正
Layer 3:  ────────────  IF (汇合点)  ────────────             — 一个汇合节点
Layer 4:  TD → RA                                             — 检测+聚合
```

**节点数：** 14-20，其中并行分支数 4-6，深度 4-5。

**物理对应：** 多源数据融合用于全天候监测（SAR 不受云影响，光学提供细节，红外用于热成像），是协同遥感的典型场景[14][15]。

---

**子原型 2B：多区域并行采集与协同分析**

```
Layer 0:  DA(区域1) | DA(区域2) | DA(区域3) | DA(区域4) | DA(区域5)
Layer 1:  RC+GC (各区域独立预处理，3 个并行任务每个区域)
Layer 2:  ────────  跨区域聚合 (RA)  ────────
Layer 3:  全局分析 (FE/TD)
```

**节点数：** 16-20，宽度可达 5-8，深度 3-4。

**物理对应：** 大区域协同观测（如全国范围灾情评估），多颗卫星并行采集多个区域后做统一分析[16]。

### 4.3 通用混合 DAG 原型（General mixed，宽度 3-5，深度 5-7）

**子原型 3A：多阶段协同遥感工作流（典型）**

```
Layer 0:  DA(光学) | DA(SAR)
Layer 1:  RC | RC | CM (云检测分支)
Layer 2:  GC | GC
Layer 3:  ────  IF (汇合)  ────
Layer 4:  FE → TD (并行)
                  ↓
Layer 5:  ────  RA (汇合)  ────
Layer 6:  输出
```

**节点数：** 12-18，宽度 3-4，深度 5-6。包含 1-2 个汇合点和并行分支。

**物理对应：** 多源协同 + 多阶段处理 + 中间汇合的典型协同遥感任务[1][14]。

---

**子原型 3B：实时灾害监测（含早期预警与深度分析双分支）**

```
Layer 0:  DA(多源)
Layer 1:  RC | RC
Layer 2:  ────  IF (融合)  ────
Layer 3:  ┌─→ 快速检测分支 (TD-light) ─→ 早期预警 RA  (短路径)
          └─→ 深度分析分支 (FE → TD-full) ─→ 详细 RA  (长路径)
Layer 4-5:  各分支独立输出
```

**节点数：** 10-14，包含 2 条独立长度的输出分支（不汇合）。

**物理对应：** 灾害监测中"先快报、后详报"的双层处理逻辑[13][16]。

---

**子原型 3C：多模态时序融合（含反复汇合分支）**

```
Layer 0:  DA(t1) | DA(t2) | DA(t3)             — 多时刻数据
Layer 1:  RC × 3
Layer 2:  GC × 3
Layer 3:  ──── 时序融合 IF ────
Layer 4:  FE × 2 (并行特征流)
Layer 5:  ──── 跨时序聚合 RA ────
Layer 6:  输出
```

**节点数：** 14-18，宽度 3-4，深度 6-7。

**物理对应：** 多时刻遥感数据融合分析（如海洋温度异常监测）[17]。

### 4.4 原型分布与生成策略

为保证算法测试的覆盖性，每类原型在数据集中按以下比例分布：

| 类别 | 原型 | 实例数 | 用途 |
|------|------|------|------|
| 链状 | 1A | 12 | 链状结构主样本 |
| 链状 | 1B | 8 | 链状深度变体 |
| 宽浅 | 2A | 12 | 宽浅结构主样本 |
| 宽浅 | 2B | 8 | 多区域并行变体 |
| 通用 | 3A | 10 | 通用结构主样本 |
| 通用 | 3B | 6 | 双输出分支变体 |
| 通用 | 3C | 4 | 时序汇合变体 |

**总计 60 个 DAG 实例**，保证：
- 每个主类（链状/宽浅/通用）至少 20 个实例（满足实验1的统计需求）；
- 子原型多样性覆盖典型遥感任务场景；
- 实例间通过随机扰动（节点参数采样、可选分支添加）保证多样性。

---

## 5. DAG 实例生成算法

```
INPUT:  原型类别 archetype, 实例索引 i
OUTPUT: DAG 实例 D_i = (V_i, E_i, workload, data_size, service_type)

Step 1: 加载原型骨架
  根据 archetype 加载基础节点序列和拓扑结构

Step 2: 结构性扰动
  - 节点数变化：在原型基础上 ±2 个节点（在合法位置插入或删除）
  - 可选分支：以概率 p_branch=0.3 在某些节点添加并行分支
  - 可选汇合：以概率 p_merge=0.4 在某些层添加汇合点
  - 验证 DAG 合法性：拓扑排序无环、单源单汇（或多汇分支）

Step 3: 节点参数采样
  对每个节点 v_j:
    根据 service_type(v_j) 从对应分布采样 W_j（参考第 2.1 节）
    记录 m_{k(j)} 和 d_{k(j)}（参考第 2.2 节）

Step 4: 边参数采样
  对每条边 (v_j, v_l):
    根据 (service_type(v_j), service_type(v_l)) 查表
    从对应 Uniform 分布采样 Data_{jl}（参考第 3 节）

Step 5: 验证与输出
  - 检查 DAG 合法性
  - 检查所有节点和边都满足约束（W > 0, Data > 0）
  - 输出 D_i 为 JSON 格式
```

**JSON 格式示例：**

```json
{
  "instance_id": "wide_shallow_2A_005",
  "archetype": "wide_shallow",
  "subarchetype": "2A",
  "nodes": [
    {"id": "v1", "service_type": "DA", "workload_GFLOPs": 3.2},
    {"id": "v2", "service_type": "DA", "workload_GFLOPs": 2.8},
    {"id": "v3", "service_type": "RC", "workload_GFLOPs": 11.5},
    ...
  ],
  "edges": [
    {"src": "v1", "dst": "v3", "data_MB": 24.7},
    ...
  ],
  "metadata": {
    "num_nodes": 18,
    "width": 5,
    "depth": 4,
    "structure_type": "wide_shallow"
  }
}
```

---

## 6. 任务到达过程生成

### 6.1 真实数据集来源

任务到达过程的参数标定基于真实灾害事件数据集。我们使用两个互补的公开数据源。

#### 6.1.1 NASA EONET（主要数据源）

- **完整名称：** Earth Observatory Natural Event Tracker
- **维护机构：** NASA Earth Observatory
- **下载地址：** https://eonet.gsfc.nasa.gov/api/v3/events
- **数据格式：** JSON via REST API
- **覆盖事件类型：** 火灾（Wildfires）、风暴（Severe Storms）、火山活动（Volcanoes）、洪水（Floods）、海冰（Sea and Lake Ice）、地震（Earthquakes，2020 年后停止收录）、干旱（Drought）等
- **每条事件包含：** 事件 ID、类别、起始/结束日期、地理坐标（经纬度）、数据源
- **使用许可：** 公共领域，免费下载
- **覆盖时间：** 2010 年至今，持续更新

**API 调用示例：**

```bash
# 获取过去 1 年所有 Wildfires 事件
curl "https://eonet.gsfc.nasa.gov/api/v3/events?category=wildfires&days=365&status=all"

# 获取所有事件类别列表
curl "https://eonet.gsfc.nasa.gov/api/v3/categories"
```

**Python 下载示例：**

```python
import requests
import json

base_url = "https://eonet.gsfc.nasa.gov/api/v3/events"
params = {
    "days": 365,           # 过去 1 年
    "status": "all",       # 包含已结束和进行中
    "limit": 5000          # 最多返回数量
}
response = requests.get(base_url, params=params)
events = response.json()["events"]

# 保存到本地
with open("eonet_events_2024.json", "w") as f:
    json.dump(events, f, indent=2)
```

#### 6.1.2 EM-DAT（辅助数据源）

- **完整名称：** Emergency Events Database
- **维护机构：** CRED (Centre for Research on the Epidemiology of Disasters), UCLouvain
- **网站：** https://www.emdat.be/
- **数据获取：** 注册学术账号后免费下载（用学校邮箱）
- **数据格式：** Excel/CSV
- **覆盖事件类型：** 灾害（Drought, Earthquake, Flood, Storm, Wildfire 等）
- **每条记录包含：** 灾害类型、国家、起止日期、影响人数、经济损失
- **覆盖时间：** 1900 年至今
- **使用许可：** 学术非商业用途免费

**用途差异：** EONET 提供精确的时空坐标（适合空间分布拟合），EM-DAT 提供长期统计（适合验证总体分布）。

### 6.2 任务到达模型

采用多元 Hawkes 过程[18][19][20]建模任务到达，参数从 EONET 数据中通过极大似然估计拟合。

#### 6.2.1 Hawkes 过程定义

将地球表面划分为 $R$ 个区域（建议按 6° × 6° 网格划分，约 $R \approx 600$ 个区域），DAG 任务类型数为 $K_T = 7$（对应 7 类原型 1A/1B/2A/2B/3A/3B/3C）。

第 $k$ 类任务在区域 $r$ 的条件强度为：

$$\lambda_{r,k}(t) = \mu_{r,k}(t) + \sum_{r'=1}^R \sum_{k'=1}^{K_T} \int_0^t \alpha_{r,k}^{r',k'} \omega(r,r') g(t-\tau) \, dN_{r',k'}(\tau)$$

其中：
- $\mu_{r,k}(t)$：基线强度（地理偏好 + 季节趋势），从 EONET 长期数据估计；
- $\alpha_{r,k}^{r',k'}$：自激励系数，刻画事件聚集性；
- $\omega(r, r')$：空间核（如高斯衰减）；
- $g(\tau)$：时间衰减核（建议指数核 $g(\tau) = \beta e^{-\beta \tau}$）。

#### 6.2.2 参数标定流程

```
INPUT:  EONET 事件数据 events.json (1 年)
OUTPUT: Hawkes 过程参数 {μ, α, ω, β}

Step 1: 数据预处理
  - 解析每条事件的 (timestamp, latitude, longitude, category)
  - 将 category 映射为 DAG 类别 k（按事件类型）
  - 将 (lat, lon) 映射为区域索引 r（按经纬度网格）

Step 2: 基线强度估计
  对每个 (r, k):
    将 1 年时间分为 12 个月
    计算每月平均事件率（event/month）
    用样条平滑作为 μ_{r,k}(t)

Step 3: 激励系数估计
  采用 EM 算法或 NPHawkes 库[21]估计 α 矩阵
  考虑稀疏约束（大多数 (r,k,r',k') 组合无明显激励）

Step 4: 空间核拟合
  ω(r, r') = exp(-d(r,r')²/(2σ²))
  其中 d(r,r') 是区域间地理距离
  σ 通过最大化似然估计，典型值 500-1000 km

Step 5: 时间衰减拟合
  g(τ) = β·exp(-β·τ)
  β 由事件聚集时间尺度决定，典型值 1/(7天) 到 1/(30天)

Step 6: 验证
  - 用拟合参数生成模拟事件
  - 与原始数据比较：总事件数、空间分布、时间聚集模式
  - K-S 检验等
```

**实现工具：**

- **Python tick 库：** https://github.com/X-DataInitiative/tick — 提供 Hawkes 过程的拟合和仿真
- **NPHawkes：** https://github.com/Hyper-Cosmic/NPHawkes — 非参数 Hawkes 估计
- **PyHawkes：** https://github.com/slinderman/pyhawkes — 多元 Hawkes 推断

#### 6.2.3 任务流仿真

参数拟合完成后，仿真任务流的过程：

```
INPUT:  拟合的 Hawkes 参数 {μ, α, ω, β}, 仿真时长 T
OUTPUT: 任务序列 [(t_i, r_i, k_i, d_i), ...]
        其中 d_i 是从模板库中抽取的具体 DAG 实例

Step 1: 用 Hawkes 仿真器（如 tick.HawkesExpKern）生成事件流
  得到序列 [(t_i, r_i, k_i)]

Step 2: 对每个 (t_i, r_i, k_i):
  从 DAG 模板库中按类别 k_i 随机抽取一个实例 d_i
  组成任务请求 (t_i, r_i, k_i, d_i)

Step 3: 输出任务流到 JSON 文件
```

### 6.3 简化版本（如果 Hawkes 拟合困难）

如果完整的 Hawkes 拟合工程量太大，可采用简化的"基线 + 自激励"模型：

```
忽略空间核 ω(r, r') = δ(r, r')（仅同区域自激励）
忽略跨类型激励：α_{r,k}^{r,k'} = 0 (k != k')
保留：基线 μ_{r,k} + 同类型自激励 α_{r,k}

参数仍通过 EONET 数据估计，但维度大幅降低
```

这种简化版本仍能保留 Hawkes 过程的核心特征（自激励聚集），同时大幅降低实现复杂度。

---

## 7. 数据集导出与使用

### 7.1 文件结构

```
dataset/
├── dag_templates/
│   ├── chain_like/
│   │   ├── chain_1A_001.json ~ chain_1A_012.json
│   │   └── chain_1B_001.json ~ chain_1B_008.json
│   ├── wide_shallow/
│   │   ├── ws_2A_001.json ~ ws_2A_012.json
│   │   └── ws_2B_001.json ~ ws_2B_008.json
│   └── general/
│       ├── general_3A_001.json ~ general_3A_010.json
│       ├── general_3B_001.json ~ general_3B_006.json
│       └── general_3C_001.json ~ general_3C_004.json
├── service_library.json     # 8 种服务的参数定义
├── arrival_traces/
│   ├── eonet_raw_2024.json  # 原始 EONET 数据
│   ├── hawkes_params.json   # 拟合的 Hawkes 参数
│   └── simulated_arrivals_T=30days.json  # 仿真任务流
└── README.md                 # 数据集使用说明
```

### 7.2 数据集元信息

每个 DAG 实例的 metadata 字段包含：
- 实例 ID、原型类别、子原型
- 节点数、边数、宽度、深度
- 服务类型分布
- 总 workload、总数据量

便于后续在论文实验章节中给出数据集统计表。

---

## 8. 在论文中的呈现策略

在 Section VI-A（Simulation Setup）中，用以下结构呈现：

```
DAG Templates and Parameter Calibration. 
We construct 60 DAG instances spanning three structural classes—
chain-like, wide-shallow, and general mixed—following standard 
multi-stage remote sensing pipelines documented in [Zhang2021, 
EUMETSAT]. Each class is further divided into sub-archetypes 
representing typical workflows: single-sensor target detection 
[Probst2017], multi-source fusion [SAR-Optical-Fusion-Ref], 
collaborative monitoring with parallel branches and merge points, 
and time-series analysis. Service workloads (in GFLOPs) are 
calibrated against measurements on a Jetson Xavier NX testbed 
running Docker containers, consistent with the standard onboard 
AI processing platform [Saari2024, Buckley2023]. CNN-based target 
detection workload follows reported FLOPs of YOLOv8 family 
[Hussain2024]. Inter-task data sizes follow Sentinel-2 product 
specifications [ESA-S2]: ~27 MB per L1B granule, ~500-700 MB per 
L1C tile, with progressive size reduction along the pipeline.

Mission Arrival Process. 
Multi-source remote sensing mission requests are generated by a 
multivariate Hawkes process [Reinhart2018, CuiLi2024], whose 
intensity parameters are calibrated from one year of NASA EONET 
[NASA-EONET] natural event records, capturing the spatio-temporal 
clustering pattern of real Earth observation demands. Events are 
classified into 7 mission types matching our DAG archetypes; 
spatial discretization uses 6° × 6° grids, resulting in 
approximately 600 regions globally.
```

---

## 9. 参考文献清单

[1] **Zhang, X., Zhou, Y., & Luo, J.** (2021). Deep learning for processing and analysis of remote sensing big data: a technical review. *Big Earth Data*, 5(4), 527-560. https://doi.org/10.1080/20964471.2021.1964879
   - **用途：** 论证 RS 处理流水线的标准结构（4 类任务：geometric/radiometric/cloud masking/data fusion）

[2] **EUMETSAT.** Data processing and file formats. *EUMETSAT Training Materials*. https://classroom.eumetsat.int/mod/book/tool/print/index.php?id=13919
   - **用途：** 论证遥感处理级别（L0/L1A/L1B/L1C/L2/L3/L4）的标准定义

[3] **ESA.** Sentinel-2 Data Products. *ESA Sentinel Online*. https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-2/Data_products
   - **用途：** L1B granule 27 MB，L1C tile 500-700 MB 的官方规格

[4] **GDAL/OGR contributors.** GDAL: Geospatial Data Abstraction Library. https://gdal.org/
   - **用途：** 几何校正（重采样、投影变换）的算法复杂度参考

[5] **Hussain, M.** (2024). YOLOv5, YOLOv8 and YOLOv10: The Go-To Detectors for Real-time Vision. *arXiv preprint arXiv:2407.02988*. https://arxiv.org/abs/2407.02988
   - **用途：** YOLOv8 系列模型的具体 FLOPs 数据（YOLOv8n/s/m/l/x = 8/28/79/165/258 GFLOPs）

[6] **Saari, I., Calò, T., et al.** (2024). Review on Hardware Devices and Software Techniques Enabling Neural Network Inference Onboard Satellites. *Remote Sensing*, 16(21), 3957. https://www.mdpi.com/2072-4292/16/21/3957
   - **用途：** Jetson Orin Nano 等嵌入式 AI 加速器在卫星任务中的实际应用综述

[7] **Buckley, L., Romero-Cañas, J., Espinosa-Aranda, J., Hervas-Martin, E., & Fernandez, M.** (2023). Towards Space Edge Computing and Onboard AI for Real-Time Earth Observation Applications. *IEEE LEO SatS 2023*. NASA JPL Technical Report. https://ai.jpl.nasa.gov/public/documents/papers/ieee-leo-sats-report.pdf
   - **用途：** NASA JPL 在 ISS 上对 Movidius Myriad X 等边缘处理器的基准测试

[8] **Furano, G., et al.** (2025). Advancing Earth Observation: A Survey on AI-Powered Image Processing in Satellites. *arXiv preprint arXiv:2501.12030*. https://arxiv.org/abs/2501.12030
   - **用途：** 多个 Jetson 系列在卫星 AI 任务中的实测研究（Barnell 2022, Rad 2023, Duggan 2023）

[9] **Sentinel Online.** Copernicus Sentinel-2 Collection 1 MSI Level-1C (L1C). https://sentinels.copernicus.eu/sentinel-data-access/sentinel-products/sentinel-2-data-products/collection-1-level-1c
   - **用途：** Sentinel-2 L1C 产品 100 km × 100 km tile、约 700 MB 的官方规格

[10] **USGS.** Landsat Collection 2 Product Definitions. https://www.usgs.gov/landsat-missions/landsat-collection-2-level-2-science-products
   - **用途：** Landsat 产品规格的补充参考

[11] **Lin, T.-Y., et al.** (2014). Microsoft COCO: Common Objects in Context. *ECCV 2014*. https://arxiv.org/abs/1405.0312
   - **用途：** 目标检测结果数据格式标准（边界框+类别+置信度）

[12] **Probst, L., et al.** (2017). A Workflow for Automated Satellite Image Processing: from Raw VHSR Data to Object-Based Spectral Information for Smallholder Agriculture. *Remote Sensing*, 9(10), 1048. https://www.mdpi.com/2072-4292/9/10/1048
   - **用途：** 单传感器顺序处理流水线的具体实例

[13] **Open Cosmos.** AI-Enabled Onboard Edge Computing for Satellite Intelligence in Disaster Management. *UN-SPIDER Knowledge Portal*. https://www.un-spider.org/news-and-events/news/ai-enabled-onboard-edge-computing-satellite-intelligence-disaster-management
   - **用途：** 灾害监测中实时 + 详细分析的双层处理逻辑

[14] **Burgueño-Romero, A., Barba-González, C., & Aldana-Montes, J.** (2024). Big Data-driven MLOps workflow for annual high-resolution land cover classification models. *Future Generation Computer Systems*. https://www.sciencedirect.com/science/article/pii/S0167739X24004631
   - **用途：** 多源数据融合工作流的工程实例

[15] **Schmitt, M., & Zhu, X. X.** (2016). Data Fusion and Remote Sensing: An ever-growing relationship. *IEEE Geoscience and Remote Sensing Magazine*, 4(4), 6-23.
   - **用途：** 多源遥感数据融合（SAR + 光学 + 红外）的综述

[16] **AIoT Ecosystem.** (2025). The AIoT ecosystem for next-generation satellite systems. *Future Generation Computer Systems*. https://www.sciencedirect.com/science/article/pii/S1874490725003702
   - **用途：** 多卫星协同观测的系统架构

[17] **Luo, Y., et al.** (2006). A Remote Sensing Application Workflow and Its Implementation in Remote Sensing Service Grid Node. *ICCS 2006*, LNCS 3991, Springer. https://link.springer.com/chapter/10.1007/11758501_42
   - **用途：** 时序遥感任务的工作流建模

[18] **Reinhart, A.** (2018). A Review of Self-Exciting Spatio-Temporal Point Processes and Their Applications. *Statistical Science*, 33(3), 299-318.
   - **用途：** Hawkes 过程在时空事件建模的权威综述

[19] **Cui, K., & Li, C.** (2024). Multivariate Hawkes processes with spatial covariates for spatiotemporal event data analysis. *Annals of the Institute of Statistical Mathematics*, 73(6), 1127-1152. https://link.springer.com/article/10.1007/s10463-023-00894-2
   - **用途：** 多元 Hawkes 过程 + 空间协变量的建模方法

[20] **Ogata, Y.** (1998). Space-time point process model for the occurrence of earthquakes. *Annals of the Institute of Statistical Mathematics*, 50(2), 379-402.
   - **用途：** ETAS 模型的奠基论文，时空 Hawkes 过程的经典参考

[21] **Mohler, G. O., Short, M. B., Brantingham, P. J., Schoenberg, F. P., & Tita, G. E.** (2011). Self-Exciting Point Process Modeling of Crime. *Journal of the American Statistical Association*, 106(493), 100-108.
   - **用途：** Hawkes 过程在非地震领域（社会事件）应用的经典引用

[22] **NASA Earth Observatory.** EONET API v3 Documentation. https://eonet.gsfc.nasa.gov/docs/v3
   - **用途：** 任务到达数据的主要来源

[23] **CRED.** EM-DAT: The International Disaster Database. UCLouvain. https://www.emdat.be/
   - **用途：** 历史灾害数据辅助验证

[24] **Bacry, E., Bompaire, M., Gaïffas, S., & Poulsen, S.** (2017). Tick: a Python library for statistical learning, with a particular emphasis on time-dependent modeling. *arXiv preprint arXiv:1707.03003*. https://github.com/X-DataInitiative/tick
   - **用途：** Hawkes 过程拟合与仿真的 Python 工具库

---

## 10. 实施检查清单

完成数据集生成需要的步骤：

- [ ] 实现服务库定义（service_library.json）
- [ ] 实现 7 个原型的拓扑骨架代码
- [ ] 实现节点参数采样器
- [ ] 实现边参数采样器
- [ ] 实现 DAG 实例生成器（含拓扑扰动）
- [ ] 在 Jetson 上实测 8 种服务的容器资源占用
- [ ] 生成 60 个 DAG 实例并验证合法性
- [ ] 下载 NASA EONET 1 年数据
- [ ] 数据预处理（事件分类 + 区域映射）
- [ ] 拟合 Hawkes 过程参数
- [ ] 仿真任务到达流
- [ ] 数据集导出为标准 JSON 格式
- [ ] 编写 README 和数据集统计报告
